#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dorigo_variant_budget_cluster.py

Goal
----
Run a laptop-friendly experiment to test the claim that six metaheuristics
(GWO, MFA, WOA, FA, BA, ALO) fall into two behavioural groups (e.g., "PSO-like" vs "ES-like")
when you look at performance profiles across instances and evaluation budgets.

What it does
------------
1) Runs each algorithm multiple times (seeds) on a small continuous benchmark suite
   under multiple evaluation budgets (number of function evaluations).
2) Aggregates to an instance_algo_budget_summary.csv with the same key columns you used before
   (successes/trials, beta_mean, beta_p05/p95, mean_best, etc.).
3) Builds success-profile vectors per (budget, target tier) and clusters algorithms (ABS and REL views).
4) Outputs:
   - out_dir/runs_detail.csv
   - out_dir/instance_algo_budget_summary.csv
   - out_dir/analysis/cluster_assignments.csv
   - out_dir/analysis/quick_kpi.txt
   - out_dir/figs/*.png (budget curves, heatmap, MDS projections, dendrograms)
   - out_dir/tables/*.tex (LaTeX-ready tables)

Dependencies
------------
pip install numpy pandas matplotlib scipy

Usage (example)
---------------
python dorigo_variant_budget_cluster.py --out_dir out_dorigo \
  --budgets 300,800,2000 --instances 40 --dim 10 --seeds 20 --pop 30 --n_jobs 1

Notes
-----
- This is a research scaffold. For a final paper, you should validate hyperparameters against
  original papers and/or run small tuning sweeps.
- The benchmark suite here is intentionally small and shift-based to create many instances cheaply.
"""

from __future__ import annotations
import argparse
import math
import os
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# plotting (headless safe)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# optional scipy imports (used for clustering + beta quantiles)
try:
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import beta as sp_beta
except Exception as e:
    raise SystemExit(
        "This script requires scipy. Install it with: pip install scipy\n"
        f"Import error: {e}"
    )


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def clip_bounds(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)

def rng_from(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def stable_hash_mod(s: str, mod: int) -> int:
    """Deterministic hash for reproducible seeds across Python runs."""
    h = zlib.adler32(s.encode('utf-8')) & 0xffffffff
    return int(h % mod)

def logit_clip(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

def zscore_across_algos_for_each_instance(mat: np.ndarray) -> np.ndarray:
    """
    mat: shape (n_algos, n_instances)
    For each instance column, z-score across algorithms.
    """
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (mat - mu) / sd

def classical_mds(D: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Classical (Torgerson) MDS from a distance matrix.
    Returns coordinates (n, n_components).
    """
    n = D.shape[0]
    # double centering
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    # eigen
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    vals = np.maximum(vals, 0.0)
    X = vecs[:, :n_components] * np.sqrt(vals[:n_components])
    return X


# -----------------------------
# Benchmark suite
# -----------------------------

def sphere(z: np.ndarray) -> float:
    return float(np.sum(z * z))

def rastrigin(z: np.ndarray) -> float:
    A = 10.0
    return float(A * z.size + np.sum(z * z - A * np.cos(2 * np.pi * z)))

def ackley(z: np.ndarray) -> float:
    a, b, c = 20.0, 0.2, 2 * np.pi
    n = z.size
    s1 = np.sum(z * z)
    s2 = np.sum(np.cos(c * z))
    term1 = -a * np.exp(-b * np.sqrt(s1 / n))
    term2 = -np.exp(s2 / n)
    return float(term1 + term2 + a + math.e)

def griewank(z: np.ndarray) -> float:
    sum_ = np.sum(z * z) / 4000.0
    prod_ = np.prod(np.cos(z / np.sqrt(np.arange(1, z.size + 1))))
    return float(sum_ - prod_ + 1.0)

def rosenbrock(z: np.ndarray) -> float:
    # classic Rosenbrock min 0 at all-ones
    x = z
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))

@dataclass(frozen=True)
class ProblemSpec:
    name: str
    func: Callable[[np.ndarray], float]
    bounds: Tuple[float, float]
    difficulty: str  # easy/med/hard tag (instance tier)
    # shift seed used to generate per-instance shift vector

def make_suite() -> List[ProblemSpec]:
    """
    Small suite with known global optimum 0 after shifting and (for Rosenbrock) re-centering.
    Difficulty tag is used only for reporting; success thresholds are controlled separately.
    """
    return [
        ProblemSpec("sphere", sphere, (-5.0, 5.0), "easy"),
        ProblemSpec("rastrigin", rastrigin, (-5.12, 5.12), "med"),
        ProblemSpec("ackley", ackley, (-5.0, 5.0), "med"),
        ProblemSpec("griewank", griewank, (-600.0, 600.0), "hard"),
        ProblemSpec("rosenbrock", rosenbrock, (-5.0, 10.0), "hard"),
    ]

def make_instance_shift(dim: int, bounds: Tuple[float, float], seed: int) -> np.ndarray:
    lo, hi = bounds
    rg = rng_from(seed)
    return rg.uniform(lo * 0.25, hi * 0.25, size=dim)

def objective_factory(spec: ProblemSpec, dim: int, shift: np.ndarray) -> Callable[[np.ndarray], float]:
    """
    Returns f(x) with known optimum f* = 0 (for our shifted definitions).
    """
    lo, hi = spec.bounds

    if spec.name == "rosenbrock":
        # optimum at x = shift + 1
        def f(x: np.ndarray) -> float:
            z = x - shift
            return spec.func(z)  # spec.func is rosenbrock; min 0 at z=1
    else:
        def f(x: np.ndarray) -> float:
            z = x - shift
            return spec.func(z)  # min 0 at z=0

    return f


# -----------------------------
# Metaheuristics (continuous)
# -----------------------------

def evaluate_population(f: Callable[[np.ndarray], float], X: np.ndarray) -> np.ndarray:
    return np.array([f(x) for x in X], dtype=float)

def run_gwo(f, dim, bounds, budget, seed, pop=30):
    """Grey Wolf Optimizer (basic reference implementation)."""
    rg = rng_from(seed)
    lo, hi = bounds
    # init wolves
    X = rg.uniform(lo, hi, size=(pop, dim))
    fit = evaluate_population(f, X)
    evals = pop

    def top3(X, fit):
        idx = np.argsort(fit)
        return idx[0], idx[1], idx[2]

    ia, ib, idd = top3(X, fit)
    Xa, Xb, Xd = X[ia].copy(), X[ib].copy(), X[idd].copy()
    fa, fb, fd = fit[ia], fit[ib], fit[idd]

    t = 0
    # each iteration evaluates pop new positions
    while evals + pop <= budget:
        a = 2.0 * (1.0 - (t / max(1, (budget / pop))))
        # update each wolf
        Xnew = np.empty_like(X)
        for i in range(pop):
            r1 = rg.random(dim); r2 = rg.random(dim)
            A1 = 2*a*r1 - a
            C1 = 2*r2
            D_alpha = np.abs(C1*Xa - X[i])
            X1 = Xa - A1*D_alpha

            r1 = rg.random(dim); r2 = rg.random(dim)
            A2 = 2*a*r1 - a
            C2 = 2*r2
            D_beta = np.abs(C2*Xb - X[i])
            X2 = Xb - A2*D_beta

            r1 = rg.random(dim); r2 = rg.random(dim)
            A3 = 2*a*r1 - a
            C3 = 2*r2
            D_delta = np.abs(C3*Xd - X[i])
            X3 = Xd - A3*D_delta

            Xnew[i] = (X1 + X2 + X3) / 3.0

        Xnew = clip_bounds(Xnew, lo, hi)
        fit_new = evaluate_population(f, Xnew)
        evals += pop
        X, fit = Xnew, fit_new

        ia, ib, idd = top3(X, fit)
        if fit[ia] < fa:
            Xa, fa = X[ia].copy(), fit[ia]
        if fit[ib] < fb:
            Xb, fb = X[ib].copy(), fit[ib]
        if fit[idd] < fd:
            Xd, fd = X[idd].copy(), fit[idd]

        t += 1

    best = float(np.min(fit))
    return best, evals

def run_woa(f, dim, bounds, budget, seed, pop=30):
    """Whale Optimization Algorithm (basic reference implementation)."""
    rg = rng_from(seed)
    lo, hi = bounds
    X = rg.uniform(lo, hi, size=(pop, dim))
    fit = evaluate_population(f, X)
    evals = pop
    ibest = int(np.argmin(fit))
    Xbest = X[ibest].copy()
    fbest = float(fit[ibest])

    t = 0
    while evals + pop <= budget:
        a = 2.0 * (1.0 - (t / max(1, (budget / pop))))
        a2 = -1.0 + (t / max(1, (budget / pop))) * (-1.0)  # for spiral
        Xnew = np.empty_like(X)
        for i in range(pop):
            p = rg.random()
            r1 = rg.random(dim)
            r2 = rg.random(dim)
            A = 2*a*r1 - a
            C = 2*r2
            if p < 0.5:
                if np.linalg.norm(A, ord=2) < 1.0:
                    D = np.abs(C*Xbest - X[i])
                    Xnew[i] = Xbest - A*D
                else:
                    j = rg.integers(0, pop)
                    Xrand = X[j]
                    D = np.abs(C*Xrand - X[i])
                    Xnew[i] = Xrand - A*D
            else:
                # spiral update
                D = np.abs(Xbest - X[i])
                l = (a2 - 1.0) * rg.random() + 1.0
                b = 1.0
                Xnew[i] = D * np.exp(b*l) * np.cos(2*np.pi*l) + Xbest

        Xnew = clip_bounds(Xnew, lo, hi)
        fit_new = evaluate_population(f, Xnew)
        evals += pop
        X, fit = Xnew, fit_new
        ibest = int(np.argmin(fit))
        if float(fit[ibest]) < fbest:
            Xbest = X[ibest].copy()
            fbest = float(fit[ibest])
        t += 1

    return fbest, evals

def run_mfo(f, dim, bounds, budget, seed, pop=30):
    """Moth-Flame Optimization (basic reference implementation)."""
    rg = rng_from(seed)
    lo, hi = bounds
    M = rg.uniform(lo, hi, size=(pop, dim))
    fit = evaluate_population(f, M)
    evals = pop

    # flames are sorted best solutions
    flames = M.copy()
    flames_fit = fit.copy()

    t = 0
    max_iter = max(1, budget // pop)
    while evals + pop <= budget:
        # sort moths to update flames
        idx = np.argsort(fit)
        M = M[idx]; fit = fit[idx]
        # update flames: best positions so far
        all_X = np.vstack([flames, M])
        all_fit = np.concatenate([flames_fit, fit])
        idx2 = np.argsort(all_fit)
        flames = all_X[idx2][:pop].copy()
        flames_fit = all_fit[idx2][:pop].copy()

        # number of flames decreases linearly
        flame_no = int(round(pop - t * ((pop - 1) / max_iter)))
        flame_no = max(1, min(pop, flame_no))

        b = 1.0
        a = -1.0 + t * (-1.0 / max_iter)  # a in [-1,-2], used in original MFO
        Mnew = np.empty_like(M)
        for i in range(pop):
            flame_idx = i if i < flame_no else flame_no - 1
            F = flames[flame_idx]
            D = np.abs(F - M[i])
            l = (a - 1.0) * rg.random() + 1.0
            Mnew[i] = D * np.exp(b*l) * np.cos(2*np.pi*l) + F

        Mnew = clip_bounds(Mnew, lo, hi)
        fit_new = evaluate_population(f, Mnew)
        evals += pop
        M, fit = Mnew, fit_new
        t += 1

    return float(np.min(fit)), evals

def run_ba(f, dim, bounds, budget, seed, pop=30):
    """Bat Algorithm (basic reference implementation)."""
    rg = rng_from(seed)
    lo, hi = bounds
    X = rg.uniform(lo, hi, size=(pop, dim))
    V = np.zeros((pop, dim), dtype=float)
    fit = evaluate_population(f, X)
    evals = pop

    ibest = int(np.argmin(fit))
    best = X[ibest].copy()
    fbest = float(fit[ibest])

    # parameters (common defaults)
    fmin, fmax = 0.0, 2.0
    A = np.full(pop, 0.9)  # loudness
    r = np.full(pop, 0.5)  # pulse rate

    while evals + pop <= budget:
        for i in range(pop):
            freq = fmin + (fmax - fmin) * rg.random()
            V[i] = V[i] + (X[i] - best) * freq
            S = X[i] + V[i]
            S = clip_bounds(S, lo, hi)

            # local search
            if rg.random() > r[i]:
                eps = rg.normal(0.0, 1.0, size=dim)
                S = best + 0.001 * eps
                S = clip_bounds(S, lo, hi)

            fS = f(S)
            evals += 1
            if (fS <= fit[i]) and (rg.random() < A[i]):
                X[i] = S
                fit[i] = fS
                A[i] *= 0.99
                r[i] = r[i] * (1.0 - math.exp(-0.01))

            if fS < fbest:
                best = S.copy()
                fbest = float(fS)

            if evals >= budget:
                break

        if evals >= budget:
            break

    return fbest, evals

def run_fa(f, dim, bounds, budget, seed, pop=30, k_neighbors=3):
    """
    Firefly Algorithm (budget-friendly approximation):
    - Standard FA is O(pop^2) evaluations/iteration if you fully compare all pairs.
    - Here we limit each firefly to move toward up to k_neighbors randomly selected better fireflies.
    """
    rg = rng_from(seed)
    lo, hi = bounds
    X = rg.uniform(lo, hi, size=(pop, dim))
    fit = evaluate_population(f, X)
    evals = pop

    beta0 = 1.0
    gamma = 1.0 / max(1e-9, (hi - lo) ** 2)
    alpha = 0.25 * (hi - lo)

    while evals + pop <= budget:
        # sort by brightness (lower is better for minimization)
        idx = np.argsort(fit)
        Xs = X[idx].copy()
        fs = fit[idx].copy()

        Xnew = Xs.copy()
        fnew = fs.copy()

        for i in range(pop):
            # candidates better than i are [0..i-1]
            if i == 0:
                # best firefly random walk
                step = alpha * rg.normal(0.0, 1.0, size=dim)
                cand = Xs[i] + step
                cand = clip_bounds(cand, lo, hi)
                fc = f(cand); evals += 1
                if fc < fnew[i]:
                    Xnew[i] = cand; fnew[i] = fc
                if evals >= budget:
                    break
                continue

            better_pool = np.arange(0, i)
            if better_pool.size == 0:
                continue
            chosen = rg.choice(better_pool, size=min(k_neighbors, better_pool.size), replace=False)
            xi = Xs[i].copy()
            fi = fnew[i]
            for j in chosen:
                xj = Xs[j]
                rij = np.linalg.norm(xi - xj)
                beta = beta0 * math.exp(-gamma * rij * rij)
                step = beta * (xj - xi) + alpha * rg.normal(0.0, 1.0, size=dim)
                cand = xi + step
                cand = clip_bounds(cand, lo, hi)
                fc = f(cand); evals += 1
                if fc < fi:
                    xi, fi = cand, fc
                if evals >= budget:
                    break
            Xnew[i] = xi; fnew[i] = fi
            if evals >= budget:
                break

        # cool alpha slowly
        alpha *= 0.98
        X, fit = Xnew, fnew
        if evals >= budget:
            break

    return float(np.min(fit)), evals

def run_alo(f, dim, bounds, budget, seed, pop=30):
    """
    Ant Lion Optimizer (simplified, evaluation-budget exact):
    - Each iteration evaluates pop ants.
    - Random walk is generated via cumulative sums and then normalized into bounds.
    """
    rg = rng_from(seed)
    lo, hi = bounds

    # initialize antlions and ants
    AL = rg.uniform(lo, hi, size=(pop, dim))
    ants = rg.uniform(lo, hi, size=(pop, dim))
    fit_AL = evaluate_population(f, AL)
    evals = pop
    fit_ants = evaluate_population(f, ants)
    evals += pop

    elite_idx = int(np.argmin(fit_AL))
    elite = AL[elite_idx].copy()
    elite_fit = float(fit_AL[elite_idx])

    max_iter = max(1, budget // pop)
    t = 0

    def roulette_select(weights: np.ndarray) -> int:
        # smaller fitness => higher weight
        w = weights.astype(float)
        w = w - np.min(w) + 1e-12
        inv = 1.0 / w
        p = inv / np.sum(inv)
        return int(rg.choice(np.arange(p.size), p=p))

    while evals + pop <= budget:
        # shrink boundaries around selected antlion + elite
        I = 1 + (t / max_iter) * (1e4 - 1)  # increasing intensity
        c = lo / I
        d = hi / I

        new_ants = np.empty_like(ants)
        for i in range(pop):
            idx = roulette_select(fit_AL)
            al = AL[idx]
            # random walk around al
            steps = rg.choice([-1.0, 1.0], size=(max_iter, dim))
            walk = np.cumsum(steps, axis=0)
            # take current time step t
            w_t = walk[min(t, max_iter - 1)]
            # normalize to [0,1]
            w_min = np.min(walk, axis=0)
            w_max = np.max(walk, axis=0)
            denom = np.where((w_max - w_min) < 1e-12, 1.0, (w_max - w_min))
            norm = (w_t - w_min) / denom
            # map to adaptive bounds around antlion
            lo_i = al + c
            hi_i = al + d
            cand1 = lo_i + norm * (hi_i - lo_i)

            # second walk around elite (same norm)
            lo_e = elite + c
            hi_e = elite + d
            cand2 = lo_e + norm * (hi_e - lo_e)

            new_ants[i] = (cand1 + cand2) / 2.0

        new_ants = clip_bounds(new_ants, lo, hi)
        fit_new = evaluate_population(f, new_ants)
        evals += pop
        ants = new_ants
        fit_ants = fit_new

        # replace antlions if ants are better
        for i in range(pop):
            if fit_ants[i] < fit_AL[i]:
                AL[i] = ants[i]
                fit_AL[i] = fit_ants[i]

        elite_idx = int(np.argmin(fit_AL))
        if float(fit_AL[elite_idx]) < elite_fit:
            elite = AL[elite_idx].copy()
            elite_fit = float(fit_AL[elite_idx])

        t += 1

    return float(min(elite_fit, np.min(fit_ants), np.min(fit_AL))), evals



def run_pso_gbest(f, dim, bounds, budget, seed, pop=30, w=0.72, c1=1.49, c2=1.49):
    """Particle Swarm Optimisation (global-best topology). Minimisation."""
    rg = rng_from(seed)
    lo, hi = bounds
    X = rg.uniform(lo, hi, size=(pop, dim))
    # velocity scale
    v_scale = 0.1 * (hi - lo)
    V = rg.uniform(-v_scale, v_scale, size=(pop, dim))
    fit = evaluate_population(f, X)
    evals = pop

    pbest = X.copy()
    pbest_fit = fit.copy()
    gbest_idx = int(np.argmin(pbest_fit))
    gbest = pbest[gbest_idx].copy()
    gbest_fit = float(pbest_fit[gbest_idx])

    while evals + pop <= budget:
        r1 = rg.random((pop, dim))
        r2 = rg.random((pop, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest[None, :] - X)
        X = X + V
        X = clip_bounds(X, lo, hi)

        fit = evaluate_population(f, X)
        evals += pop

        improved = fit < pbest_fit
        pbest[improved] = X[improved]
        pbest_fit[improved] = fit[improved]

        gbest_idx = int(np.argmin(pbest_fit))
        if float(pbest_fit[gbest_idx]) < gbest_fit:
            gbest_fit = float(pbest_fit[gbest_idx])
            gbest = pbest[gbest_idx].copy()

    return float(gbest_fit), evals

def run_pso_ring(f, dim, bounds, budget, seed, pop=30, w=0.72, c1=1.49, c2=1.49):
    """PSO with ring topology (local-best). Minimisation."""
    rg = rng_from(seed)
    lo, hi = bounds
    X = rg.uniform(lo, hi, size=(pop, dim))
    v_scale = 0.1 * (hi - lo)
    V = rg.uniform(-v_scale, v_scale, size=(pop, dim))
    fit = evaluate_population(f, X)
    evals = pop

    pbest = X.copy()
    pbest_fit = fit.copy()

    def lbest_positions(pbest, pbest_fit):
        # ring neighborhood: i-1, i, i+1
        lbest = np.empty_like(pbest)
        for i in range(pop):
            idxs = [(i - 1) % pop, i, (i + 1) % pop]
            j = idxs[int(np.argmin(pbest_fit[idxs]))]
            lbest[i] = pbest[j]
        return lbest

    while evals + pop <= budget:
        lbest = lbest_positions(pbest, pbest_fit)
        r1 = rg.random((pop, dim))
        r2 = rg.random((pop, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (lbest - X)
        X = X + V
        X = clip_bounds(X, lo, hi)

        fit = evaluate_population(f, X)
        evals += pop

        improved = fit < pbest_fit
        pbest[improved] = X[improved]
        pbest_fit[improved] = fit[improved]

    best = float(np.min(pbest_fit))
    return best, evals

def run_es_1p1(f, dim, bounds, budget, seed, pop=30, sigma0: float = None):
    """
    (1+1)-ES with simple 1/5th success rule step-size adaptation.
    We initialise from 'pop' random samples to match other population methods' initial sampling.
    """
    rg = rng_from(seed)
    lo, hi = bounds
    X0 = rg.uniform(lo, hi, size=(pop, dim))
    f0 = evaluate_population(f, X0)
    evals = pop
    x = X0[int(np.argmin(f0))].copy()
    fx = float(np.min(f0))

    if sigma0 is None:
        sigma = 0.2 * (hi - lo)
    else:
        sigma = float(sigma0)

    success_window = 20
    succ = 0
    steps = 0
    # remaining budget is consumed one evaluation per iteration
    while evals < budget:
        x2 = x + sigma * rg.normal(0.0, 1.0, size=dim)
        x2 = clip_bounds(x2, lo, hi)
        f2 = float(f(x2))
        evals += 1
        steps += 1
        if f2 < fx:
            x, fx = x2, f2
            succ += 1

        if steps % success_window == 0:
            rate = succ / success_window
            # 1/5 success rule: if success rate > 0.2 increase step size, else decrease
            if rate > 0.2:
                sigma *= 1.2
            else:
                sigma *= 0.82
            # keep sigma in sane range
            sigma = float(np.clip(sigma, 1e-12, 0.5 * (hi - lo)))
            succ = 0

    return float(fx), evals
ALGO_RUNNERS = {
    "PSO_GBEST": run_pso_gbest,
    "PSO_RING": run_pso_ring,
    "ES_1P1": run_es_1p1,
    "GWO": run_gwo,
    "WOA": run_woa,
    "MFO": run_mfo,
    "FA":  run_fa,
    "BA":  run_ba,
    "ALO": run_alo,
}

EXPECTED_GROUPS_DEFAULT = {
    # This encodes the *claim you want to test*. Adjust if your hypothesis differs.
    # "PSO-like" group:
    "GWO": "PSO",
    "WOA": "PSO",
    "MFO": "PSO",
    "FA":  "PSO",
    "BA":  "PSO",
    # "ES-like" group:
    "ALO": "ES",
}


# -----------------------------
# Experiment runner
# -----------------------------

def run_one(algo: str, f: Callable[[np.ndarray], float], dim: int, bounds: Tuple[float, float],
            budget: int, seed: int, pop: int) -> float:
    runner = ALGO_RUNNERS[algo]
    if algo == "FA":
        best, _evals = runner(f, dim, bounds, budget, seed, pop=pop, k_neighbors=3)
    else:
        best, _evals = runner(f, dim, bounds, budget, seed, pop=pop)
    return float(best)

def make_run_plan(budgets: List[int], suite: List[ProblemSpec], instances: int, seeds: int,
                  algos: List[str]) -> List[Tuple]:
    """
    Returns list of tuples to run:
    (instance_id, problem_name, bounds, shift_seed, algo, budget, seed)
    instance_id is a stable integer within each problem.
    """
    plan = []
    for spec in suite:
        for inst in range(instances):
            shift_seed = 10_000 * stable_hash_mod(spec.name, 1000) + inst
            for budget in budgets:
                for algo in algos:
                    for r in range(seeds):
                        seed = 1_000_000 * stable_hash_mod(algo, 1000) + 10_000 * inst + 37 * budget + r
                        plan.append((inst, spec.name, spec.bounds, shift_seed, algo, budget, seed))
    return plan

def run_experiments(out_dir: Path, budgets: List[int], instances: int, dim: int,
                    seeds: int, pop: int, algos: List[str]) -> Path:
    """
    Runs all planned runs sequentially (n_jobs is intentionally 1 for Windows/laptop safety).
    Writes runs_detail.csv and returns its path.
    """
    suite = make_suite()
    ensure_dir(out_dir)
    t0 = time.time()

    plan = make_run_plan(budgets, suite, instances, seeds, algos)
    total = len(plan)

    rows = []
    for idx, (inst_id, pname, bounds, shift_seed, algo, budget, seed) in enumerate(plan, start=1):
        spec = next(s for s in suite if s.name == pname)
        shift = make_instance_shift(dim, bounds, shift_seed)
        f = objective_factory(spec, dim, shift)

        best = run_one(algo, f, dim, bounds, budget, seed, pop=pop)

        rows.append({
            "instance_id": int(inst_id),
            "problem": pname,
            "domain": "cont",
            "budget": int(budget),
            "algo_variant": algo,   # keep simple label; add hyperparams if you expand
            "algo_base": algo,
            "seed": int(seed),
            "best": float(best),
        })

        if idx % 500 == 0 or idx == total:
            elapsed = time.time() - t0
            print(f"[PROG] {idx:,}/{total:,} runs  elapsed={elapsed:.1f}s")

    df = pd.DataFrame(rows)
    out_path = out_dir / "runs_detail.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")
    return out_path

def aggregate_to_summary(runs_csv: Path, out_dir: Path,
                         target_tols: Dict[str, float] = None,
                         alpha0: float = 1.0, beta0: float = 1.0) -> Path:
    """
    Expand runs into targets (easy/med/hard) via tolerance thresholds on best value,
    then aggregate to instance_algo_budget_summary.csv with Beta posterior summaries.
    """
    if target_tols is None:
        target_tols = {"easy": 1e-3, "med": 1e-2, "hard": 1e-1}

    df = pd.read_csv(runs_csv)
    # expand targets without rerunning
    exp = []
    for tgt, tol in target_tols.items():
        d2 = df.copy()
        d2["target"] = tgt
        d2["success"] = (d2["best"] <= tol).astype(int)
        exp.append(d2)
    df2 = pd.concat(exp, ignore_index=True)

    group_cols = ["instance_id", "problem", "domain", "budget", "target", "algo_variant", "algo_base"]
    g = df2.groupby(group_cols, as_index=False)

    def beta_mean(succ: int, trials: int) -> float:
        return (alpha0 + succ) / (alpha0 + beta0 + trials)

    rows = []
    for _, sub in g:
        succ = int(sub["success"].sum())
        trials = int(sub.shape[0])
        bm = beta_mean(succ, trials)

        # beta quantiles (5th/95th)
        a_post = alpha0 + succ
        b_post = beta0 + trials - succ
        p05 = float(sp_beta.ppf(0.05, a_post, b_post))
        p95 = float(sp_beta.ppf(0.95, a_post, b_post))

        best_vals = sub["best"].to_numpy(dtype=float)
        rows.append({
            **{c: sub.iloc[0][c] for c in group_cols},
            "successes": succ,
            "trials": trials,
            "succ_rate": succ / trials if trials else np.nan,
            "beta_mean": bm,
            "beta_p05": p05,
            "beta_p95": p95,
            "mean_best": float(np.mean(best_vals)),
            "median_best": float(np.median(best_vals)),
            "min_best": float(np.min(best_vals)),
            "max_best": float(np.max(best_vals)),
        })

    out = pd.DataFrame(rows)
    out_path = out_dir / "instance_algo_budget_summary.csv"
    out.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")
    return out_path


# -----------------------------
# Analysis: profiles, clustering, tables, plots
# -----------------------------

def build_profile_matrix(df: pd.DataFrame, budget: int, target: str,
                         algos: List[str]) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Returns:
      mat: (n_algos, n_instances) beta_mean
      algo_order: list of algo names in row order
      instance_keys: list of instance identifiers "problem|instance_id"
    """
    sub = df[(df["budget"] == budget) & (df["target"] == target)].copy()
    sub["inst_key"] = sub["problem"].astype(str) + "|" + sub["instance_id"].astype(str)

    instance_keys = sorted(sub["inst_key"].unique().tolist())
    algo_order = [a for a in algos if a in sub["algo_base"].unique()]

    mat = np.full((len(algo_order), len(instance_keys)), np.nan, dtype=float)

    pivot = sub.pivot_table(index="algo_base", columns="inst_key", values="beta_mean", aggfunc="mean")
    for i, a in enumerate(algo_order):
        if a in pivot.index:
            row = pivot.loc[a]
            for j, k in enumerate(instance_keys):
                if k in row.index:
                    mat[i, j] = float(row[k])

    # fill missing with column means (shouldn't happen if complete)
    col_mean = np.nanmean(mat, axis=0, keepdims=True)
    inds = np.where(np.isnan(mat))
    mat[inds] = np.take_along_axis(col_mean, inds[1][None, :], axis=1).flatten()

    return mat, algo_order, instance_keys

def cluster_algorithms(mat: np.ndarray, algo_order: List[str], view: str,
                       tau: float = 6.0, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute distance matrix and return cluster labels for k clusters.
    view in {"ABS","REL"}:
      - ABS: uses raw beta_mean
      - REL: z-score per instance across algorithms, then logit_clip (after mapping through sigmoid?)
    """
    X = mat.copy()

    if view.upper() == "REL":
        X = zscore_across_algos_for_each_instance(X)
        # map z-scores to (0,1) via sigmoid so we can use logit-like stabilization
        X = 1.0 / (1.0 + np.exp(-X))
    # stabilize boundaries for both
    X = logit_clip(X, eps=1e-6)

    # pairwise distances between algorithms
    D_vec = pdist(X, metric="euclidean")
    D = squareform(D_vec)

    # hierarchical clustering
    Z = linkage(D_vec, method="average")
    labels = fcluster(Z, t=k, criterion="maxclust")
    return labels, D

def pick_family_medoids(D: np.ndarray, algo_order: List[str], expected_groups: Dict[str, str]) -> Dict[str, str]:
    """
    Pick medoid algorithm for each expected group using within-group distance sums.
    Returns group_name -> algo_name
    """
    group_to_algos: Dict[str, List[str]] = {}
    for a in algo_order:
        g = expected_groups.get(a, "UNK")
        group_to_algos.setdefault(g, []).append(a)

    medoids = {}
    for g, algos in group_to_algos.items():
        idxs = [algo_order.index(a) for a in algos]
        if len(idxs) == 1:
            medoids[g] = algos[0]
            continue
        # compute sum distances within group
        subD = D[np.ix_(idxs, idxs)]
        sums = subD.sum(axis=1)
        medoid_local = idxs[int(np.argmin(sums))]
        medoids[g] = algo_order[medoid_local]
    return medoids

def soft_membership(D: np.ndarray, algo_order: List[str], anchors: Dict[str, str], tau: float) -> pd.DataFrame:
    """
    Soft assignment to anchor families using exp(-tau * distance).
    anchors: family -> algo_name
    Returns df with columns: algo, family, weight
    """
    families = list(anchors.keys())
    anchor_idxs = [algo_order.index(anchors[f]) for f in families]

    out_rows = []
    for i, algo in enumerate(algo_order):
        d = np.array([D[i, j] for j in anchor_idxs], dtype=float)
        w = np.exp(-tau * d)
        w = w / np.sum(w)
        for fam, wi in zip(families, w):
            out_rows.append({"algo": algo, "family": fam, "weight": float(wi)})
    return pd.DataFrame(out_rows)

def quick_kpi(df: pd.DataFrame, budgets: List[int], targets: List[str], algos: List[str]) -> str:
    lines = []
    for b in budgets:
        for t in targets:
            sub = df[(df["budget"] == b) & (df["target"] == t)]
            if sub.empty:
                continue
            mean_by_algo = sub.groupby("algo_base")["beta_mean"].mean().sort_values(ascending=False)
            top5 = ", ".join([f"{a}={mean_by_algo[a]:.3f}" for a in mean_by_algo.index[:5]])
            lines.append(f"CONT  budget={b:<5}  target={t:<4} top5: {top5}")
    return "\n".join(lines)

def write_latex_topk(df: pd.DataFrame, out_tex: Path, budgets: List[int], targets: List[str], k: int = 6) -> None:
    rows = []
    for b in budgets:
        for t in targets:
            sub = df[(df["budget"] == b) & (df["target"] == t)]
            mean_by_algo = sub.groupby("algo_base")["beta_mean"].mean().sort_values(ascending=False)
            items = [f"{a} ({mean_by_algo[a]:.3f})" for a in mean_by_algo.index[:k]]
            rows.append((b, t, " ; ".join(items)))

    tex = []
    tex.append(r"\begin{tabular}{r l p{0.72\linewidth}}")
    tex.append(r"\hline")
    tex.append(r"Budget & Target & Top algorithms by mean posterior success ($\bar{p}$) \\")
    tex.append(r"\hline")
    for b, t, s in rows:
        tex.append(f"{b} & {t} & {s} \\\\")
    tex.append(r"\hline")
    tex.append(r"\end{tabular}")
    out_tex.write_text("\n".join(tex), encoding="utf-8")

def plot_budget_curves(df: pd.DataFrame, out_png: Path, budgets: List[int], targets: List[str], algos: List[str]) -> None:
    # mean beta_mean over instances
    pivot = df.groupby(["algo_base", "budget", "target"])["beta_mean"].mean().reset_index()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    for algo in algos:
        sub = pivot[pivot["algo_base"] == algo]
        if sub.empty:
            continue
        # plot separate line per target, but keep readable
        for t in targets:
            sub2 = sub[sub["target"] == t].sort_values("budget")
            if sub2.empty:
                continue
            ax.plot(sub2["budget"], sub2["beta_mean"], marker="o", label=f"{algo}-{t}")
    ax.set_xlabel("Budget (function evaluations)")
    ax.set_ylabel("Mean posterior success (beta_mean)")
    ax.set_title("Success vs budget (by algorithm and target tier)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_heatmap(df: pd.DataFrame, out_png: Path, budgets: List[int], targets: List[str], algos: List[str]) -> None:
    # matrix algos x (budget,target)
    cols = [(b, t) for b in budgets for t in targets]
    M = np.zeros((len(algos), len(cols)), dtype=float)
    for i, algo in enumerate(algos):
        for j, (b, t) in enumerate(cols):
            sub = df[(df["algo_base"] == algo) & (df["budget"] == b) & (df["target"] == t)]
            M[i, j] = float(sub["beta_mean"].mean()) if not sub.empty else np.nan

    fig = plt.figure(figsize=(12, 0.5 + 0.35 * len(algos)))
    ax = fig.add_subplot(111)
    im = ax.imshow(M, aspect="auto")
    ax.set_yticks(np.arange(len(algos)))
    ax.set_yticklabels(algos)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([f"{b}-{t}" for b, t in cols], rotation=45, ha="right")
    ax.set_title("Heatmap: mean beta_mean across (budget,target)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

def plot_mds(D: np.ndarray, algo_order: List[str], labels: np.ndarray, out_png: Path, title: str) -> None:
    X = classical_mds(D, n_components=2)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    # scatter without explicit colors; matplotlib defaults will handle cycling if we plot per-cluster
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        ax.scatter(X[idx, 0], X[idx, 1], label=f"cluster {c}")
        for i in idx:
            ax.text(X[i, 0], X[i, 1], algo_order[i], fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

def plot_dendrogram(D: np.ndarray, algo_order: List[str], out_png: Path, title: str) -> None:
    D_vec = squareform(D, checks=False)
    Z = linkage(D_vec, method="average")
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    dendrogram(Z, labels=algo_order, leaf_rotation=45, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

def analyze_and_report(summary_csv: Path, out_dir: Path, budgets: List[int],
                       targets: List[str], algos: List[str],
                       anchor_pso: str, anchor_es: str, tau: float = 6.0) -> None:
    df = pd.read_csv(summary_csv)

    ana_dir = out_dir / "analysis"
    fig_dir = out_dir / "figs"
    tab_dir = out_dir / "tables"
    ensure_dir(ana_dir); ensure_dir(fig_dir); ensure_dir(tab_dir)

    # quick KPI
    kpi = quick_kpi(df, budgets, targets, algos)
    (ana_dir / "quick_kpi.txt").write_text(kpi + "\n", encoding="utf-8")
    print("\n=== Quick KPI (mean beta_mean) ===")
    print(kpi)

    # paper top-k table
    write_latex_topk(df, tab_dir / "topk_by_budget_target.tex", budgets, targets, k=6)

    # plots
    plot_budget_curves(df, fig_dir / "success_vs_budget.png", budgets, targets, algos)
    plot_heatmap(df, fig_dir / "heatmap_success.png", budgets, targets, algos)

    cluster_rows = []
    assign_rows = []
    for b in budgets:
        for t in targets:
            mat, algo_order, inst_keys = build_profile_matrix(df, b, t, algos)
            for view in ["ABS", "REL"]:
                labels, D = cluster_algorithms(mat, algo_order, view=view, tau=tau, k=2)
                                # membership to fixed anchors (non-circular)
                if (anchor_pso not in algo_order) or (anchor_es not in algo_order):
                    raise SystemExit(f"Anchors not in algo list for slice budget={b} target={t}: "
                                     f"anchor_pso={anchor_pso} anchor_es={anchor_es}. "
                                     "Include them in --algos or change --anchor_* flags.")
                anchors = {"PSO": anchor_pso, "ES": anchor_es}
                mem = soft_membership(D, algo_order, anchors, tau=tau)

                # dump clustering + membership
                for algo, lab in zip(algo_order, labels):
                    cluster_rows.append({
                        "budget": b, "target": t, "view": view,
                        "algo": algo, "cluster": int(lab),
                        "anchor_PSO": anchor_pso,
                        "anchor_ES": anchor_es,
                    })

                mem_out = ana_dir / f"membership_budget{b}_target{t}_{view}.csv"
                mem.to_csv(mem_out, index=False)
                # derive hard family assignment from weights
                mem_w = mem.pivot_table(index='algo', columns='family', values='weight', aggfunc='mean').reset_index()
                for _, rr in mem_w.iterrows():
                    w_pso = float(rr.get('PSO', np.nan))
                    w_es  = float(rr.get('ES', np.nan))
                    fam = 'PSO' if w_pso >= w_es else 'ES'
                    assign_rows.append({
                        'budget': b, 'target': t, 'view': view,
                        'algo': rr['algo'], 'assign_family': fam,
                        'w_PSO': w_pso, 'w_ES': w_es,
                        'anchor_pso': anchor_pso, 'anchor_es': anchor_es,
                    })

                # plots per slice/view
                plot_mds(D, algo_order, labels, fig_dir / f"mds_budget{b}_target{t}_{view}.png",
                         title=f"MDS ({view}) budget={b} target={t}")
                plot_dendrogram(D, algo_order, fig_dir / f"dendro_budget{b}_target{t}_{view}.png",
                                title=f"Dendrogram ({view}) budget={b} target={t}")

    clus_df = pd.DataFrame(cluster_rows)
    clus_df.to_csv(ana_dir / "cluster_assignments.csv", index=False)
    print(f"[OK] wrote {ana_dir / 'cluster_assignments.csv'}")

    # claim test summary: per algo, how often it assigns to PSO vs ES across slices
    assign_df = pd.DataFrame(assign_rows)
    assign_df.to_csv(ana_dir / 'family_assignments.csv', index=False)
    # aggregate stability summary
    summ = (assign_df.groupby(['algo'])
            .agg(n_slices=('assign_family','size'),
                 frac_PSO=('assign_family', lambda s: float(np.mean(s=='PSO'))),
                 mean_w_PSO=('w_PSO','mean'),
                 mean_w_ES=('w_ES','mean'))
            .reset_index())
    summ['dominant_family'] = np.where(summ['frac_PSO']>=0.5,'PSO','ES')
    summ.to_csv(ana_dir / 'claim_stability_summary.csv', index=False)
    print(f"[OK] wrote {ana_dir / 'family_assignments.csv'}")
    print(f"[OK] wrote {ana_dir / 'claim_stability_summary.csv'}")

    # summary LaTeX table: cluster by budget/target/view
    tex_lines = []
    tex_lines.append(r"\begin{tabular}{r l l l l}")
    tex_lines.append(r"\hline")
    tex_lines.append(r"Budget & Target & View & Algorithm & Cluster \\")
    tex_lines.append(r"\hline")
    for _, r in clus_df.sort_values(["budget", "target", "view", "algo"]).iterrows():
        tex_lines.append(f"{int(r['budget'])} & {r['target']} & {r['view']} & {r['algo']} & {int(r['cluster'])} \\\\")
    tex_lines.append(r"\hline")
    tex_lines.append(r"\end{tabular}")
    (tab_dir / "clusters_by_budget_target_view.tex").write_text("\n".join(tex_lines), encoding="utf-8")


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--budgets", type=str, default="300,800,2000", help="Comma-separated budgets")
    ap.add_argument("--instances", type=int, default=40, help="Number of instances per function")
    ap.add_argument("--dim", type=int, default=10, help="Dimension")
    ap.add_argument("--seeds", type=int, default=20, help="Independent runs per (instance, algo, budget)")
    ap.add_argument("--pop", type=int, default=30, help="Population size for population methods")
    ap.add_argument("--algos", type=str, default="PSO_GBEST,PSO_RING,ES_1P1,GWO,MFO,WOA,FA,BA,ALO", help="Comma-separated algos to test (include anchors)")
    ap.add_argument("--tau", type=float, default=6.0, help="Membership temperature (higher => sharper)")
    ap.add_argument("--anchor_pso", type=str, default="PSO_GBEST", help="Anchor algorithm name for PSO family")
    ap.add_argument("--anchor_es", type=str, default="ES_1P1", help="Anchor algorithm name for ES family")
    ap.add_argument("--analysis_only", action="store_true", help="Skip running; analyze existing summary CSV")
    ap.add_argument("--in_summary_csv", type=str, default="", help="If analysis_only, path to instance_algo_budget_summary.csv")
    ap.add_argument("--runs_csv", type=str, default="", help="If you already have runs_detail.csv, reuse it")
    return ap.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    targets = ["easy", "med", "hard"]

    if args.analysis_only:
        if not args.in_summary_csv:
            raise SystemExit("--analysis_only requires --in_summary_csv")
        summary_csv = Path(args.in_summary_csv)
        analyze_and_report(summary_csv, out_dir, budgets, targets, algos, anchor_pso=args.anchor_pso, anchor_es=args.anchor_es, tau=args.tau)
        return

    ensure_dir(out_dir)

    if args.runs_csv:
        runs_csv = Path(args.runs_csv)
        if not runs_csv.exists():
            raise SystemExit(f"--runs_csv not found: {runs_csv}")
    else:
        runs_csv = run_experiments(out_dir, budgets, args.instances, args.dim, args.seeds, args.pop, algos)

    summary_csv = aggregate_to_summary(runs_csv, out_dir)

    analyze_and_report(summary_csv, out_dir, budgets, targets, algos, anchor_pso=args.anchor_pso, anchor_es=args.anchor_es, tau=args.tau)

    print("\n[DONE] dorigo variant budget clustering complete.")

if __name__ == "__main__":
    main()
