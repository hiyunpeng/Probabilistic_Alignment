#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dorigo_tune_then_eval_with_progress.py

End-to-end pipeline:
1) Tune hyperparameters on a DEV split (successive halving / racing).
2) Freeze best params.
3) Evaluate success profiles on HOLDOUT split with repeated runs across budgets/targets.
4) Write reproducible CSV artifacts.

Adds progress bars via tqdm (with safe fallback if tqdm not installed).

Dependencies
-----------
pip install numpy pandas scipy matplotlib tqdm

Example
-------
python dorigo_tune_then_eval_with_progress.py \
  --out_dir out_dorigo_tuned \
  --algos GWO,WOA,MFO,FA,BA,ALO,PSO_GBEST,ES_1P1 \
  --budgets 300,500,800,1000 \
  --instances_per_problem 20 \
  --dev_frac 0.4 \
  --dim 10 \
  --R_eval 10 \
  --n_candidates 60 \
  --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import beta as sp_beta
except Exception as e:
    raise SystemExit("This script requires scipy. Install with: pip install scipy\n" + str(e))

# -----------------------------
# Progress bar (tqdm) with fallback
# -----------------------------
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:
    _tqdm = None

def tqdm(iterable=None, total=None, desc=None, leave=True, position=None, disable=False, **kwargs):
    """A thin wrapper around tqdm that degrades gracefully if tqdm isn't installed."""
    if disable or _tqdm is None:
        if iterable is None:
            class _Dummy:
                def update(self, n=1): pass
                def close(self): pass
                def set_postfix_str(self, s): pass
            return _Dummy()
        return iterable
    return _tqdm(iterable=iterable, total=total, desc=desc, leave=leave, position=position, **kwargs)

# -----------------------------
# Repro + helpers
# -----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def rng_from(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))

def stable_hash_mod(s: str, mod: int) -> int:
    h = zlib.adler32(s.encode("utf-8")) & 0xffffffff
    return int(h % mod)

def clip_bounds(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)

# -----------------------------
# Benchmark suite (continuous, shifted)
# -----------------------------

@dataclass(frozen=True)
class ProblemSpec:
    name: str
    bounds: Tuple[float, float]
    fn: Callable[[np.ndarray], float]   # expects x already shifted (optimum at 0)
    optimum: float = 0.0

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))

def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    return float(A * x.size + np.sum(x * x - A * np.cos(2 * math.pi * x)))

def ackley(x: np.ndarray) -> float:
    a, b, c = 20.0, 0.2, 2*math.pi
    d = x.size
    s1 = np.sum(x*x)
    s2 = np.sum(np.cos(c*x))
    return float(-a*np.exp(-b*np.sqrt(s1/d)) - np.exp(s2/d) + a + math.e)

def griewank(x: np.ndarray) -> float:
    i = np.arange(1, x.size+1, dtype=float)
    return float(1.0 + np.sum(x*x)/4000.0 - np.prod(np.cos(x/np.sqrt(i))))

def make_suite() -> List[ProblemSpec]:
    return [
        ProblemSpec("Sphere", (-5.0, 5.0), sphere),
        ProblemSpec("Rosenbrock", (-5.0, 5.0), rosenbrock),
        ProblemSpec("Rastrigin", (-5.12, 5.12), rastrigin),
        ProblemSpec("Ackley", (-5.0, 5.0), ackley),
        ProblemSpec("Griewank", (-6.0, 6.0), griewank),
    ]

def make_shifted_objective(spec: ProblemSpec, dim: int, shift_seed: int) -> Callable[[np.ndarray], float]:
    rg = rng_from(shift_seed)
    lo, hi = spec.bounds
    shift = rg.uniform(-1.0, 1.0, size=dim) * 0.3 * (hi - lo)
    def f(x: np.ndarray) -> float:
        return spec.fn(x - shift)
    return f

# -----------------------------
# Core evaluation helper
# -----------------------------

def evaluate_population(f: Callable[[np.ndarray], float], X: np.ndarray) -> np.ndarray:
    return np.array([f(x) for x in X], dtype=float)

# -----------------------------
# Algorithms (parameterized)
# -----------------------------

def run_gwo(f, dim, bounds, budget, seed, pop=30, a0: float = 2.0):
    rg = rng_from(seed)
    lo, hi = bounds
    X = rg.uniform(lo, hi, size=(pop, dim))
    fit = evaluate_population(f, X)
    evals = pop

    def top3(fit):
        idx = np.argsort(fit)
        return idx[0], idx[1], idx[2]

    ia, ib, ic = top3(fit)
    Xa, Xb, Xc = X[ia].copy(), X[ib].copy(), X[ic].copy()

    t = 0
    T = max(1, budget // pop)
    while evals + pop <= budget:
        a = float(a0) * (1.0 - (t / T))
        Xnew = np.empty_like(X)
        for i in range(pop):
            r1 = rg.random(dim); r2 = rg.random(dim)
            A1 = 2*a*r1 - a; C1 = 2*r2
            D1 = np.abs(C1*Xa - X[i])
            X1 = Xa - A1*D1

            r1 = rg.random(dim); r2 = rg.random(dim)
            A2 = 2*a*r1 - a; C2 = 2*r2
            D2 = np.abs(C2*Xb - X[i])
            X2 = Xb - A2*D2

            r1 = rg.random(dim); r2 = rg.random(dim)
            A3 = 2*a*r1 - a; C3 = 2*r2
            D3 = np.abs(C3*Xc - X[i])
            X3 = Xc - A3*D3

            Xnew[i] = (X1 + X2 + X3) / 3.0

        X = clip_bounds(Xnew, lo, hi)
        fit = evaluate_population(f, X)
        evals += pop

        ia, ib, ic = top3(fit)
        Xa, Xb, Xc = X[ia].copy(), X[ib].copy(), X[ic].copy()
        t += 1

    return float(np.min(fit)), evals

def run_woa(f, dim, bounds, budget, seed, pop=30, a0: float = 2.0, spiral_b: float = 1.0):
    rg = rng_from(seed)
    lo, hi = bounds
    X = rg.uniform(lo, hi, size=(pop, dim))
    fit = evaluate_population(f, X)
    evals = pop
    ib = int(np.argmin(fit))
    Xbest = X[ib].copy()
    fbest = float(fit[ib])

    t = 0
    T = max(1, budget // pop)
    while evals + pop <= budget:
        a = float(a0) * (1.0 - (t / T))
        a2 = -1.0 + (t / T) * (-1.0)
        Xnew = np.empty_like(X)
        for i in range(pop):
            p = rg.random()
            r1 = rg.random(dim); r2 = rg.random(dim)
            A = 2*a*r1 - a
            C = 2*r2
            if p < 0.5:
                if np.linalg.norm(A) < 1.0:
                    D = np.abs(C*Xbest - X[i])
                    Xnew[i] = Xbest - A*D
                else:
                    j = int(rg.integers(0, pop))
                    Xrand = X[j]
                    D = np.abs(C*Xrand - X[i])
                    Xnew[i] = Xrand - A*D
            else:
                D = np.abs(Xbest - X[i])
                l = (a2 - 1.0) * rg.random() + 1.0
                Xnew[i] = D * np.exp(float(spiral_b)*l) * np.cos(2*math.pi*l) + Xbest

        X = clip_bounds(Xnew, lo, hi)
        fit = evaluate_population(f, X)
        evals += pop
        ib = int(np.argmin(fit))
        if float(fit[ib]) < fbest:
            fbest = float(fit[ib])
            Xbest = X[ib].copy()
        t += 1

    return float(min(fbest, float(np.min(fit)))), evals

def run_mfo(f, dim, bounds, budget, seed, pop=30, b: float = 1.0):
    rg = rng_from(seed)
    lo, hi = bounds
    M = rg.uniform(lo, hi, size=(pop, dim))
    fit = evaluate_population(f, M)
    evals = pop

    flames = M.copy()
    flames_fit = fit.copy()

    t = 0
    max_iter = max(1, budget // pop)
    while evals + pop <= budget:
        idx = np.argsort(fit)
        M = M[idx]; fit = fit[idx]

        all_X = np.vstack([flames, M])
        all_fit = np.concatenate([flames_fit, fit])
        idx2 = np.argsort(all_fit)
        flames = all_X[idx2][:pop].copy()
        flames_fit = all_fit[idx2][:pop].copy()

        flame_no = int(round(pop - t * ((pop - 1) / max_iter)))
        flame_no = max(1, min(pop, flame_no))

        Mnew = np.empty_like(M)
        for i in range(pop):
            flame_idx = min(i, flame_no - 1)
            F = flames[flame_idx]
            D = np.abs(F - M[i])
            l = (rg.random() * 2.0 - 1.0)
            Mnew[i] = D * np.exp(float(b) * l) * np.cos(2*math.pi*l) + F

        M = clip_bounds(Mnew, lo, hi)
        fit = evaluate_population(f, M)
        evals += pop
        t += 1

    return float(min(float(np.min(fit)), float(np.min(flames_fit)))), evals

def run_fa(f, dim, bounds, budget, seed, pop=30,
           beta0: float = 1.0,
           gamma_scale: float = 1.0,
           alpha_scale: float = 0.25,
           k_neighbors: int = 3):
    rg = rng_from(seed)
    lo, hi = bounds
    X = rg.uniform(lo, hi, size=(pop, dim))
    fit = evaluate_population(f, X)
    evals = pop

    beta0 = float(beta0)
    gamma = float(gamma_scale) / max(1e-9, (hi - lo) ** 2)
    alpha = float(alpha_scale) * (hi - lo)
    k_neighbors = int(max(1, min(pop-1, k_neighbors)))

    while evals + pop <= budget:
        idx = np.argsort(fit)
        Xs = X[idx].copy()

        Xnew = Xs.copy()
        for i in range(pop):
            if i == 0:
                continue
            candidates = rg.choice(np.arange(i), size=min(k_neighbors, i), replace=False)
            for j in candidates:
                rij = np.linalg.norm(Xnew[i] - Xs[j])
                beta = beta0 * np.exp(-gamma * rij * rij)
                step = beta * (Xs[j] - Xnew[i]) + alpha * rg.normal(0.0, 1.0, size=dim)
                Xnew[i] = Xnew[i] + step

        X = clip_bounds(Xnew, lo, hi)
        fit = evaluate_population(f, X)
        evals += pop

    return float(np.min(fit)), evals

def run_ba(f, dim, bounds, budget, seed, pop=30,
           fmax: float = 2.0,
           A0: float = 0.9,
           r0: float = 0.5,
           local_step: float = 0.001):
    rg = rng_from(seed)
    lo, hi = bounds
    X = rg.uniform(lo, hi, size=(pop, dim))
    V = np.zeros((pop, dim), dtype=float)
    fit = evaluate_population(f, X)
    evals = pop

    ib = int(np.argmin(fit))
    best = X[ib].copy()
    fbest = float(fit[ib])

    fmin, fmax = 0.0, float(max(1e-9, fmax))
    A = np.full(pop, float(A0))
    r = np.full(pop, float(r0))
    local_step = float(local_step)

    while evals + pop <= budget:
        for i in range(pop):
            freq = fmin + (fmax - fmin) * rg.random()
            V[i] = V[i] + (X[i] - best) * freq
            S = X[i] + V[i]
            S = clip_bounds(S, lo, hi)

            if rg.random() > r[i]:
                eps = rg.normal(0.0, 1.0, size=dim)
                S = best + local_step * eps
                S = clip_bounds(S, lo, hi)

            fS = float(f(S))
            evals += 1

            if fS < fit[i] and rg.random() < A[i]:
                X[i] = S
                fit[i] = fS
                A[i] *= 0.95

            if fS < fbest:
                best = S.copy()
                fbest = fS

            if evals >= budget:
                break

        if evals >= budget:
            break

    return float(min(fbest, float(np.min(fit)))), evals

def run_alo(f, dim, bounds, budget, seed, pop=30, intensity_max: float = 1e4):
    rg = rng_from(seed)
    lo, hi = bounds
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
    intensity_max = float(max(10.0, intensity_max))

    def roulette_select(weights: np.ndarray) -> int:
        w = weights.astype(float)
        w = w - np.min(w) + 1e-12
        inv = 1.0 / w
        p = inv / np.sum(inv)
        return int(rg.choice(np.arange(p.size), p=p))

    while evals + pop <= budget:
        I = 1.0 + (t / max_iter) * (intensity_max - 1.0)
        c = lo / I
        d = hi / I

        new_ants = np.empty_like(ants)
        for i in range(pop):
            sel = roulette_select(fit_AL)
            RW1 = np.cumsum(rg.choice([-1, 1], size=(dim,)), axis=0)
            RW2 = np.cumsum(rg.choice([-1, 1], size=(dim,)), axis=0)

            def norm_walk(RW):
                RW = (RW - RW.min()) / (RW.max() - RW.min() + 1e-12)
                return c + RW * (d - c)

            w1 = norm_walk(RW1)
            w2 = norm_walk(RW2)
            ant = (AL[sel] + w1 + elite + w2) / 2.0
            new_ants[i] = ant

        ants = clip_bounds(new_ants, lo, hi)
        fit_ants = evaluate_population(f, ants)
        evals += pop

        improved = fit_ants < fit_AL
        AL[improved] = ants[improved]
        fit_AL[improved] = fit_ants[improved]

        elite_idx = int(np.argmin(fit_AL))
        if float(fit_AL[elite_idx]) < elite_fit:
            elite = AL[elite_idx].copy()
            elite_fit = float(fit_AL[elite_idx])

        t += 1

    return float(min(elite_fit, float(np.min(fit_ants)), float(np.min(fit_AL)))), evals

def run_pso_gbest(f, dim, bounds, budget, seed, pop=30, w=0.72, c1=1.49, c2=1.49):
    rg = rng_from(seed)
    lo, hi = bounds
    X = rg.uniform(lo, hi, size=(pop, dim))
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

def run_es_1p1(f, dim, bounds, budget, seed, pop=30, sigma0: float = None):
    rg = rng_from(seed)
    lo, hi = bounds
    X0 = rg.uniform(lo, hi, size=(pop, dim))
    f0 = evaluate_population(f, X0)
    evals = pop
    x = X0[int(np.argmin(f0))].copy()
    fx = float(np.min(f0))

    sigma = 0.2 * (hi - lo) if sigma0 is None else float(sigma0)
    success_window = 20
    succ = 0
    steps = 0
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
            sigma *= 1.2 if rate > 0.2 else 0.82
            sigma = float(np.clip(sigma, 1e-12, 0.5 * (hi - lo)))
            succ = 0

    return float(fx), evals

ALGO_RUNNERS = {
    "GWO": run_gwo,
    "WOA": run_woa,
    "MFO": run_mfo,
    "FA":  run_fa,
    "BA":  run_ba,
    "ALO": run_alo,
    "PSO_GBEST": run_pso_gbest,
    "ES_1P1": run_es_1p1,
}

# -----------------------------
# Targets
# -----------------------------

def parse_target_tols(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=")
        out[k.strip()] = float(v.strip())
    return out

def beta_posterior(succ: int, trials: int):
    a = 1 + succ
    b = 1 + trials - succ
    mean = a / (a + b)
    p05 = float(sp_beta.ppf(0.05, a, b))
    p95 = float(sp_beta.ppf(0.95, a, b))
    return float(mean), p05, p95

# -----------------------------
# Parameter spaces (tuning)
# -----------------------------

DEFAULT_SPACES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "GWO": {"pop": {"type": "int", "lo": 20, "hi": 80}, "a0": {"type": "float", "lo": 1.5, "hi": 3.0}},
    "WOA": {"pop": {"type": "int", "lo": 20, "hi": 80}, "a0": {"type": "float", "lo": 1.5, "hi": 3.0},
            "spiral_b": {"type": "float", "lo": 0.5, "hi": 2.0}},
    "MFO": {"pop": {"type": "int", "lo": 20, "hi": 80}, "b": {"type": "float", "lo": 0.5, "hi": 2.0}},
    "FA":  {"pop": {"type": "int", "lo": 20, "hi": 80}, "beta0": {"type": "float", "lo": 0.2, "hi": 2.5},
            "gamma_scale": {"type": "float_log", "lo": 1e-2, "hi": 50.0},
            "alpha_scale": {"type": "float_log", "lo": 1e-3, "hi": 0.5},
            "k_neighbors": {"type": "int", "lo": 1, "hi": 10}},
    "BA":  {"pop": {"type": "int", "lo": 20, "hi": 80}, "fmax": {"type": "float", "lo": 0.5, "hi": 5.0},
            "A0": {"type": "float", "lo": 0.3, "hi": 0.95}, "r0": {"type": "float", "lo": 0.1, "hi": 0.95},
            "local_step": {"type": "float_log", "lo": 1e-5, "hi": 1e-1}},
    "ALO": {"pop": {"type": "int", "lo": 20, "hi": 80},
            "intensity_max": {"type": "float_log", "lo": 1e2, "hi": 1e6}},
    "PSO_GBEST": {"pop": {"type": "int", "lo": 20, "hi": 80}, "w": {"type": "float", "lo": 0.3, "hi": 0.9},
                 "c1": {"type": "float", "lo": 0.5, "hi": 2.5}, "c2": {"type": "float", "lo": 0.5, "hi": 2.5}},
    "ES_1P1": {"pop": {"type": "int", "lo": 10, "hi": 60}, "sigma0": {"type": "float_log", "lo": 1e-3, "hi": 3.0}},
}

def sample_params(rng: np.random.Generator, space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for k, spec in space.items():
        typ = spec["type"]
        if typ == "float_log":
            lo, hi = float(spec["lo"]), float(spec["hi"])
            x = 10 ** rng.uniform(np.log10(lo), np.log10(hi))
            params[k] = float(x)
        elif typ == "float":
            lo, hi = float(spec["lo"]), float(spec["hi"])
            params[k] = float(rng.uniform(lo, hi))
        elif typ == "int":
            lo, hi = int(spec["lo"]), int(spec["hi"])
            params[k] = int(rng.integers(lo, hi + 1))
        else:
            raise ValueError(f"Unknown param type: {typ}")
    return params

def coerce_params_for_budget(params: Dict[str, Any], min_budget: int) -> Dict[str, Any]:
    p = dict(params)
    if "pop" in p:
        p["pop"] = int(min(p["pop"], max(5, min_budget // 2)))
    return p

def default_params_midpoint(space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    p = {}
    for k, spec in space.items():
        typ = spec["type"]
        if typ == "int":
            p[k] = int(round((int(spec["lo"]) + int(spec["hi"])) / 2))
        elif typ == "float":
            p[k] = float((float(spec["lo"]) + float(spec["hi"])) / 2.0)
        elif typ == "float_log":
            lo, hi = float(spec["lo"]), float(spec["hi"])
            p[k] = float(10 ** ((math.log10(lo) + math.log10(hi)) / 2.0))
        else:
            raise ValueError(typ)
    return p

# -----------------------------
# Running & aggregation
# -----------------------------

def run_one(algo: str, f: Callable[[np.ndarray], float], dim: int, bounds: Tuple[float, float],
            budget: int, seed: int, params: Dict[str, Any]) -> float:
    runner = ALGO_RUNNERS[algo]
    best, _evals = runner(f, dim, bounds, budget, seed, **params)
    return float(best)

def stable_run_seed(algo: str, params: Dict[str, Any], problem: str, instance_id: int, budget: int, r: int, base_seed: int) -> int:
    blob = json.dumps(params, sort_keys=True)
    key = f"{algo}|{blob}|{problem}|{instance_id}|{budget}|{r}|{base_seed}"
    return 1_000_000 * stable_hash_mod(key, 1_000_000_000)

def make_instances(suite: List[ProblemSpec], instances_per_problem: int):
    insts = []
    for spec in suite:
        for inst in range(instances_per_problem):
            shift_seed = 10_000 * stable_hash_mod(spec.name, 1000) + inst
            insts.append((spec.name, inst, spec, shift_seed))
    return insts

def split_instances(insts, dev_frac: float, seed: int):
    rg = rng_from(seed)
    idx = np.arange(len(insts))
    rg.shuffle(idx)
    k = int(round(len(insts) * dev_frac))
    dev = [insts[i] for i in idx[:k]]
    hold = [insts[i] for i in idx[k:]]
    return dev, hold

def run_grid_for_algo(algo: str, params: Dict[str, Any], insts, dim: int, budgets: List[int], R: int, base_seed: int,
                      progress_desc: str, progress_enable: bool):
    rows = []
    total = len(insts) * len(budgets) * R
    pb = tqdm(total=total, desc=progress_desc, leave=False, disable=not progress_enable)
    try:
        for (pname, inst_id, spec, shift_seed) in insts:
            f = make_shifted_objective(spec, dim=dim, shift_seed=shift_seed)
            for b in budgets:
                for r in range(R):
                    seed = stable_run_seed(algo, params, pname, inst_id, b, r, base_seed)
                    best = run_one(algo, f, dim, spec.bounds, b, seed, params)
                    rows.append({
                        "domain": "cont",
                        "problem": pname,
                        "instance_id": inst_id,
                        "shift_seed": shift_seed,
                        "algo_variant": algo,
                        "budget": int(b),
                        "rep": int(r),
                        "seed": int(seed),
                        "best": float(best),
                    })
                    pb.update(1)
    finally:
        pb.close()
    return rows

def build_instance_table(run_rows, target_tols: Dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame(run_rows)
    out_rows = []
    for (domain, problem, instance_id, algo, budget), g in df.groupby(
        ["domain", "problem", "instance_id", "algo_variant", "budget"], as_index=False
    ):
        trials = int(g.shape[0])
        mean_best = float(g["best"].mean())
        median_best = float(g["best"].median())
        min_best = float(g["best"].min())
        max_best = float(g["best"].max())

        for tgt, tol in target_tols.items():
            succ = int(np.sum(g["best"].to_numpy() <= float(tol)))
            beta_mean, beta_p05, beta_p95 = beta_posterior(succ, trials)
            out_rows.append({
                "domain": domain,
                "problem": problem,
                "instance_id": instance_id,
                "algo_variant": algo,
                "budget": int(budget),
                "target": tgt,
                "successes": succ,
                "trials": trials,
                "beta_mean": beta_mean,
                "beta_p05": beta_p05,
                "beta_p95": beta_p95,
                "mean_best": mean_best,
                "median_best": median_best,
                "min_best": min_best,
                "max_best": max_best,
            })
    return pd.DataFrame(out_rows)

# -----------------------------
# Tuning objective + successive halving
# -----------------------------

def auc_over_budgets(budgets: List[int], y: np.ndarray) -> float:
    b = np.asarray(budgets, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    bb = b[m]
    yy = y[m]
    return float(np.trapz(yy, bb) / (bb.max() - bb.min()))

def parse_kv_weights(s: str) -> Dict[str, float]:
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=")
        out[k.strip()] = float(v.strip())
    return out

def score_candidate(instance_table: pd.DataFrame,
                    budgets: List[int],
                    targets: List[str],
                    target_weights: Dict[str, float],
                    objective: str = "hybrid") -> float:
    g = instance_table.groupby(["budget", "target"], as_index=False).agg(
        mean_beta=("beta_mean", "mean"),
        mean_best=("mean_best", "mean"),
    )
    total = 0.0
    for tgt in targets:
        gt = g[g["target"] == tgt].set_index("budget")
        vals = np.array([gt.loc[b, "mean_beta"] if b in gt.index else np.nan for b in budgets], dtype=float)
        auc = auc_over_budgets(budgets, vals)
        if np.isfinite(auc):
            total += target_weights.get(tgt, 1.0) * auc

        if objective == "hybrid":
            qvals = np.array([gt.loc[b, "mean_best"] if b in gt.index else np.nan for b in budgets], dtype=float)
            q = np.nanmean(qvals)
            if np.isfinite(q):
                total += 0.05 * (-math.log(1.0 + max(0.0, float(q))))
    return float(total)

@dataclass(frozen=True)
class Rung:
    name: str
    instances_cap: int
    budgets: List[int]
    R: int
    keep_frac: float

def default_rungs(all_budgets: List[int]) -> List[Rung]:
    b = sorted(all_budgets)
    b_small = [b[0], b[-1]] if len(b) >= 2 else b
    return [
        Rung("rung1", instances_cap=20, budgets=b_small, R=3, keep_frac=0.33),
        Rung("rung2", instances_cap=999999, budgets=b, R=5, keep_frac=0.33),
    ]

def tune_algo(algo: str,
              space: Dict[str, Dict[str, Any]],
              dev_insts,
              dim: int,
              budgets_all: List[int],
              targets: List[str],
              target_tols: Dict[str, float],
              target_weights: Dict[str, float],
              n_candidates: int,
              seed: int,
              out_dir: Path,
              tune_objective: str,
              progress_enable: bool) -> Dict[str, Any]:

    rg = rng_from(seed + 1000 * stable_hash_mod(algo, 1000))
    rungs = default_rungs(budgets_all)

    candidates = [coerce_params_for_budget(sample_params(rg, space), min(budgets_all)) for _ in range(n_candidates)]
    history_rows = []

    alive = candidates
    for rung in rungs:
        insts = dev_insts[: min(len(dev_insts), rung.instances_cap)]

        desc = f"TUNE {algo} | {rung.name} ({len(alive)} cand)"
        pb = tqdm(total=len(alive), desc=desc, leave=True, disable=not progress_enable)

        scored = []
        try:
            for params in alive:
                t0 = time.time()
                run_rows = run_grid_for_algo(
                    algo, params, insts, dim, rung.budgets, rung.R, base_seed=seed,
                    progress_desc=f"  runs {algo}/{rung.name}", progress_enable=progress_enable
                )
                inst_table = build_instance_table(run_rows, target_tols)
                score = score_candidate(inst_table, rung.budgets, targets, target_weights, objective=tune_objective)
                dt = time.time() - t0
                scored.append((score, params))
                history_rows.append({
                    "algo": algo,
                    "rung": rung.name,
                    "score": float(score),
                    "seconds": float(dt),
                    "params_json": json.dumps(params, sort_keys=True),
                    **params,
                })
                pb.update(1)
        finally:
            pb.close()

        scored.sort(key=lambda x: x[0], reverse=True)
        keep = max(1, int(math.ceil(len(scored) * rung.keep_frac)))
        alive = [p for _, p in scored[:keep]]

    best = alive[0]
    algo_dir = out_dir / algo
    ensure_dir(algo_dir)
    (algo_dir / "best_params.json").write_text(json.dumps(best, indent=2, sort_keys=True))
    pd.DataFrame(history_rows).to_csv(algo_dir / "tuning_history.csv", index=False)
    return best

# -----------------------------
# CLI + orchestration
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--algos", type=str, default="GWO,WOA,MFO,FA,BA,ALO",
                    help="Comma-separated algo list. Available: " + ",".join(ALGO_RUNNERS.keys()))
    ap.add_argument("--budgets", type=str, default="300,500,800,1000")
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--instances_per_problem", type=int, default=20)
    ap.add_argument("--dev_frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--R_eval", type=int, default=10)
    ap.add_argument("--n_candidates", type=int, default=60)
    ap.add_argument("--tune_objective", type=str, default="hybrid", choices=["hybrid", "success_auc"])

    ap.add_argument("--target_tols", type=str, default="easy=1e-1,med=1e-2,hard=1e-3",
                    help="Comma list k=v. easy should be easiest (largest tol).")
    ap.add_argument("--target_weights", type=str, default="easy=1,med=2,hard=3")
    ap.add_argument("--skip_tuning", action="store_true")

    ap.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    return ap.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    progress_enable = (not args.no_progress)

    budgets = sorted({int(x) for x in args.budgets.split(",") if x.strip()})
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    for a in algos:
        if a not in ALGO_RUNNERS:
            raise SystemExit(f"Unknown algo: {a}")

    suite = make_suite()
    insts = make_instances(suite, args.instances_per_problem)
    dev_insts, hold_insts = split_instances(insts, args.dev_frac, args.seed)

    target_tols = parse_target_tols(args.target_tols)
    targets = list(target_tols.keys())
    target_weights = parse_kv_weights(args.target_weights)

    (out_dir / "README.txt").write_text(
        f"algos={algos}\n"
        f"budgets={budgets}\n"
        f"dim={args.dim}\n"
        f"instances_per_problem={args.instances_per_problem}\n"
        f"dev_frac={args.dev_frac} (dev={len(dev_insts)}, holdout={len(hold_insts)})\n"
        f"R_eval={args.R_eval}\n"
        f"target_tols={target_tols}\n"
        f"tune_objective={args.tune_objective}\n"
        f"progress={'on' if progress_enable else 'off'}\n"
    )

    tuning_root = out_dir / "tuning"
    ensure_dir(tuning_root)

    best_params_map: Dict[str, Dict[str, Any]] = {}

    algo_pb = tqdm(algos, desc="ALGORITHMS", leave=True, disable=not progress_enable)
    for algo in algo_pb:
        if hasattr(algo_pb, "set_postfix_str"):
            algo_pb.set_postfix_str(algo)

        if args.skip_tuning:
            space = DEFAULT_SPACES.get(algo, {})
            best = coerce_params_for_budget(default_params_midpoint(space), min(budgets)) if space else {}
            best_params_map[algo] = best
            algo_dir = tuning_root / algo
            ensure_dir(algo_dir)
            (algo_dir / "best_params.json").write_text(json.dumps(best, indent=2, sort_keys=True))
            (algo_dir / "tuning_history.csv").write_text("skipped_tuning\n")
        else:
            space = DEFAULT_SPACES.get(algo, {})
            if not space:
                best_params_map[algo] = {}
                continue
            best = tune_algo(
                algo, space, dev_insts, args.dim, budgets, targets, target_tols,
                target_weights, args.n_candidates, args.seed, tuning_root,
                args.tune_objective, progress_enable=progress_enable
            )
            best_params_map[algo] = best

    eval_root = out_dir / "eval"
    ensure_dir(eval_root)

    all_rows = []
    eval_pb = tqdm(algos, desc="EVAL (holdout)", leave=True, disable=not progress_enable)
    for algo in eval_pb:
        params = coerce_params_for_budget(best_params_map.get(algo, {}), min(budgets))
        rows = run_grid_for_algo(
            algo, params, hold_insts, args.dim, budgets, args.R_eval, base_seed=args.seed + 999,
            progress_desc=f"EVAL runs {algo}", progress_enable=progress_enable
        )
        all_rows.extend(rows)

    runs_df = pd.DataFrame(all_rows)
    runs_df.to_csv(eval_root / "runs_detail.csv", index=False)

    inst_table = build_instance_table(all_rows, target_tols)
    inst_table.to_csv(eval_root / "instance_algo_budget_summary.csv", index=False)

    fig_dir = eval_root / "figs"
    ensure_dir(fig_dir)
    g = inst_table.groupby(["algo_variant", "budget", "target"], as_index=False)["beta_mean"].mean()
    for tgt in targets:
        sub = g[g["target"] == tgt]
        fig = plt.figure(figsize=(8, 4))
        ax = plt.gca()
        for algo in algos:
            s2 = sub[sub["algo_variant"] == algo].sort_values("budget")
            ax.plot(s2["budget"].to_numpy(), s2["beta_mean"].to_numpy(), marker="o", linewidth=1, label=algo)
        ax.set_title(f"Mean posterior success vs budget ({tgt})")
        ax.set_xlabel("Budget")
        ax.set_ylabel("beta_mean")
        ax.legend(ncol=3, fontsize=8, frameon=True)
        fig.tight_layout()
        fig.savefig(fig_dir / f"success_vs_budget_{tgt}.png", dpi=200)
        plt.close(fig)

    print("[OK] wrote:", eval_root / "instance_algo_budget_summary.csv")
    if _tqdm is None:
        print("[NOTE] tqdm not installed; progress bars were disabled. Install with: pip install tqdm")

if __name__ == "__main__":
    main()
