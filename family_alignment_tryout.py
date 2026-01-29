#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Family Alignment Tryout (Win-frequency + Probabilistic clustering)

Goal:
  Examine claim: GWO/MFO/WOA/FA/BA behave like PSO-family,
                ALO behaves like Evolution-family
  by running many trials and using win-frequency signatures.

Outputs:
  - runs.csv                 : per run (instance, algo, seed, budget, best)
  - instance_summary.csv     : per (instance, algo, budget) aggregated (median over seeds)
  - wins_by_group.csv        : win counts per group (func, dim-bin, budget-tier)
  - membership.csv           : P(PSO-family) vs P(Evolution-family) for each algorithm

Dependencies: numpy only

Run:
  python family_alignment_tryout.py run --out_dir out
  python family_alignment_tryout.py analyze --out_dir out

You can also do both in one:
  python family_alignment_tryout.py all --out_dir out
"""

from __future__ import annotations
import argparse, csv, os, json, math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np


# ----------------------------
# Utilities
# ----------------------------
def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen–Shannon divergence between distributions p and q."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))

def dim_bin(d: int) -> str:
    if d <= 5:
        return "d<=5"
    if d <= 10:
        return "d<=10"
    return "d>10"

def make_rotation_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthogonal matrix via QR."""
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    # Fix sign to make determinant positive more often (optional)
    diag = np.sign(np.diag(R))
    Q = Q * diag
    return Q

class EvalCounter:
    """Objective wrapper counting evaluations and tracking best."""
    def __init__(self, obj, budget: int):
        self.obj = obj
        self.budget = int(budget)
        self.evals = 0
        self.best = float("inf")

    def __call__(self, x: np.ndarray) -> float:
        if self.evals >= self.budget:
            # caller should avoid this; we return +inf for safety
            return float("inf")
        self.evals += 1
        y = float(self.obj(x))
        if y < self.best:
            self.best = y
        return y

    def remaining(self) -> int:
        return self.budget - self.evals


def dirichlet_posterior_samples(counts: np.ndarray, alpha0: float, n_mc: int, rng: np.random.Generator):
    alpha = counts.astype(float) + float(alpha0)
    return rng.dirichlet(alpha, size=n_mc)  # shape [n_mc, K]


# ----------------------------
# BBOB-lite objective suite (shift + rotation)
# ----------------------------
def f_sphere(z):      # min at 0
    return float(np.sum(z * z))

def f_rosenbrock(z):  # standard Rosenbrock has min at ones; use shift inside
    y = z + 1.0
    return float(np.sum(100.0 * (y[1:] - y[:-1] ** 2) ** 2 + (1.0 - y[:-1]) ** 2))

def f_rastrigin(z):
    A = 10.0
    return float(A * z.size + np.sum(z * z - A * np.cos(2.0 * np.pi * z)))

def f_ackley(z):
    d = z.size
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    s1 = np.sum(z * z)
    s2 = np.sum(np.cos(c * z))
    return float(-a * np.exp(-b * np.sqrt(s1 / d)) - np.exp(s2 / d) + a + math.e)

def f_griewank(z):
    d = z.size
    sum_sq = np.sum(z * z) / 4000.0
    prod_cos = 1.0
    for i in range(d):
        prod_cos *= math.cos(z[i] / math.sqrt(i + 1))
    return float(sum_sq - prod_cos + 1.0)

FUNC_MAP = {
    "sphere": f_sphere,
    "rosenbrock": f_rosenbrock,
    "rastrigin": f_rastrigin,
    "ackley": f_ackley,
    "griewank": f_griewank,
}


@dataclass
class Instance:
    instance_id: str
    func: str
    dim: int
    bounds: np.ndarray  # [d,2]
    shift: np.ndarray   # [d]
    rot: np.ndarray     # [d,d]

    def make_objective(self):
        base = FUNC_MAP[self.func]
        shift = self.shift
        rot = self.rot

        def obj(x: np.ndarray) -> float:
            z = rot @ (x - shift)
            return base(z)
        return obj


def make_instances(funcs: List[str], dims: List[int], n_per_cell: int, master_seed: int) -> List[Instance]:
    rng = np.random.default_rng(master_seed)
    instances: List[Instance] = []
    for fn in funcs:
        for d in dims:
            bounds = np.array([[-5.0, 5.0]] * d, dtype=float)
            for k in range(n_per_cell):
                shift = rng.uniform(bounds[:, 0], bounds[:, 1])
                rot = make_rotation_matrix(d, rng)
                iid = f"{fn}_d{d}_i{k:03d}"
                instances.append(Instance(
                    instance_id=iid,
                    func=fn,
                    dim=d,
                    bounds=bounds,
                    shift=shift,
                    rot=rot,
                ))
    return instances


# ----------------------------
# Optimizers: PSO anchors
# ----------------------------
def pso_standard(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    V = rng.normal(scale=0.1, size=(pop, d))
    pbest = X.copy()
    pbest_y = np.array([evalf(X[i]) for i in range(pop)], dtype=float)
    g_idx = int(np.argmin(pbest_y))
    gbest = pbest[g_idx].copy()
    gbest_y = float(pbest_y[g_idx])

    w, c1, c2 = 0.7, 1.5, 1.5
    while evalf.remaining() >= pop:
        r1 = rng.random((pop, d))
        r2 = rng.random((pop, d))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = clamp(X + V, lo, hi)

        y = np.array([evalf(X[i]) for i in range(pop)], dtype=float)
        improved = y < pbest_y
        pbest[improved] = X[improved]
        pbest_y[improved] = y[improved]
        g_idx = int(np.argmin(pbest_y))
        if pbest_y[g_idx] < gbest_y:
            gbest_y = float(pbest_y[g_idx])
            gbest = pbest[g_idx].copy()

    return evalf.best, evalf.evals


def pso_constriction(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    # Clerc constriction PSO params (commonly used)
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    V = rng.normal(scale=0.1, size=(pop, d))
    pbest = X.copy()
    pbest_y = np.array([evalf(X[i]) for i in range(pop)], dtype=float)
    g_idx = int(np.argmin(pbest_y))
    gbest = pbest[g_idx].copy()

    chi = 0.7298
    c1, c2 = 1.49618, 1.49618

    while evalf.remaining() >= pop:
        r1 = rng.random((pop, d))
        r2 = rng.random((pop, d))
        V = chi * (V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X))
        X = clamp(X + V, lo, hi)

        y = np.array([evalf(X[i]) for i in range(pop)], dtype=float)
        improved = y < pbest_y
        pbest[improved] = X[improved]
        pbest_y[improved] = y[improved]
        g_idx = int(np.argmin(pbest_y))
        gbest = pbest[g_idx].copy()

    return evalf.best, evalf.evals


def pso_ring(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    # Ring topology PSO: neighborhood best from left/right
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    V = rng.normal(scale=0.1, size=(pop, d))
    pbest = X.copy()
    pbest_y = np.array([evalf(X[i]) for i in range(pop)], dtype=float)

    w, c1, c2 = 0.65, 1.6, 1.6

    def neigh_best(i: int):
        left = (i - 1) % pop
        right = (i + 1) % pop
        idxs = [left, i, right]
        j = idxs[int(np.argmin(pbest_y[idxs]))]
        return pbest[j]

    while evalf.remaining() >= pop:
        for i in range(pop):
            nb = neigh_best(i)
            r1 = rng.random(d)
            r2 = rng.random(d)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (nb - X[i])
            X[i] = clamp(X[i] + V[i], lo, hi)
        y = np.array([evalf(X[i]) for i in range(pop)], dtype=float)
        improved = y < pbest_y
        pbest[improved] = X[improved]
        pbest_y[improved] = y[improved]

    return evalf.best, evalf.evals


# ----------------------------
# Evolution-family anchors: DE, GA, ES
# ----------------------------
def de_rand_1_bin(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    F, CR = 0.8, 0.9
    fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)

    while evalf.remaining() > 0:
        for i in range(pop):
            if evalf.remaining() <= 0:
                break
            idxs = [j for j in range(pop) if j != i]
            a, b, c = rng.choice(idxs, size=3, replace=False)
            mutant = X[a] + F * (X[b] - X[c])
            mutant = clamp(mutant, lo, hi)
            trial = X[i].copy()
            jrand = int(rng.integers(0, d))
            for j in range(d):
                if rng.random() < CR or j == jrand:
                    trial[j] = mutant[j]
            y = evalf(trial)
            if y < fit[i]:
                X[i] = trial
                fit[i] = y

    return evalf.best, evalf.evals


def ga_real_coded(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 40):
    # Simple real-coded GA: tournament + SBX-like blend + Gaussian mutation
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)

    def tournament(k=3):
        idx = rng.integers(0, pop, size=k)
        return int(idx[np.argmin(fit[idx])])

    while evalf.remaining() >= pop:
        newX = np.empty_like(X)
        for i in range(pop):
            p1 = X[tournament()]
            p2 = X[tournament()]
            alpha = rng.random(d)
            child = alpha * p1 + (1 - alpha) * p2
            # mutation
            if rng.random() < 0.9:
                child += rng.normal(0, 0.1 * (hi - lo), size=d)
            child = clamp(child, lo, hi)
            newX[i] = child
        X = newX
        fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)

    return evalf.best, evalf.evals


def es_mu_lambda(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, mu: int = 10, lam: int = 40):
    # (mu, lambda)-ES with log-normal self-adaptation (global step-size per individual)
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(mu, d))
    sigma = np.full(mu, 0.3 * np.mean(hi - lo))
    fit = np.array([evalf(X[i]) for i in range(mu)], dtype=float)

    tau = 1.0 / math.sqrt(d)

    while evalf.remaining() >= lam:
        # offspring
        kids = []
        kids_sigma = []
        for _ in range(lam):
            p = int(rng.integers(0, mu))
            sig = sigma[p] * math.exp(tau * rng.normal())
            child = X[p] + rng.normal(0, sig, size=d)
            child = clamp(child, lo, hi)
            kids.append(child)
            kids_sigma.append(sig)
        kids = np.array(kids, dtype=float)
        kids_sigma = np.array(kids_sigma, dtype=float)
        kid_fit = np.array([evalf(kids[i]) for i in range(lam)], dtype=float)

        # select best mu
        idx = np.argsort(kid_fit)[:mu]
        X = kids[idx]
        sigma = kids_sigma[idx]
        fit = kid_fit[idx]

    return evalf.best, evalf.evals


# ----------------------------
# "Bestial" algorithms under test
# ----------------------------
def gwo(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    # Grey Wolf Optimizer (simplified canonical update)
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)

    it = 0
    max_it = 1_000_000  # bounded by eval budget anyway
    while evalf.remaining() >= pop:
        idx = np.argsort(fit)
        alpha, beta, delta = X[idx[0]], X[idx[1]], X[idx[2]]
        a = 2.0 * (1.0 - it / max(1, max_it))
        newX = np.empty_like(X)
        for i in range(pop):
            A1 = 2 * a * rng.random(d) - a
            C1 = 2 * rng.random(d)
            D_alpha = np.abs(C1 * alpha - X[i])
            X1 = alpha - A1 * D_alpha

            A2 = 2 * a * rng.random(d) - a
            C2 = 2 * rng.random(d)
            D_beta = np.abs(C2 * beta - X[i])
            X2 = beta - A2 * D_beta

            A3 = 2 * a * rng.random(d) - a
            C3 = 2 * rng.random(d)
            D_delta = np.abs(C3 * delta - X[i])
            X3 = delta - A3 * D_delta

            newX[i] = (X1 + X2 + X3) / 3.0
        X = clamp(newX, lo, hi)
        fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)
        it += 1

    return evalf.best, evalf.evals


def woa(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    # Whale Optimization Algorithm (simplified canonical)
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)
    best_idx = int(np.argmin(fit))
    best = X[best_idx].copy()

    t = 0
    max_t = 1_000_000
    while evalf.remaining() >= pop:
        a = 2.0 * (1.0 - t / max(1, max_t))
        for i in range(pop):
            r1 = rng.random(d)
            r2 = rng.random(d)
            A = 2 * a * r1 - a
            C = 2 * r2
            p = rng.random()
            if p < 0.5:
                if np.linalg.norm(A) < 1:
                    D = np.abs(C * best - X[i])
                    X[i] = best - A * D
                else:
                    j = int(rng.integers(0, pop))
                    Xrand = X[j]
                    D = np.abs(C * Xrand - X[i])
                    X[i] = Xrand - A * D
            else:
                # spiral
                D = np.abs(best - X[i])
                l = rng.uniform(-1, 1, size=d)
                b = 1.0
                X[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best
            X[i] = clamp(X[i], lo, hi)

        fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)
        best_idx = int(np.argmin(fit))
        best = X[best_idx].copy()
        t += 1

    return evalf.best, evalf.evals


def mfo(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    # Moth-Flame Optimization (simplified)
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    moths = rng.uniform(lo, hi, size=(pop, d))
    moth_fit = np.array([evalf(moths[i]) for i in range(pop)], dtype=float)

    t = 0
    max_t = 1_000_000
    while evalf.remaining() >= pop:
        idx = np.argsort(moth_fit)
        flames = moths[idx].copy()
        flames_fit = moth_fit[idx].copy()

        a = -1 + t * (-1 / max(1, max_t))  # from  -1 to -2 roughly; simplified
        new_moths = np.empty_like(moths)
        for i in range(pop):
            flame = flames[min(i, pop - 1)]
            D = np.abs(flame - moths[i])
            l = rng.uniform(-1, 1, size=d)
            b = 1.0
            new_moths[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + flame
        moths = clamp(new_moths, lo, hi)
        moth_fit = np.array([evalf(moths[i]) for i in range(pop)], dtype=float)
        t += 1

    return evalf.best, evalf.evals


def firefly(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    # Firefly Algorithm (compute-managed version)
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)

    beta0 = 1.0
    gamma = 1.0 / (np.mean(hi - lo) ** 2 + 1e-12)
    alpha = 0.2 * np.mean(hi - lo)

    while evalf.remaining() >= pop:
        # move each firefly toward a randomly chosen better one among top-K
        idx = np.argsort(fit)
        topk = idx[:max(3, pop // 5)]
        newX = X.copy()
        for i in range(pop):
            # pick brighter (lower fit)
            candidates = topk[fit[topk] < fit[i]]
            if candidates.size == 0:
                # random walk
                newX[i] = X[i] + rng.normal(0, alpha, size=d)
            else:
                j = int(rng.choice(candidates))
                rij = np.linalg.norm(X[i] - X[j])
                beta = beta0 * math.exp(-gamma * rij * rij)
                step = beta * (X[j] - X[i]) + rng.normal(0, alpha, size=d)
                newX[i] = X[i] + step
            newX[i] = clamp(newX[i], lo, hi)

        X = newX
        fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)
        alpha *= 0.999  # mild cooling

    return evalf.best, evalf.evals


def bat(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    # Bat Algorithm (simplified)
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    V = np.zeros((pop, d), dtype=float)
    fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)

    best_idx = int(np.argmin(fit))
    best = X[best_idx].copy()
    best_y = float(fit[best_idx])

    fmin, fmax = 0.0, 2.0
    A = np.full(pop, 0.9)
    r = np.full(pop, 0.5)

    while evalf.remaining() >= pop:
        for i in range(pop):
            freq = fmin + (fmax - fmin) * rng.random()
            V[i] = V[i] + (X[i] - best) * freq
            cand = X[i] + V[i]
            cand = clamp(cand, lo, hi)

            if rng.random() > r[i]:
                # local random walk around best
                eps = rng.normal(0, 0.1 * np.mean(hi - lo), size=d)
                cand = best + eps
                cand = clamp(cand, lo, hi)

            yc = evalf(cand)
            if yc < fit[i] and rng.random() < A[i]:
                X[i] = cand
                fit[i] = yc
                A[i] *= 0.99
                r[i] = r[i] * (1 - math.exp(-0.01))
            if fit[i] < best_y:
                best_y = float(fit[i])
                best = X[i].copy()

    return evalf.best, evalf.evals


def alo(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    # Ant Lion Optimizer (managed-cost approximation of canonical ALO)
    # Core idea: ants sample around selected antlions + elite with shrinking radius.
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]

    antlions = rng.uniform(lo, hi, size=(pop, d))
    al_fit = np.array([evalf(antlions[i]) for i in range(pop)], dtype=float)

    elite_idx = int(np.argmin(al_fit))
    elite = antlions[elite_idx].copy()
    elite_fit = float(al_fit[elite_idx])

    t = 0
    max_t = 1_000_000
    while evalf.remaining() >= pop:
        # roulette selection weights (better => higher weight)
        inv = 1.0 / (al_fit - np.min(al_fit) + 1e-12)
        probs = inv / np.sum(inv)

        shrink = 0.5 * (1 - t / max(1, max_t))  # decreases over time
        radius = shrink * np.mean(hi - lo)

        ants = np.empty((pop, d), dtype=float)
        for i in range(pop):
            j = int(rng.choice(pop, p=probs))
            center = 0.5 * (antlions[j] + elite)
            ants[i] = center + rng.normal(0, radius, size=d)
            ants[i] = clamp(ants[i], lo, hi)

        ant_fit = np.array([evalf(ants[i]) for i in range(pop)], dtype=float)

        # Replace antlions if ants are better (elitism-ish)
        for i in range(pop):
            if ant_fit[i] < al_fit[i]:
                antlions[i] = ants[i]
                al_fit[i] = ant_fit[i]

        elite_idx = int(np.argmin(al_fit))
        if al_fit[elite_idx] < elite_fit:
            elite_fit = float(al_fit[elite_idx])
            elite = antlions[elite_idx].copy()

        t += 1

    return evalf.best, evalf.evals


# ----------------------------
# Algorithm registry
# ----------------------------
ALGORITHMS = {
    # PSO anchors
    "PSO_STD": pso_standard,
    "PSO_CONSTR": pso_constriction,
    "PSO_RING": pso_ring,

    # Evolution anchors (broad: DE/GA/ES)
    "DE": de_rand_1_bin,
    "GA": ga_real_coded,
    "ES_ML": es_mu_lambda,

    # Under-test (paper list)
    "GWO": gwo,
    "MFO": mfo,
    "WOA": woa,
    "FA": firefly,
    "BA": bat,
    "ALO": alo,
}

PSO_ANCHORS = ["PSO_STD", "PSO_CONSTR", "PSO_RING"]
EVO_ANCHORS = ["DE", "GA", "ES_ML"]


# ----------------------------
# Runner: produce runs.csv + instance_summary.csv
# ----------------------------
def run_trials(out_dir: str,
               funcs: List[str],
               dims: List[int],
               n_per_cell: int,
               seeds: List[int],
               budget_mults: List[int],
               master_seed: int,
               pop: int):
    os.makedirs(out_dir, exist_ok=True)
    runs_path = os.path.join(out_dir, "runs.csv")
    summary_path = os.path.join(out_dir, "instance_summary.csv")

    instances = make_instances(funcs, dims, n_per_cell, master_seed)

    # Per-run log
    with open(runs_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "instance_id", "func", "dim", "budget", "budget_tier",
            "algo", "seed", "best", "evals",
            "meta_json"
        ])
        w.writeheader()

        for inst in instances:
            obj = inst.make_objective()
            bounds = inst.bounds
            for mult in budget_mults:
                budget = int(mult * inst.dim)
                tier = f"{mult}xD"
                for algo_name, algo_fn in ALGORITHMS.items():
                    for s in seeds:
                        rng = np.random.default_rng((hash((master_seed, inst.instance_id, tier, algo_name, s)) & 0xFFFFFFFF))
                        evalf = EvalCounter(obj, budget=budget)
                        # Some algos need population; pass pop for consistency where supported
                        try:
                            best, evals = algo_fn(evalf=evalf, bounds=bounds, rng=rng, pop=pop)
                        except TypeError:
                            best, evals = algo_fn(evalf=evalf, bounds=bounds, rng=rng)  # for ES params etc.
                        w.writerow({
                            "instance_id": inst.instance_id,
                            "func": inst.func,
                            "dim": inst.dim,
                            "budget": budget,
                            "budget_tier": tier,
                            "algo": algo_name,
                            "seed": s,
                            "best": best,
                            "evals": evals,
                            "meta_json": json.dumps({"shift": True, "rot": True}, ensure_ascii=False),
                        })

    # Aggregate to instance_summary: median over seeds
    # We'll do streaming read to avoid extra deps.
    rows = []
    with open(runs_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["dim"] = int(row["dim"])
            row["budget"] = int(row["budget"])
            row["best"] = float(row["best"])
            row["seed"] = int(row["seed"])
            rows.append(row)

    key = lambda rr: (rr["instance_id"], rr["budget_tier"], rr["algo"])
    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    for rr in rows:
        grouped.setdefault(key(rr), []).append(rr["best"])
        meta[rr["instance_id"]] = {"func": rr["func"], "dim": rr["dim"]}

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "instance_id", "func", "dim", "dim_bin",
            "budget_tier", "algo", "best_median"
        ])
        w.writeheader()
        for (iid, tier, algo), vals in grouped.items():
            fn = meta[iid]["func"]
            d = meta[iid]["dim"]
            w.writerow({
                "instance_id": iid,
                "func": fn,
                "dim": d,
                "dim_bin": dim_bin(d),
                "budget_tier": tier,
                "algo": algo,
                "best_median": float(np.median(vals)),
            })

    print(f"[OK] wrote {runs_path}")
    print(f"[OK] wrote {summary_path}")


# ----------------------------
# Analysis: win counts -> Dirichlet -> fingerprint -> membership prob
# ----------------------------
def analyze(out_dir: str, alpha0: float, mc: int, seed: int):
    summary_path = os.path.join(out_dir, "instance_summary.csv")
    wins_path = os.path.join(out_dir, "wins_by_group.csv")
    membership_path = os.path.join(out_dir, "membership.csv")

    # Load summary
    rows = []
    with open(summary_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["dim"] = int(row["dim"])
            row["best_median"] = float(row["best_median"])
            rows.append(row)

    algos = list(ALGORITHMS.keys())
    algo_index = {a: i for i, a in enumerate(algos)}

    # Determine winner per (instance_id, budget_tier)
    # We have one row per (instance, algo, tier).
    inst_keys = sorted(set((rr["instance_id"], rr["budget_tier"]) for rr in rows))
    winners = []  # (group_key, winner_algo)
    for iid, tier in inst_keys:
        subset = [rr for rr in rows if rr["instance_id"] == iid and rr["budget_tier"] == tier]
        subset.sort(key=lambda rr: rr["best_median"])
        winner_algo = subset[0]["algo"]
        fn = subset[0]["func"]
        db = subset[0]["dim_bin"]
        group_key = (fn, db, tier)  # bucket = function × dim_bin × budget_tier
        winners.append((group_key, winner_algo))

    groups = sorted(set(g for g, _ in winners))
    G = len(groups)
    K = len(algos)

    # Win counts per group
    counts_by_group: Dict[Tuple[str, str, str], np.ndarray] = {}
    for g in groups:
        counts_by_group[g] = np.zeros(K, dtype=int)
    for g, a in winners:
        counts_by_group[g][algo_index[a]] += 1

    # Write wins_by_group.csv
    with open(wins_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "func", "dim_bin", "budget_tier", "algo", "win_count", "n_instances_in_group"
        ])
        w.writeheader()
        for g in groups:
            fn, db, tier = g
            c = counts_by_group[g]
            n = int(np.sum(c))
            for a in algos:
                w.writerow({
                    "func": fn,
                    "dim_bin": db,
                    "budget_tier": tier,
                    "algo": a,
                    "win_count": int(c[algo_index[a]]),
                    "n_instances_in_group": n,
                })

    # Dirichlet posterior samples per group: P_best[g] ~ Dirichlet(alpha0 + counts)
    rng = np.random.default_rng(seed)
    # P_samples shape: [mc, K, G]
    P_samples = np.zeros((mc, K, G), dtype=float)
    for gi, g in enumerate(groups):
        c = counts_by_group[g]
        P_samples[:, :, gi] = dirichlet_posterior_samples(c, alpha0=alpha0, n_mc=mc, rng=rng)

    # Fingerprint for algo a: vector over groups = p_best(a|group); normalize over groups to form distribution
    # V_samples shape: [mc, K, G] -> normalize along G
    V = P_samples.copy()
    Vsum = np.sum(V, axis=2, keepdims=True)
    V = V / np.clip(Vsum, 1e-12, None)

    # Centroids per family: average fingerprints of anchor algos
    pso_ids = [algo_index[a] for a in PSO_ANCHORS]
    evo_ids = [algo_index[a] for a in EVO_ANCHORS]

    C_pso = np.mean(V[:, pso_ids, :], axis=1)  # [mc, G]
    C_evo = np.mean(V[:, evo_ids, :], axis=1)  # [mc, G]
    # normalize again
    C_pso = C_pso / np.clip(C_pso.sum(axis=1, keepdims=True), 1e-12, None)
    C_evo = C_evo / np.clip(C_evo.sum(axis=1, keepdims=True), 1e-12, None)

    # Membership probability: P( JSD(algo, PSO) < JSD(algo, EVO) )
    report = []
    for a in algos:
        ai = algo_index[a]
        d_pso = np.array([js_divergence(V[s, ai, :], C_pso[s, :]) for s in range(mc)], dtype=float)
        d_evo = np.array([js_divergence(V[s, ai, :], C_evo[s, :]) for s in range(mc)], dtype=float)
        p_pso = float(np.mean(d_pso < d_evo))
        # credible interval for p_pso (binomial approx via beta posterior)
        # (for reporting only; keeps dependencies minimal)
        wins = int(np.sum(d_pso < d_evo))
        alpha = 1 + wins
        beta = 1 + (mc - wins)
        # Monte Carlo interval from Beta
        samp = rng.beta(alpha, beta, size=5000)
        p05, p95 = float(np.quantile(samp, 0.05)), float(np.quantile(samp, 0.95))

        report.append({
            "algo": a,
            "P_PSO_family": p_pso,
            "P_PSO_p05": p05,
            "P_PSO_p95": p95,
            "P_EVO_family": 1.0 - p_pso,
        })

    # Write membership.csv
    with open(membership_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(sorted(report, key=lambda rr: rr["P_PSO_family"], reverse=True))

    print(f"[OK] wrote {wins_path}")
    print(f"[OK] wrote {membership_path}\n")

    # Console summary: focus on the paper list + anchors
    focus = PSO_ANCHORS + EVO_ANCHORS + ["GWO", "MFO", "WOA", "FA", "BA", "ALO"]
    focus = [a for a in focus if a in ALGORITHMS]
    print("=== Membership summary (higher P_PSO => more PSO-like) ===")
    rep_map = {r["algo"]: r for r in report}
    for a in focus:
        r = rep_map[a]
        print(f"{a:10s}  P(PSO)={r['P_PSO_family']:.3f}  [p05={r['P_PSO_p05']:.3f}, p95={r['P_PSO_p95']:.3f}]")

    print("\nInterpretation rule-of-thumb:")
    print("  P(PSO) > 0.95  => strongly PSO-family")
    print("  P(PSO) < 0.05  => strongly Evolution-family")
    print("  otherwise      => ambiguous / needs more instances or better features")


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_run = sub.add_parser("run")
    ap_run.add_argument("--out_dir", type=str, default="out")
    ap_run.add_argument("--funcs", type=str, default="sphere,rosenbrock,rastrigin,ackley,griewank")
    ap_run.add_argument("--dims", type=str, default="5,10")
    ap_run.add_argument("--n_per_cell", type=int, default=5)
    ap_run.add_argument("--seeds", type=str, default="0,1,2")
    ap_run.add_argument("--budget_mults", type=str, default="200,1000")  # budgets = mult * D
    ap_run.add_argument("--master_seed", type=int, default=123)
    ap_run.add_argument("--pop", type=int, default=30)

    ap_an = sub.add_parser("analyze")
    ap_an.add_argument("--out_dir", type=str, default="out")
    ap_an.add_argument("--alpha0", type=float, default=1.0)
    ap_an.add_argument("--mc", type=int, default=4000)
    ap_an.add_argument("--seed", type=int, default=2026)

    ap_all = sub.add_parser("all")
    ap_all.add_argument("--out_dir", type=str, default="out")
    ap_all.add_argument("--funcs", type=str, default="sphere,rosenbrock,rastrigin,ackley,griewank")
    ap_all.add_argument("--dims", type=str, default="5,10")
    ap_all.add_argument("--n_per_cell", type=int, default=5)
    ap_all.add_argument("--seeds", type=str, default="0,1,2")
    ap_all.add_argument("--budget_mults", type=str, default="200,1000")
    ap_all.add_argument("--master_seed", type=int, default=123)
    ap_all.add_argument("--pop", type=int, default=30)
    ap_all.add_argument("--alpha0", type=float, default=1.0)
    ap_all.add_argument("--mc", type=int, default=4000)
    ap_all.add_argument("--seed", type=int, default=2026)

    args = ap.parse_args()

    if args.cmd in ("run", "all"):
        funcs = [s.strip() for s in args.funcs.split(",") if s.strip()]
        dims = [int(s.strip()) for s in args.dims.split(",") if s.strip()]
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        mults = [int(s.strip()) for s in args.budget_mults.split(",") if s.strip()]

        run_trials(
            out_dir=args.out_dir,
            funcs=funcs,
            dims=dims,
            n_per_cell=args.n_per_cell,
            seeds=seeds,
            budget_mults=mults,
            master_seed=args.master_seed,
            pop=args.pop,
        )

    if args.cmd in ("analyze", "all"):
        analyze(
            out_dir=args.out_dir,
            alpha0=args.alpha0,
            mc=args.mc,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
