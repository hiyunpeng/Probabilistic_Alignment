#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Family Alignment Tryout v3 (improved competitiveness + robust membership)

Key Improvements vs v2
----------------------
A) Fix under-test algorithms' schedules (GWO/WOA/MFO/ALO):
   - max_iter tied to eval budget (evalf.budget // pop), so 'a' and shrink schedules actually progress.

B) Add rank-based profile membership (ShapeRank), more robust than regret normalization.

C) Add evidence gating:
   - If algorithm has too few Top-k hits AND too few Top-1 wins, label as INSUFFICIENT.
   - Avoid over-interpreting algorithms that are consistently non-competitive.

Outputs
-------
out_dir/
  runs.csv
  instance_summary.csv
  wins_by_group_top1.csv
  hits_by_group_top{k}.csv
  membership.csv  (Top1 + Topk + ShapeRegret + ShapeRank + Combined + status)

Dependencies: numpy only

Run
---
python family_alignment_tryout.py all --out_dir out --topk 3
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


# ============================================================
# Utilities
# ============================================================

def clamp(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
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


def cosine_sim(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def dim_bin(d: int) -> str:
    if d <= 5:
        return "d<=5"
    if d <= 10:
        return "d<=10"
    return "d>10"


def make_rotation_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    """Random orthogonal matrix via QR decomposition."""
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    diag = np.sign(np.diag(R))
    Q = Q * diag
    return Q


class EvalCounter:
    """Objective wrapper counting evaluations and tracking best-so-far."""
    def __init__(self, obj, budget: int):
        self.obj = obj
        self.budget = int(budget)
        self.evals = 0
        self.best = float("inf")

    def __call__(self, x: np.ndarray) -> float:
        if self.evals >= self.budget:
            return float("inf")
        self.evals += 1
        y = float(self.obj(x))
        if y < self.best:
            self.best = y
        return y

    def remaining(self) -> int:
        return self.budget - self.evals


def dirichlet_samples(counts: np.ndarray, alpha0: float, n_mc: int, rng: np.random.Generator) -> np.ndarray:
    alpha = counts.astype(float) + float(alpha0)
    return rng.dirichlet(alpha, size=n_mc)  # [n_mc, K]


def beta_samples(success: int, total: int, alpha0: float, n_mc: int, rng: np.random.Generator) -> np.ndarray:
    a = float(alpha0 + success)
    b = float(alpha0 + (total - success))
    return rng.beta(a, b, size=n_mc)  # [n_mc]


# ============================================================
# Objective suite (BBOB-lite): shift + rotation
# ============================================================

def f_sphere(z: np.ndarray) -> float:
    return float(np.sum(z * z))


def f_rosenbrock(z: np.ndarray) -> float:
    y = z + 1.0
    return float(np.sum(100.0 * (y[1:] - y[:-1] ** 2) ** 2 + (1.0 - y[:-1]) ** 2))


def f_rastrigin(z: np.ndarray) -> float:
    A = 10.0
    return float(A * z.size + np.sum(z * z - A * np.cos(2.0 * np.pi * z)))


def f_ackley(z: np.ndarray) -> float:
    d = z.size
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    s1 = np.sum(z * z)
    s2 = np.sum(np.cos(c * z))
    return float(-a * np.exp(-b * np.sqrt(s1 / d)) - np.exp(s2 / d) + a + math.e)


def f_griewank(z: np.ndarray) -> float:
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
    bounds: np.ndarray
    shift: np.ndarray
    rot: np.ndarray

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
                instances.append(Instance(iid, fn, d, bounds, shift, rot))
    return instances


# ============================================================
# Algorithms
# ============================================================

# -------- PSO anchors --------
def pso_standard(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    V = rng.normal(scale=0.1, size=(pop, d))
    pbest = X.copy()
    pbest_y = np.array([evalf(X[i]) for i in range(pop)], dtype=float)
    gbest = pbest[int(np.argmin(pbest_y))].copy()

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
        gbest = pbest[int(np.argmin(pbest_y))].copy()

    return evalf.best, evalf.evals


def pso_constriction(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    V = rng.normal(scale=0.1, size=(pop, d))
    pbest = X.copy()
    pbest_y = np.array([evalf(X[i]) for i in range(pop)], dtype=float)
    gbest = pbest[int(np.argmin(pbest_y))].copy()

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
        gbest = pbest[int(np.argmin(pbest_y))].copy()

    return evalf.best, evalf.evals


def pso_ring(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    V = rng.normal(scale=0.1, size=(pop, d))
    pbest = X.copy()
    pbest_y = np.array([evalf(X[i]) for i in range(pop)], dtype=float)

    w, c1, c2 = 0.65, 1.6, 1.6

    def neigh_best(i: int) -> np.ndarray:
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


# -------- Evolution anchors --------
def de_rand_1_bin(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)

    F, CR = 0.8, 0.9
    while evalf.remaining() > 0:
        for i in range(pop):
            if evalf.remaining() <= 0:
                break
            idxs = [j for j in range(pop) if j != i]
            a, b, c = rng.choice(idxs, size=3, replace=False)
            mutant = clamp(X[a] + F * (X[b] - X[c]), lo, hi)
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
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)

    def tournament(k=3) -> int:
        idx = rng.integers(0, pop, size=k)
        return int(idx[np.argmin(fit[idx])])

    while evalf.remaining() >= pop:
        newX = np.empty_like(X)
        for i in range(pop):
            p1 = X[tournament()]
            p2 = X[tournament()]
            alpha = rng.random(d)
            child = alpha * p1 + (1 - alpha) * p2
            if rng.random() < 0.9:
                child += rng.normal(0, 0.1 * (hi - lo), size=d)
            newX[i] = clamp(child, lo, hi)
        X = newX
        fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)

    return evalf.best, evalf.evals


def es_mu_lambda(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, mu: int = 10, lam: int = 40):
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(mu, d))
    sigma = np.full(mu, 0.3 * np.mean(hi - lo))
    _ = np.array([evalf(X[i]) for i in range(mu)], dtype=float)

    tau = 1.0 / math.sqrt(d)
    while evalf.remaining() >= lam:
        kids = []
        kids_sigma = []
        for _ in range(lam):
            p = int(rng.integers(0, mu))
            sig = sigma[p] * math.exp(tau * rng.normal())
            child = clamp(X[p] + rng.normal(0, sig, size=d), lo, hi)
            kids.append(child)
            kids_sigma.append(sig)
        kids = np.array(kids, dtype=float)
        kids_sigma = np.array(kids_sigma, dtype=float)
        kid_fit = np.array([evalf(kids[i]) for i in range(lam)], dtype=float)
        idx = np.argsort(kid_fit)[:mu]
        X = kids[idx]
        sigma = kids_sigma[idx]

    return evalf.best, evalf.evals


# -------- Under-test algorithms (FIXED schedules) --------
def gwo(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)

    max_iter = max(1, evalf.budget // pop)  # FIX: schedule tied to budget
    it = 0
    while evalf.remaining() >= pop:
        idx = np.argsort(fit)
        alpha, beta, delta = X[idx[0]], X[idx[1]], X[idx[2]]
        a = 2.0 - 2.0 * (it / max_iter)

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
        if it >= max_iter:
            break

    return evalf.best, evalf.evals


def woa(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)
    best = X[int(np.argmin(fit))].copy()

    max_iter = max(1, evalf.budget // pop)  # FIX
    t = 0
    while evalf.remaining() >= pop:
        a = 2.0 - 2.0 * (t / max_iter)
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
                D = np.abs(best - X[i])
                l = float(rng.uniform(-1, 1))  # closer to standard WOA: scalar l
                b = 1.0
                X[i] = D * math.exp(b * l) * math.cos(2 * math.pi * l) + best

            X[i] = clamp(X[i], lo, hi)

        fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)
        best = X[int(np.argmin(fit))].copy()
        t += 1
        if t >= max_iter:
            break

    return evalf.best, evalf.evals


def mfo(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    moths = rng.uniform(lo, hi, size=(pop, d))
    moth_fit = np.array([evalf(moths[i]) for i in range(pop)], dtype=float)

    max_iter = max(1, evalf.budget // pop)  # FIX
    t = 0
    while evalf.remaining() >= pop:
        idx = np.argsort(moth_fit)
        flames = moths[idx].copy()

        # typical MFO uses a decreasing parameter; we keep a minimal but progressing schedule
        new_moths = np.empty_like(moths)
        for i in range(pop):
            flame = flames[min(i, pop - 1)]
            D = np.abs(flame - moths[i])
            l = float(rng.uniform(-1, 1))  # scalar l often used
            b = 1.0
            new_moths[i] = D * math.exp(b * l) * math.cos(2 * math.pi * l) + flame

        moths = clamp(new_moths, lo, hi)
        moth_fit = np.array([evalf(moths[i]) for i in range(pop)], dtype=float)
        t += 1
        if t >= max_iter:
            break

    return evalf.best, evalf.evals


def firefly(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)

    beta0 = 1.0
    gamma = 1.0 / (np.mean(hi - lo) ** 2 + 1e-12)
    alpha = 0.2 * np.mean(hi - lo)

    while evalf.remaining() >= pop:
        idx = np.argsort(fit)
        elite = idx[:max(3, pop // 5)]
        newX = X.copy()
        for i in range(pop):
            better = elite[fit[elite] < fit[i]]
            if better.size == 0:
                newX[i] = X[i] + rng.normal(0, alpha, size=d)
            else:
                j = int(rng.choice(better))
                rij = np.linalg.norm(X[i] - X[j])
                beta = beta0 * math.exp(-gamma * rij * rij)
                newX[i] = X[i] + beta * (X[j] - X[i]) + rng.normal(0, alpha, size=d)
            newX[i] = clamp(newX[i], lo, hi)
        X = newX
        fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)
        alpha *= 0.995  # slightly faster decay

    return evalf.best, evalf.evals


def bat(evalf: EvalCounter, bounds: np.ndarray, rng: np.random.Generator, pop: int = 30):
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    V = np.zeros((pop, d), dtype=float)
    fit = np.array([evalf(X[i]) for i in range(pop)], dtype=float)

    best = X[int(np.argmin(fit))].copy()
    best_y = float(np.min(fit))

    fmin, fmax = 0.0, 2.0
    A = np.full(pop, 0.9)
    r = np.full(pop, 0.5)

    while evalf.remaining() >= pop:
        for i in range(pop):
            freq = fmin + (fmax - fmin) * rng.random()
            V[i] = V[i] + (X[i] - best) * freq
            cand = clamp(X[i] + V[i], lo, hi)

            if rng.random() > r[i]:
                cand = clamp(best + rng.normal(0, 0.1 * np.mean(hi - lo), size=d), lo, hi)

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
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    antlions = rng.uniform(lo, hi, size=(pop, d))
    al_fit = np.array([evalf(antlions[i]) for i in range(pop)], dtype=float)

    elite_idx = int(np.argmin(al_fit))
    elite = antlions[elite_idx].copy()
    elite_fit = float(al_fit[elite_idx])

    max_iter = max(1, evalf.budget // pop)  # FIX
    t = 0
    while evalf.remaining() >= pop:
        inv = 1.0 / (al_fit - np.min(al_fit) + 1e-12)
        probs = inv / np.sum(inv)

        # shrink radius across iterations (now meaningful)
        shrink = 1.0 - (t / max_iter)
        radius = max(1e-9, shrink) * np.mean(hi - lo)

        ants = np.empty((pop, d), dtype=float)
        for i in range(pop):
            j = int(rng.choice(pop, p=probs))
            center = 0.5 * (antlions[j] + elite)
            ants[i] = clamp(center + rng.normal(0, 0.5 * radius, size=d), lo, hi)

        ant_fit = np.array([evalf(ants[i]) for i in range(pop)], dtype=float)
        improved = ant_fit < al_fit
        antlions[improved] = ants[improved]
        al_fit[improved] = ant_fit[improved]

        elite_idx = int(np.argmin(al_fit))
        if al_fit[elite_idx] < elite_fit:
            elite_fit = float(al_fit[elite_idx])
            elite = antlions[elite_idx].copy()

        t += 1
        if t >= max_iter:
            break

    return evalf.best, evalf.evals


ALGORITHMS = {
    "PSO_STD": pso_standard,
    "PSO_CONSTR": pso_constriction,
    "PSO_RING": pso_ring,
    "DE": de_rand_1_bin,
    "GA": ga_real_coded,
    "ES_ML": es_mu_lambda,
    "GWO": gwo,
    "MFO": mfo,
    "WOA": woa,
    "FA": firefly,
    "BA": bat,
    "ALO": alo,
}

PSO_ANCHORS = ["PSO_STD", "PSO_CONSTR", "PSO_RING"]
EVO_ANCHORS = ["DE", "GA", "ES_ML"]


# ============================================================
# Runner
# ============================================================

def run_trials(
    out_dir: str,
    funcs: List[str],
    dims: List[int],
    n_per_cell: int,
    seeds: List[int],
    budget_mults: List[int],
    master_seed: int,
    pop: int,
):
    os.makedirs(out_dir, exist_ok=True)
    runs_path = os.path.join(out_dir, "runs.csv")
    summary_path = os.path.join(out_dir, "instance_summary.csv")

    instances = make_instances(funcs, dims, n_per_cell, master_seed)

    with open(runs_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "instance_id", "func", "dim", "budget", "budget_tier",
                "algo", "seed", "best", "evals", "meta_json"
            ],
        )
        w.writeheader()

        for inst in instances:
            obj = inst.make_objective()
            bounds = inst.bounds
            for mult in budget_mults:
                budget = int(mult * inst.dim)
                tier = f"{mult}xD"
                for algo_name, algo_fn in ALGORITHMS.items():
                    for s in seeds:
                        seed_int = (hash((master_seed, inst.instance_id, tier, algo_name, s)) & 0xFFFFFFFF)
                        rng = np.random.default_rng(seed_int)

                        evalf = EvalCounter(obj, budget=budget)
                        try:
                            best, evals = algo_fn(evalf=evalf, bounds=bounds, rng=rng, pop=pop)
                        except TypeError:
                            best, evals = algo_fn(evalf=evalf, bounds=bounds, rng=rng)

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

    # aggregate median over seeds
    rows = []
    with open(runs_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "instance_id": row["instance_id"],
                "func": row["func"],
                "dim": int(row["dim"]),
                "budget_tier": row["budget_tier"],
                "algo": row["algo"],
                "best": float(row["best"]),
            })

    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for rr in rows:
        grouped.setdefault((rr["instance_id"], rr["budget_tier"], rr["algo"]), []).append(rr["best"])
        meta[rr["instance_id"]] = {"func": rr["func"], "dim": rr["dim"]}

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["instance_id", "func", "dim", "dim_bin", "budget_tier", "algo", "best_median"],
        )
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


# ============================================================
# Analysis
# ============================================================

def analyze(out_dir: str, alpha0: float, mc: int, seed: int, topk: int, min_hits: int, min_wins: int):
    if topk < 1:
        raise ValueError("--topk must be >= 1")

    summary_path = os.path.join(out_dir, "instance_summary.csv")
    wins_top1_path = os.path.join(out_dir, "wins_by_group_top1.csv")
    hits_topk_path = os.path.join(out_dir, f"hits_by_group_top{topk}.csv")
    membership_path = os.path.join(out_dir, "membership.csv")

    rows = []
    with open(summary_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "instance_id": row["instance_id"],
                "func": row["func"],
                "dim": int(row["dim"]),
                "dim_bin": row["dim_bin"],
                "budget_tier": row["budget_tier"],
                "algo": row["algo"],
                "best_median": float(row["best_median"]),
            })

    algos = list(ALGORITHMS.keys())
    algo_index = {a: i for i, a in enumerate(algos)}
    K = len(algos)

    inst_tier_keys = sorted(set((rr["instance_id"], rr["budget_tier"]) for rr in rows))

    per_inst_tier: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    group_of_inst_tier: Dict[Tuple[str, str], Tuple[str, str, str]] = {}

    for iid, tier in inst_tier_keys:
        subset = [rr for rr in rows if rr["instance_id"] == iid and rr["budget_tier"] == tier]
        per_inst_tier[(iid, tier)] = subset
        fn = subset[0]["func"]
        db = subset[0]["dim_bin"]
        group_of_inst_tier[(iid, tier)] = (fn, db, tier)

    groups = sorted(set(group_of_inst_tier.values()))
    G = len(groups)
    group_index = {g: i for i, g in enumerate(groups)}

    rng = np.random.default_rng(seed)

    top1_counts_by_group = {g: np.zeros(K, dtype=int) for g in groups}
    topk_hits_by_group = {g: np.zeros(K, dtype=int) for g in groups}
    n_inst_by_group = {g: 0 for g in groups}

    # ShapeRegret: score = 1 - normalized regret
    shape_regret_by_group: Dict[Tuple[str, str, str], List[np.ndarray]] = {g: [] for g in groups}
    # ShapeRank: score = (K - rank)/(K-1)
    shape_rank_by_group: Dict[Tuple[str, str, str], List[np.ndarray]] = {g: [] for g in groups}

    for it_key, subset in per_inst_tier.items():
        g = group_of_inst_tier[it_key]
        n_inst_by_group[g] += 1

        subset_sorted = sorted(subset, key=lambda rr: rr["best_median"])
        winner = subset_sorted[0]["algo"]
        top1_counts_by_group[g][algo_index[winner]] += 1

        for rr in subset_sorted[:min(topk, len(subset_sorted))]:
            topk_hits_by_group[g][algo_index[rr["algo"]]] += 1

        # Build per-algo best array
        vals = np.full(K, np.nan, dtype=float)
        for rr in subset:
            vals[algo_index[rr["algo"]]] = rr["best_median"]

        # ShapeRegret
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        norm_regret = (vals - vmin) / (vmax - vmin + 1e-12)
        score_regret = 1.0 - norm_regret
        shape_regret_by_group[g].append(score_regret)

        # ShapeRank
        rank_score = np.zeros(K, dtype=float)
        # ranks from sorted order
        for rank, rr in enumerate(subset_sorted, start=1):  # 1 is best
            ai = algo_index[rr["algo"]]
            rank_score[ai] = (K - rank) / (K - 1 + 1e-12)
        shape_rank_by_group[g].append(rank_score)

    # write wins_by_group_top1.csv
    with open(wins_top1_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["func", "dim_bin", "budget_tier", "algo", "win_count", "n_instances_in_group"])
        w.writeheader()
        for g in groups:
            fn, db, tier = g
            c = top1_counts_by_group[g]
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

    # write hits_by_group_topk.csv
    with open(hits_topk_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["func", "dim_bin", "budget_tier", "algo", f"hit_count_top{topk}", "n_instances_in_group"])
        w.writeheader()
        for g in groups:
            fn, db, tier = g
            hits = topk_hits_by_group[g]
            n = int(n_inst_by_group[g])
            for a in algos:
                w.writerow({
                    "func": fn,
                    "dim_bin": db,
                    "budget_tier": tier,
                    "algo": a,
                    f"hit_count_top{topk}": int(hits[algo_index[a]]),
                    "n_instances_in_group": n,
                })

    pso_ids = [algo_index[a] for a in ["PSO_STD", "PSO_CONSTR", "PSO_RING"]]
    evo_ids = [algo_index[a] for a in ["DE", "GA", "ES_ML"]]

    # ---------- Top-1 membership ----------
    P_top1 = np.zeros((mc, K, G), dtype=float)
    for g in groups:
        gi = group_index[g]
        P_top1[:, :, gi] = dirichlet_samples(top1_counts_by_group[g], alpha0=alpha0, n_mc=mc, rng=rng)
    V_top1 = P_top1 / np.clip(P_top1.sum(axis=2, keepdims=True), 1e-12, None)

    C_pso_top1 = np.mean(V_top1[:, pso_ids, :], axis=1)
    C_evo_top1 = np.mean(V_top1[:, evo_ids, :], axis=1)
    C_pso_top1 = C_pso_top1 / np.clip(C_pso_top1.sum(axis=1, keepdims=True), 1e-12, None)
    C_evo_top1 = C_evo_top1 / np.clip(C_evo_top1.sum(axis=1, keepdims=True), 1e-12, None)

    P_PSO_top1_mc = np.zeros((mc, K), dtype=float)
    for s in range(mc):
        for ai in range(K):
            dp = js_divergence(V_top1[s, ai, :], C_pso_top1[s, :])
            de = js_divergence(V_top1[s, ai, :], C_evo_top1[s, :])
            P_PSO_top1_mc[s, ai] = 1.0 if dp < de else 0.0

    # ---------- Top-k membership ----------
    P_hit = np.zeros((mc, K, G), dtype=float)
    for g in groups:
        gi = group_index[g]
        n = int(n_inst_by_group[g])
        hits = topk_hits_by_group[g]
        for ai in range(K):
            P_hit[:, ai, gi] = beta_samples(int(hits[ai]), n, alpha0=alpha0, n_mc=mc, rng=rng)

    V_topk = P_hit / np.clip(P_hit.sum(axis=2, keepdims=True), 1e-12, None)
    C_pso_topk = np.mean(V_topk[:, pso_ids, :], axis=1)
    C_evo_topk = np.mean(V_topk[:, evo_ids, :], axis=1)
    C_pso_topk = C_pso_topk / np.clip(C_pso_topk.sum(axis=1, keepdims=True), 1e-12, None)
    C_evo_topk = C_evo_topk / np.clip(C_evo_topk.sum(axis=1, keepdims=True), 1e-12, None)

    P_PSO_topk_mc = np.zeros((mc, K), dtype=float)
    for s in range(mc):
        for ai in range(K):
            dp = js_divergence(V_topk[s, ai, :], C_pso_topk[s, :])
            de = js_divergence(V_topk[s, ai, :], C_evo_topk[s, :])
            P_PSO_topk_mc[s, ai] = 1.0 if dp < de else 0.0

    # ---------- Shape membership (Regret + Rank) ----------
    def bootstrap_shape(shape_by_group: Dict[Tuple[str, str, str], List[np.ndarray]]) -> np.ndarray:
        mats = {}
        for g in groups:
            mats[g] = np.stack(shape_by_group[g], axis=0) if len(shape_by_group[g]) else np.zeros((0, K), dtype=float)

        out = np.zeros((mc, K), dtype=float)
        for s in range(mc):
            M = np.zeros((K, G), dtype=float)
            for g in groups:
                gi = group_index[g]
                mat = mats[g]
                if mat.shape[0] == 0:
                    continue
                idx = rng.integers(0, mat.shape[0], size=mat.shape[0])
                M[:, gi] = mat[idx].mean(axis=0)

            # z-score per algo across groups
            mean_a = M.mean(axis=1, keepdims=True)
            std_a = M.std(axis=1, keepdims=True) + 1e-12
            Mz = (M - mean_a) / std_a

            Cp = Mz[pso_ids].mean(axis=0)
            Ce = Mz[evo_ids].mean(axis=0)

            for ai in range(K):
                sp = cosine_sim(Mz[ai], Cp)
                se = cosine_sim(Mz[ai], Ce)
                out[s, ai] = 1.0 if sp > se else 0.0
        return out

    P_PSO_shape_regret_mc = bootstrap_shape(shape_regret_by_group)
    P_PSO_shape_rank_mc = bootstrap_shape(shape_rank_by_group)

    # ---------- Summaries ----------
    def summarize_binary(mc_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = mc_mat.mean(axis=0)
        lo = np.quantile(mc_mat, 0.05, axis=0)
        hi = np.quantile(mc_mat, 0.95, axis=0)
        return mean, lo, hi

    p1_mean, p1_lo, p1_hi = summarize_binary(P_PSO_top1_mc)
    pk_mean, pk_lo, pk_hi = summarize_binary(P_PSO_topk_mc)
    sr_mean, sr_lo, sr_hi = summarize_binary(P_PSO_shape_regret_mc)
    rk_mean, rk_lo, rk_hi = summarize_binary(P_PSO_shape_rank_mc)

    # Combined shape (simple average)
    shape_comb_mc = 0.5 * P_PSO_shape_regret_mc + 0.5 * P_PSO_shape_rank_mc
    sc_mean, sc_lo, sc_hi = summarize_binary(shape_comb_mc)

    # totals
    total_wins = np.zeros(K, dtype=int)
    total_hits = np.zeros(K, dtype=int)
    for g in groups:
        total_wins += top1_counts_by_group[g]
        total_hits += topk_hits_by_group[g]

    # evidence gate
    report = []
    for ai, a in enumerate(algos):
        wins = int(total_wins[ai])
        hits = int(total_hits[ai])
        sufficient = (wins >= min_wins) or (hits >= min_hits)
        status = "OK" if sufficient else "INSUFFICIENT"

        report.append({
            "algo": a,
            "status": status,
            "total_top1_wins": wins,
            f"total_top{topk}_hits": hits,

            "P_PSO_top1": float(p1_mean[ai]),
            "P_PSO_top1_p05": float(p1_lo[ai]),
            "P_PSO_top1_p95": float(p1_hi[ai]),

            f"P_PSO_top{topk}": float(pk_mean[ai]),
            f"P_PSO_top{topk}_p05": float(pk_lo[ai]),
            f"P_PSO_top{topk}_p95": float(pk_hi[ai]),

            "P_PSO_shape_regret": float(sr_mean[ai]),
            "P_PSO_shape_regret_p05": float(sr_lo[ai]),
            "P_PSO_shape_regret_p95": float(sr_hi[ai]),

            "P_PSO_shape_rank": float(rk_mean[ai]),
            "P_PSO_shape_rank_p05": float(rk_lo[ai]),
            "P_PSO_shape_rank_p95": float(rk_hi[ai]),

            "P_PSO_shape_combined": float(sc_mean[ai]),
            "P_PSO_shape_combined_p05": float(sc_lo[ai]),
            "P_PSO_shape_combined_p95": float(sc_hi[ai]),
        })

    with open(membership_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(sorted(report, key=lambda rr: rr["P_PSO_shape_combined"], reverse=True))

    print(f"[OK] wrote {wins_top1_path}")
    print(f"[OK] wrote {hits_topk_path}")
    print(f"[OK] wrote {membership_path}\n")

    focus = ["PSO_STD", "PSO_CONSTR", "PSO_RING", "DE", "GA", "ES_ML", "GWO", "MFO", "WOA", "FA", "BA", "ALO"]
    rep_map = {r["algo"]: r for r in report}

    print("=== Membership summary (with evidence gate) ===")
    for a in focus:
        r = rep_map[a]
        print(
            f"{a:10s}  status={r['status']:12s}  "
            f"Top1={r['P_PSO_top1']:.3f}  "
            f"Top{topk}={r[f'P_PSO_top{topk}']:.3f}  "
            f"ShapeComb={r['P_PSO_shape_combined']:.3f}  "
            f"(wins={r['total_top1_wins']}, hits={r[f'total_top{topk}_hits']})"
        )


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_run = sub.add_parser("run")
    ap_run.add_argument("--out_dir", type=str, default="out")
    ap_run.add_argument("--funcs", type=str, default="sphere,rosenbrock,rastrigin,ackley,griewank")
    ap_run.add_argument("--dims", type=str, default="5,10")
    ap_run.add_argument("--n_per_cell", type=int, default=5)
    ap_run.add_argument("--seeds", type=str, default="0,1,2")
    ap_run.add_argument("--budget_mults", type=str, default="200,1000")
    ap_run.add_argument("--master_seed", type=int, default=123)
    ap_run.add_argument("--pop", type=int, default=30)

    ap_an = sub.add_parser("analyze")
    ap_an.add_argument("--out_dir", type=str, default="out")
    ap_an.add_argument("--alpha0", type=float, default=1.0)
    ap_an.add_argument("--mc", type=int, default=4000)
    ap_an.add_argument("--seed", type=int, default=2026)
    ap_an.add_argument("--topk", type=int, default=3)
    ap_an.add_argument("--min_hits", type=int, default=5, help="evidence gate: minimum total Top-k hits")
    ap_an.add_argument("--min_wins", type=int, default=1, help="evidence gate: minimum total Top-1 wins")

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
    ap_all.add_argument("--topk", type=int, default=3)
    ap_all.add_argument("--min_hits", type=int, default=5)
    ap_all.add_argument("--min_wins", type=int, default=1)

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
            topk=args.topk,
            min_hits=args.min_hits,
            min_wins=args.min_wins,
        )


if __name__ == "__main__":
    main()
