#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
family_signature_tryout_v2.py
=============================

Non-probabilistic "family" discovery via BEHAVIORAL SIGNATURES,
fixed to avoid NaN-driven clustering by ensuring every algorithm runs in BOTH domains.

Algorithms (same set in continuous + combinatorial)
--------------------------------------------------
PSO, DE, ES, GWO, WOA, SA, HC

Continuous problems: sphere, rastrigin, ackley (shift + rotation)
Combinatorial problems: maxcut (bitstring), knapsack (bitstring)

Outputs (out_dir)
-----------------
runs_features.csv   # per run feature vector
algo_features.csv   # aggregated signature per algorithm (mean over runs)
dist_cosine.csv     # cosine distance matrix on standardized signatures
clusters.csv        # cluster assignment (cut of hierarchical clustering)
dendrogram.csv      # merge steps (average-linkage)

Run
---
python family_signature_tryout_v2.py all --out_dir out_sig2 --k 3
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Any

import numpy as np

EPS = 1e-12


# ============================================================
# Utils
# ============================================================

def make_outdir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def clamp(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def auc_trapz(y: np.ndarray) -> float:
    if y.size <= 1:
        return float(y[-1]) if y.size == 1 else 0.0
    x = np.linspace(0.0, 1.0, num=y.size)
    return float(np.trapz(y, x))


def zscore(X: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0) + EPS
    return (X - mu) / sd


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


# ============================================================
# Trace + Features
# ============================================================

@dataclass
class Trace:
    best_so_far: List[float]
    diversity: List[float]
    step: List[float]
    evals: List[int]

    @staticmethod
    def empty() -> "Trace":
        return Trace(best_so_far=[], diversity=[], step=[], evals=[])

    def log(self, best: float, div: float, step: float, evals: int) -> None:
        self.best_so_far.append(float(best))
        self.diversity.append(float(div))
        self.step.append(float(step))
        self.evals.append(int(evals))


def trace_to_features(tr: Trace) -> Dict[str, float]:
    """
    Deterministic signature features (no probability):
      - progress AUC (robust across negative/positive objectives)
      - early/mid/late progress
      - stagnation fraction
      - diversity/step stats
      - relative improvement
    """
    if len(tr.best_so_far) == 0:
        return {k: float("nan") for k in [
            "final_best", "improve_rel", "auc_progress", "p10_progress", "p50_progress", "p90_progress",
            "stagnation_frac", "div_mean", "div_decay", "step_mean", "step_decay", "step_burst"
        ]}

    best = np.array(tr.best_so_far, dtype=float)
    # enforce monotone best-so-far (safety)
    best = np.minimum.accumulate(best)

    start = float(best[0])
    final = float(best[-1])
    improve_abs = start - final  # minimization: positive means improved
    improve_rel = float(improve_abs / (abs(start) + 1.0))

    # progress in [0,1]: 0 at start, 1 at final (if improved)
    if start > final + 1e-12:
        progress = (start - best) / (start - final + EPS)
        progress = np.clip(progress, 0.0, 1.0)
    else:
        progress = np.zeros_like(best)

    auc_progress = auc_trapz(progress)
    n = progress.size
    p10 = float(progress[int(0.1 * (n - 1))]) if n > 1 else float(progress[0])
    p50 = float(progress[int(0.5 * (n - 1))]) if n > 1 else float(progress[0])
    p90 = float(progress[int(0.9 * (n - 1))]) if n > 1 else float(progress[0])

    # stagnation: fraction of checkpoints with no improvement in best-so-far
    tol = 1e-12
    d_improve = (best[:-1] - best[1:]) > tol
    stagnation_frac = float(1.0 - np.mean(d_improve)) if d_improve.size else 1.0

    div = np.array(tr.diversity, dtype=float)
    step = np.array(tr.step, dtype=float)

    div_mean = float(np.mean(div))
    div_decay = float(div[-1] / (div[0] + EPS)) if div.size else 0.0

    step_mean = float(np.mean(step))
    step_decay = float(step[-1] / (step[0] + EPS)) if step.size else 0.0
    step_burst = float(np.std(step) / (np.mean(step) + EPS)) if step.size else 0.0

    return {
        "final_best": final,
        "improve_rel": improve_rel,
        "auc_progress": auc_progress,
        "p10_progress": p10,
        "p50_progress": p50,
        "p90_progress": p90,
        "stagnation_frac": stagnation_frac,
        "div_mean": div_mean,
        "div_decay": div_decay,
        "step_mean": step_mean,
        "step_decay": step_decay,
        "step_burst": step_burst,
    }


# ============================================================
# Continuous benchmarks (shift + rotation)
# ============================================================

def make_rotation_matrix(d: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    diag = np.sign(np.diag(R))
    return Q * diag


def f_sphere(z: np.ndarray) -> float:
    return float(np.sum(z * z))


def f_rastrigin(z: np.ndarray) -> float:
    A = 10.0
    return float(A * z.size + np.sum(z * z - A * np.cos(2.0 * np.pi * z)))


def f_ackley(z: np.ndarray) -> float:
    d = z.size
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    s1 = np.sum(z * z)
    s2 = np.sum(np.cos(c * z))
    return float(-a * np.exp(-b * np.sqrt(s1 / d)) - np.exp(s2 / d) + a + math.e)


CONT_FUNCS: Dict[str, Callable[[np.ndarray], float]] = {
    "sphere": f_sphere,
    "rastrigin": f_rastrigin,
    "ackley": f_ackley,
}


@dataclass
class ContInstance:
    iid: str
    func: str
    dim: int
    bounds: np.ndarray
    shift: np.ndarray
    rot: np.ndarray

    def obj(self, x: np.ndarray) -> float:
        z = self.rot @ (x - self.shift)
        return CONT_FUNCS[self.func](z)


def make_cont_instances(funcs: List[str], dims: List[int], n_per_cell: int, seed: int) -> List[ContInstance]:
    rng = np.random.default_rng(seed)
    insts: List[ContInstance] = []
    for fn in funcs:
        for d in dims:
            bounds = np.array([[-5.0, 5.0]] * d, dtype=float)
            for k in range(n_per_cell):
                shift = rng.uniform(bounds[:, 0], bounds[:, 1])
                rot = make_rotation_matrix(d, rng)
                insts.append(ContInstance(
                    iid=f"{fn}_d{d}_i{k:03d}",
                    func=fn,
                    dim=d,
                    bounds=bounds,
                    shift=shift,
                    rot=rot
                ))
    return insts


def cont_diversity(X: np.ndarray, bounds: np.ndarray) -> float:
    # normalized by mean range to be comparable across dims
    span = float(np.mean(bounds[:, 1] - bounds[:, 0]))
    return float(np.mean(np.std(X, axis=0)) / (span + EPS))


# ============================================================
# Combinatorial benchmarks (bitstring)
# ============================================================

@dataclass
class MaxCutInstance:
    iid: str
    n: int
    W: np.ndarray  # symmetric

    def obj(self, x: np.ndarray) -> float:
        xi = x.astype(int)
        diff = (xi[:, None] != xi[None, :]).astype(float)
        cut = 0.5 * float(np.sum(self.W * diff))
        return -cut  # minimize


@dataclass
class KnapsackInstance:
    iid: str
    w: np.ndarray
    v: np.ndarray
    cap: float

    def obj(self, x: np.ndarray) -> float:
        take = x.astype(int)
        tw = float(np.dot(self.w, take))
        tv = float(np.dot(self.v, take))
        penalty = 0.0
        if tw > self.cap:
            penalty = (tw - self.cap) ** 2
        return -(tv) + 0.01 * penalty


def make_comb_instances(n_graph: int, n_knap: int, n_per: int, seed: int) -> Tuple[List[MaxCutInstance], List[KnapsackInstance]]:
    rng = np.random.default_rng(seed)
    maxcuts, knaps = [], []

    for k in range(n_per):
        W = rng.random((n_graph, n_graph))
        W = 0.5 * (W + W.T)
        np.fill_diagonal(W, 0.0)
        maxcuts.append(MaxCutInstance(iid=f"maxcut_n{n_graph}_i{k:03d}", n=n_graph, W=W))

    for k in range(n_per):
        w = rng.uniform(1.0, 30.0, size=n_knap)
        v = rng.uniform(1.0, 50.0, size=n_knap)
        cap = 0.4 * float(np.sum(w))
        knaps.append(KnapsackInstance(iid=f"knap_n{n_knap}_i{k:03d}", w=w, v=v, cap=cap))

    return maxcuts, knaps


def ham_diversity(X: np.ndarray) -> float:
    # average per-bit std in [0,0.5]
    return float(np.mean(np.std(X, axis=0)))


def ham_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a.astype(int) != b.astype(int)))  # fraction of flipped bits


# ============================================================
# Algorithms (Continuous)
# ============================================================

def algo_pso_cont(inst: ContInstance, budget: int, rng: np.random.Generator, pop: int, log_points: int) -> Tuple[float, Trace]:
    d = inst.dim
    lo, hi = inst.bounds[:, 0], inst.bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    V = rng.normal(scale=0.1, size=(pop, d))

    y = np.array([inst.obj(X[i]) for i in range(pop)], dtype=float)
    evals = pop

    pbest, pbest_y = X.copy(), y.copy()
    g = int(np.argmin(pbest_y))
    best_so_far = float(pbest_y[g])
    gbest = pbest[g].copy()

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def log(step_mean: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best_so_far, cont_diversity(X, inst.bounds), step_mean, evals)

    w, c1, c2 = 0.7, 1.5, 1.5
    log(0.0)

    while evals + pop <= budget:
        X_prev = X.copy()
        r1 = rng.random((pop, d))
        r2 = rng.random((pop, d))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = clamp(X + V, lo, hi)

        y = np.array([inst.obj(X[i]) for i in range(pop)], dtype=float)
        evals += pop

        improved = y < pbest_y
        pbest[improved] = X[improved]
        pbest_y[improved] = y[improved]

        g = int(np.argmin(pbest_y))
        if pbest_y[g] < best_so_far:
            best_so_far = float(pbest_y[g])
            gbest = pbest[g].copy()

        span = float(np.mean(hi - lo))
        step_mean = float(np.mean(np.linalg.norm(X - X_prev, axis=1)) / (span + EPS))
        log(step_mean)

    log(0.0)
    return best_so_far, tr


def algo_de_cont(inst: ContInstance, budget: int, rng: np.random.Generator, pop: int, log_points: int) -> Tuple[float, Trace]:
    d = inst.dim
    lo, hi = inst.bounds[:, 0], inst.bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    fit = np.array([inst.obj(X[i]) for i in range(pop)], dtype=float)
    evals = pop
    best_so_far = float(np.min(fit))

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def log(step_mean: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best_so_far, cont_diversity(X, inst.bounds), step_mean, evals)

    F, CR = 0.8, 0.9
    log(0.0)

    while evals < budget:
        step_sizes = []
        for i in range(pop):
            if evals >= budget:
                break
            idxs = [j for j in range(pop) if j != i]
            a, b, c = rng.choice(idxs, size=3, replace=False)
            mutant = clamp(X[a] + F * (X[b] - X[c]), lo, hi)
            trial = X[i].copy()
            jrand = int(rng.integers(0, d))
            for j in range(d):
                if rng.random() < CR or j == jrand:
                    trial[j] = mutant[j]

            y = inst.obj(trial)
            evals += 1
            step_sizes.append(float(np.linalg.norm(trial - X[i])))

            if y < fit[i]:
                X[i] = trial
                fit[i] = y
                if y < best_so_far:
                    best_so_far = float(y)

        span = float(np.mean(hi - lo))
        step_mean = float(np.mean(step_sizes) / (span + EPS)) if step_sizes else 0.0
        log(step_mean)

    log(0.0)
    return best_so_far, tr


def algo_es_cont(inst: ContInstance, budget: int, rng: np.random.Generator, pop: int, log_points: int) -> Tuple[float, Trace]:
    # (mu, lambda)-ES with elitism (keeps best-so-far stable)
    d = inst.dim
    lo, hi = inst.bounds[:, 0], inst.bounds[:, 1]
    mu = max(5, pop // 3)
    lam = max(mu * 4, pop)

    parents = rng.uniform(lo, hi, size=(mu, d))
    pfit = np.array([inst.obj(parents[i]) for i in range(mu)], dtype=float)
    evals = mu
    best_so_far = float(np.min(pfit))

    sigma = 0.3 * float(np.mean(hi - lo))
    tau = 1.0 / math.sqrt(d)

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def log(step_mean: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best_so_far, cont_diversity(parents, inst.bounds), step_mean, evals)

    log(0.0)

    while evals + lam <= budget:
        steps = []
        kids = np.empty((lam, d), dtype=float)
        kfit = np.empty(lam, dtype=float)

        # self-adapt sigma (global, MVP)
        sigma = float(np.clip(sigma * math.exp(tau * rng.normal()), 1e-6, 1e6))

        for k in range(lam):
            p = int(rng.integers(0, mu))
            child = parents[p] + rng.normal(0, sigma, size=d)
            child = clamp(child, lo, hi)
            kids[k] = child
            steps.append(float(np.linalg.norm(child - parents[p])))
            kfit[k] = inst.obj(child)

        evals += lam
        # elitism: include parents + kids, select best mu
        allX = np.vstack([parents, kids])
        allF = np.concatenate([pfit, kfit])
        idx = np.argsort(allF)[:mu]
        parents = allX[idx]
        pfit = allF[idx]

        if pfit[0] < best_so_far:
            best_so_far = float(pfit[0])

        span = float(np.mean(hi - lo))
        step_mean = float(np.mean(steps) / (span + EPS)) if steps else 0.0
        log(step_mean)

    log(0.0)
    return best_so_far, tr


def algo_gwo_cont(inst: ContInstance, budget: int, rng: np.random.Generator, pop: int, log_points: int) -> Tuple[float, Trace]:
    d = inst.dim
    lo, hi = inst.bounds[:, 0], inst.bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    fit = np.array([inst.obj(X[i]) for i in range(pop)], dtype=float)
    evals = pop
    best_so_far = float(np.min(fit))

    max_iter = max(1, budget // pop)
    it = 0

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def log(step_mean: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best_so_far, cont_diversity(X, inst.bounds), step_mean, evals)

    log(0.0)

    while evals + pop <= budget and it < max_iter:
        idx = np.argsort(fit)
        alpha, beta, delta = X[idx[0]], X[idx[1]], X[idx[2]]
        a = 2.0 - 2.0 * (it / max_iter)

        X_prev = X.copy()
        X_new = np.empty_like(X)
        for i in range(pop):
            A1 = 2 * a * rng.random(d) - a
            C1 = 2 * rng.random(d)
            D1 = np.abs(C1 * alpha - X[i])
            X1 = alpha - A1 * D1

            A2 = 2 * a * rng.random(d) - a
            C2 = 2 * rng.random(d)
            D2 = np.abs(C2 * beta - X[i])
            X2 = beta - A2 * D2

            A3 = 2 * a * rng.random(d) - a
            C3 = 2 * rng.random(d)
            D3 = np.abs(C3 * delta - X[i])
            X3 = delta - A3 * D3

            X_new[i] = (X1 + X2 + X3) / 3.0

        X = clamp(X_new, lo, hi)
        fit = np.array([inst.obj(X[i]) for i in range(pop)], dtype=float)
        evals += pop
        best_so_far = min(best_so_far, float(np.min(fit)))

        span = float(np.mean(hi - lo))
        step_mean = float(np.mean(np.linalg.norm(X - X_prev, axis=1)) / (span + EPS))
        log(step_mean)
        it += 1

    log(0.0)
    return best_so_far, tr


def algo_woa_cont(inst: ContInstance, budget: int, rng: np.random.Generator, pop: int, log_points: int) -> Tuple[float, Trace]:
    d = inst.dim
    lo, hi = inst.bounds[:, 0], inst.bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    fit = np.array([inst.obj(X[i]) for i in range(pop)], dtype=float)
    evals = pop

    j = int(np.argmin(fit))
    best = X[j].copy()
    best_so_far = float(fit[j])

    max_iter = max(1, budget // pop)
    t = 0

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def log(step_mean: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best_so_far, cont_diversity(X, inst.bounds), step_mean, evals)

    log(0.0)

    while evals + pop <= budget and t < max_iter:
        a = 2.0 - 2.0 * (t / max_iter)
        X_prev = X.copy()

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
                    j2 = int(rng.integers(0, pop))
                    Xrand = X[j2]
                    D = np.abs(C * Xrand - X[i])
                    X[i] = Xrand - A * D
            else:
                D = np.abs(best - X[i])
                l = float(rng.uniform(-1, 1))
                b = 1.0
                X[i] = D * math.exp(b * l) * math.cos(2 * math.pi * l) + best

            X[i] = clamp(X[i], lo, hi)

        fit = np.array([inst.obj(X[i]) for i in range(pop)], dtype=float)
        evals += pop
        j = int(np.argmin(fit))
        if fit[j] < best_so_far:
            best_so_far = float(fit[j])
            best = X[j].copy()

        span = float(np.mean(hi - lo))
        step_mean = float(np.mean(np.linalg.norm(X - X_prev, axis=1)) / (span + EPS))
        log(step_mean)
        t += 1

    log(0.0)
    return best_so_far, tr


def algo_sa_cont(inst: ContInstance, budget: int, rng: np.random.Generator, pop: int, log_points: int) -> Tuple[float, Trace]:
    # single-solution SA on continuous
    d = inst.dim
    lo, hi = inst.bounds[:, 0], inst.bounds[:, 1]
    x = rng.uniform(lo, hi, size=d)
    fx = float(inst.obj(x))
    best_so_far = fx

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())
    evals = 1

    T0, Tend = 1.0, 1e-3
    alpha = (Tend / T0) ** (1.0 / max(1, budget))

    span = float(np.mean(hi - lo))
    step_sigma0 = 0.2 * span

    def log(step_norm: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best_so_far, 0.0, step_norm, evals)

    log(0.0)

    while evals < budget:
        T = T0 * (alpha ** evals)
        step_sigma = step_sigma0 * (0.99 ** (evals / (budget + EPS)))
        x2 = clamp(x + rng.normal(0, step_sigma, size=d), lo, hi)
        f2 = float(inst.obj(x2))
        evals += 1

        delta = f2 - fx
        accept = (delta <= 0.0) or (rng.random() < math.exp(-delta / (T + EPS)))
        if accept:
            x, fx = x2, f2
            best_so_far = min(best_so_far, fx)

        step_norm = float(np.linalg.norm(x2 - x) / (span + EPS))
        log(step_norm)

    log(0.0)
    return best_so_far, tr


def algo_hc_cont(inst: ContInstance, budget: int, rng: np.random.Generator, pop: int, log_points: int) -> Tuple[float, Trace]:
    # simple hill-climb: accept only improvements
    d = inst.dim
    lo, hi = inst.bounds[:, 0], inst.bounds[:, 1]
    x = rng.uniform(lo, hi, size=d)
    fx = float(inst.obj(x))
    best_so_far = fx

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())
    evals = 1

    span = float(np.mean(hi - lo))
    step_sigma0 = 0.25 * span

    def log(step_norm: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best_so_far, 0.0, step_norm, evals)

    log(0.0)

    while evals < budget:
        step_sigma = step_sigma0 * (0.995 ** (evals / (budget + EPS)))
        x2 = clamp(x + rng.normal(0, step_sigma, size=d), lo, hi)
        f2 = float(inst.obj(x2))
        evals += 1

        if f2 < fx:
            x, fx = x2, f2
            best_so_far = min(best_so_far, fx)

        step_norm = float(np.linalg.norm(x2 - x) / (span + EPS))
        log(step_norm)

    log(0.0)
    return best_so_far, tr


# ============================================================
# Algorithms (Combinatorial / bitstring)
# We run domain-appropriate variants but keep SAME family names.
# ============================================================

def algo_pso_bin(inst_obj: Callable[[np.ndarray], float], n: int, budget: int, rng: np.random.Generator, pop: int, log_points: int) -> Tuple[float, Trace]:
    # binary PSO (family=PSO)
    X = rng.integers(0, 2, size=(pop, n), dtype=int)
    V = rng.normal(scale=0.5, size=(pop, n))

    fit = np.array([inst_obj(X[i]) for i in range(pop)], dtype=float)
    evals = pop

    pbest, pbest_y = X.copy(), fit.copy()
    g = int(np.argmin(pbest_y))
    best_so_far = float(pbest_y[g])
    gbest = pbest[g].copy()

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def log(step_mean: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best_so_far, ham_diversity(X), step_mean, evals)

    w, c1, c2 = 0.7, 1.7, 1.7
    log(0.0)

    while evals + pop <= budget:
        X_prev = X.copy()
        r1 = rng.random((pop, n))
        r2 = rng.random((pop, n))

        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        P = sigmoid(V)
        flips = (rng.random((pop, n)) < P).astype(int)
        X = (X ^ flips).astype(int)

        fit = np.array([inst_obj(X[i]) for i in range(pop)], dtype=float)
        evals += pop

        improved = fit < pbest_y
        pbest[improved] = X[improved]
        pbest_y[improved] = fit[improved]

        g = int(np.argmin(pbest_y))
        if pbest_y[g] < best_so_far:
            best_so_far = float(pbest_y[g])
            gbest = pbest[g].copy()

        step_mean = float(np.mean([ham_dist(X[i], X_prev[i]) for i in range(pop)]))
        log(step_mean)

    log(0.0)
    return best_so_far, tr


def algo_de_bin(inst_obj: Callable[[np.ndarray], float], n: int, budget: int, rng: np.random.Generator, pop: int, log_points: int) -> Tuple[float, Trace]:
    # DE-like bit operator (family=DE): flip where (b != c) with prob F, then bin crossover
    X = rng.integers(0, 2, size=(pop, n), dtype=int)
    fit = np.array([inst_obj(X[i]) for i in range(pop)], dtype=float)
    evals = pop
    best_so_far = float(np.min(fit))

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def log(step_mean: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best_so_far, ham_diversity(X), step_mean, evals)

    F, CR = 0.5, 0.9
    log(0.0)

    while evals < budget:
        steps = []
        for i in range(pop):
            if evals >= budget:
                break
            idxs = [j for j in range(pop) if j != i]
            a, b, c = rng.choice(idxs, size=3, replace=False)
            mutant = X[a].copy()
            diff = (X[b] != X[c])
            flip = (rng.random(n) < F) & diff
            mutant = mutant ^ flip.astype(int)

            trial = X[i].copy()
            jrand = int(rng.integers(0, n))
            cross = (rng.random(n) < CR)
            cross[jrand] = True
            trial[cross] = mutant[cross]

            y = float(inst_obj(trial))
            evals += 1
            steps.append(ham_dist(trial, X[i]))

            if y < fit[i]:
                X[i] = trial
                fit[i] = y
                if y < best_so_far:
                    best_so_far = y

        log(float(np.mean(steps)) if steps else 0.0)

    log(0.0)
    return best_so_far, tr


def algo_es_bin(inst_obj: Callable[[np.ndarray], float], n: int, budget: int, rng: np.random.Generator, pop: int, log_points: int) -> Tuple[float, Trace]:
    # (mu, lambda)-ES analogue on bits (family=ES) with elitism
    mu = max(5, pop // 3)
    lam = max(mu * 4, pop)

    parents = rng.integers(0, 2, size=(mu, n), dtype=int)
    pfit = np.array([inst_obj(parents[i]) for i in range(mu)], dtype=float)
    evals = mu
    best_so_far = float(np.min(pfit))

    # mutation rate self-adapt (global, MVP)
    pm = 0.03

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def log(step_mean: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best_so_far, ham_diversity(parents), step_mean, evals)

    log(0.0)

    while evals + lam <= budget:
        # adapt pm a bit (keep bounded)
        pm = float(np.clip(pm * math.exp(0.1 * rng.normal()), 0.001, 0.2))

        kids = np.empty((lam, n), dtype=int)
        kfit = np.empty(lam, dtype=float)
        steps = []
        for k in range(lam):
            p = int(rng.integers(0, mu))
            child = parents[p].copy()
            flip = (rng.random(n) < pm)
            child = (child ^ flip.astype(int)).astype(int)
            kids[k] = child
            steps.append(float(np.mean(flip)))
            kfit[k] = float(inst_obj(child))

        evals += lam
        allX = np.vstack([parents, kids])
        allF = np.concatenate([pfit, kfit])
        idx = np.argsort(allF)[:mu]
        parents = allX[idx]
        pfit = allF[idx]
        best_so_far = min(best_so_far, float(pfit[0]))

        log(float(np.mean(steps)) if steps else 0.0)

    log(0.0)
    return best_so_far, tr


def algo_gwo_bin(inst_obj: Callable[[np.ndarray], float], n: int, budget: int, rng: np.random.Generator, pop: int, log_points: int) -> Tuple[float, Trace]:
    # Binary GWO via latent real vectors -> sigmoid -> bits
    U = rng.normal(size=(pop, n))
    X = (rng.random((pop, n)) < (1.0 / (1.0 + np.exp(-U)))).astype(int)

    fit = np.array([inst_obj(X[i]) for i in range(pop)], dtype=float)
    evals = pop
    best_so_far = float(np.min(fit))

    max_iter = max(1, budget // pop)
    it = 0

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def sig(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def log(step_mean: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best_so_far, ham_diversity(X), step_mean, evals)

    log(0.0)

    while evals + pop <= budget and it < max_iter:
        idx = np.argsort(fit)
        alpha, beta, delta = U[idx[0]], U[idx[1]], U[idx[2]]
        a = 2.0 - 2.0 * (it / max_iter)

        U_prev = U.copy()
        U_new = np.empty_like(U)
        for i in range(pop):
            A1 = 2 * a * rng.random(n) - a
            C1 = 2 * rng.random(n)
            D1 = np.abs(C1 * alpha - U[i])
            U1 = alpha - A1 * D1

            A2 = 2 * a * rng.random(n) - a
            C2 = 2 * rng.random(n)
            D2 = np.abs(C2 * beta - U[i])
            U2 = beta - A2 * D2

            A3 = 2 * a * rng.random(n) - a
            C3 = 2 * rng.random(n)
            D3 = np.abs(C3 * delta - U[i])
            U3 = delta - A3 * D3

            U_new[i] = (U1 + U2 + U3) / 3.0

        U = U_new
        P = sig(U)
        X_prev = X.copy()
        X = (rng.random((pop, n)) < P).astype(int)

        fit = np.array([inst_obj(X[i]) for i in range(pop)], dtype=float)
        evals += pop
        best_so_far = min(best_so_far, float(np.min(fit)))

        step_mean = float(np.mean([ham_dist(X[i], X_prev[i]) for i in range(pop)]))
        log(step_mean)
        it += 1

    log(0.0)
    return best_so_far, tr


def algo_woa_bin(inst_obj: Callable[[np.ndarray], float], n: int, budget: int, rng: np.random.Generator, pop: int, log_points: int) -> Tuple[float, Trace]:
    # Binary WOA via latent real vectors -> sigmoid -> bits
    U = rng.normal(size=(pop, n))
    P = 1.0 / (1.0 + np.exp(-U))
    X = (rng.random((pop, n)) < P).astype(int)

    fit = np.array([inst_obj(X[i]) for i in range(pop)], dtype=float)
    evals = pop
    j = int(np.argmin(fit))
    best_u = U[j].copy()
    best_x = X[j].copy()
    best_so_far = float(fit[j])

    max_iter = max(1, budget // pop)
    t = 0

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def sig(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def log(step_mean: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best_so_far, ham_diversity(X), step_mean, evals)

    log(0.0)

    while evals + pop <= budget and t < max_iter:
        a = 2.0 - 2.0 * (t / max_iter)

        X_prev = X.copy()
        for i in range(pop):
            r1 = rng.random(n)
            r2 = rng.random(n)
            A = 2 * a * r1 - a
            C = 2 * r2
            p = rng.random()

            if p < 0.5:
                if np.linalg.norm(A) < 1:
                    D = np.abs(C * best_u - U[i])
                    U[i] = best_u - A * D
                else:
                    j2 = int(rng.integers(0, pop))
                    U_rand = U[j2]
                    D = np.abs(C * U_rand - U[i])
                    U[i] = U_rand - A * D
            else:
                D = np.abs(best_u - U[i])
                l = float(rng.uniform(-1, 1))
                b = 1.0
                U[i] = D * math.exp(b * l) * math.cos(2 * math.pi * l) + best_u

        P = sig(U)
        X = (rng.random((pop, n)) < P).astype(int)

        fit = np.array([inst_obj(X[i]) for i in range(pop)], dtype=float)
        evals += pop
        j = int(np.argmin(fit))
        if fit[j] < best_so_far:
            best_so_far = float(fit[j])
            best_u = U[j].copy()
            best_x = X[j].copy()

        step_mean = float(np.mean([ham_dist(X[i], X_prev[i]) for i in range(pop)]))
        log(step_mean)
        t += 1

    log(0.0)
    return best_so_far, tr


def algo_sa_bin(inst_obj: Callable[[np.ndarray], float], n: int, budget: int, rng: np.random.Generator, pop: int, log_points: int) -> Tuple[float, Trace]:
    x = rng.integers(0, 2, size=n, dtype=int)
    fx = float(inst_obj(x))
    best_so_far = fx

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())
    evals = 1

    T0, Tend = 1.0, 1e-3
    alpha = (Tend / T0) ** (1.0 / max(1, budget))

    def log(step_val: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best_so_far, 0.0, step_val, evals)

    log(0.0)

    while evals < budget:
        T = T0 * (alpha ** evals)
        j = int(rng.integers(0, n))
        x2 = x.copy()
        x2[j] ^= 1
        f2 = float(inst_obj(x2))
        evals += 1

        delta = f2 - fx
        accept = (delta <= 0.0) or (rng.random() < math.exp(-delta / (T + EPS)))
        if accept:
            x, fx = x2, f2
            best_so_far = min(best_so_far, fx)

        # step feature: acceptance (0/1) is useful for SA “temperature footprint”
        log(1.0 if accept else 0.0)

    log(0.0)
    return best_so_far, tr


def algo_hc_bin(inst_obj: Callable[[np.ndarray], float], n: int, budget: int, rng: np.random.Generator, pop: int, log_points: int) -> Tuple[float, Trace]:
    x = rng.integers(0, 2, size=n, dtype=int)
    fx = float(inst_obj(x))
    best_so_far = fx

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())
    evals = 1

    def log(step_val: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best_so_far, 0.0, step_val, evals)

    log(0.0)

    while evals < budget:
        j = int(rng.integers(0, n))
        x2 = x.copy()
        x2[j] ^= 1
        f2 = float(inst_obj(x2))
        evals += 1
        improved = f2 < fx
        if improved:
            x, fx = x2, f2
            best_so_far = min(best_so_far, fx)
        log(1.0 if improved else 0.0)

    log(0.0)
    return best_so_far, tr


# ============================================================
# Experiment orchestration
# ============================================================

ALGOS_CONT = {
    "PSO": algo_pso_cont,
    "DE":  algo_de_cont,
    "ES":  algo_es_cont,
    "GWO": algo_gwo_cont,
    "WOA": algo_woa_cont,
    "SA":  algo_sa_cont,
    "HC":  algo_hc_cont,
}

ALGOS_BIN = {
    "PSO": algo_pso_bin,
    "DE":  algo_de_bin,
    "ES":  algo_es_bin,
    "GWO": algo_gwo_bin,
    "WOA": algo_woa_bin,
    "SA":  algo_sa_bin,
    "HC":  algo_hc_bin,
}


def run(out_dir: str,
        seeds: List[int],
        cont_funcs: List[str],
        cont_dims: List[int],
        cont_n_per: int,
        cont_budget_mult: int,
        comb_n_per: int,
        maxcut_n: int,
        knap_n: int,
        comb_budget: int,
        master_seed: int,
        pop: int,
        log_points: int):
    make_outdir(out_dir)
    runs_path = os.path.join(out_dir, "runs_features.csv")

    cont_insts = make_cont_instances(cont_funcs, cont_dims, cont_n_per, seed=master_seed)
    maxcuts, knaps = make_comb_instances(maxcut_n, knap_n, comb_n_per, seed=master_seed + 7)

    rows: List[Dict[str, Any]] = []

    # continuous
    for inst in cont_insts:
        budget = int(cont_budget_mult * inst.dim)
        for algo, fn in ALGOS_CONT.items():
            for s in seeds:
                run_seed = (hash((master_seed, "cont", inst.iid, algo, s)) & 0xFFFFFFFF)
                rng = np.random.default_rng(run_seed)
                best, tr = fn(inst=inst, budget=budget, rng=rng, pop=pop, log_points=log_points)
                feats = trace_to_features(tr)
                row = {
                    "domain": "cont", "problem": inst.func, "instance_id": inst.iid,
                    "n": inst.dim, "budget": budget, "algo": algo, "seed": s
                }
                row.update(feats)
                rows.append(row)

    # combinatorial: maxcut + knapsack
    for inst in maxcuts:
        for algo, fn in ALGOS_BIN.items():
            for s in seeds:
                run_seed = (hash((master_seed, "comb", inst.iid, algo, s)) & 0xFFFFFFFF)
                rng = np.random.default_rng(run_seed)
                best, tr = fn(inst_obj=inst.obj, n=inst.n, budget=comb_budget, rng=rng, pop=pop, log_points=log_points)
                feats = trace_to_features(tr)
                row = {
                    "domain": "comb", "problem": "maxcut", "instance_id": inst.iid,
                    "n": inst.n, "budget": comb_budget, "algo": algo, "seed": s
                }
                row.update(feats)
                rows.append(row)

    for inst in knaps:
        n = int(inst.w.size)
        for algo, fn in ALGOS_BIN.items():
            for s in seeds:
                run_seed = (hash((master_seed, "comb", inst.iid, algo, s)) & 0xFFFFFFFF)
                rng = np.random.default_rng(run_seed)
                best, tr = fn(inst_obj=inst.obj, n=n, budget=comb_budget, rng=rng, pop=pop, log_points=log_points)
                feats = trace_to_features(tr)
                row = {
                    "domain": "comb", "problem": "knapsack", "instance_id": inst.iid,
                    "n": n, "budget": comb_budget, "algo": algo, "seed": s
                }
                row.update(feats)
                rows.append(row)

    # write
    fieldnames = list(rows[0].keys())
    with open(runs_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] wrote {runs_path}")


def aggregate(out_dir: str) -> Tuple[List[str], List[str], np.ndarray]:
    runs_path = os.path.join(out_dir, "runs_features.csv")
    algo_path = os.path.join(out_dir, "algo_features.csv")

    rows = []
    with open(runs_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # features we aggregate (exclude final_best; keep improve_rel + signature features)
    feat_cols = [
        "improve_rel", "auc_progress", "p10_progress", "p50_progress", "p90_progress", "stagnation_frac",
        "div_mean", "div_decay", "step_mean", "step_decay", "step_burst",
    ]

    algos = sorted(set(rr["algo"] for rr in rows))
    domains = ["cont", "comb"]

    feat_names = []
    for dom in domains:
        for fc in feat_cols:
            feat_names.append(f"{dom}_{fc}")

    X = np.zeros((len(algos), len(feat_names)), dtype=float)

    def mean_for(algo: str, dom: str, fc: str) -> float:
        vals = []
        for rr in rows:
            if rr["algo"] == algo and rr["domain"] == dom:
                v = float(rr[fc])
                if not math.isnan(v):
                    vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    for i, a in enumerate(algos):
        for j, fn in enumerate(feat_names):
            dom, fc = fn.split("_", 1)
            X[i, j] = mean_for(a, dom, fc)

    # no NaNs expected; still guard:
    X = np.nan_to_num(X, nan=np.nanmean(X))

    with open(algo_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["algo"] + feat_names)
        w.writeheader()
        for i, a in enumerate(algos):
            row = {"algo": a}
            for j, fn in enumerate(feat_names):
                row[fn] = float(X[i, j])
            w.writerow(row)

    print(f"[OK] wrote {algo_path}")
    return algos, feat_names, X


def dist_and_cluster(out_dir: str, k: int):
    algos, feat_names, X = aggregate(out_dir)

    Xz = zscore(X)
    n = len(algos)

    # cosine distance matrix
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            D[i, j] = cosine_distance(Xz[i], Xz[j])

    dist_path = os.path.join(out_dir, "dist_cosine.csv")
    with open(dist_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algo"] + algos)
        for i, a in enumerate(algos):
            w.writerow([a] + [f"{D[i, j]:.6f}" for j in range(n)])
    print(f"[OK] wrote {dist_path}")

    # Hierarchical average-linkage clustering
    # clusters are lists of indices
    clusters = [[i] for i in range(n)]
    cluster_ids = list(range(n))
    next_id = n

    def avg_link(ca: List[int], cb: List[int]) -> float:
        vals = [D[i, j] for i in ca for j in cb]
        return float(np.mean(vals)) if vals else 0.0

    merges = []
    while len(clusters) > 1:
        best = (None, None, float("inf"))
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = avg_link(clusters[i], clusters[j])
                if d < best[2]:
                    best = (i, j, d)

        i, j, d = best
        ca, cb = clusters[i], clusters[j]
        ida, idb = cluster_ids[i], cluster_ids[j]
        new = ca + cb

        merges.append({
            "new_cluster_id": next_id,
            "left_id": ida,
            "right_id": idb,
            "distance": d,
            "size": len(new),
        })

        # merge
        clusters.pop(j)
        clusters.pop(i)
        cluster_ids.pop(j)
        cluster_ids.pop(i)

        clusters.append(new)
        cluster_ids.append(next_id)
        next_id += 1

    dendro_path = os.path.join(out_dir, "dendrogram.csv")
    with open(dendro_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["new_cluster_id", "left_id", "right_id", "distance", "size"])
        w.writeheader()
        w.writerows(merges)
    print(f"[OK] wrote {dendro_path}")

    # Cut tree at k clusters (rebuild merges from leaves)
    # Since n is small, easiest: replay merges and stop when k clusters remain.
    current = {i: [i] for i in range(n)}
    active_ids = set(range(n))

    for m in merges:
        if len(active_ids) <= k:
            break
        left, right, new_id = m["left_id"], m["right_id"], m["new_cluster_id"]
        if left in active_ids and right in active_ids:
            new_members = current[left] + current[right]
            current[new_id] = new_members
            active_ids.remove(left)
            active_ids.remove(right)
            active_ids.add(new_id)

    # assign labels
    label_map = {}
    for ci, cid in enumerate(sorted(active_ids)):
        for idx in current[cid]:
            label_map[idx] = ci

    clusters_path = os.path.join(out_dir, "clusters.csv")
    with open(clusters_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["algo", "cluster", "nearest_neighbor", "nn_distance"])
        w.writeheader()
        for i, a in enumerate(algos):
            order = np.argsort(D[i])
            nn = int(order[1]) if n > 1 else i
            w.writerow({
                "algo": a,
                "cluster": int(label_map[i]),
                "nearest_neighbor": algos[nn],
                "nn_distance": float(D[i, nn]),
            })
    print(f"[OK] wrote {clusters_path}")

    # Print summary
    print("\n=== Cluster summary (hierarchical cut) ===")
    for c in range(k):
        mem = [algos[i] for i in range(n) if label_map[i] == c]
        print(f"Cluster {c}: {', '.join(mem) if mem else '(empty)'}")


# ============================================================
# CLI
# ============================================================

def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_run = sub.add_parser("run")
    ap_run.add_argument("--out_dir", type=str, default="out_sig2")
    ap_run.add_argument("--seeds", type=str, default="0,1,2")
    ap_run.add_argument("--master_seed", type=int, default=123)
    ap_run.add_argument("--pop", type=int, default=40)
    ap_run.add_argument("--log_points", type=int, default=50)

    ap_run.add_argument("--cont_funcs", type=str, default="sphere,rastrigin,ackley")
    ap_run.add_argument("--cont_dims", type=str, default="5,10")
    ap_run.add_argument("--cont_n_per", type=int, default=5)
    ap_run.add_argument("--cont_budget_mult", type=int, default=500)

    ap_run.add_argument("--comb_n_per", type=int, default=5)
    ap_run.add_argument("--maxcut_n", type=int, default=40)
    ap_run.add_argument("--knap_n", type=int, default=50)
    ap_run.add_argument("--comb_budget", type=int, default=4000)

    ap_an = sub.add_parser("analyze")
    ap_an.add_argument("--out_dir", type=str, default="out_sig2")
    ap_an.add_argument("--k", type=int, default=3)

    ap_all = sub.add_parser("all")
    ap_all.add_argument("--out_dir", type=str, default="out_sig2")
    ap_all.add_argument("--seeds", type=str, default="0,1,2")
    ap_all.add_argument("--master_seed", type=int, default=123)
    ap_all.add_argument("--pop", type=int, default=40)
    ap_all.add_argument("--log_points", type=int, default=50)

    ap_all.add_argument("--cont_funcs", type=str, default="sphere,rastrigin,ackley")
    ap_all.add_argument("--cont_dims", type=str, default="5,10")
    ap_all.add_argument("--cont_n_per", type=int, default=5)
    ap_all.add_argument("--cont_budget_mult", type=int, default=500)

    ap_all.add_argument("--comb_n_per", type=int, default=5)
    ap_all.add_argument("--maxcut_n", type=int, default=40)
    ap_all.add_argument("--knap_n", type=int, default=50)
    ap_all.add_argument("--comb_budget", type=int, default=4000)

    ap_all.add_argument("--k", type=int, default=3)

    args = ap.parse_args()

    if args.cmd in ("run", "all"):
        run(
            out_dir=args.out_dir,
            seeds=parse_int_list(args.seeds),
            cont_funcs=parse_str_list(args.cont_funcs),
            cont_dims=parse_int_list(args.cont_dims),
            cont_n_per=args.cont_n_per,
            cont_budget_mult=args.cont_budget_mult,
            comb_n_per=args.comb_n_per,
            maxcut_n=args.maxcut_n,
            knap_n=args.knap_n,
            comb_budget=args.comb_budget,
            master_seed=args.master_seed,
            pop=args.pop,
            log_points=args.log_points,
        )

    if args.cmd in ("analyze", "all"):
        dist_and_cluster(out_dir=args.out_dir, k=args.k)


if __name__ == "__main__":
    main()
