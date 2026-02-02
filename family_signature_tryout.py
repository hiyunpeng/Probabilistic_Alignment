#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
family_signature_tryout.py
==========================

Goal
----
Non-probabilistic algorithm "family" discovery via BEHAVIORAL SIGNATURES:
- Run algorithms on continuous + combinatorial problems
- Collect time-series traces (best-so-far, diversity, step/move size)
- Convert traces into a fixed feature vector per run
- Aggregate per algorithm -> algorithm signature
- Cluster algorithms by signature (cosine distance + k-means)

Dependencies
------------
- numpy only

Quick start
-----------
python family_signature_tryout.py all --out_dir out_sig
# then inspect:
#   out_sig/runs_features.csv
#   out_sig/algo_features.csv
#   out_sig/dist_cosine.csv
#   out_sig/clusters.csv

Notes
-----
- This is an MVP, built for signal, not perfect algorithm fidelity.
- Minimization convention throughout.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Any

import numpy as np


# ============================================================
# Common utilities
# ============================================================

EPS = 1e-12


def clamp(x: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < EPS or nb < EPS:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def zscore_matrix(X: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0) + EPS
    return (X - mu) / sd


def impute_nan_with_colmean(X: np.ndarray) -> np.ndarray:
    X2 = X.copy()
    col_mean = np.nanmean(X2, axis=0)
    inds = np.where(np.isnan(X2))
    X2[inds] = np.take(col_mean, inds[1])
    # If a column is entirely NaN, nanmean gives NaN; fix with 0
    X2 = np.nan_to_num(X2, nan=0.0)
    return X2


def auc_trapz(y: np.ndarray) -> float:
    # AUC over normalized time [0,1] with uniform spacing
    if y.size <= 1:
        return float(y[-1]) if y.size == 1 else 0.0
    x = np.linspace(0.0, 1.0, num=y.size)
    return float(np.trapz(y, x))


def safe_ratio(a: float, b: float) -> float:
    return float(a / (b + EPS))


def make_outdir(d: str) -> None:
    os.makedirs(d, exist_ok=True)


# ============================================================
# Trace and feature extraction
# ============================================================

@dataclass
class Trace:
    best_hist: List[float]
    div_hist: List[float]
    step_hist: List[float]
    eval_hist: List[int]

    @staticmethod
    def empty() -> "Trace":
        return Trace(best_hist=[], div_hist=[], step_hist=[], eval_hist=[])

    def log(self, best: float, diversity: float, step: float, evals: int) -> None:
        self.best_hist.append(float(best))
        self.div_hist.append(float(diversity))
        self.step_hist.append(float(step))
        self.eval_hist.append(int(evals))


def trace_to_features(tr: Trace) -> Dict[str, float]:
    """
    Convert trace to deterministic features (no Bayesian/probabilistic inference).
    Features are designed to be family-revealing:
      - anytime curve shape (AUC, early gain)
      - stagnation fraction
      - diversity level + decay
      - step/move magnitude + decay
    """
    if len(tr.best_hist) == 0:
        # Shouldn't happen, but keep robust
        return {
            "final_best": np.nan,
            "auc_best": np.nan,
            "early_gain": np.nan,
            "stagnation_frac": np.nan,
            "div_mean": np.nan,
            "div_decay": np.nan,
            "step_mean": np.nan,
            "step_decay": np.nan,
        }

    best = np.array(tr.best_hist, dtype=float)
    div = np.array(tr.div_hist, dtype=float)
    step = np.array(tr.step_hist, dtype=float)

    # Normalize best curve to [0,1] by (best - final)/(start - final)
    start = float(best[0])
    final = float(best[-1])
    denom = (start - final)
    if abs(denom) < 1e-9:
        best_norm = np.zeros_like(best)
    else:
        best_norm = (best - final) / (denom + EPS)  # 1 at start, 0 at end if improving

    # Anytime curve AUC (lower is better; 0 means immediate convergence)
    auc_best = auc_trapz(best_norm)

    # Early gain at 10% checkpoints
    idx10 = max(0, int(0.1 * (len(best) - 1)))
    best10 = float(best[idx10])
    early_gain = safe_ratio(start - best10, start - final) if abs(start - final) > 1e-9 else 0.0
    early_gain = float(np.clip(early_gain, 0.0, 1.0))

    # Stagnation fraction (no improvement beyond tiny tolerance)
    tol = 1e-9
    improvements = (best[:-1] - best[1:]) > tol
    stagnation_frac = float(1.0 - np.mean(improvements)) if improvements.size > 0 else 1.0

    # Diversity features
    div_mean = float(np.mean(div))
    div_decay = safe_ratio(div[-1], div[0]) if abs(div[0]) > 1e-12 else 0.0

    # Step features
    step_mean = float(np.mean(step))
    step_decay = safe_ratio(step[-1], step[0]) if abs(step[0]) > 1e-12 else 0.0

    return {
        "final_best": final,
        "auc_best": auc_best,
        "early_gain": early_gain,
        "stagnation_frac": stagnation_frac,
        "div_mean": div_mean,
        "div_decay": div_decay,
        "step_mean": step_mean,
        "step_decay": step_decay,
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
    bounds: np.ndarray  # [d,2]
    shift: np.ndarray   # [d]
    rot: np.ndarray     # [d,d]

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
                iid = f"{fn}_d{d}_i{k:03d}"
                insts.append(ContInstance(iid=iid, func=fn, dim=d, bounds=bounds, shift=shift, rot=rot))
    return insts


# ============================================================
# Combinatorial benchmarks (bitstring)
#   - MaxCut (minimize -cut_weight)
#   - 0/1 Knapsack (minimize -(value) + penalty overweight)
# ============================================================

@dataclass
class MaxCutInstance:
    iid: str
    n: int
    W: np.ndarray  # symmetric weights matrix

    def obj(self, x: np.ndarray) -> float:
        # x in {0,1}^n
        # cut = sum_{i<j} W_ij * [x_i != x_j]
        xi = x.astype(int)
        diff = (xi[:, None] != xi[None, :]).astype(float)
        cut = 0.5 * float(np.sum(self.W * diff))
        return -cut  # minimize negative


@dataclass
class KnapsackInstance:
    iid: str
    w: np.ndarray
    v: np.ndarray
    cap: float

    def obj(self, x: np.ndarray) -> float:
        # x in {0,1}^n
        take = x.astype(int)
        tw = float(np.dot(self.w, take))
        tv = float(np.dot(self.v, take))
        penalty = 0.0
        if tw > self.cap:
            # quadratic penalty
            penalty = (tw - self.cap) ** 2
        return -(tv) + 0.01 * penalty  # minimize


def make_comb_instances(n_graph: int, n_knap: int, n_per: int, seed: int) -> Tuple[List[MaxCutInstance], List[KnapsackInstance]]:
    rng = np.random.default_rng(seed)
    maxcuts: List[MaxCutInstance] = []
    knaps: List[KnapsackInstance] = []

    # MaxCut
    for k in range(n_per):
        W = rng.random((n_graph, n_graph))
        W = 0.5 * (W + W.T)
        np.fill_diagonal(W, 0.0)
        iid = f"maxcut_n{n_graph}_i{k:03d}"
        maxcuts.append(MaxCutInstance(iid=iid, n=n_graph, W=W))

    # Knapsack
    for k in range(n_per):
        w = rng.uniform(1.0, 30.0, size=n_knap)
        v = rng.uniform(1.0, 50.0, size=n_knap)
        cap = 0.4 * float(np.sum(w))  # medium-tight capacity
        iid = f"knap_n{n_knap}_i{k:03d}"
        knaps.append(KnapsackInstance(iid=iid, w=w, v=v, cap=cap))

    return maxcuts, knaps


# ============================================================
# Instrumented algorithms (continuous)
# Each returns: final_best, Trace
# ============================================================

def cont_diversity(X: np.ndarray) -> float:
    # average per-dimension std as a diversity proxy
    return float(np.mean(np.std(X, axis=0)))


def pso_cont(inst: ContInstance, budget: int, rng: np.random.Generator, pop: int = 30, log_points: int = 50) -> Tuple[float, Trace]:
    d = inst.dim
    lo, hi = inst.bounds[:, 0], inst.bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    V = rng.normal(scale=0.1, size=(pop, d))

    y = np.array([inst.obj(X[i]) for i in range(pop)], dtype=float)
    evals = pop
    pbest = X.copy()
    pbest_y = y.copy()
    gbest = pbest[int(np.argmin(pbest_y))].copy()
    gbest_y = float(np.min(pbest_y))

    tr = Trace.empty()
    last_mean = np.mean(X, axis=0)
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def log_state(step_mean: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best=gbest_y, diversity=cont_diversity(X), step=step_mean, evals=evals)

    w, c1, c2 = 0.7, 1.5, 1.5
    log_state(step_mean=0.0)

    while evals + pop <= budget:
        r1 = rng.random((pop, d))
        r2 = rng.random((pop, d))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X_new = clamp(X + V, lo, hi)

        step_mean = float(np.mean(np.linalg.norm(X_new - X, axis=1)))
        X = X_new

        y = np.array([inst.obj(X[i]) for i in range(pop)], dtype=float)
        evals += pop

        improved = y < pbest_y
        pbest[improved] = X[improved]
        pbest_y[improved] = y[improved]

        j = int(np.argmin(pbest_y))
        if pbest_y[j] < gbest_y:
            gbest_y = float(pbest_y[j])
            gbest = pbest[j].copy()

        _ = last_mean
        last_mean = np.mean(X, axis=0)
        log_state(step_mean=step_mean)

    return gbest_y, tr


def de_cont(inst: ContInstance, budget: int, rng: np.random.Generator, pop: int = 30, log_points: int = 50) -> Tuple[float, Trace]:
    d = inst.dim
    lo, hi = inst.bounds[:, 0], inst.bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    fit = np.array([inst.obj(X[i]) for i in range(pop)], dtype=float)
    evals = pop

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def log_state(step_mean: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best=float(np.min(fit)), diversity=cont_diversity(X), step=step_mean, evals=evals)

    F, CR = 0.8, 0.9
    log_state(0.0)

    while evals + pop <= budget:
        step_sizes = []
        for i in range(pop):
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
            if evals >= budget:
                break

        log_state(step_mean=float(np.mean(step_sizes)) if step_sizes else 0.0)

        if evals >= budget:
            break

    return float(np.min(fit)), tr


def es_cont(inst: ContInstance, budget: int, rng: np.random.Generator, mu: int = 10, lam: int = 40, log_points: int = 50) -> Tuple[float, Trace]:
    d = inst.dim
    lo, hi = inst.bounds[:, 0], inst.bounds[:, 1]
    X = rng.uniform(lo, hi, size=(mu, d))
    sigma = np.full(mu, 0.3 * np.mean(hi - lo))

    fit = np.array([inst.obj(X[i]) for i in range(mu)], dtype=float)
    evals = mu

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def log_state(step_mean: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best=float(np.min(fit)), diversity=cont_diversity(X), step=step_mean, evals=evals)

    tau = 1.0 / math.sqrt(d)
    log_state(0.0)

    while evals + lam <= budget:
        kids = np.empty((lam, d), dtype=float)
        steps = []
        for k in range(lam):
            p = int(rng.integers(0, mu))
            sig = sigma[p] * math.exp(tau * rng.normal())
            child = X[p] + rng.normal(0, sig, size=d)
            child = clamp(child, lo, hi)
            kids[k] = child
            steps.append(float(np.linalg.norm(child - X[p])))

        kid_fit = np.array([inst.obj(kids[i]) for i in range(lam)], dtype=float)
        evals += lam

        idx = np.argsort(kid_fit)[:mu]
        X = kids[idx]
        fit = kid_fit[idx]
        sigma = np.clip(sigma.mean() * np.ones(mu), 1e-6, 1e6)  # keep simple (MVP)

        log_state(step_mean=float(np.mean(steps)) if steps else 0.0)

    return float(np.min(fit)), tr


def gwo_cont(inst: ContInstance, budget: int, rng: np.random.Generator, pop: int = 30, log_points: int = 50) -> Tuple[float, Trace]:
    d = inst.dim
    lo, hi = inst.bounds[:, 0], inst.bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    fit = np.array([inst.obj(X[i]) for i in range(pop)], dtype=float)
    evals = pop

    max_iter = max(1, budget // pop)  # schedule tied to budget
    it = 0

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def log_state(step_mean: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best=float(np.min(fit)), diversity=cont_diversity(X), step=step_mean, evals=evals)

    log_state(0.0)

    while evals + pop <= budget and it < max_iter:
        idx = np.argsort(fit)
        alpha, beta, delta = X[idx[0]], X[idx[1]], X[idx[2]]
        a = 2.0 - 2.0 * (it / max_iter)

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

        X_new = clamp(X_new, lo, hi)
        step_mean = float(np.mean(np.linalg.norm(X_new - X, axis=1)))
        X = X_new
        fit = np.array([inst.obj(X[i]) for i in range(pop)], dtype=float)
        evals += pop

        log_state(step_mean)
        it += 1

    return float(np.min(fit)), tr


def woa_cont(inst: ContInstance, budget: int, rng: np.random.Generator, pop: int = 30, log_points: int = 50) -> Tuple[float, Trace]:
    d = inst.dim
    lo, hi = inst.bounds[:, 0], inst.bounds[:, 1]
    X = rng.uniform(lo, hi, size=(pop, d))
    fit = np.array([inst.obj(X[i]) for i in range(pop)], dtype=float)
    evals = pop
    best = X[int(np.argmin(fit))].copy()
    best_y = float(np.min(fit))

    max_iter = max(1, budget // pop)
    t = 0

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def log_state(step_mean: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best=best_y, diversity=cont_diversity(X), step=step_mean, evals=evals)

    log_state(0.0)

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
                    j = int(rng.integers(0, pop))
                    Xrand = X[j]
                    D = np.abs(C * Xrand - X[i])
                    X[i] = Xrand - A * D
            else:
                D = np.abs(best - X[i])
                l = float(rng.uniform(-1, 1))
                b = 1.0
                X[i] = D * math.exp(b * l) * math.cos(2 * math.pi * l) + best

            X[i] = clamp(X[i], lo, hi)

        step_mean = float(np.mean(np.linalg.norm(X - X_prev, axis=1)))
        fit = np.array([inst.obj(X[i]) for i in range(pop)], dtype=float)
        evals += pop

        j = int(np.argmin(fit))
        if fit[j] < best_y:
            best_y = float(fit[j])
            best = X[j].copy()

        log_state(step_mean)
        t += 1

    return best_y, tr


CONT_ALGOS: Dict[str, Callable[..., Tuple[float, Trace]]] = {
    "PSO": pso_cont,
    "DE": de_cont,
    "ES": es_cont,
    "GWO": gwo_cont,
    "WOA": woa_cont,
}


# ============================================================
# Instrumented algorithms (combinatorial: bitstring)
# ============================================================

def hamming_diversity(X: np.ndarray) -> float:
    # X shape [pop,n] in {0,1}
    # diversity proxy: average per-bit std
    return float(np.mean(np.std(X, axis=0)))


def bs_step_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(a.astype(int) != b.astype(int)))


def ga_bin(inst_obj: Callable[[np.ndarray], float], n: int, budget: int, rng: np.random.Generator,
           pop: int = 60, pm: float = 0.02, log_points: int = 50) -> Tuple[float, Trace]:
    # classic evolutionary signature: selection + crossover + mutation
    X = rng.integers(0, 2, size=(pop, n), dtype=int)
    fit = np.array([inst_obj(X[i]) for i in range(pop)], dtype=float)
    evals = pop

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def tournament(k=3) -> int:
        idx = rng.integers(0, pop, size=k)
        return int(idx[np.argmin(fit[idx])])

    def log_state(step_mean: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best=float(np.min(fit)), diversity=hamming_diversity(X), step=step_mean, evals=evals)

    log_state(0.0)

    while evals + pop <= budget:
        X_new = np.empty_like(X)
        steps = []
        for i in range(pop):
            p1 = X[tournament()]
            p2 = X[tournament()]
            # 1-point crossover
            cp = int(rng.integers(1, n))
            child = np.concatenate([p1[:cp], p2[cp:]]).astype(int)
            # mutation
            mut = rng.random(n) < pm
            child = (child ^ mut.astype(int)).astype(int)
            X_new[i] = child
            steps.append(bs_step_dist(child, X[i]))

        X = X_new
        fit = np.array([inst_obj(X[i]) for i in range(pop)], dtype=float)
        evals += pop
        log_state(step_mean=float(np.mean(steps)) if steps else 0.0)

    return float(np.min(fit)), tr


def sa_bin(inst_obj: Callable[[np.ndarray], float], n: int, budget: int, rng: np.random.Generator,
           log_points: int = 50) -> Tuple[float, Trace]:
    # single-solution signature: local moves + acceptance of worse
    x = rng.integers(0, 2, size=n, dtype=int)
    fx = float(inst_obj(x))
    best = fx

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def log_state(step_mean: float, evals: int):
        if evals in checkpoints or evals >= budget:
            tr.log(best=best, diversity=0.0, step=step_mean, evals=evals)

    # geometric temperature schedule
    T0, Tend = 1.0, 1e-3
    steps_total = max(1, budget)
    alpha = (Tend / T0) ** (1.0 / steps_total)

    evals = 0
    log_state(0.0, evals)

    step_accum = []
    while evals < budget:
        T = T0 * (alpha ** evals)
        # flip one bit
        j = int(rng.integers(0, n))
        x2 = x.copy()
        x2[j] ^= 1
        f2 = float(inst_obj(x2))
        evals += 1

        delta = f2 - fx
        accept = (delta <= 0.0) or (rng.random() < math.exp(-delta / (T + EPS)))
        if accept:
            step_accum.append(1.0)
            x, fx = x2, f2
            if fx < best:
                best = fx
        else:
            step_accum.append(0.0)

        # log step as acceptance ratio over last window (behavior proxy)
        if evals in checkpoints or evals >= budget:
            window = step_accum[-max(1, len(step_accum)//10):]
            step_mean = float(np.mean(window)) if window else 0.0
            log_state(step_mean, evals)

    return float(best), tr


def hillclimb_bin(inst_obj: Callable[[np.ndarray], float], n: int, budget: int, rng: np.random.Generator,
                  log_points: int = 50) -> Tuple[float, Trace]:
    x = rng.integers(0, 2, size=n, dtype=int)
    fx = float(inst_obj(x))
    best = fx

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    evals = 0
    tr.log(best=best, diversity=0.0, step=0.0, evals=evals)

    while evals < budget:
        # best-improving single-bit flip (first improvement)
        improved = False
        order = rng.permutation(n)
        for j in order:
            x2 = x.copy()
            x2[j] ^= 1
            f2 = float(inst_obj(x2))
            evals += 1
            if f2 < fx:
                x, fx = x2, f2
                best = min(best, fx)
                improved = True
                break
            if evals >= budget:
                break

        step_mean = 1.0 if improved else 0.0
        if evals in checkpoints or evals >= budget:
            tr.log(best=best, diversity=0.0, step=step_mean, evals=evals)

    return float(best), tr


def bpso_bin(inst_obj: Callable[[np.ndarray], float], n: int, budget: int, rng: np.random.Generator,
             pop: int = 50, log_points: int = 50) -> Tuple[float, Trace]:
    # binary PSO signature: global/personal best-driven flips via sigmoid(velocity)
    X = rng.integers(0, 2, size=(pop, n), dtype=int)
    V = rng.normal(scale=0.5, size=(pop, n))

    fit = np.array([inst_obj(X[i]) for i in range(pop)], dtype=float)
    evals = pop

    pbest = X.copy()
    pbest_y = fit.copy()
    g = int(np.argmin(pbest_y))
    gbest = pbest[g].copy()
    gbest_y = float(pbest_y[g])

    tr = Trace.empty()
    checkpoints = set(np.linspace(0, budget, num=log_points, dtype=int).tolist())

    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def log_state(step_mean: float):
        if evals in checkpoints or evals >= budget:
            tr.log(best=gbest_y, diversity=hamming_diversity(X), step=step_mean, evals=evals)

    w, c1, c2 = 0.7, 1.7, 1.7
    log_state(0.0)

    while evals + pop <= budget:
        X_prev = X.copy()
        r1 = rng.random((pop, n))
        r2 = rng.random((pop, n))

        # velocity update towards pbest/gbest in bit space (0/1)
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
        if pbest_y[g] < gbest_y:
            gbest_y = float(pbest_y[g])
            gbest = pbest[g].copy()

        step_mean = float(np.mean([bs_step_dist(X[i], X_prev[i]) for i in range(pop)]))
        log_state(step_mean)

    return gbest_y, tr


COMB_ALGOS: Dict[str, Callable[..., Tuple[float, Trace]]] = {
    "GA_BIN": ga_bin,
    "SA": sa_bin,
    "HC": hillclimb_bin,
    "BPSO": bpso_bin,
}


# ============================================================
# Experiment runner
# ============================================================

def run_all(
    out_dir: str,
    seeds: List[int],
    # continuous setup
    cont_funcs: List[str],
    cont_dims: List[int],
    cont_n_per_cell: int,
    cont_budget_mult: int,
    # combinatorial setup
    comb_n_per: int,
    maxcut_n: int,
    knap_n: int,
    comb_budget: int,
    master_seed: int,
):
    make_outdir(out_dir)
    runs_path = os.path.join(out_dir, "runs_features.csv")

    # Make instances
    cont_insts = make_cont_instances(cont_funcs, cont_dims, cont_n_per_cell, seed=master_seed)
    maxcuts, knaps = make_comb_instances(n_graph=maxcut_n, n_knap=knap_n, n_per=comb_n_per, seed=master_seed + 7)

    rows: List[Dict[str, Any]] = []

    # ---------- Continuous runs ----------
    for inst in cont_insts:
        budget = int(cont_budget_mult * inst.dim)
        for algo_name, algo_fn in CONT_ALGOS.items():
            for s in seeds:
                run_seed = (hash((master_seed, "cont", inst.iid, algo_name, s)) & 0xFFFFFFFF)
                rng = np.random.default_rng(run_seed)
                final_best, tr = algo_fn(inst=inst, budget=budget, rng=rng)
                feats = trace_to_features(tr)

                row = {
                    "domain": "cont",
                    "problem": inst.func,
                    "instance_id": inst.iid,
                    "n": inst.dim,
                    "budget": budget,
                    "algo": algo_name,
                    "seed": s,
                }
                row.update(feats)
                rows.append(row)

    # ---------- Combinatorial runs ----------
    # MaxCut
    for inst in maxcuts:
        for algo_name, algo_fn in COMB_ALGOS.items():
            for s in seeds:
                run_seed = (hash((master_seed, "comb", inst.iid, algo_name, s)) & 0xFFFFFFFF)
                rng = np.random.default_rng(run_seed)
                final_best, tr = algo_fn(inst_obj=inst.obj, n=inst.n, budget=comb_budget, rng=rng)
                feats = trace_to_features(tr)

                row = {
                    "domain": "comb",
                    "problem": "maxcut",
                    "instance_id": inst.iid,
                    "n": inst.n,
                    "budget": comb_budget,
                    "algo": algo_name,
                    "seed": s,
                }
                row.update(feats)
                rows.append(row)

    # Knapsack
    for inst in knaps:
        n = int(inst.w.size)
        for algo_name, algo_fn in COMB_ALGOS.items():
            for s in seeds:
                run_seed = (hash((master_seed, "comb", inst.iid, algo_name, s)) & 0xFFFFFFFF)
                rng = np.random.default_rng(run_seed)
                final_best, tr = algo_fn(inst_obj=inst.obj, n=n, budget=comb_budget, rng=rng)
                feats = trace_to_features(tr)

                row = {
                    "domain": "comb",
                    "problem": "knapsack",
                    "instance_id": inst.iid,
                    "n": n,
                    "budget": comb_budget,
                    "algo": algo_name,
                    "seed": s,
                }
                row.update(feats)
                rows.append(row)

    # Write runs_features.csv
    fieldnames = list(rows[0].keys())
    with open(runs_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] wrote {runs_path}")


def aggregate_features(out_dir: str) -> Tuple[List[str], np.ndarray, List[str]]:
    runs_path = os.path.join(out_dir, "runs_features.csv")
    out_path = os.path.join(out_dir, "algo_features.csv")

    rows: List[Dict[str, Any]] = []
    with open(runs_path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # Feature columns
    base_feats = [
        "auc_best", "early_gain", "stagnation_frac",
        "div_mean", "div_decay", "step_mean", "step_decay",
    ]

    algos = sorted(set(rr["algo"] for rr in rows))
    domains = ["cont", "comb"]

    # aggregate per algo and domain, then concatenate: cont_* + comb_*
    feat_names: List[str] = []
    for dom in domains:
        for bf in base_feats:
            feat_names.append(f"{dom}_{bf}")

    X = np.full((len(algos), len(feat_names)), np.nan, dtype=float)

    # helper
    def mean_of(algo: str, dom: str, feat: str) -> float:
        vals = []
        for rr in rows:
            if rr["algo"] == algo and rr["domain"] == dom:
                v = float(rr[feat])
                if not math.isnan(v):
                    vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    for i, a in enumerate(algos):
        for j, name in enumerate(feat_names):
            dom, bf = name.split("_", 1)
            X[i, j] = mean_of(a, dom, bf)

    # write algo_features.csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["algo"] + feat_names)
        w.writeheader()
        for i, a in enumerate(algos):
            row = {"algo": a}
            for j, fn in enumerate(feat_names):
                row[fn] = float(X[i, j]) if not math.isnan(X[i, j]) else ""
            w.writerow(row)

    print(f"[OK] wrote {out_path}")
    return algos, X, feat_names


def pairwise_cosine_distance_matrix(algos: List[str], X: np.ndarray, out_dir: str) -> np.ndarray:
    out_path = os.path.join(out_dir, "dist_cosine.csv")

    X2 = impute_nan_with_colmean(X)
    Xz = zscore_matrix(X2)

    n = len(algos)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            D[i, j] = cosine_distance(Xz[i], Xz[j])

    # write
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algo"] + algos)
        for i, a in enumerate(algos):
            w.writerow([a] + [f"{D[i, j]:.6f}" for j in range(n)])

    print(f"[OK] wrote {out_path}")
    return D


def kmeans(X: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 10, max_iter: int = 200) -> np.ndarray:
    """
    Simple k-means on standardized features.
    """
    X2 = impute_nan_with_colmean(X)
    Xz = zscore_matrix(X2)

    best_labels = None
    best_inertia = float("inf")

    n = Xz.shape[0]
    if k <= 1:
        return np.zeros(n, dtype=int)

    for _ in range(n_init):
        centers = Xz[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)

        for _it in range(max_iter):
            # assign
            dists = np.sum((Xz[:, None, :] - centers[None, :, :]) ** 2, axis=2)  # [n,k]
            new_labels = np.argmin(dists, axis=1)

            if np.all(new_labels == labels):
                break
            labels = new_labels

            # update centers
            for c in range(k):
                pts = Xz[labels == c]
                if pts.shape[0] == 0:
                    centers[c] = Xz[rng.integers(0, n)]
                else:
                    centers[c] = np.mean(pts, axis=0)

        inertia = float(np.sum((Xz - centers[labels]) ** 2))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    return best_labels if best_labels is not None else np.zeros(X.shape[0], dtype=int)


def analyze(out_dir: str, k: int, seed: int):
    algos, X, feat_names = aggregate_features(out_dir)
    D = pairwise_cosine_distance_matrix(algos, X, out_dir)

    rng = np.random.default_rng(seed)
    labels = kmeans(X, k=k, rng=rng)

    # write clusters
    out_path = os.path.join(out_dir, "clusters.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["algo", "cluster", "nearest_neighbor", "nn_distance"])
        w.writeheader()

        for i, a in enumerate(algos):
            # nearest neighbor by cosine distance
            nn_j = int(np.argsort(D[i])[1]) if len(algos) > 1 else i
            w.writerow({
                "algo": a,
                "cluster": int(labels[i]),
                "nearest_neighbor": algos[nn_j],
                "nn_distance": float(D[i, nn_j]),
            })

    print(f"[OK] wrote {out_path}")

    # console summary
    print("\n=== Cluster summary (signature-based, non-probabilistic) ===")
    for c in range(k):
        members = [algos[i] for i in range(len(algos)) if labels[i] == c]
        print(f"Cluster {c}: {', '.join(members) if members else '(empty)'}")

    print("\n=== Nearest neighbors (cosine distance on signatures) ===")
    for i, a in enumerate(algos):
        order = np.argsort(D[i])
        top = [algos[j] for j in order[1:4]]  # 3 nearest excluding self
        dist = [D[i, j] for j in order[1:4]]
        pairs = ", ".join([f"{top[t]}({dist[t]:.3f})" for t in range(len(top))])
        print(f"{a:8s} -> {pairs}")


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
    ap_run.add_argument("--out_dir", type=str, default="out_sig")
    ap_run.add_argument("--seeds", type=str, default="0,1,2")
    ap_run.add_argument("--master_seed", type=int, default=123)

    ap_run.add_argument("--cont_funcs", type=str, default="sphere,rastrigin,ackley")
    ap_run.add_argument("--cont_dims", type=str, default="5,10")
    ap_run.add_argument("--cont_n_per_cell", type=int, default=5)
    ap_run.add_argument("--cont_budget_mult", type=int, default=500)  # budget = mult * D

    ap_run.add_argument("--comb_n_per", type=int, default=5)
    ap_run.add_argument("--maxcut_n", type=int, default=40)
    ap_run.add_argument("--knap_n", type=int, default=50)
    ap_run.add_argument("--comb_budget", type=int, default=4000)

    ap_an = sub.add_parser("analyze")
    ap_an.add_argument("--out_dir", type=str, default="out_sig")
    ap_an.add_argument("--k", type=int, default=3, help="k-means clusters")
    ap_an.add_argument("--seed", type=int, default=2026)

    ap_all = sub.add_parser("all")
    ap_all.add_argument("--out_dir", type=str, default="out_sig")
    ap_all.add_argument("--seeds", type=str, default="0,1,2")
    ap_all.add_argument("--master_seed", type=int, default=123)

    ap_all.add_argument("--cont_funcs", type=str, default="sphere,rastrigin,ackley")
    ap_all.add_argument("--cont_dims", type=str, default="5,10")
    ap_all.add_argument("--cont_n_per_cell", type=int, default=5)
    ap_all.add_argument("--cont_budget_mult", type=int, default=500)

    ap_all.add_argument("--comb_n_per", type=int, default=5)
    ap_all.add_argument("--maxcut_n", type=int, default=40)
    ap_all.add_argument("--knap_n", type=int, default=50)
    ap_all.add_argument("--comb_budget", type=int, default=4000)

    ap_all.add_argument("--k", type=int, default=3)
    ap_all.add_argument("--seed", type=int, default=2026)

    args = ap.parse_args()

    if args.cmd in ("run", "all"):
        run_all(
            out_dir=args.out_dir,
            seeds=parse_int_list(args.seeds),
            cont_funcs=parse_str_list(args.cont_funcs),
            cont_dims=parse_int_list(args.cont_dims),
            cont_n_per_cell=args.cont_n_per_cell,
            cont_budget_mult=args.cont_budget_mult,
            comb_n_per=args.comb_n_per,
            maxcut_n=args.maxcut_n,
            knap_n=args.knap_n,
            comb_budget=args.comb_budget,
            master_seed=args.master_seed,
        )

    if args.cmd in ("analyze", "all"):
        analyze(out_dir=args.out_dir, k=args.k, seed=args.seed)


if __name__ == "__main__":
    main()
