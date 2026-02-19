#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
success_profile_laptop_runner_v4.py

Laptop-friendly benchmark runner for "success profiles" experiments.

What it does
------------
1) Generates BIN and CONT problem instances.
2) Builds instance-wise targets (easy/med/hard) using Random Search (RS) calibration
   and per-instance quantiles of RS best-of-budget scores.
3) Runs multiple algorithms for multiple repetitions at specified budgets.
4) Aggregates to instance-level success statistics using a Beta(1,1) prior:
      beta_mean = (1 + successes) / (2 + trials)
   and produces beta_p05/beta_p95 (Monte Carlo, SciPy optional).
5) Writes:
   - runs_detail.csv
   - targets_by_instance.csv
   - instance_algo_budget_summary.csv   (schema compatible with your analysis scripts)

Design choices for laptop stability
-----------------------------------
- Default scale is modest (instances ~20 per domain, reps=5).
- Tabu uses sampled neighbourhood (k candidates/step) to respect evaluation budgets.
- No multiprocessing by default (Windows-safe).

Usage (PowerShell)
------------------
python success_profile_laptop_runner_v4.py ^
  --out_dir out_succ_laptop_v4 ^
  --budgets 300,800 ^
  --bin_instances 20 --cont_instances 20 ^
  --reps 5 --calib_reps 20
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import pandas as pd


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def beta_mean(successes: int, trials: int, a0: float = 1.0, b0: float = 1.0) -> float:
    return (a0 + successes) / (a0 + b0 + trials)


def beta_quantiles_mc(successes: int, trials: int, q_lo=0.05, q_hi=0.95, draws=4000, seed=0) -> Tuple[float, float]:
    a = 1.0 + successes
    b = 1.0 + (trials - successes)
    rng = np.random.default_rng(seed)
    samples = rng.beta(a, b, size=draws)
    return float(np.quantile(samples, q_lo)), float(np.quantile(samples, q_hi))


def beta_quantiles(successes: int, trials: int, q_lo=0.05, q_hi=0.95, draws=4000, seed=0) -> Tuple[float, float]:
    try:
        from scipy.stats import beta as sp_beta  # type: ignore
        a = 1.0 + successes
        b = 1.0 + (trials - successes)
        return float(sp_beta.ppf(q_lo, a, b)), float(sp_beta.ppf(q_hi, a, b))
    except Exception:
        return beta_quantiles_mc(successes, trials, q_lo=q_lo, q_hi=q_hi, draws=draws, seed=seed)


# -----------------------------
# Problem definitions
# -----------------------------

@dataclass(frozen=True)
class BinInstance:
    instance_id: str
    problem: str
    n_bits: int
    payload: Dict[str, Any]

    def evaluate(self, x: np.ndarray) -> float:
        # Score to MAXIMISE
        if self.problem == "onemax":
            return float(np.sum(x))
        if self.problem == "leadingones":
            idx0 = np.argmax(x == 0) if np.any(x == 0) else self.n_bits
            return float(idx0)
        if self.problem == "knapsack01":
            w = self.payload["weights"]
            v = self.payload["values"]
            cap = self.payload["capacity"]
            total_w = float(np.dot(w, x))
            total_v = float(np.dot(v, x))
            if total_w <= cap:
                return total_v
            penalty = self.payload["penalty"]
            return total_v - penalty * (total_w - cap)
        raise ValueError(f"Unknown BIN problem: {self.problem}")


@dataclass(frozen=True)
class ContInstance:
    instance_id: str
    problem: str
    dim: int
    lo: float
    hi: float
    payload: Dict[str, Any]

    def evaluate(self, x: np.ndarray) -> float:
        # Score to MAXIMISE: score = -objective for minimisation problems
        if self.problem == "sphere":
            return -float(np.sum(x * x))
        if self.problem == "rastrigin":
            A = 10.0
            f = A * self.dim + float(np.sum(x * x - A * np.cos(2.0 * math.pi * x)))
            return -f
        raise ValueError(f"Unknown CONT problem: {self.problem}")


def make_bin_instances(rng: np.random.Generator, n_instances: int, n_bits: int) -> List[BinInstance]:
    instances: List[BinInstance] = []
    probs = ["onemax", "leadingones", "knapsack01"]
    for i in range(n_instances):
        prob = probs[i % len(probs)]
        iid = f"bin_{prob}_{i:03d}"
        if prob == "knapsack01":
            n_items = n_bits
            weights = rng.integers(1, 20, size=n_items).astype(float)
            values = rng.integers(1, 30, size=n_items).astype(float)
            capacity = float(0.25 * np.sum(weights))
            penalty = float(np.max(values / weights) * 5.0)
            payload = {"weights": weights, "values": values, "capacity": capacity, "penalty": penalty}
            instances.append(BinInstance(iid, prob, n_items, payload))
        else:
            instances.append(BinInstance(iid, prob, n_bits, payload={}))
    return instances


def make_cont_instances(rng: np.random.Generator, n_instances: int, dim: int, lo: float, hi: float) -> List[ContInstance]:
    instances: List[ContInstance] = []
    probs = ["sphere", "rastrigin"]
    for i in range(n_instances):
        prob = probs[i % len(probs)]
        iid = f"cont_{prob}_{i:03d}"
        instances.append(ContInstance(iid, prob, dim, lo, hi, payload={}))
    return instances


# -----------------------------
# Random Search (baseline + contender)
# -----------------------------

def rs_bin(inst: BinInstance, budget: int, rng: np.random.Generator) -> float:
    best = -1e18
    for _ in range(budget):
        x = rng.integers(0, 2, size=inst.n_bits, dtype=np.int8)
        s = inst.evaluate(x)
        if s > best:
            best = s
    return best


def rs_cont(inst: ContInstance, budget: int, rng: np.random.Generator) -> float:
    best = -1e18
    for _ in range(budget):
        x = rng.uniform(inst.lo, inst.hi, size=inst.dim)
        s = inst.evaluate(x)
        if s > best:
            best = s
    return best


# -----------------------------
# Binary algorithms
# -----------------------------

def hc_bin(inst: BinInstance, budget: int, rng: np.random.Generator, flips: int = 1) -> float:
    x = rng.integers(0, 2, size=inst.n_bits, dtype=np.int8)
    best = inst.evaluate(x)
    evals = 1
    while evals < budget:
        cand = x.copy()
        idx = rng.choice(inst.n_bits, size=flips, replace=False)
        cand[idx] = 1 - cand[idx]
        s = inst.evaluate(cand)
        evals += 1
        if s >= best:
            x = cand
            best = s
    return best


def sa_bin(inst: BinInstance, budget: int, rng: np.random.Generator, flips: int = 1,
           T0: float = 1.0, alpha: float = 0.995) -> float:
    x = rng.integers(0, 2, size=inst.n_bits, dtype=np.int8)
    curr = inst.evaluate(x)
    best = curr
    evals = 1
    k = 0
    T = T0
    while evals < budget:
        cand = x.copy()
        idx = rng.choice(inst.n_bits, size=flips, replace=False)
        cand[idx] = 1 - cand[idx]
        s = inst.evaluate(cand)
        evals += 1
        delta = s - curr
        if delta >= 0 or rng.random() < math.exp(delta / max(T, 1e-9)):
            x = cand
            curr = s
            if s > best:
                best = s
        k += 1
        T = T0 * (alpha ** k)
    return best


def tabu_bin(inst: BinInstance, budget: int, rng: np.random.Generator, tenure: int = 7, k: int = 20) -> float:
    n = inst.n_bits
    x = rng.integers(0, 2, size=n, dtype=np.int8)
    curr = inst.evaluate(x)
    best = curr
    evals = 1

    tabu = np.zeros(n, dtype=np.int32)  # expiry iteration
    it = 0
    k = min(k, n)

    while evals < budget:
        it += 1
        idxs = rng.choice(n, size=k, replace=False)
        best_s = -1e18
        best_idx = None
        for j in idxs:
            if tabu[j] > it:
                continue
            cand = x.copy()
            cand[j] = 1 - cand[j]
            s = inst.evaluate(cand)
            evals += 1
            if s > best_s:
                best_s = s
                best_idx = int(j)
            if evals >= budget:
                break

        if best_idx is None:
            j = int(rng.integers(0, n))
            cand = x.copy()
            cand[j] = 1 - cand[j]
            best_s = inst.evaluate(cand)
            evals += 1
            best_idx = j

        x[best_idx] = 1 - x[best_idx]
        curr = best_s
        tabu[best_idx] = it + tenure
        if curr > best:
            best = curr

    return best


def ga_bin(inst: BinInstance, budget: int, rng: np.random.Generator,
           pop: int = 50, pc: float = 0.9, pm: float = 0.02, tourn_k: int = 3) -> float:
    n = inst.n_bits
    pop = int(pop)
    P = rng.integers(0, 2, size=(pop, n), dtype=np.int8)
    fitness = np.array([inst.evaluate(P[i]) for i in range(pop)], dtype=float)
    evals = pop
    best = float(np.max(fitness))

    def tournament() -> int:
        idx = rng.integers(0, pop, size=tourn_k)
        return int(idx[np.argmax(fitness[idx])])

    while evals < budget:
        newP = np.empty_like(P)
        for i in range(0, pop, 2):
            p1 = P[tournament()].copy()
            p2 = P[tournament()].copy()
            if rng.random() < pc:
                cx = int(rng.integers(1, n))
                c1 = np.concatenate([p1[:cx], p2[cx:]])
                c2 = np.concatenate([p2[:cx], p1[cx:]])
            else:
                c1, c2 = p1, p2

            m1 = rng.random(n) < pm
            m2 = rng.random(n) < pm
            c1[m1] = 1 - c1[m1]
            c2[m2] = 1 - c2[m2]

            newP[i] = c1
            if i + 1 < pop:
                newP[i + 1] = c2

        remaining = budget - evals
        to_eval = min(pop, remaining)
        new_fit = np.empty(pop, dtype=float)
        for i in range(to_eval):
            new_fit[i] = inst.evaluate(newP[i])
        evals += to_eval
        if to_eval < pop:
            new_fit[to_eval:] = -1e18
            P = newP
            fitness = new_fit
            best = max(best, float(np.max(new_fit[:to_eval])))
            break

        P = newP
        fitness = new_fit
        best = max(best, float(np.max(fitness)))

    return best


def umda_bin(inst: BinInstance, budget: int, rng: np.random.Generator,
             pop: int = 50, elite: float = 0.2, eps: float = 1e-3) -> float:
    n = inst.n_bits
    pop = int(pop)
    elite_n = max(1, int(round(pop * elite)))
    p = np.full(n, 0.5, dtype=float)

    best = -1e18
    evals = 0
    while evals < budget:
        remaining = budget - evals
        to_sample = min(pop, remaining)
        X = (rng.random((to_sample, n)) < p).astype(np.int8)
        fit = np.array([inst.evaluate(X[i]) for i in range(to_sample)], dtype=float)
        evals += to_sample
        best = max(best, float(np.max(fit)))

        elite_idx = np.argsort(fit)[-min(elite_n, len(fit)):]
        p = np.mean(X[elite_idx], axis=0)
        p = np.clip(p, eps, 1.0 - eps)

        if to_sample < pop:
            break

    return best


# -----------------------------
# Continuous algorithms
# -----------------------------

def hc_cont(inst: ContInstance, budget: int, rng: np.random.Generator, sigma: float = 0.2) -> float:
    x = rng.uniform(inst.lo, inst.hi, size=inst.dim)
    best = inst.evaluate(x)
    evals = 1
    while evals < budget:
        cand = clamp(x + rng.normal(0.0, sigma, size=inst.dim), inst.lo, inst.hi)
        s = inst.evaluate(cand)
        evals += 1
        if s >= best:
            x = cand
            best = s
    return best


def sa_cont(inst: ContInstance, budget: int, rng: np.random.Generator,
            sigma: float = 0.3, T0: float = 1.0, alpha: float = 0.995) -> float:
    x = rng.uniform(inst.lo, inst.hi, size=inst.dim)
    curr = inst.evaluate(x)
    best = curr
    evals = 1
    k = 0
    T = T0
    while evals < budget:
        cand = clamp(x + rng.normal(0.0, sigma, size=inst.dim), inst.lo, inst.hi)
        s = inst.evaluate(cand)
        evals += 1
        delta = s - curr
        if delta >= 0 or rng.random() < math.exp(delta / max(T, 1e-9)):
            x = cand
            curr = s
            if s > best:
                best = s
        k += 1
        T = T0 * (alpha ** k)
    return best


def pso_cont(inst: ContInstance, budget: int, rng: np.random.Generator,
             p: int = 20, w: float = 0.72, c1: float = 1.49, c2: float = 1.49,
             topology: str = "global") -> float:
    dim = inst.dim
    p = int(p)
    X = rng.uniform(inst.lo, inst.hi, size=(p, dim))
    V = rng.normal(0.0, 0.1, size=(p, dim))
    fit = np.array([inst.evaluate(X[i]) for i in range(p)], dtype=float)
    evals = p

    pbest = X.copy()
    pbest_fit = fit.copy()
    gbest_idx = int(np.argmax(fit))
    gbest = X[gbest_idx].copy()
    gbest_fit = float(fit[gbest_idx])

    def ring_best(i: int) -> np.ndarray:
        left = (i - 1) % p
        right = (i + 1) % p
        idxs = [left, i, right]
        j = int(idxs[np.argmax(pbest_fit[idxs])])
        return pbest[j]

    while evals < budget:
        remaining = budget - evals
        to_update = min(p, remaining)
        for i in range(to_update):
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            social = ring_best(i) if topology == "ring" else gbest
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (social - X[i])
            X[i] = clamp(X[i] + V[i], inst.lo, inst.hi)
            s = inst.evaluate(X[i])
            evals += 1
            if s > pbest_fit[i]:
                pbest[i] = X[i].copy()
                pbest_fit[i] = s
                if s > gbest_fit:
                    gbest_fit = float(s)
                    gbest = X[i].copy()
            if evals >= budget:
                break
        if to_update < p:
            break
    return gbest_fit


def de_cont(inst: ContInstance, budget: int, rng: np.random.Generator,
            pop: int = 20, F: float = 0.8, CR: float = 0.9) -> float:
    dim = inst.dim
    pop = int(pop)
    X = rng.uniform(inst.lo, inst.hi, size=(pop, dim))
    fit = np.array([inst.evaluate(X[i]) for i in range(pop)], dtype=float)
    evals = pop
    best = float(np.max(fit))

    while evals < budget:
        for i in range(pop):
            if evals >= budget:
                break
            idxs = [j for j in range(pop) if j != i]
            a, b, c = rng.choice(idxs, size=3, replace=False)
            mutant = clamp(X[a] + F * (X[b] - X[c]), inst.lo, inst.hi)
            cross = rng.random(dim) < CR
            if not np.any(cross):
                cross[int(rng.integers(0, dim))] = True
            trial = np.where(cross, mutant, X[i])
            s = inst.evaluate(trial)
            evals += 1
            if s >= fit[i]:
                X[i] = trial
                fit[i] = s
                if s > best:
                    best = float(s)
    return best


def es_cont(inst: ContInstance, budget: int, rng: np.random.Generator,
            mu: int = 10, lam: int = 40, sigma: float = 0.3) -> float:
    dim = inst.dim
    mu = int(mu)
    lam = int(lam)
    parents = rng.uniform(inst.lo, inst.hi, size=(mu, dim))
    fit = np.array([inst.evaluate(parents[i]) for i in range(mu)], dtype=float)
    evals = mu
    best = float(np.max(fit))

    while evals < budget:
        remaining = budget - evals
        to_make = min(lam, remaining)
        off = np.empty((to_make, dim), dtype=float)
        for i in range(to_make):
            p1, p2 = rng.integers(0, mu, size=2)
            base = 0.5 * (parents[p1] + parents[p2])
            off[i] = clamp(base + rng.normal(0.0, sigma, size=dim), inst.lo, inst.hi)
        off_fit = np.array([inst.evaluate(off[i]) for i in range(to_make)], dtype=float)
        evals += to_make
        best = max(best, float(np.max(off_fit)))
        if to_make >= mu:
            idx = np.argsort(off_fit)[-mu:]
            parents = off[idx]
            fit = off_fit[idx]
        else:
            break
    return best


# -----------------------------
# Registry
# -----------------------------

@dataclass(frozen=True)
class AlgoSpec:
    domain: str
    algo_base: str
    algo_variant: str
    fn: Any
    kwargs: Dict[str, Any]


def make_default_algos(tabu_k: int) -> List[AlgoSpec]:
    algos: List[AlgoSpec] = []
    algos += [
        # BIN
        AlgoSpec("bin", "RS_BIN", "RS_BIN", rs_bin, {}),
        AlgoSpec("bin", "HC", "HC(flips=1)", hc_bin, {"flips": 1}),
        AlgoSpec("bin", "HC", "HC(flips=2)", hc_bin, {"flips": 2}),
        AlgoSpec("bin", "SA", "SA(T0=1.0,alpha=0.995,flips=1)", sa_bin, {"T0": 1.0, "alpha": 0.995, "flips": 1}),
        AlgoSpec("bin", "SA", "SA(T0=2.0,alpha=0.99,flips=1)", sa_bin, {"T0": 2.0, "alpha": 0.99, "flips": 1}),
        AlgoSpec("bin", "SA", "SA(T0=2.0,alpha=0.995,flips=2)", sa_bin, {"T0": 2.0, "alpha": 0.995, "flips": 2}),
        AlgoSpec("bin", "TABU", "TABU(tenure=7)", tabu_bin, {"tenure": 7, "k": tabu_k}),
        AlgoSpec("bin", "TABU", "TABU(tenure=12)", tabu_bin, {"tenure": 12, "k": tabu_k}),
        AlgoSpec("bin", "GA", "GA(pop=50,pc=0.9,pm=0.02)", ga_bin, {"pop": 50, "pc": 0.9, "pm": 0.02}),
        AlgoSpec("bin", "GA", "GA(pop=50,pc=0.9,pm=0.05)", ga_bin, {"pop": 50, "pc": 0.9, "pm": 0.05}),
        AlgoSpec("bin", "GA", "GA(pop=100,pc=0.9,pm=0.02)", ga_bin, {"pop": 100, "pc": 0.9, "pm": 0.02}),
        AlgoSpec("bin", "UMDA", "UMDA(pop=50,elite=0.2)", umda_bin, {"pop": 50, "elite": 0.2}),
        AlgoSpec("bin", "UMDA", "UMDA(pop=100,elite=0.2)", umda_bin, {"pop": 100, "elite": 0.2}),
        # CONT
        AlgoSpec("cont", "RS_CONT", "RS_CONT", rs_cont, {}),
        AlgoSpec("cont", "HC_CONT", "HC_CONT(sigma=0.2)", hc_cont, {"sigma": 0.2}),
        AlgoSpec("cont", "SA_CONT", "SA_CONT(sigma=0.3,T0=1.0,alpha=0.995)", sa_cont, {"sigma": 0.3, "T0": 1.0, "alpha": 0.995}),
        AlgoSpec("cont", "PSO_STD", "PSO_STD(p=20,w=0.72,c1=1.49,c2=1.49)", pso_cont, {"p": 20, "w": 0.72, "c1": 1.49, "c2": 1.49, "topology": "global"}),
        AlgoSpec("cont", "PSO_STD", "PSO_STD(p=40,w=0.72,c1=1.49,c2=1.49)", pso_cont, {"p": 40, "w": 0.72, "c1": 1.49, "c2": 1.49, "topology": "global"}),
        AlgoSpec("cont", "PSO_RING", "PSO_RING(p=20,ring,w=0.72,c1=1.49,c2=1.49)", pso_cont, {"p": 20, "w": 0.72, "c1": 1.49, "c2": 1.49, "topology": "ring"}),
        AlgoSpec("cont", "PSO_RING", "PSO_RING(p=40,ring,w=0.72,c1=1.49,c2=1.49)", pso_cont, {"p": 40, "w": 0.72, "c1": 1.49, "c2": 1.49, "topology": "ring"}),
        AlgoSpec("cont", "DE", "DE(pop=20,F=0.8,CR=0.9)", de_cont, {"pop": 20, "F": 0.8, "CR": 0.9}),
        AlgoSpec("cont", "DE", "DE(pop=40,F=0.8,CR=0.9)", de_cont, {"pop": 40, "F": 0.8, "CR": 0.9}),
        AlgoSpec("cont", "ES_ML", "ES_ML(mu=10,lam=40,sigma=0.3)", es_cont, {"mu": 10, "lam": 40, "sigma": 0.3}),
        AlgoSpec("cont", "ES_ML", "ES_ML(mu=20,lam=80,sigma=0.2)", es_cont, {"mu": 20, "lam": 80, "sigma": 0.2}),
    ]
    return algos


def build_targets_for_instance(inst: Any, domain: str, budgets: List[int], calib_reps: int,
                               q_easy: float, q_med: float, q_hard: float, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    rs = rs_bin if domain == "bin" else rs_cont
    for b in budgets:
        bests = []
        for _ in range(calib_reps):
            rrng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
            bests.append(rs(inst, b, rrng))
        bests = np.array(bests, dtype=float)
        rows += [
            {"instance_id": inst.instance_id, "domain": domain, "problem": inst.problem, "budget": int(b),
             "target": "easy", "threshold": float(np.quantile(bests, q_easy)), "q": q_easy, "calib_reps": calib_reps},
            {"instance_id": inst.instance_id, "domain": domain, "problem": inst.problem, "budget": int(b),
             "target": "med", "threshold": float(np.quantile(bests, q_med)), "q": q_med, "calib_reps": calib_reps},
            {"instance_id": inst.instance_id, "domain": domain, "problem": inst.problem, "budget": int(b),
             "target": "hard", "threshold": float(np.quantile(bests, q_hard)), "q": q_hard, "calib_reps": calib_reps},
        ]
    return pd.DataFrame(rows)


def run_experiment(out_dir: Path, seed: int, budgets: List[int],
                   bin_instances: int, cont_instances: int,
                   bin_nbits: int, cont_dim: int, cont_lo: float, cont_hi: float,
                   reps: int, calib_reps: int,
                   q_easy: float, q_med: float, q_hard: float,
                   tabu_k: int, beta_mc_draws: int,
                   progress_every: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Instances
    bin_insts = make_bin_instances(rng, bin_instances, bin_nbits)
    cont_insts = make_cont_instances(rng, cont_instances, cont_dim, cont_lo, cont_hi)

    # Algorithms
    algos = make_default_algos(tabu_k=tabu_k)
    bin_algos = [a for a in algos if a.domain == "bin"]
    cont_algos = [a for a in algos if a.domain == "cont"]

    # Targets
    print(f"[INFO] generating targets via RS-quantiles: BIN={len(bin_insts)} CONT={len(cont_insts)} calib_reps={calib_reps}")
    t0 = time.time()
    frames = []
    for inst in bin_insts:
        frames.append(build_targets_for_instance(inst, "bin", budgets, calib_reps, q_easy, q_med, q_hard,
                                                 seed=int(rng.integers(0, 2**32 - 1))))
    for inst in cont_insts:
        frames.append(build_targets_for_instance(inst, "cont", budgets, calib_reps, q_easy, q_med, q_hard,
                                                 seed=int(rng.integers(0, 2**32 - 1))))
    targets_df = pd.concat(frames, ignore_index=True)
    targets_path = out_dir / "targets_by_instance.csv"
    targets_df.to_csv(targets_path, index=False)
    print(f"[OK] wrote {targets_path.as_posix()}  elapsed={time.time() - t0:.1f}s")

    thr_map = {(r.instance_id, int(r.budget), r.target): float(r.threshold) for r in targets_df.itertuples(index=False)}

    # Plan count
    total_runs = (
        len(bin_insts) * len(budgets) * 3 * len(bin_algos) * reps +
        len(cont_insts) * len(budgets) * 3 * len(cont_algos) * reps
    )
    print(f"[INFO] total runs planned: {total_runs:,} (reps={reps}, budgets={budgets})")

    runs_rows: List[Dict[str, Any]] = []
    done = 0
    t_start = time.time()

    def run_one(inst: Any, domain: str, spec: AlgoSpec, budget: int, target: str, rep_idx: int) -> Dict[str, Any]:
        thr = thr_map[(inst.instance_id, budget, target)]
        rrng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        best = spec.fn(inst, budget, rrng, **spec.kwargs)
        succ = 1 if best >= thr else 0
        return {
            "instance_id": inst.instance_id,
            "problem": inst.problem,
            "domain": domain,
            "budget": int(budget),
            "target": target,
            "algo_variant": spec.algo_variant,
            "algo_base": spec.algo_base,
            "rep": int(rep_idx),
            "best_score": float(best),
            "threshold": float(thr),
            "success": int(succ),
        }

    # BIN
    for inst in bin_insts:
        for b in budgets:
            for target in ("easy", "med", "hard"):
                for spec in bin_algos:
                    for r in range(reps):
                        runs_rows.append(run_one(inst, "bin", spec, b, target, r))
                        done += 1
                        if progress_every and done % progress_every == 0:
                            print(f"[PROG] {done:,}/{total_runs:,} runs  elapsed={time.time() - t_start:.1f}s")

    # CONT
    for inst in cont_insts:
        for b in budgets:
            for target in ("easy", "med", "hard"):
                for spec in cont_algos:
                    for r in range(reps):
                        runs_rows.append(run_one(inst, "cont", spec, b, target, r))
                        done += 1
                        if progress_every and done % progress_every == 0:
                            print(f"[PROG] {done:,}/{total_runs:,} runs  elapsed={time.time() - t_start:.1f}s")

    runs_df = pd.DataFrame(runs_rows)
    runs_path = out_dir / "runs_detail.csv"
    runs_df.to_csv(runs_path, index=False)
    print(f"[OK] wrote {runs_path.as_posix()}")

    # Aggregate
    grp_cols = ["instance_id", "problem", "domain", "budget", "target", "algo_variant", "algo_base"]
    agg = runs_df.groupby(grp_cols, as_index=False).agg(
        successes=("success", "sum"),
        trials=("success", "count"),
        succ_rate=("success", "mean"),
        mean_best=("best_score", "mean"),
        median_best=("best_score", "median"),
        min_best=("best_score", "min"),
        max_best=("best_score", "max"),
    )

    # Beta stats
    bmeans, p05s, p95s = [], [], []
    base_qseed = int(np.random.default_rng(seed).integers(0, 2**32 - 1))
    for idx, row in agg.iterrows():
        s = int(row["successes"])
        R = int(row["trials"])
        bmeans.append(beta_mean(s, R))
        p05, p95 = beta_quantiles(s, R, draws=beta_mc_draws, seed=base_qseed + idx)
        p05s.append(p05)
        p95s.append(p95)
    agg["beta_mean"] = bmeans
    agg["beta_p05"] = p05s
    agg["beta_p95"] = p95s

    out_summary = out_dir / "instance_algo_budget_summary.csv"
    cols = [
        "instance_id", "problem", "domain", "budget", "target",
        "algo_variant", "algo_base",
        "successes", "trials", "succ_rate",
        "beta_mean", "beta_p05", "beta_p95",
        "mean_best", "median_best", "min_best", "max_best"
    ]
    agg = agg[cols]
    agg.to_csv(out_summary, index=False)
    print(f"[OK] wrote {out_summary.as_posix()}")

    # KPI
    print("\n=== Quick KPI (mean beta_mean by domain/budget/target) ===")
    for (dom, b, t), sub in agg.groupby(["domain", "budget", "target"]):
        sub2 = sub.groupby("algo_variant", as_index=False)["beta_mean"].mean().sort_values("beta_mean", ascending=False)
        top5 = ", ".join([f"{r.algo_variant}={r.beta_mean:.3f}" for r in sub2.head(5).itertuples(index=False)])
        print(f"{dom.upper():4s} budget={int(b):<4d} target={t:<4s} top5: {top5}")

    print(f"\n[DONE] total runs={total_runs:,}  elapsed={time.time() - t_start:.1f}s")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--budgets", type=str, default="300,800", help="Comma-separated budgets, e.g. 300,800")
    ap.add_argument("--bin_instances", type=int, default=20)
    ap.add_argument("--cont_instances", type=int, default=20)
    ap.add_argument("--bin_nbits", type=int, default=80)
    ap.add_argument("--cont_dim", type=int, default=10)
    ap.add_argument("--cont_lo", type=float, default=-5.12)
    ap.add_argument("--cont_hi", type=float, default=5.12)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--calib_reps", type=int, default=20)
    ap.add_argument("--q_easy", type=float, default=0.50)
    ap.add_argument("--q_med", type=float, default=0.75)
    ap.add_argument("--q_hard", type=float, default=0.90)
    ap.add_argument("--tabu_k", type=int, default=20)
    ap.add_argument("--beta_mc_draws", type=int, default=4000)
    ap.add_argument("--progress_every", type=int, default=2000)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    run_experiment(
        out_dir=Path(args.out_dir),
        seed=args.seed,
        budgets=budgets,
        bin_instances=args.bin_instances,
        cont_instances=args.cont_instances,
        bin_nbits=args.bin_nbits,
        cont_dim=args.cont_dim,
        cont_lo=args.cont_lo,
        cont_hi=args.cont_hi,
        reps=args.reps,
        calib_reps=args.calib_reps,
        q_easy=args.q_easy,
        q_med=args.q_med,
        q_hard=args.q_hard,
        tabu_k=args.tabu_k,
        beta_mc_draws=args.beta_mc_draws,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
