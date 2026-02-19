#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Large-scale success-profile experiment runner (MVP, self-contained).

What it does
------------
1) Generates many optimisation problem instances (binary + continuous).
2) Runs many algorithms / variants for many repetitions under multiple budgets.
3) Logs per-run results (final best + AUC-type trajectory metrics).
4) Aggregates to an "instance_algo_budget_summary.csv" compatible with your
   success_profile_analysis_v2_2.py script (same key columns, plus extras).

Why this version
----------------
- Scales to more instances / algorithms / budgets without changing your analysis pipeline.
- Adds multi-target success (easy/med/hard) + optional top-quantile targets.
- Adds "trajectory quality" signals (AUC on best-so-far, log-AUC) so you can go
  beyond single success probabilities when writing the paper.

No external deps beyond: numpy, pandas.

Example
-------
python success_profile_large_runner_v3.py \\
  --out_dir out_succ_large_v3 \\
  --seed 0 \\
  --instances 60 \\
  --reps 30 \\
  --calib 200 \\
  --budgets 500 2000 5000 \\
  --bin_bits 200 500 \\
  --cont_dims 10 30 \\
  --bin_problems onemax leadingones trap5 knapsack01 \\
  --cont_problems sphere rastrigin ackley rosenbrock \\
  --add_topq 0.1 0.25

Notes (pragmatic)
-----------------
- If you go truly large (e.g., 100 instances × 50 reps × 20 algos × 4 budgets),
  you will want parallelism. This file keeps it single-process to remain portable.
  You can later bolt on multiprocessing around run_one().
"""

from __future__ import annotations
import argparse
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ---------------------------
# Utilities
# ---------------------------

def set_global_seed(seed: int) -> None:
    np.random.seed(seed)

def now_ms() -> int:
    return int(time.time() * 1000)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def clamp(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))

def safe_logit(p: float, eps: float = 1e-6) -> float:
    p = clamp(p, eps, 1 - eps)
    return float(math.log(p / (1 - p)))

def beta_posterior_stats(successes: int, trials: int, a0: float = 1.0, b0: float = 1.0,
                         mc_samples: int = 20000, seed: int = 0) -> Tuple[float, float, float]:
    """
    Returns (mean, p05, p95) for Beta(a0+succ, b0+fail).
    Uses Monte Carlo to avoid SciPy dependency.
    """
    a = a0 + successes
    b = b0 + (trials - successes)
    rng = np.random.default_rng(seed)
    samples = rng.beta(a, b, size=mc_samples)
    mean = float(a / (a + b))
    p05 = float(np.quantile(samples, 0.05))
    p95 = float(np.quantile(samples, 0.95))
    return mean, p05, p95

def auc_best(best_so_far: np.ndarray, mode: str, eps: float = 1e-12) -> Tuple[float, float]:
    """
    Two trajectory summaries:
      - auc_raw  : mean(best_so_far)  (larger better for max, smaller better for min)
      - auc_log  : mean(log(best_so_far+eps)) (same direction as raw for min if positive)

    For MAX problems, we also provide auc_log_regret if we can infer an upper bound.
    In this MVP, we return only auc_raw and auc_log(abs(best)+eps) in a consistent way.
    """
    if best_so_far.size == 0:
        return float("nan"), float("nan")
    auc_raw = float(np.mean(best_so_far))
    auc_log = float(np.mean(np.log(np.abs(best_so_far) + eps)))
    # For "min" tasks, smaller auc_raw is better; for "max", larger is better.
    return auc_raw, auc_log

def parse_variant_base(variant: str) -> str:
    # algo_base is everything before first '(' if present.
    i = variant.find("(")
    return variant if i < 0 else variant[:i].strip()


# ---------------------------
# Problem instances
# ---------------------------

@dataclass
class BinInstance:
    instance_id: str
    problem: str
    n_bits: int
    # knapsack params
    weights: Optional[np.ndarray] = None
    values: Optional[np.ndarray] = None
    capacity: Optional[int] = None
    # misc seed
    seed: int = 0

@dataclass
class ContInstance:
    instance_id: str
    problem: str
    dim: int
    shift: np.ndarray
    rot: Optional[np.ndarray]
    bounds: Tuple[float, float]
    seed: int = 0

# Binary objectives (maximize)
def f_onemax(x: np.ndarray, inst: BinInstance) -> float:
    return float(np.sum(x))

def f_leadingones(x: np.ndarray, inst: BinInstance) -> float:
    # count consecutive ones from start
    idx0 = np.where(x == 0)[0]
    return float(idx0[0]) if idx0.size > 0 else float(x.size)

def f_trap5(x: np.ndarray, inst: BinInstance, k: int = 5) -> float:
    n = x.size
    m = n // k
    score = 0.0
    for i in range(m):
        block = x[i*k:(i+1)*k]
        u = int(np.sum(block))
        score += k if u == k else (k - 1 - u)  # deceptive trap
    # leftover bits contribute linearly
    if m*k < n:
        score += float(np.sum(x[m*k:]))
    return float(score)

def f_knapsack01(x: np.ndarray, inst: BinInstance) -> float:
    assert inst.weights is not None and inst.values is not None and inst.capacity is not None
    w = int(np.dot(x, inst.weights))
    v = float(np.dot(x, inst.values))
    if w <= inst.capacity:
        return v
    # penalty for infeasible: harsh but not zero, still comparable
    return float(v - 10.0 * (w - inst.capacity))

BIN_FUNCS = {
    "onemax": f_onemax,
    "leadingones": f_leadingones,
    "trap5": f_trap5,
    "knapsack01": f_knapsack01,
}

def make_knapsack_instance(n_bits: int, seed: int) -> Tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    # weights in [1, 100], values correlated with weights
    weights = rng.integers(1, 101, size=n_bits, dtype=np.int32)
    values = (weights + rng.integers(0, 50, size=n_bits)).astype(np.int32)
    capacity = int(0.5 * np.sum(weights))
    return weights, values, capacity

def knapsack_upper_bound(weights: np.ndarray, values: np.ndarray, capacity: int) -> float:
    # fractional greedy UB
    ratio = values / np.maximum(weights, 1e-9)
    order = np.argsort(-ratio)
    wsum = 0.0
    vsum = 0.0
    for i in order:
        if wsum + weights[i] <= capacity:
            wsum += float(weights[i])
            vsum += float(values[i])
        else:
            remain = capacity - wsum
            if remain <= 0:
                break
            frac = float(remain) / float(weights[i])
            vsum += float(values[i]) * frac
            break
    return float(vsum)

# Continuous objectives (minimize)
def _apply_shift_rot(x: np.ndarray, inst: ContInstance) -> np.ndarray:
    z = x - inst.shift
    if inst.rot is not None:
        z = inst.rot @ z
    return z

def f_sphere(x: np.ndarray, inst: ContInstance) -> float:
    z = _apply_shift_rot(x, inst)
    return float(np.sum(z*z))

def f_rastrigin(x: np.ndarray, inst: ContInstance) -> float:
    z = _apply_shift_rot(x, inst)
    return float(10.0*inst.dim + np.sum(z*z - 10.0*np.cos(2*math.pi*z)))

def f_ackley(x: np.ndarray, inst: ContInstance) -> float:
    z = _apply_shift_rot(x, inst)
    d = inst.dim
    a = 20.0
    b = 0.2
    c = 2*math.pi
    s1 = np.sum(z*z)
    s2 = np.sum(np.cos(c*z))
    return float(-a*np.exp(-b*math.sqrt(s1/d)) - np.exp(s2/d) + a + math.e)

def f_rosenbrock(x: np.ndarray, inst: ContInstance) -> float:
    z = _apply_shift_rot(x, inst)
    return float(np.sum(100.0*(z[1:] - z[:-1]**2)**2 + (1 - z[:-1])**2))

CONT_FUNCS = {
    "sphere": f_sphere,
    "rastrigin": f_rastrigin,
    "ackley": f_ackley,
    "rosenbrock": f_rosenbrock,
}

CONT_BOUNDS = {
    "sphere": (-5.0, 5.0),
    "rastrigin": (-5.12, 5.12),
    "ackley": (-32.768, 32.768),
    "rosenbrock": (-5.0, 5.0),
}

def random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    # QR-based random orthogonal
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    # Fix sign
    diag = np.sign(np.diag(R))
    Q = Q * diag
    return Q

def generate_bin_instances(problems: List[str], n_bits_list: List[int], n_instances: int, seed: int) -> List[BinInstance]:
    rng = np.random.default_rng(seed)
    insts: List[BinInstance] = []
    for pb in problems:
        for n_bits in n_bits_list:
            for j in range(n_instances):
                s = int(rng.integers(0, 2**31-1))
                instance_id = f"bin::{pb}::n{n_bits}::i{j:03d}"
                if pb == "knapsack01":
                    w, v, cap = make_knapsack_instance(n_bits, s)
                    insts.append(BinInstance(instance_id=instance_id, problem=pb, n_bits=n_bits,
                                             weights=w, values=v, capacity=cap, seed=s))
                else:
                    insts.append(BinInstance(instance_id=instance_id, problem=pb, n_bits=n_bits, seed=s))
    return insts

def generate_cont_instances(problems: List[str], dims: List[int], n_instances: int, seed: int,
                            use_rotation: bool = True) -> List[ContInstance]:
    rng = np.random.default_rng(seed)
    insts: List[ContInstance] = []
    for pb in problems:
        lo, hi = CONT_BOUNDS.get(pb, (-5.0, 5.0))
        for d in dims:
            for j in range(n_instances):
                s = int(rng.integers(0, 2**31-1))
                r = np.random.default_rng(s)
                shift = r.uniform(lo, hi, size=d)
                rot = random_orthogonal(d, r) if use_rotation and d <= 50 else None
                instance_id = f"cont::{pb}::d{d}::i{j:03d}"
                insts.append(ContInstance(instance_id=instance_id, problem=pb, dim=d, shift=shift,
                                          rot=rot, bounds=(lo, hi), seed=s))
    return insts


# ---------------------------
# Targets (multi-level + optional top-q)
# ---------------------------

def calibrate_targets_bin(inst: BinInstance, calib_evals: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    f = BIN_FUNCS[inst.problem]
    # baseline best from random sampling
    best = -1e18
    for _ in range(calib_evals):
        x = rng.integers(0, 2, size=inst.n_bits, dtype=np.int8)
        val = f(x, inst)
        if val > best:
            best = val

    # upper bound
    if inst.problem in ("onemax", "leadingones", "trap5"):
        ub = float(inst.n_bits)
    else:
        # fractional UB for knapsack
        ub = knapsack_upper_bound(inst.weights, inst.values, inst.capacity)  # type: ignore

    b0 = float(best)
    # Ensure b0 <= ub (numerical)
    ub = max(ub, b0)

    easy = b0 + 0.30*(ub - b0)
    med  = b0 + 0.60*(ub - b0)
    hard = b0 + 0.85*(ub - b0)
    # Clip to ub
    easy = min(easy, ub)
    med = min(med, ub)
    hard = min(hard, ub)
    return {"easy": float(easy), "med": float(med), "hard": float(hard), "ub": float(ub), "b0": float(b0)}

def calibrate_targets_cont(inst: ContInstance, calib_evals: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    f = CONT_FUNCS[inst.problem]
    lo, hi = inst.bounds
    best = 1e18
    for _ in range(calib_evals):
        x = rng.uniform(lo, hi, size=inst.dim)
        val = f(x, inst)
        if val < best:
            best = val
    b0 = float(best)
    # targets are fractions of baseline best (lower is better)
    eps = 1e-12
    easy = max(b0 * 0.70, eps)
    med  = max(b0 * 0.40, eps)
    hard = max(b0 * 0.20, eps)
    return {"easy": float(easy), "med": float(med), "hard": float(hard), "b0": float(b0)}


# ---------------------------
# Algorithms (binary)
# ---------------------------

def rs_bin(inst: BinInstance, budget: int, seed: int) -> Tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    f = BIN_FUNCS[inst.problem]
    best = -1e18
    traj = np.empty(budget, dtype=float)
    for t in range(budget):
        x = rng.integers(0, 2, size=inst.n_bits, dtype=np.int8)
        val = f(x, inst)
        if val > best:
            best = val
        traj[t] = best
    return float(best), traj

def hc_bin(inst: BinInstance, budget: int, seed: int, flips: int = 1) -> Tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    f = BIN_FUNCS[inst.problem]
    x = rng.integers(0, 2, size=inst.n_bits, dtype=np.int8)
    best = f(x, inst)
    traj = np.empty(budget, dtype=float)
    for t in range(budget):
        y = x.copy()
        idx = rng.choice(inst.n_bits, size=flips, replace=False)
        y[idx] = 1 - y[idx]
        val = f(y, inst)
        if val >= best:
            x = y
            best = val
        traj[t] = best
    return float(best), traj

def sa_bin(inst: BinInstance, budget: int, seed: int, T0: float = 2.0, alpha: float = 0.995, flips: int = 1) -> Tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    f = BIN_FUNCS[inst.problem]
    x = rng.integers(0, 2, size=inst.n_bits, dtype=np.int8)
    fx = f(x, inst)
    best = fx
    T = float(T0)
    traj = np.empty(budget, dtype=float)
    for t in range(budget):
        y = x.copy()
        idx = rng.choice(inst.n_bits, size=flips, replace=False)
        y[idx] = 1 - y[idx]
        fy = f(y, inst)
        if fy >= fx or rng.random() < math.exp((fy - fx) / max(T, 1e-12)):
            x, fx = y, fy
        if fx > best:
            best = fx
        traj[t] = best
        T *= alpha
    return float(best), traj

def tabu_bin(inst: BinInstance, budget: int, seed: int, tenure: int = 7) -> Tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    f = BIN_FUNCS[inst.problem]
    x = rng.integers(0, 2, size=inst.n_bits, dtype=np.int8)
    fx = f(x, inst)
    best = fx
    # tabu list for bit indices
    tabu_until = np.zeros(inst.n_bits, dtype=np.int32)
    traj = np.empty(budget, dtype=float)
    for t in range(budget):
        # evaluate a small neighborhood sample to keep cost stable
        cand_idx = rng.choice(inst.n_bits, size=min(30, inst.n_bits), replace=False)
        best_move = None
        best_val = -1e18
        for i in cand_idx:
            y = x.copy()
            y[i] = 1 - y[i]
            val = f(y, inst)
            is_tabu = t < tabu_until[i]
            # aspiration: allow tabu if improves global best
            if is_tabu and val <= best:
                continue
            if val > best_val:
                best_val = val
                best_move = i
        if best_move is not None:
            i = int(best_move)
            x[i] = 1 - x[i]
            fx = best_val
            tabu_until[i] = t + tenure
            if fx > best:
                best = fx
        traj[t] = best
    return float(best), traj

def ga_bin(inst: BinInstance, budget: int, seed: int, pop: int = 50, pc: float = 0.9, pm: float = 0.02) -> Tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    f = BIN_FUNCS[inst.problem]
    pop_size = int(pop)
    # Each generation evaluates pop individuals.
    # Ensure at least 1 generation.
    gens = max(1, budget // pop_size)
    # init
    P = rng.integers(0, 2, size=(pop_size, inst.n_bits), dtype=np.int8)
    fit = np.array([f(P[i], inst) for i in range(pop_size)], dtype=float)
    best = float(np.max(fit))
    evals = pop_size
    traj = []
    traj.extend([best] * min(budget, evals))
    while evals < budget:
        # tournament selection
        def tour():
            a, b = rng.integers(0, pop_size), rng.integers(0, pop_size)
            return P[a] if fit[a] >= fit[b] else P[b]
        newP = np.empty_like(P)
        for i in range(0, pop_size, 2):
            p1 = tour().copy()
            p2 = tour().copy()
            if rng.random() < pc:
                cx = int(rng.integers(1, inst.n_bits))
                c1 = np.concatenate([p1[:cx], p2[cx:]])
                c2 = np.concatenate([p2[:cx], p1[cx:]])
            else:
                c1, c2 = p1, p2
            # mutation
            m1 = rng.random(inst.n_bits) < pm
            m2 = rng.random(inst.n_bits) < pm
            c1[m1] = 1 - c1[m1]
            c2[m2] = 1 - c2[m2]
            newP[i] = c1
            if i + 1 < pop_size:
                newP[i+1] = c2
        P = newP
        fit = np.array([f(P[i], inst) for i in range(pop_size)], dtype=float)
        evals += pop_size
        best = max(best, float(np.max(fit)))
        traj.extend([best] * min(pop_size, budget - (evals - pop_size)))
        if evals >= budget:
            break
    traj_arr = np.array(traj[:budget], dtype=float)
    return float(best), traj_arr

def umda_bin(inst: BinInstance, budget: int, seed: int, pop: int = 50, elite: float = 0.2) -> Tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    f = BIN_FUNCS[inst.problem]
    pop_size = int(pop)
    mu = max(1, int(pop_size * elite))
    p = np.full(inst.n_bits, 0.5, dtype=float)
    best = -1e18
    evals = 0
    traj = []
    while evals < budget:
        # sample
        P = (rng.random((pop_size, inst.n_bits)) < p).astype(np.int8)
        fit = np.array([f(P[i], inst) for i in range(pop_size)], dtype=float)
        evals += pop_size
        best = max(best, float(np.max(fit)))
        traj.extend([best] * min(pop_size, budget - (evals - pop_size)))
        # update probs from elites
        elite_idx = np.argsort(-fit)[:mu]
        p = np.mean(P[elite_idx], axis=0)
        # prevent collapse
        p = np.clip(p, 0.02, 0.98)
    return float(best), np.array(traj[:budget], dtype=float)

BIN_ALGOS = {
    "RS_BIN": ("Random", rs_bin, {}),
    "HC": ("LocalSearch", hc_bin, {}),
    "SA": ("LocalSearch", sa_bin, {}),
    "TABU": ("LocalSearch", tabu_bin, {}),
    "GA": ("Evolution", ga_bin, {}),
    "UMDA": ("Evolution", umda_bin, {}),
}


# ---------------------------
# Algorithms (continuous)
# ---------------------------

def rs_cont(inst: ContInstance, budget: int, seed: int) -> Tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    f = CONT_FUNCS[inst.problem]
    lo, hi = inst.bounds
    best = 1e18
    traj = np.empty(budget, dtype=float)
    for t in range(budget):
        x = rng.uniform(lo, hi, size=inst.dim)
        val = f(x, inst)
        if val < best:
            best = val
        traj[t] = best
    return float(best), traj

def hc_cont(inst: ContInstance, budget: int, seed: int, sigma: float = 0.2) -> Tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    f = CONT_FUNCS[inst.problem]
    lo, hi = inst.bounds
    x = rng.uniform(lo, hi, size=inst.dim)
    fx = f(x, inst)
    best = fx
    traj = np.empty(budget, dtype=float)
    for t in range(budget):
        y = x + rng.normal(scale=sigma, size=inst.dim)
        y = np.clip(y, lo, hi)
        fy = f(y, inst)
        if fy <= fx:
            x, fx = y, fy
        best = min(best, fx)
        traj[t] = best
    return float(best), traj

def sa_cont(inst: ContInstance, budget: int, seed: int, sigma: float = 0.3, T0: float = 1.0, alpha: float = 0.995) -> Tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    f = CONT_FUNCS[inst.problem]
    lo, hi = inst.bounds
    x = rng.uniform(lo, hi, size=inst.dim)
    fx = f(x, inst)
    best = fx
    T = float(T0)
    traj = np.empty(budget, dtype=float)
    for t in range(budget):
        y = x + rng.normal(scale=sigma, size=inst.dim)
        y = np.clip(y, lo, hi)
        fy = f(y, inst)
        if fy <= fx or rng.random() < math.exp(-(fy - fx) / max(T, 1e-12)):
            x, fx = y, fy
        if fx < best:
            best = fx
        traj[t] = best
        T *= alpha
    return float(best), traj

def de_cont(inst: ContInstance, budget: int, seed: int, pop: int = 20, F: float = 0.8, CR: float = 0.9) -> Tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    f = CONT_FUNCS[inst.problem]
    lo, hi = inst.bounds
    pop_size = int(pop)
    X = rng.uniform(lo, hi, size=(pop_size, inst.dim))
    fit = np.array([f(X[i], inst) for i in range(pop_size)], dtype=float)
    best = float(np.min(fit))
    evals = pop_size
    traj = []
    traj.extend([best] * min(budget, evals))
    while evals < budget:
        for i in range(pop_size):
            idxs = rng.choice([j for j in range(pop_size) if j != i], size=3, replace=False)
            a, b, c = X[idxs]
            mutant = np.clip(a + F * (b - c), lo, hi)
            cross = rng.random(inst.dim) < CR
            if not np.any(cross):
                cross[rng.integers(0, inst.dim)] = True
            trial = np.where(cross, mutant, X[i])
            ftrial = f(trial, inst)
            evals += 1
            if ftrial <= fit[i]:
                X[i] = trial
                fit[i] = ftrial
                if ftrial < best:
                    best = float(ftrial)
            traj.append(best)
            if evals >= budget:
                break
        if evals >= budget:
            break
    traj_arr = np.array(traj[:max(0, budget - min(budget, pop_size))], dtype=float)
    if traj_arr.size == 0:
        full = np.array(traj[:budget], dtype=float)
        if full.size < budget:
            full = np.pad(full, (0, budget-full.size), constant_values=best)
        return best, full
    # prefix already filled
    prefix = np.array([best] * min(budget, pop_size), dtype=float)
    full = np.concatenate([prefix, traj_arr])[:budget]
    if full.size < budget:
        full = np.pad(full, (0, budget-full.size), constant_values=best)
    return float(best), full

def pso_cont(inst: ContInstance, budget: int, seed: int, p: int = 20, w: float = 0.72, c1: float = 1.49, c2: float = 1.49,
             topology: str = "global") -> Tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    f = CONT_FUNCS[inst.problem]
    lo, hi = inst.bounds
    n = int(p)
    X = rng.uniform(lo, hi, size=(n, inst.dim))
    V = rng.normal(scale=0.1*(hi-lo), size=(n, inst.dim))
    pbest = X.copy()
    pbest_fit = np.array([f(X[i], inst) for i in range(n)], dtype=float)
    gbest_idx = int(np.argmin(pbest_fit))
    gbest = pbest[gbest_idx].copy()
    gbest_fit = float(pbest_fit[gbest_idx])

    evals = n
    traj = []
    traj.extend([gbest_fit] * min(budget, evals))

    # neighbor best for ring
    def neigh_best(i: int) -> Tuple[np.ndarray, float]:
        if topology == "global":
            return gbest, gbest_fit
        # ring: neighborhood (i-1, i, i+1)
        idxs = [(i-1) % n, i, (i+1) % n]
        j = idxs[int(np.argmin(pbest_fit[idxs]))]
        return pbest[j], float(pbest_fit[j])

    while evals < budget:
        for i in range(n):
            nb, nb_fit = neigh_best(i)
            r1 = rng.random(inst.dim)
            r2 = rng.random(inst.dim)
            V[i] = w*V[i] + c1*r1*(pbest[i] - X[i]) + c2*r2*(nb - X[i])
            X[i] = np.clip(X[i] + V[i], lo, hi)
            fx = f(X[i], inst)
            evals += 1
            if fx < pbest_fit[i]:
                pbest[i] = X[i].copy()
                pbest_fit[i] = fx
                if fx < gbest_fit:
                    gbest_fit = float(fx)
                    gbest = X[i].copy()
            traj.append(gbest_fit)
            if evals >= budget:
                break
        if evals >= budget:
            break
    # pad / truncate
    full = np.array(traj[:budget], dtype=float)
    if full.size < budget:
        full = np.pad(full, (0, budget-full.size), constant_values=gbest_fit)
    return float(gbest_fit), full

def es_ml_cont(inst: ContInstance, budget: int, seed: int, mu: int = 10, lam: int = 40, sigma: float = 0.3) -> Tuple[float, np.ndarray]:
    rng = np.random.default_rng(seed)
    f = CONT_FUNCS[inst.problem]
    lo, hi = inst.bounds
    mu = int(mu); lam = int(lam)
    x_mean = rng.uniform(lo, hi, size=inst.dim)
    best = f(x_mean, inst)
    evals = 1
    traj = [best]
    while evals < budget:
        # sample offspring
        X = x_mean + rng.normal(scale=sigma, size=(lam, inst.dim))
        X = np.clip(X, lo, hi)
        fit = np.array([f(X[i], inst) for i in range(lam)], dtype=float)
        evals += lam
        # select
        idx = np.argsort(fit)[:mu]
        x_mean = np.mean(X[idx], axis=0)
        best = min(best, float(np.min(fit)))
        traj.extend([best] * min(lam, max(0, budget - (evals - lam))))
        if evals >= budget:
            break
        # mild sigma schedule (simple)
        sigma = max(1e-6, sigma * 0.999)
    full = np.array(traj[:budget], dtype=float)
    if full.size < budget:
        full = np.pad(full, (0, budget-full.size), constant_values=best)
    return float(best), full

CONT_ALGOS = {
    "RS_CONT": ("Random", rs_cont, {}),
    "HC_CONT": ("LocalSearch", hc_cont, {}),
    "SA_CONT": ("LocalSearch", sa_cont, {}),
    "DE": ("Evolution", de_cont, {}),
    "PSO_STD": ("PSO_global", pso_cont, {"topology": "global"}),
    "PSO_RING": ("PSO_local", pso_cont, {"topology": "ring"}),
    "ES_ML": ("Evolution", es_ml_cont, {}),
}


# ---------------------------
# Running one (instance, algo_variant, rep, budget)
# ---------------------------

@dataclass
class RunResult:
    instance_id: str
    domain: str
    problem: str
    budget: int
    algo_variant: str
    algo_base: str
    rep: int
    seed: int
    best: float
    auc_raw: float
    auc_log: float
    wall_ms: int
    # successes per target_name
    success: Dict[str, int]

def run_one_bin(inst: BinInstance, budget: int, algo_variant: str, rep: int, seed: int,
                targets: Dict[str, float], topq_targets: Optional[Dict[str, float]] = None) -> RunResult:
    base = parse_variant_base(algo_variant)
    start = now_ms()

    # dispatch
    if algo_variant == "RS_BIN":
        best, traj = rs_bin(inst, budget, seed)
    elif algo_variant.startswith("HC("):
        flips = int(algo_variant.split("flips=")[1].split(")")[0])
        best, traj = hc_bin(inst, budget, seed, flips=flips)
    elif algo_variant.startswith("SA("):
        # SA(T0=2.0,alpha=0.995,flips=1)
        s = algo_variant
        T0 = float(s.split("T0=")[1].split(",")[0])
        alpha = float(s.split("alpha=")[1].split(",")[0])
        flips = int(s.split("flips=")[1].split(")")[0])
        best, traj = sa_bin(inst, budget, seed, T0=T0, alpha=alpha, flips=flips)
    elif algo_variant.startswith("TABU("):
        tenure = int(algo_variant.split("tenure=")[1].split(")")[0])
        best, traj = tabu_bin(inst, budget, seed, tenure=tenure)
    elif algo_variant.startswith("GA("):
        # GA(pop=50,pc=0.9,pm=0.02)
        s = algo_variant
        pop = int(s.split("pop=")[1].split(",")[0])
        pc = float(s.split("pc=")[1].split(",")[0])
        pm = float(s.split("pm=")[1].split(")")[0])
        best, traj = ga_bin(inst, budget, seed, pop=pop, pc=pc, pm=pm)
    elif algo_variant.startswith("UMDA("):
        # UMDA(pop=50,elite=0.2)
        s = algo_variant
        pop = int(s.split("pop=")[1].split(",")[0])
        elite = float(s.split("elite=")[1].split(")")[0])
        best, traj = umda_bin(inst, budget, seed, pop=pop, elite=elite)
    else:
        raise ValueError(f"Unknown binary algo_variant: {algo_variant}")

    wall = now_ms() - start
    auc_raw, auc_log = auc_best(traj, mode="max")

    succ: Dict[str, int] = {}
    for k in ("easy", "med", "hard"):
        thr = targets[k]
        succ[k] = 1 if best >= thr else 0
    if topq_targets:
        for name, thr in topq_targets.items():
            succ[name] = 1 if best >= thr else 0

    return RunResult(
        instance_id=inst.instance_id,
        domain="bin",
        problem=inst.problem,
        budget=int(budget),
        algo_variant=algo_variant,
        algo_base=base,
        rep=int(rep),
        seed=int(seed),
        best=float(best),
        auc_raw=float(auc_raw),
        auc_log=float(auc_log),
        wall_ms=int(wall),
        success=succ,
    )

def run_one_cont(inst: ContInstance, budget: int, algo_variant: str, rep: int, seed: int,
                 targets: Dict[str, float], topq_targets: Optional[Dict[str, float]] = None) -> RunResult:
    base = parse_variant_base(algo_variant)
    start = now_ms()

    if algo_variant == "RS_CONT":
        best, traj = rs_cont(inst, budget, seed)
    elif algo_variant.startswith("HC_CONT("):
        sigma = float(algo_variant.split("sigma=")[1].split(")")[0])
        best, traj = hc_cont(inst, budget, seed, sigma=sigma)
    elif algo_variant.startswith("SA_CONT("):
        s = algo_variant
        sigma = float(s.split("sigma=")[1].split(",")[0])
        T0 = float(s.split("T0=")[1].split(",")[0])
        alpha = float(s.split("alpha=")[1].split(")")[0])
        best, traj = sa_cont(inst, budget, seed, sigma=sigma, T0=T0, alpha=alpha)
    elif algo_variant.startswith("DE("):
        s = algo_variant
        pop = int(s.split("pop=")[1].split(",")[0])
        F = float(s.split("F=")[1].split(",")[0])
        CR = float(s.split("CR=")[1].split(")")[0])
        best, traj = de_cont(inst, budget, seed, pop=pop, F=F, CR=CR)
    elif algo_variant.startswith("PSO_STD("):
        s = algo_variant
        p = int(s.split("p=")[1].split(",")[0])
        w = float(s.split("w=")[1].split(",")[0])
        c1 = float(s.split("c1=")[1].split(",")[0])
        c2 = float(s.split("c2=")[1].split(")")[0])
        best, traj = pso_cont(inst, budget, seed, p=p, w=w, c1=c1, c2=c2, topology="global")
    elif algo_variant.startswith("PSO_RING("):
        s = algo_variant
        p = int(s.split("p=")[1].split(",")[0])
        w = float(s.split("w=")[1].split(",")[0])
        c1 = float(s.split("c1=")[1].split(",")[0])
        c2 = float(s.split("c2=")[1].split(")")[0])
        best, traj = pso_cont(inst, budget, seed, p=p, w=w, c1=c1, c2=c2, topology="ring")
    elif algo_variant.startswith("ES_ML("):
        s = algo_variant
        mu = int(s.split("mu=")[1].split(",")[0])
        lam = int(s.split("lam=")[1].split(",")[0])
        sigma = float(s.split("sigma=")[1].split(")")[0])
        best, traj = es_ml_cont(inst, budget, seed, mu=mu, lam=lam, sigma=sigma)
    else:
        raise ValueError(f"Unknown continuous algo_variant: {algo_variant}")

    wall = now_ms() - start
    auc_raw, auc_log = auc_best(traj, mode="min")

    succ: Dict[str, int] = {}
    for k in ("easy", "med", "hard"):
        thr = targets[k]
        succ[k] = 1 if best <= thr else 0
    if topq_targets:
        for name, thr in topq_targets.items():
            succ[name] = 1 if best <= thr else 0

    return RunResult(
        instance_id=inst.instance_id,
        domain="cont",
        problem=inst.problem,
        budget=int(budget),
        algo_variant=algo_variant,
        algo_base=base,
        rep=int(rep),
        seed=int(seed),
        best=float(best),
        auc_raw=float(auc_raw),
        auc_log=float(auc_log),
        wall_ms=int(wall),
        success=succ,
    )


# ---------------------------
# Variant factory (expand algorithm variants)
# ---------------------------

def build_bin_variants() -> List[str]:
    v = []
    v.append("RS_BIN")
    v.extend([f"HC(flips={k})" for k in (1, 2)])
    v.extend([
        "SA(T0=1.0,alpha=0.995,flips=1)",
        "SA(T0=2.0,alpha=0.99,flips=1)",
        "SA(T0=2.0,alpha=0.995,flips=2)",
    ])
    v.extend([f"TABU(tenure={t})" for t in (7, 12)])
    v.extend([
        "GA(pop=50,pc=0.9,pm=0.02)",
        "GA(pop=50,pc=0.9,pm=0.05)",
        "GA(pop=100,pc=0.9,pm=0.02)",
    ])
    v.extend([
        "UMDA(pop=50,elite=0.2)",
        "UMDA(pop=100,elite=0.2)",
    ])
    return v

def build_cont_variants() -> List[str]:
    v = []
    v.append("RS_CONT")
    v.extend([f"HC_CONT(sigma={s})" for s in (0.2,)])
    v.extend([f"SA_CONT(sigma=0.3,T0=1.0,alpha=0.995)"])
    v.extend([
        "DE(pop=20,F=0.8,CR=0.9)",
        "DE(pop=40,F=0.8,CR=0.9)",
    ])
    v.extend([
        "PSO_STD(p=20,w=0.72,c1=1.49,c2=1.49)",
        "PSO_STD(p=40,w=0.72,c1=1.49,c2=1.49)",
        "PSO_RING(p=20,ring,w=0.72,c1=1.49,c2=1.49)",
        "PSO_RING(p=40,ring,w=0.72,c1=1.49,c2=1.49)",
    ])
    v.extend([
        "ES_ML(mu=10,lam=40,sigma=0.3)",
        "ES_ML(mu=20,lam=80,sigma=0.2)",
    ])
    return v


# ---------------------------
# Aggregation
# ---------------------------

def aggregate_results(runs: List[RunResult],
                      target_thresholds: Dict[str, Dict[str, float]],
                      a0: float = 1.0,
                      b0: float = 1.0,
                      mc_samples: int = 20000,
                      seed: int = 0) -> pd.DataFrame:
    """
    Produces one row per (instance_id, problem, domain, budget, target, algo_variant).
    'target' is numeric threshold (float). 'target_name' indicates easy/med/hard/topq.

    Output includes the classic columns you already have:
      instance_id, problem, domain, budget, target, algo_variant, algo_base,
      successes, trials, succ_rate, beta_mean, beta_p05, beta_p95,
      mean_best, median_best, min_best, max_best
    plus:
      mean_auc_raw, median_auc_raw, mean_auc_log, median_auc_log, mean_wall_ms
    """
    rows = []
    # group by (instance, algo, budget)
    by_key: Dict[Tuple[str, str, int], List[RunResult]] = {}
    for r in runs:
        key = (r.instance_id, r.algo_variant, r.budget)
        by_key.setdefault(key, []).append(r)

    for (instance_id, algo_variant, budget), lst in by_key.items():
        domain = lst[0].domain
        problem = lst[0].problem
        algo_base = lst[0].algo_base
        bests = np.array([x.best for x in lst], dtype=float)
        auc_raws = np.array([x.auc_raw for x in lst], dtype=float)
        auc_logs = np.array([x.auc_log for x in lst], dtype=float)
        wall_ms = np.array([x.wall_ms for x in lst], dtype=float)

        trials = len(lst)
        succ_rate = None  # per target below

        # targets available for this instance+domain
        tkey = f"{instance_id}"
        thr_map = target_thresholds[tkey]  # contains easy/med/hard + optional TOPq
        # For each target name, aggregate successes
        # (all runs share same thresholds for that instance, by construction)
        # We store target numeric threshold in 'target' for compatibility.
        target_names = [k for k in thr_map.keys() if k not in ("b0", "ub")]
        for tname in target_names:
            thr = float(thr_map[tname])
            succ = int(sum(x.success.get(tname, 0) for x in lst))
            succ_rate = float(succ) / float(trials) if trials > 0 else float("nan")
            beta_mean, beta_p05, beta_p95 = beta_posterior_stats(
                successes=succ, trials=trials, a0=a0, b0=b0, mc_samples=mc_samples, seed=(seed + hash((instance_id, algo_variant, budget, tname)) % 1000000)
            )
            rows.append({
                "instance_id": instance_id,
                "problem": problem,
                "domain": domain,
                "budget": int(budget),
                "target": thr,
                "target_name": tname,
                "algo_variant": algo_variant,
                "algo_base": algo_base,
                "successes": int(succ),
                "trials": int(trials),
                "succ_rate": succ_rate,
                "beta_mean": beta_mean,
                "beta_p05": beta_p05,
                "beta_p95": beta_p95,
                "mean_best": float(np.mean(bests)),
                "median_best": float(np.median(bests)),
                "min_best": float(np.min(bests)),
                "max_best": float(np.max(bests)),
                "mean_auc_raw": float(np.mean(auc_raws)),
                "median_auc_raw": float(np.median(auc_raws)),
                "mean_auc_log": float(np.mean(auc_logs)),
                "median_auc_log": float(np.median(auc_logs)),
                "mean_wall_ms": float(np.mean(wall_ms)),
            })
    df = pd.DataFrame(rows)
    # stable ordering
    df = df.sort_values(["domain", "problem", "budget", "instance_id", "algo_variant", "target_name"]).reset_index(drop=True)
    return df


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instances", type=int, default=60, help="instances per (problem, dim/n)")
    ap.add_argument("--reps", type=int, default=30, help="repetitions per (instance, algo, budget)")
    ap.add_argument("--calib", type=int, default=200, help="random calibration evals per instance to set easy/med/hard targets")
    ap.add_argument("--budgets", type=int, nargs="+", default=[500, 2000, 5000])

    ap.add_argument("--bin_bits", type=int, nargs="+", default=[200, 500])
    ap.add_argument("--cont_dims", type=int, nargs="+", default=[10, 30])

    ap.add_argument("--bin_problems", type=str, nargs="+", default=["onemax", "leadingones", "trap5", "knapsack01"])
    ap.add_argument("--cont_problems", type=str, nargs="+", default=["sphere", "rastrigin", "ackley", "rosenbrock"])

    ap.add_argument("--no_rotation", action="store_true", help="disable rotation for continuous instances")
    ap.add_argument("--add_topq", type=float, nargs="*", default=[], help="also add targets based on pooled top-q thresholds (e.g., 0.1 0.25)")
    ap.add_argument("--progress_every", type=int, default=2000, help="print progress every N runs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Build instances
    bin_insts = generate_bin_instances(args.bin_problems, args.bin_bits, args.instances, seed=args.seed + 11)
    cont_insts = generate_cont_instances(args.cont_problems, args.cont_dims, args.instances,
                                         seed=args.seed + 29, use_rotation=(not args.no_rotation))

    # Build variants
    bin_variants = build_bin_variants()
    cont_variants = build_cont_variants()

    # Pre-calibrate targets per instance (easy/med/hard + store b0/ub)
    targets_by_instance: Dict[str, Dict[str, float]] = {}

    print(f"[INFO] generating targets: bin={len(bin_insts)} cont={len(cont_insts)} calib={args.calib}")
    for inst in bin_insts:
        t = calibrate_targets_bin(inst, calib_evals=args.calib, seed=inst.seed + 123)
        targets_by_instance[inst.instance_id] = t
    for inst in cont_insts:
        t = calibrate_targets_cont(inst, calib_evals=args.calib, seed=inst.seed + 456)
        targets_by_instance[inst.instance_id] = t

    # Run all experiments (store RunResult for aggregation)
    runs: List[RunResult] = []

    # We'll also temporarily store per (instance,budget,domain) all bests for TOP-q targets.
    # key -> list of (algo_variant, best) for each rep run (pooled)
    pooled_best: Dict[Tuple[str, str, int], List[float]] = {}

    total_runs = (
        len(bin_insts) * len(args.budgets) * len(bin_variants) * args.reps +
        len(cont_insts) * len(args.budgets) * len(cont_variants) * args.reps
    )
    print(f"[INFO] total runs planned: {total_runs:,}")

    run_counter = 0
    start_all = now_ms()

    # Binary runs
    for inst in bin_insts:
        inst_targets = targets_by_instance[inst.instance_id]
        for budget in args.budgets:
            for algo_variant in bin_variants:
                for rep in range(args.reps):
                    seed = int(rng.integers(0, 2**31-1))
                    rr = run_one_bin(inst, budget, algo_variant, rep, seed, targets=inst_targets, topq_targets=None)
                    runs.append(rr)
                    pooled_best.setdefault((inst.instance_id, "bin", int(budget)), []).append(rr.best)
                    run_counter += 1
                    if args.progress_every > 0 and run_counter % args.progress_every == 0:
                        elapsed = (now_ms() - start_all) / 1000.0
                        print(f"[PROG] {run_counter:,}/{total_runs:,} runs  elapsed={elapsed:.1f}s")

    # Continuous runs
    for inst in cont_insts:
        inst_targets = targets_by_instance[inst.instance_id]
        for budget in args.budgets:
            for algo_variant in cont_variants:
                for rep in range(args.reps):
                    seed = int(rng.integers(0, 2**31-1))
                    rr = run_one_cont(inst, budget, algo_variant, rep, seed, targets=inst_targets, topq_targets=None)
                    runs.append(rr)
                    pooled_best.setdefault((inst.instance_id, "cont", int(budget)), []).append(rr.best)
                    run_counter += 1
                    if args.progress_every > 0 and run_counter % args.progress_every == 0:
                        elapsed = (now_ms() - start_all) / 1000.0
                        print(f"[PROG] {run_counter:,}/{total_runs:,} runs  elapsed={elapsed:.1f}s")

    # Add TOP-q targets if requested (computed from pooled bests)
    if args.add_topq:
        qs = sorted(list(set([float(q) for q in args.add_topq if 0 < q < 1])))
        print(f"[INFO] computing TOP-q targets: {qs}")
        for (instance_id, domain, budget), vals in pooled_best.items():
            arr = np.array(vals, dtype=float)
            if domain == "bin":
                # top-q for maximization => threshold is (1-q) quantile
                for q in qs:
                    thr = float(np.quantile(arr, 1.0 - q))
                    targets_by_instance[instance_id][f"TOP{int(q*100):02d}"] = thr
            else:
                # top-q for minimization => threshold is q quantile (lower is better)
                for q in qs:
                    thr = float(np.quantile(arr, q))
                    targets_by_instance[instance_id][f"TOP{int(q*100):02d}"] = thr

        # retroactively fill success flags for TOP-q targets in each run
        for r in runs:
            tmap = targets_by_instance[r.instance_id]
            for q in qs:
                name = f"TOP{int(q*100):02d}"
                thr = tmap[name]
                if r.domain == "bin":
                    r.success[name] = 1 if r.best >= thr else 0
                else:
                    r.success[name] = 1 if r.best <= thr else 0

    # Write per-run detail CSV
    runs_path = out_dir / "runs_detail.csv"
    print(f"[INFO] writing runs_detail.csv -> {runs_path}")
    with runs_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        # header
        # store easy/med/hard + any TOPxx present in first run (safe)
        extra_targets = sorted([k for k in runs[0].success.keys() if k not in ("easy", "med", "hard")]) if runs else []
        header = [
            "instance_id", "domain", "problem", "budget", "algo_variant", "algo_base",
            "rep", "seed", "best", "auc_raw", "auc_log", "wall_ms",
            "succ_easy", "succ_med", "succ_hard",
        ] + [f"succ_{k}" for k in extra_targets]
        w.writerow(header)
        for r in runs:
            row = [
                r.instance_id, r.domain, r.problem, r.budget, r.algo_variant, r.algo_base,
                r.rep, r.seed, r.best, r.auc_raw, r.auc_log, r.wall_ms,
                r.success.get("easy", 0), r.success.get("med", 0), r.success.get("hard", 0),
            ] + [r.success.get(k, 0) for k in extra_targets]
            w.writerow(row)

    # Write targets (for paper reproducibility)
    targets_path = out_dir / "targets_by_instance.csv"
    print(f"[INFO] writing targets_by_instance.csv -> {targets_path}")
    t_rows = []
    for iid, mp in targets_by_instance.items():
        base = {"instance_id": iid}
        base.update({k: float(v) for k, v in mp.items()})
        t_rows.append(base)
    pd.DataFrame(t_rows).to_csv(targets_path, index=False)

    # Aggregate and write instance_algo_budget_summary.csv (compatible)
    print("[INFO] aggregating to instance_algo_budget_summary.csv ...")
    df_sum = aggregate_results(
        runs=runs,
        target_thresholds=targets_by_instance,
        a0=1.0,
        b0=1.0,
        mc_samples=20000,
        seed=args.seed + 999,
    )
    out_sum = out_dir / "instance_algo_budget_summary.csv"
    df_sum.to_csv(out_sum, index=False)
    print(f"[OK] wrote {out_sum}")

    # Quick executive summary (budget-sliced)
    # (This is NOT the clustering; it's a sanity KPI for scale runs.)
    print("\n=== Quick KPI (mean beta_mean by domain/budget/target_name) ===")
    kpi = (
        df_sum
        .groupby(["domain", "budget", "target_name", "algo_variant"], as_index=False)["beta_mean"]
        .mean()
    )
    # show top-5 per slice (direction differs, but beta is success so higher is better)
    for (domain, budget, target_name), g in kpi.groupby(["domain", "budget", "target_name"]):
        g = g.sort_values("beta_mean", ascending=False).head(5)
        top = ", ".join([f"{row.algo_variant}={row.beta_mean:.3f}" for row in g.itertuples()])
        print(f"{domain.upper():4s} budget={budget:<6d} target={target_name:<6s} top5: {top}")

    elapsed_all = (now_ms() - start_all) / 1000.0
    print(f"\n[DONE] total runs={len(runs):,}  elapsed={elapsed_all:.1f}s")
    print(f"[NEXT] Feed the summary into your analyzer, e.g.:")
    print(f"  python success_profile_analysis_v2_2.py --in_csv {out_sum.as_posix()} --out_dir {out_dir.as_posix()}_analysis \\\n"
          f"    --value_col beta_mean --make_views auto --rel_norm zscore --anchor_select rel_medoid --tau 6.0 --n_boot 400")


if __name__ == "__main__":
    main()
