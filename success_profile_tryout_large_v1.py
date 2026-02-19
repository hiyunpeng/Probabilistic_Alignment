
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
success_profile_tryout_large_v1.py

Scalable success-profile benchmark generator (more algorithms, more bounds, more test sets).

Outputs (in --out_dir)
----------------------
1) runs.csv
2) instance_algo_budget_summary.csv

These match the schema your success_profile_analysis_v2_*.py scripts expect
(e.g., columns: instance_id, problem, domain, budget, target, algo_variant, algo_base,
successes, trials, succ_rate, beta_mean, beta_p05, beta_p95, mean_best, median_best, min_best, max_best).

PowerShell usage (use backticks ` for line continuation, NOT backslashes):
---------------------------------------------------------------------------
python success_profile_tryout_large_v1.py `
  --out_dir out_succ_large `
  --n_instances_bin 40 --n_instances_cont 40 `
  --n_runs 20 `
  --budgets 500,2000 `
  --include_algos default `
  --target_mode abs `
  --seed 0

Then feed to your analysis:
python success_profile_analysis_v2_2.py --in_csv out_succ_large/instance_algo_budget_summary.csv --out_dir out_succ_large_analysis --value_col beta_mean ...

Notes
-----
- Convention: ALWAYS minimize.
  * Binary problems use cost = (optimum - fitness) with optimum cost = 0.
  * Continuous problems are standard minimization with global min = 0.
- target_mode:
  * abs: fixed practical thresholds per problem/budget (fast, stable).
  * rel: target = rel_quantile of observed best values per instance+budget (hardness-controlled).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from pathlib import Path

# ----------------------------
# Utilities
# ----------------------------

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def parse_csv_list(s: str, cast=float):
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [cast(p) for p in parts]

def beta_ci_mc(successes: int, trials: int, a0: float = 1.0, b0: float = 1.0, n_mc: int = 2000,
               rng: Optional[np.random.Generator] = None) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    a = a0 + successes
    b = b0 + (trials - successes)
    samples = rng.beta(a, b, size=n_mc)
    return float(np.quantile(samples, 0.05)), float(np.quantile(samples, 0.95))

def summarize_best(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    return {
        "mean_best": float(np.mean(arr)),
        "median_best": float(np.median(arr)),
        "min_best": float(np.min(arr)),
        "max_best": float(np.max(arr)),
    }


# ----------------------------
# Problem definitions (MINIMIZATION)
# ----------------------------

@dataclass(frozen=True)
class ProblemSpec:
    domain: str                 # "bin" or "cont"
    problem: str
    dim_or_n: int
    bounds: Tuple[float, float] # for cont
    seed: int                   # instance seed

@dataclass
class ProblemInstance:
    spec: ProblemSpec
    eval_fn: Callable[[np.ndarray], float]   # expects a vector; bin vector in {0,1}
    optimum_cost: float = 0.0
    extra: Dict[str, Any] = None


def make_bin_instance(spec: ProblemSpec) -> ProblemInstance:
    rng = np.random.default_rng(spec.seed)
    n = spec.dim_or_n

    if spec.problem == "onemax":
        def f(x):
            return float(n - np.sum(x))
        return ProblemInstance(spec, f, optimum_cost=0.0, extra={})

    if spec.problem == "leadingones":
        def f(x):
            k = 0
            for i in range(n):
                if x[i] == 1:
                    k += 1
                else:
                    break
            return float(n - k)
        return ProblemInstance(spec, f, optimum_cost=0.0, extra={})

    if spec.problem == "trap5":
        k = 5
        if n % k != 0:
            raise ValueError("trap5 requires n divisible by 5")
        def f(x):
            cost = 0.0
            for b in range(0, n, k):
                u = int(np.sum(x[b:b+k]))
                fit = k if u == k else (k - 1 - u)     # in [0,k]
                cost += (k - fit)                      # convert to cost with min 0
            return float(cost)
        return ProblemInstance(spec, f, optimum_cost=0.0, extra={"k": k})

    if spec.problem == "knapsack01":
        weights = rng.integers(1, 40, size=n)
        values  = rng.integers(1, 100, size=n)
        cap = int(0.40 * int(np.sum(weights)))

        dp = np.zeros(cap + 1, dtype=int)
        for i in range(n):
            w = int(weights[i]); v = int(values[i])
            dp[w:] = np.maximum(dp[w:], dp[:-w] + v)
        opt_val = int(np.max(dp))

        def f(x):
            tot_w = int(np.dot(weights, x))
            tot_v = int(np.dot(values, x))
            if tot_w <= cap:
                return float(opt_val - tot_v)
            return float(opt_val - tot_v + 10.0 * (tot_w - cap))
        return ProblemInstance(spec, f, optimum_cost=0.0, extra={"cap": cap, "opt_val": opt_val})

    raise ValueError(f"Unknown binary problem: {spec.problem}")


def make_cont_instance(spec: ProblemSpec) -> ProblemInstance:
    d = spec.dim_or_n

    if spec.problem == "sphere":
        def f(x):
            return float(np.sum(x*x))
        return ProblemInstance(spec, f, optimum_cost=0.0, extra={})

    if spec.problem == "rastrigin":
        def f(x):
            return float(10.0*d + np.sum(x*x - 10.0*np.cos(2*np.pi*x)))
        return ProblemInstance(spec, f, optimum_cost=0.0, extra={})

    if spec.problem == "ackley":
        def f(x):
            a=20.0; b=0.2; c=2*np.pi
            s1 = np.sum(x*x)
            s2 = np.sum(np.cos(c*x))
            term1 = -a*np.exp(-b*np.sqrt(s1/d))
            term2 = -np.exp(s2/d)
            return float(term1 + term2 + a + np.e)
        return ProblemInstance(spec, f, optimum_cost=0.0, extra={})

    if spec.problem == "griewank":
        def f(x):
            sum_sq = np.sum(x*x)/4000.0
            prod = 1.0
            for i in range(d):
                prod *= np.cos(x[i]/np.sqrt(i+1))
            return float(sum_sq - prod + 1.0)
        return ProblemInstance(spec, f, optimum_cost=0.0, extra={})

    if spec.problem == "rosenbrock":
        def f(x):
            return float(np.sum(100.0*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2))
        return ProblemInstance(spec, f, optimum_cost=0.0, extra={})

    if spec.problem == "levy":
        def f(x):
            w = 1.0 + (x - 1.0)/4.0
            term1 = np.sin(np.pi*w[0])**2
            term3 = (w[-1]-1.0)**2*(1.0 + np.sin(2*np.pi*w[-1])**2)
            wi = w[:-1]
            term2 = np.sum((wi-1.0)**2 * (1.0 + 10.0*np.sin(np.pi*wi+1.0)**2))
            return float(term1 + term2 + term3)
        return ProblemInstance(spec, f, optimum_cost=0.0, extra={})

    raise ValueError(f"Unknown continuous problem: {spec.problem}")


# ----------------------------
# Algorithms
# ----------------------------

def rs_bin(eval_fn, n, budget, rng):
    best = float("inf"); best_i = 0
    for i in range(1, budget+1):
        x = rng.integers(0, 2, size=n, dtype=int)
        v = eval_fn(x)
        if v < best:
            best = v; best_i = i
            if best <= 0:
                break
    return best, best_i

def hc_bin(eval_fn, n, budget, rng, flips=1):
    x = rng.integers(0, 2, size=n, dtype=int)
    best = eval_fn(x); best_i = 1
    evals = 1
    while evals < budget and best > 0:
        y = x.copy()
        idx = rng.choice(n, size=flips, replace=False)
        y[idx] = 1 - y[idx]
        evals += 1
        v = eval_fn(y)
        if v <= best:
            x = y; best = v; best_i = evals
    return best, best_i

def sa_bin(eval_fn, n, budget, rng, flips=1, T0=2.0, alpha=0.995):
    x = rng.integers(0, 2, size=n, dtype=int)
    fx = eval_fn(x); best = fx; best_i = 1
    T = T0
    evals = 1
    while evals < budget and best > 0:
        y = x.copy()
        idx = rng.choice(n, size=flips, replace=False)
        y[idx] = 1 - y[idx]
        evals += 1
        fy = eval_fn(y)
        dE = fy - fx
        if dE <= 0 or rng.random() < np.exp(-dE / max(1e-12, T)):
            x, fx = y, fy
            if fx < best:
                best = fx; best_i = evals
        T *= alpha
    return best, best_i

def tabu_bin(eval_fn, n, budget, rng, tenure=7):
    x = rng.integers(0, 2, size=n, dtype=int)
    fx = eval_fn(x); best = fx; best_i = 1
    evals = 1
    tabu = np.zeros(n, dtype=int)
    while evals < budget and best > 0:
        cand_k = min(10, n)
        cands = rng.choice(n, size=cand_k, replace=False)
        best_move = None
        best_move_val = float("inf")
        for j in cands:
            y = x.copy()
            y[j] = 1 - y[j]
            v = eval_fn(y)
            evals += 1
            if (tabu[j] > 0) and (v >= best):  # aspiration
                continue
            if v < best_move_val:
                best_move_val = v
                best_move = j
            if evals >= budget:
                break
        tabu = np.maximum(tabu - 1, 0)
        if best_move is not None:
            x[best_move] = 1 - x[best_move]
            fx = best_move_val
            tabu[best_move] = tenure
            if fx < best:
                best = fx; best_i = evals
        if evals >= budget:
            break
    return best, best_i

def ga_bin(eval_fn, n, budget, rng, pop=50, pc=0.9, pm=0.02, tour=2):
    def rand_ind():
        return rng.integers(0, 2, size=n, dtype=int)

    def tournament(P, F):
        idx = rng.integers(0, len(P), size=tour)
        b = idx[0]
        for j in idx[1:]:
            if F[j] < F[b]:
                b = j
        return P[b].copy()

    P = [rand_ind() for _ in range(pop)]
    evals = 0
    F = []
    best = float("inf"); best_i = 0

    for i in range(pop):
        evals += 1
        v = eval_fn(P[i])
        F.append(v)
        if v < best:
            best = v; best_i = evals
    if evals >= budget or best <= 0:
        return best, best_i

    while evals < budget and best > 0:
        newP = []
        while len(newP) < pop and evals < budget:
            p1 = tournament(P, F)
            p2 = tournament(P, F)
            if rng.random() < pc:
                cp = rng.integers(1, n)
                c1 = np.concatenate([p1[:cp], p2[cp:]])
                c2 = np.concatenate([p2[:cp], p1[cp:]])
            else:
                c1 = p1.copy(); c2 = p2.copy()

            for c in (c1, c2):
                m = rng.random(size=n) < pm
                c[m] = 1 - c[m]
                newP.append(c)
                if len(newP) >= pop:
                    break

        P = newP[:pop]
        F = []
        for i in range(len(P)):
            if evals >= budget:
                break
            evals += 1
            v = eval_fn(P[i])
            F.append(v)
            if v < best:
                best = v; best_i = evals
                if best <= 0:
                    break
    return best, best_i

def umda_bin(eval_fn, n, budget, rng, pop=50, elite_frac=0.2):
    p = np.full(n, 0.5, dtype=float)
    evals = 0
    best = float("inf"); best_i = 0
    elite = max(1, int(pop * elite_frac))

    while evals < budget and best > 0:
        P = (rng.random((pop, n)) < p[None, :]).astype(int)
        F = np.zeros(pop, dtype=float)
        for i in range(pop):
            if evals >= budget:
                break
            evals += 1
            F[i] = eval_fn(P[i])
            if F[i] < best:
                best = float(F[i]); best_i = evals
        idx = np.argsort(F)[:elite]
        p = np.clip(np.mean(P[idx], axis=0), 0.01, 0.99)
    return best, best_i


def rs_cont(eval_fn, d, budget, rng, bounds):
    lo, hi = bounds
    best = float("inf"); best_i = 0
    for i in range(1, budget+1):
        x = rng.uniform(lo, hi, size=d)
        v = eval_fn(x)
        if v < best:
            best = v; best_i = i
            if best <= 0:
                break
    return best, best_i

def hc_cont(eval_fn, d, budget, rng, bounds, sigma=0.2):
    lo, hi = bounds
    x = rng.uniform(lo, hi, size=d)
    fx = eval_fn(x); best = fx; best_i = 1
    evals = 1
    while evals < budget and best > 0:
        y = clamp(x + rng.normal(0.0, sigma, size=d), lo, hi)
        evals += 1
        fy = eval_fn(y)
        if fy <= fx:
            x, fx = y, fy
            if fx < best:
                best = fx; best_i = evals
    return best, best_i

def sa_cont(eval_fn, d, budget, rng, bounds, sigma=0.3, T0=1.0, alpha=0.995):
    lo, hi = bounds
    x = rng.uniform(lo, hi, size=d)
    fx = eval_fn(x); best = fx; best_i = 1
    T = T0
    evals = 1
    while evals < budget and best > 0:
        y = clamp(x + rng.normal(0.0, sigma, size=d), lo, hi)
        evals += 1
        fy = eval_fn(y)
        dE = fy - fx
        if dE <= 0 or rng.random() < np.exp(-dE / max(1e-12, T)):
            x, fx = y, fy
            if fx < best:
                best = fx; best_i = evals
        T *= alpha
    return best, best_i

def de_cont(eval_fn, d, budget, rng, bounds, pop=20, F=0.8, CR=0.9):
    lo, hi = bounds
    P = rng.uniform(lo, hi, size=(pop, d))
    evals = 0
    Fvals = np.zeros(pop, dtype=float)
    best = float("inf"); best_i = 0

    for i in range(pop):
        evals += 1
        Fvals[i] = eval_fn(P[i])
        if Fvals[i] < best:
            best = float(Fvals[i]); best_i = evals

    if evals >= budget or best <= 0:
        return best, best_i

    while evals < budget and best > 0:
        for i in range(pop):
            if evals >= budget:
                break
            idxs = [j for j in range(pop) if j != i]
            a, b, c = rng.choice(idxs, size=3, replace=False)
            v = clamp(P[a] + F * (P[b] - P[c]), lo, hi)
            jrand = rng.integers(0, d)
            mask = (rng.random(d) < CR)
            mask[jrand] = True
            u = np.where(mask, v, P[i])
            evals += 1
            fu = eval_fn(u)
            if fu <= Fvals[i]:
                P[i] = u
                Fvals[i] = fu
                if fu < best:
                    best = float(fu); best_i = evals
                    if best <= 0:
                        break
    return best, best_i

def pso_cont(eval_fn, d, budget, rng, bounds, p=20, w=0.72, c1=1.49, c2=1.49, ring=False):
    lo, hi = bounds
    X = rng.uniform(lo, hi, size=(p, d))
    V = rng.normal(0.0, (hi-lo)*0.05, size=(p, d))
    pbest = X.copy()
    pbest_val = np.array([eval_fn(x) for x in X], dtype=float)
    evals = p
    gbest_idx = int(np.argmin(pbest_val))
    gbest_val = float(pbest_val[gbest_idx])
    best_i = evals

    if evals >= budget or gbest_val <= 0:
        return gbest_val, best_i

    if ring:
        neigh = [[(i-1) % p, i, (i+1) % p] for i in range(p)]

    while evals < budget and gbest_val > 0:
        for i in range(p):
            if evals >= budget:
                break
            if ring:
                nb = neigh[i]
                lbest_idx = nb[int(np.argmin(pbest_val[nb]))]
                g = pbest[lbest_idx]
            else:
                g = pbest[int(np.argmin(pbest_val))]
            r1 = rng.random(d)
            r2 = rng.random(d)
            V[i] = w*V[i] + c1*r1*(pbest[i]-X[i]) + c2*r2*(g-X[i])
            X[i] = clamp(X[i] + V[i], lo, hi)
            evals += 1
            v = eval_fn(X[i])
            if v <= pbest_val[i]:
                pbest[i] = X[i].copy()
                pbest_val[i] = v
                if v < gbest_val:
                    gbest_val = float(v)
                    best_i = evals
                    if gbest_val <= 0:
                        break
    return gbest_val, best_i

def es_cont(eval_fn, d, budget, rng, bounds, mu=10, lam=40, sigma=0.3):
    lo, hi = bounds
    parents = rng.uniform(lo, hi, size=(mu, d))
    fpar = np.array([eval_fn(x) for x in parents], dtype=float)
    evals = mu
    best = float(np.min(fpar))
    best_i = evals

    if evals >= budget or best <= 0:
        return best, best_i

    while evals < budget and best > 0:
        off = []
        foff = []
        success_ct = 0
        for _ in range(lam):
            if evals >= budget:
                break
            pidx = int(rng.integers(0, mu))
            child = clamp(parents[pidx] + rng.normal(0.0, sigma, size=d), lo, hi)
            evals += 1
            fv = eval_fn(child)
            off.append(child)
            foff.append(fv)
            if fv < fpar[pidx]:
                success_ct += 1
            if fv < best:
                best = float(fv); best_i = evals
                if best <= 0:
                    break
        if not off:
            break
        off = np.asarray(off); foff = np.asarray(foff)
        idx = np.argsort(foff)[:mu]
        parents = off[idx]
        fpar = foff[idx]
        rate = success_ct / max(1, len(off))
        sigma *= 1.05 if rate > 0.2 else 0.95
        sigma = float(np.clip(sigma, 1e-3, (hi-lo)))
    return best, best_i


# ----------------------------
# Portfolio selection
# ----------------------------

@dataclass(frozen=True)
class AlgoSpec:
    algo_base: str
    algo_variant: str
    domain: str
    params: Dict[str, Any]

def build_portfolio(mode: str) -> List[AlgoSpec]:
    bin_algos = [
        AlgoSpec("RS_BIN", "RS_BIN", "bin", {}),
        AlgoSpec("HC", "HC(flips=1)", "bin", {"flips": 1}),
        AlgoSpec("HC", "HC(flips=2)", "bin", {"flips": 2}),
        AlgoSpec("SA", "SA(T0=1.0,alpha=0.995,flips=1)", "bin", {"T0": 1.0, "alpha": 0.995, "flips": 1}),
        AlgoSpec("SA", "SA(T0=2.0,alpha=0.99,flips=1)", "bin", {"T0": 2.0, "alpha": 0.99, "flips": 1}),
        AlgoSpec("SA", "SA(T0=2.0,alpha=0.995,flips=2)", "bin", {"T0": 2.0, "alpha": 0.995, "flips": 2}),
        AlgoSpec("GA", "GA(pop=50,pc=0.9,pm=0.02)", "bin", {"pop": 50, "pc": 0.9, "pm": 0.02}),
        AlgoSpec("GA", "GA(pop=50,pc=0.9,pm=0.05)", "bin", {"pop": 50, "pc": 0.9, "pm": 0.05}),
        AlgoSpec("GA", "GA(pop=100,pc=0.9,pm=0.02)", "bin", {"pop": 100, "pc": 0.9, "pm": 0.02}),
    ]
    cont_algos = [
        AlgoSpec("RS_CONT", "RS_CONT", "cont", {}),
        AlgoSpec("HC_CONT", "HC_CONT(sigma=0.2)", "cont", {"sigma": 0.2}),
        AlgoSpec("SA_CONT", "SA_CONT(sigma=0.3,T0=1.0,alpha=0.995)", "cont", {"sigma": 0.3, "T0": 1.0, "alpha": 0.995}),
        AlgoSpec("DE", "DE(pop=20,F=0.8,CR=0.9)", "cont", {"pop": 20, "F": 0.8, "CR": 0.9}),
        AlgoSpec("DE", "DE(pop=40,F=0.8,CR=0.9)", "cont", {"pop": 40, "F": 0.8, "CR": 0.9}),
        AlgoSpec("PSO_STD", "PSO_STD(p=20,w=0.72,c1=1.49,c2=1.49)", "cont", {"p": 20, "w": 0.72, "c1": 1.49, "c2": 1.49, "ring": False}),
        AlgoSpec("PSO_STD", "PSO_STD(p=40,w=0.72,c1=1.49,c2=1.49)", "cont", {"p": 40, "w": 0.72, "c1": 1.49, "c2": 1.49, "ring": False}),
        AlgoSpec("PSO_RING", "PSO_RING(p=20,ring,w=0.72,c1=1.49,c2=1.49)", "cont", {"p": 20, "w": 0.72, "c1": 1.49, "c2": 1.49, "ring": True}),
        AlgoSpec("PSO_RING", "PSO_RING(p=40,ring,w=0.72,c1=1.49,c2=1.49)", "cont", {"p": 40, "w": 0.72, "c1": 1.49, "c2": 1.49, "ring": True}),
    ]

    if mode in ("default", "full"):
        bin_algos += [
            AlgoSpec("TABU", "TABU(tenure=7)", "bin", {"tenure": 7}),
            AlgoSpec("UMDA", "UMDA(pop=50,elite=0.2)", "bin", {"pop": 50, "elite_frac": 0.2}),
            AlgoSpec("UMDA", "UMDA(pop=100,elite=0.2)", "bin", {"pop": 100, "elite_frac": 0.2}),
        ]
        cont_algos += [
            AlgoSpec("ES_ML", "ES_ML(mu=10,lam=40,sigma=0.3)", "cont", {"mu": 10, "lam": 40, "sigma": 0.3}),
        ]

    if mode == "full":
        bin_algos += [
            AlgoSpec("TABU", "TABU(tenure=12)", "bin", {"tenure": 12}),
        ]
        cont_algos += [
            AlgoSpec("ES_ML", "ES_ML(mu=20,lam=80,sigma=0.2)", "cont", {"mu": 20, "lam": 80, "sigma": 0.2}),
        ]

    if mode == "fast":
        bin_algos = [a for a in bin_algos if a.algo_base in ("RS_BIN", "HC", "SA", "GA")]
        cont_algos = [a for a in cont_algos if a.algo_base in ("RS_CONT", "DE", "PSO_STD", "PSO_RING")]

    return bin_algos + cont_algos


def run_algo_on_instance(inst: ProblemInstance, algo: AlgoSpec, budget: int, seed: int) -> Tuple[float, int]:
    rng = np.random.default_rng(seed)
    if inst.spec.domain == "bin":
        n = inst.spec.dim_or_n
        if algo.algo_base == "RS_BIN":
            return rs_bin(inst.eval_fn, n, budget, rng)
        if algo.algo_base == "HC":
            return hc_bin(inst.eval_fn, n, budget, rng, flips=int(algo.params.get("flips", 1)))
        if algo.algo_base == "SA":
            return sa_bin(inst.eval_fn, n, budget, rng,
                          flips=int(algo.params.get("flips", 1)),
                          T0=float(algo.params.get("T0", 2.0)),
                          alpha=float(algo.params.get("alpha", 0.995)))
        if algo.algo_base == "TABU":
            return tabu_bin(inst.eval_fn, n, budget, rng, tenure=int(algo.params.get("tenure", 7)))
        if algo.algo_base == "GA":
            return ga_bin(inst.eval_fn, n, budget, rng,
                          pop=int(algo.params.get("pop", 50)),
                          pc=float(algo.params.get("pc", 0.9)),
                          pm=float(algo.params.get("pm", 0.02)))
        if algo.algo_base == "UMDA":
            return umda_bin(inst.eval_fn, n, budget, rng,
                            pop=int(algo.params.get("pop", 50)),
                            elite_frac=float(algo.params.get("elite_frac", 0.2)))
        raise ValueError(f"Unknown bin algo_base: {algo.algo_base}")

    d = inst.spec.dim_or_n
    bounds = inst.spec.bounds
    if algo.algo_base == "RS_CONT":
        return rs_cont(inst.eval_fn, d, budget, rng, bounds)
    if algo.algo_base == "HC_CONT":
        return hc_cont(inst.eval_fn, d, budget, rng, bounds, sigma=float(algo.params.get("sigma", 0.2)))
    if algo.algo_base == "SA_CONT":
        return sa_cont(inst.eval_fn, d, budget, rng, bounds,
                       sigma=float(algo.params.get("sigma", 0.3)),
                       T0=float(algo.params.get("T0", 1.0)),
                       alpha=float(algo.params.get("alpha", 0.995)))
    if algo.algo_base == "DE":
        return de_cont(inst.eval_fn, d, budget, rng, bounds,
                       pop=int(algo.params.get("pop", 20)),
                       F=float(algo.params.get("F", 0.8)),
                       CR=float(algo.params.get("CR", 0.9)))
    if algo.algo_base in ("PSO_STD", "PSO_RING"):
        return pso_cont(inst.eval_fn, d, budget, rng, bounds,
                        p=int(algo.params.get("p", 20)),
                        w=float(algo.params.get("w", 0.72)),
                        c1=float(algo.params.get("c1", 1.49)),
                        c2=float(algo.params.get("c2", 1.49)),
                        ring=bool(algo.params.get("ring", False)))
    if algo.algo_base == "ES_ML":
        return es_cont(inst.eval_fn, d, budget, rng, bounds,
                       mu=int(algo.params.get("mu", 10)),
                       lam=int(algo.params.get("lam", 40)),
                       sigma=float(algo.params.get("sigma", 0.3)))
    raise ValueError(f"Unknown cont algo_base: {algo.algo_base}")


# ----------------------------
# Instance generation
# ----------------------------

def generate_instances_bin(n_instances: int, seed: int, sizes: List[int], problems: List[str]) -> List[ProblemInstance]:
    rng = np.random.default_rng(seed)
    insts = []
    for _ in range(n_instances):
        n = int(rng.choice(sizes))
        prob = str(rng.choice(problems))
        if prob == "trap5":
            n = max(10, (n // 5) * 5)
        s = int(rng.integers(0, 10**9))
        spec = ProblemSpec(domain="bin", problem=prob, dim_or_n=n, bounds=(0.0, 1.0), seed=s)
        insts.append(make_bin_instance(spec))
    return insts

def generate_instances_cont(n_instances: int, seed: int, dims: List[int], bounds_list: List[Tuple[float,float]], problems: List[str]) -> List[ProblemInstance]:
    rng = np.random.default_rng(seed)
    insts = []
    for _ in range(n_instances):
        d = int(rng.choice(dims))
        prob = str(rng.choice(problems))
        lo, hi = bounds_list[int(rng.integers(0, len(bounds_list)))]
        s = int(rng.integers(0, 10**9))
        spec = ProblemSpec(domain="cont", problem=prob, dim_or_n=d, bounds=(float(lo), float(hi)), seed=s)
        insts.append(make_cont_instance(spec))
    return insts


# ----------------------------
# Targets
# ----------------------------

def abs_target_for_instance(inst: ProblemInstance, budget: int) -> float:
    if inst.spec.domain == "bin":
        return 0.0

    # budget-tolerance schedule
    if budget <= 500:
        base = 1e-2
    elif budget <= 2000:
        base = 1e-4
    else:
        base = 1e-6

    prob = inst.spec.problem
    if prob == "sphere":
        return base
    if prob in ("rastrigin", "ackley", "griewank"):
        return max(base * 10.0, 1e-3)
    if prob in ("rosenbrock", "levy"):
        return max(base * 100.0, 1e-2)
    return max(base * 10.0, 1e-3)

def compute_rel_targets(runs_df: pd.DataFrame, rel_quantile: float = 0.10) -> Dict[Tuple[str,int], float]:
    targets: Dict[Tuple[str,int], float] = {}
    grp = runs_df.groupby(["instance_id", "budget"], sort=False)
    for (iid, budget), g in grp:
        targets[(iid, int(budget))] = float(np.quantile(g["best_value"].values, rel_quantile))
    return targets


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--n_instances_bin", type=int, default=40)
    ap.add_argument("--n_instances_cont", type=int, default=40)
    ap.add_argument("--n_runs", type=int, default=20)

    ap.add_argument("--budgets", type=str, default="500,2000")
    ap.add_argument("--include_algos", choices=["fast", "default", "full"], default="default")

    ap.add_argument("--bin_sizes", type=str, default="50,100,150")
    ap.add_argument("--bin_problems", type=str, default="onemax,leadingones,trap5,knapsack01")

    ap.add_argument("--cont_dims", type=str, default="5,10,20")
    ap.add_argument("--cont_bounds", type=str, default="-5,5;-10,10;-30,30")
    ap.add_argument("--cont_problems", type=str, default="sphere,rastrigin,ackley,griewank,rosenbrock,levy")

    ap.add_argument("--target_mode", choices=["abs", "rel"], default="abs")
    ap.add_argument("--rel_quantile", type=float, default=0.10)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta_mc", type=int, default=2000)

    args = ap.parse_args()
    ensure_dir(args.out_dir)

    budgets = [int(b) for b in parse_csv_list(args.budgets, int)]
    bin_sizes = [int(x) for x in parse_csv_list(args.bin_sizes, int)]
    bin_probs = [s.strip() for s in args.bin_problems.split(",") if s.strip()]

    cont_dims = [int(x) for x in parse_csv_list(args.cont_dims, int)]
    bounds_list: List[Tuple[float,float]] = []
    for block in args.cont_bounds.split(";"):
        a, b = block.split(",")
        bounds_list.append((float(a), float(b)))
    cont_probs = [s.strip() for s in args.cont_problems.split(",") if s.strip()]

    # Generate instances
    insts_bin = generate_instances_bin(args.n_instances_bin, args.seed + 11, bin_sizes, bin_probs)
    insts_cont = generate_instances_cont(args.n_instances_cont, args.seed + 23, cont_dims, bounds_list, cont_probs)
    insts = insts_bin + insts_cont

    # Portfolio
    portfolio = build_portfolio(args.include_algos)

    # Run
    run_rows = []
    run_seed_base = int(args.seed * 10_000 + 7)

    for budget in budgets:
        for inst_idx, inst in enumerate(insts):
            if inst.spec.domain == "bin":
                iid = f"BIN::{inst.spec.problem}::n={inst.spec.dim_or_n}::seed={inst.spec.seed}"
                bounds_str = ""
            else:
                lo, hi = inst.spec.bounds
                iid = f"CONT::{inst.spec.problem}::d={inst.spec.dim_or_n}::b=({lo},{hi})::seed={inst.spec.seed}"
                bounds_str = f"{lo},{hi}"

            abs_target = abs_target_for_instance(inst, budget)

            for algo in portfolio:
                if algo.domain != inst.spec.domain:
                    continue
                for r in range(args.n_runs):
                    seed = run_seed_base + (budget * 100_000) + (inst_idx * 1000) + (hash(algo.algo_variant) % 1000) + r
                    best, best_i = run_algo_on_instance(inst, algo, budget=budget, seed=seed)
                    run_rows.append({
                        "instance_id": iid,
                        "problem": inst.spec.problem,
                        "domain": inst.spec.domain,
                        "budget": int(budget),
                        "bounds": bounds_str,
                        "dim_or_n": int(inst.spec.dim_or_n),
                        "target_abs": float(abs_target),
                        "algo_variant": algo.algo_variant,
                        "algo_base": algo.algo_base,
                        "run_id": int(r),
                        "seed": int(seed),
                        "best_value": float(best),
                        "best_eval": int(best_i),
                    })

    runs_df = pd.DataFrame(run_rows)
    runs_path = Path(args.out_dir) / "runs.csv"
    runs_df.to_csv(runs_path, index=False)
    print(f"[OK] wrote {runs_path}")

    # Targets & success
    if args.target_mode == "abs":
        runs_df["target"] = runs_df["target_abs"]
    else:
        rel_targets = compute_rel_targets(runs_df, rel_quantile=float(args.rel_quantile))
        runs_df["target"] = runs_df.apply(lambda row: rel_targets[(row["instance_id"], int(row["budget"]))], axis=1)

    runs_df["success"] = (runs_df["best_value"] <= runs_df["target"]).astype(int)
    runs_df.drop(columns=["target_abs"], inplace=True)
    runs_df.to_csv(runs_path, index=False)
    print(f"[OK] updated {runs_path} (target/success)")

    # Aggregate
    agg_rows = []
    mc_rng = np.random.default_rng(args.seed + 999)

    group_cols = ["instance_id", "problem", "domain", "budget", "target", "algo_variant", "algo_base"]
    for keys, g in runs_df.groupby(group_cols, sort=False):
        successes = int(g["success"].sum())
        trials = int(len(g))
        succ_rate = float(successes / max(1, trials))
        beta_mean = float((successes + 1.0) / (trials + 2.0))
        p05, p95 = beta_ci_mc(successes, trials, n_mc=int(args.beta_mc), rng=mc_rng)
        stats = summarize_best(list(g["best_value"].values))

        agg_rows.append({
            "instance_id": keys[0],
            "problem": keys[1],
            "domain": keys[2],
            "budget": int(keys[3]),
            "target": float(keys[4]),
            "algo_variant": keys[5],
            "algo_base": keys[6],
            "successes": successes,
            "trials": trials,
            "succ_rate": succ_rate,
            "beta_mean": beta_mean,
            "beta_p05": float(p05),
            "beta_p95": float(p95),
            **stats
        })

    summ_df = pd.DataFrame(agg_rows)
    out_path = Path(args.out_dir) / "instance_algo_budget_summary.csv"
    summ_df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path}")

    print("\n=== Quick aggregate (mean beta_mean by algo_variant) ===")
    q = summ_df.groupby("algo_variant")["beta_mean"].mean().sort_values(ascending=False)
    for name, v in q.items():
        print(f"{name:35s} mean_beta_success={v:.3f}")

    print("\n[DONE] large-scale tryout complete.")


if __name__ == "__main__":
    main()
