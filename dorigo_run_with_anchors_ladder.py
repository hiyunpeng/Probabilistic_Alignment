#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
dorigo_run_with_anchors_ladder.py

Purpose
-------
Run the full HOLDOUT experiment with:
  - 6 metaheuristics: GWO, MFO, WOA, FA, BA, ALO
  - 2 anchor families: PSO_GBEST, ES_1P1
  - an extended budget ladder (default: 300,500,800,1000,2000,5000,10000)
  - repeated runs (default: R=10)
and produce:
  - runs_detail.csv
  - instance_algo_budget_summary.csv  (success stats + best-value stats)

Dependencies
------------
pip install numpy pandas scipy tqdm matplotlib

Windows cmd example
-------------------
python dorigo_run_with_anchors_ladder.py ^
  --out_dir .\out_dorigo_new ^
  --algos GWO,MFO,WOA,FA,BA,ALO,PSO_GBEST,ES_1P1 ^
  --budgets 300,500,800,1000,2000,5000,10000 ^
  --dim 10 --instances_per_problem 20 --dev_frac 0.4 ^
  --R 10 --seed 0

Notes / Practical guidance
--------------------------
- This is CPU-only pure Python. The extended ladder can be compute-heavy.
  Start with a smoke test:
    --smoke_test
  Then scale up.
- Targets affect success-only metrics, not objective best-value itself.
  We compute best-value once per run, then compute success for each target offline.
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import beta as sp_beta
except Exception as e:
    raise SystemExit(
        "This runner requires scipy for Beta posterior quantiles.\n"
        "Install with: pip install scipy\n"
        f"Error: {e}"
    )

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x=None, total=None, desc=None, **kw):
        return x if x is not None else range(total or 0)


# =========================================================
# Utilities
# =========================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def rng_from(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))

def stable_hash_mod(s: str, mod: int) -> int:
    h = zlib.adler32(s.encode("utf-8")) & 0xffffffff
    return int(h % mod)

def stable_run_seed(algo: str, params: Dict[str, Any], problem: str, instance_id: int, budget: int, r: int, base_seed: int) -> int:
    blob = json.dumps(params, sort_keys=True)
    key = f"{algo}|{blob}|{problem}|{instance_id}|{budget}|{r}|{base_seed}"
    return 1_000_000 * stable_hash_mod(key, 1_000_000_000)

def clip_bounds(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)

def eval_pop(f: Callable[[np.ndarray], float], X: np.ndarray) -> np.ndarray:
    return np.array([f(x) for x in X], dtype=float)

def beta_posterior(succ: int, trials: int) -> Tuple[float, float, float]:
    a = 1 + succ
    b = 1 + trials - succ
    mean = a / (a + b)
    p05 = float(sp_beta.ppf(0.05, a, b))
    p95 = float(sp_beta.ppf(0.95, a, b))
    return float(mean), p05, p95


# =========================================================
# Benchmark suite (continuous) + shifted instances
# =========================================================

@dataclass(frozen=True)
class ProblemSpec:
    name: str
    bounds: Tuple[float, float]
    fn: Callable[[np.ndarray], float]
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

def make_instances(suite: List[ProblemSpec], instances_per_problem: int):
    insts = []
    for spec in suite:
        for inst_id in range(instances_per_problem):
            shift_seed = 10_000 * stable_hash_mod(spec.name, 1000) + inst_id
            insts.append((spec.name, inst_id, spec, shift_seed))
    return insts

def split_instances(insts, dev_frac: float, seed: int):
    if dev_frac <= 0.0:
        return [], insts
    rg = rng_from(seed)
    idx = np.arange(len(insts))
    rg.shuffle(idx)
    k = int(round(len(insts) * dev_frac))
    dev = [insts[i] for i in idx[:k]]
    hold = [insts[i] for i in idx[k:]]
    return dev, hold


# =========================================================
# Algorithms
# =========================================================

def run_gwo(f, dim, bounds, budget, seed, pop=30, a0: float = 2.0):
    rg = rng_from(seed)
    lo, hi = bounds
    X = rg.uniform(lo, hi, size=(pop, dim))
    fit = eval_pop(f, X)
    evals = pop

    def top3(ff):
        idx = np.argsort(ff)
        return int(idx[0]), int(idx[1]), int(idx[2])

    ia, ib, ic = top3(fit)
    Xa, Xb, Xc = X[ia].copy(), X[ib].copy(), X[ic].copy()

    t = 0
    T = max(1, budget // max(pop, 1))
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
        fit = eval_pop(f, X)
        evals += pop

        ia, ib, ic = top3(fit)
        Xa, Xb, Xc = X[ia].copy(), X[ib].copy(), X[ic].copy()
        t += 1

    return float(np.min(fit)), evals

def run_woa(f, dim, bounds, budget, seed, pop=30, a0: float = 2.0, spiral_b: float = 1.0):
    rg = rng_from(seed)
    lo, hi = bounds
    X = rg.uniform(lo, hi, size=(pop, dim))
    fit = eval_pop(f, X)
    evals = pop
    ib = int(np.argmin(fit))
    Xbest = X[ib].copy()
    fbest = float(fit[ib])

    t = 0
    T = max(1, budget // max(pop, 1))
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
        fit = eval_pop(f, X)
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
    fit = eval_pop(f, M)
    evals = pop

    flames = M.copy()
    flames_fit = fit.copy()

    t = 0
    max_iter = max(1, budget // max(pop, 1))
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
        fit = eval_pop(f, M)
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
    fit = eval_pop(f, X)
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
        fit = eval_pop(f, X)
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
    fit = eval_pop(f, X)
    evals = pop

    ib = int(np.argmin(fit))
    best = X[ib].copy()
    fbest = float(fit[ib])

    fmin, fmax = 0.0, float(max(1e-9, fmax))
    A = np.full(pop, float(A0))
    r = np.full(pop, float(r0))
    local_step = float(local_step)

    while evals < budget:
        for i in range(pop):
            if evals >= budget:
                break
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

    return float(min(fbest, float(np.min(fit)))), evals

def run_alo(f, dim, bounds, budget, seed, pop=30, intensity_max: float = 1e4):
    rg = rng_from(seed)
    lo, hi = bounds
    AL = rg.uniform(lo, hi, size=(pop, dim))
    ants = rg.uniform(lo, hi, size=(pop, dim))
    fit_AL = eval_pop(f, AL)
    evals = pop
    fit_ants = eval_pop(f, ants)
    evals += pop

    elite_idx = int(np.argmin(fit_AL))
    elite = AL[elite_idx].copy()
    elite_fit = float(fit_AL[elite_idx])

    max_iter = max(1, budget // max(pop, 1))
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
        fit_ants = eval_pop(f, ants)
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
    fit = eval_pop(f, X)
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
        fit = eval_pop(f, X)
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
    f0 = eval_pop(f, X0)
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

DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "GWO": {"pop": 30, "a0": 2.0},
    "WOA": {"pop": 30, "a0": 2.0, "spiral_b": 1.0},
    "MFO": {"pop": 30, "b": 1.0},
    "FA":  {"pop": 30, "beta0": 1.0, "gamma_scale": 1.0, "alpha_scale": 0.25, "k_neighbors": 3},
    "BA":  {"pop": 30, "fmax": 2.0, "A0": 0.9, "r0": 0.5, "local_step": 0.001},
    "ALO": {"pop": 30, "intensity_max": 1e4},
    "PSO_GBEST": {"pop": 30, "w": 0.72, "c1": 1.49, "c2": 1.49},
    "ES_1P1": {"pop": 30, "sigma0": None},
}

def build_instance_summary(runs_df: pd.DataFrame, target_tols: Dict[str, float]) -> pd.DataFrame:
    out_rows = []
    for (problem, instance_id, algo, budget), g in runs_df.groupby(
        ["problem", "instance_id", "algo_variant", "budget"], as_index=False
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
                "domain": "cont",
                "problem": problem,
                "instance_id": int(instance_id),
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

def parse_target_tols(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=")
        out[k.strip()] = float(v.strip())
    return out

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, type=str)

    ap.add_argument("--algos", type=str, default="GWO,MFO,WOA,FA,BA,ALO,PSO_GBEST,ES_1P1")
    ap.add_argument("--budgets", type=str, default="300,500,800,1000,2000,5000,10000")
    ap.add_argument("--targets", type=str, default="easy=1e-1,med=1e-2,hard=1e-3")

    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--instances_per_problem", type=int, default=20)
    ap.add_argument("--dev_frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--R", type=int, default=10)

    ap.add_argument("--smoke_test", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    fig_dir = out_dir / "figs"
    ensure_dir(fig_dir)

    if args.smoke_test:
        args.instances_per_problem = 3
        args.R = 2
        args.dev_frac = 0.4
        args.budgets = "300,1000"

    budgets = sorted({int(x) for x in args.budgets.split(",") if x.strip()})
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    for a in algos:
        if a not in ALGO_RUNNERS:
            raise SystemExit(f"Unknown algo: {a}. Available: {list(ALGO_RUNNERS.keys())}")

    target_tols = parse_target_tols(args.targets)

    suite = make_suite()
    insts = make_instances(suite, args.instances_per_problem)
    _, hold = split_instances(insts, args.dev_frac, args.seed)

    params_map = {a: DEFAULT_PARAMS.get(a, {}) for a in algos}

    total = len(hold) * len(algos) * len(budgets) * args.R
    pb = tqdm(total=total, desc="RUN HOLDOUT", leave=True)

    rows = []
    for (pname, inst_id, spec, shift_seed) in hold:
        f = make_shifted_objective(spec, dim=args.dim, shift_seed=shift_seed)
        lo, hi = spec.bounds
        for algo in algos:
            params = params_map[algo]
            runner = ALGO_RUNNERS[algo]
            for b in budgets:
                for r in range(args.R):
                    seed = stable_run_seed(algo, params, pname, inst_id, b, r, base_seed=args.seed + 999)
                    best, evals = runner(f, args.dim, (lo, hi), b, seed, **params)
                    rows.append({
                        "domain": "cont",
                        "problem": pname,
                        "instance_id": int(inst_id),
                        "shift_seed": int(shift_seed),
                        "algo_variant": algo,
                        "budget": int(b),
                        "rep": int(r),
                        "seed": int(seed),
                        "evals": int(evals),
                        "best": float(best),
                    })
                    pb.update(1)
    pb.close()

    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(out_dir / "runs_detail.csv", index=False)

    summary_df = build_instance_summary(runs_df, target_tols)
    summary_df.to_csv(out_dir / "instance_algo_budget_summary.csv", index=False)

    # quick plots: success curves
    g = summary_df.groupby(["algo_variant", "budget", "target"], as_index=False)["beta_mean"].mean()
    for tgt in target_tols.keys():
        sub = g[g["target"] == tgt]
        fig = plt.figure(figsize=(9, 4))
        ax = plt.gca()
        for algo in algos:
            s = sub[sub["algo_variant"] == algo].sort_values("budget")
            ax.plot(s["budget"].to_numpy(), s["beta_mean"].to_numpy(), marker="o", linewidth=1, label=algo)
        ax.set_title(f"Mean posterior success vs budget ({tgt})")
        ax.set_xlabel("Budget")
        ax.set_ylabel("beta_mean")
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / f"success_vs_budget_{tgt}.png", dpi=200)
        plt.close(fig)

    print("[OK] wrote:", out_dir / "instance_algo_budget_summary.csv")
    print("[OK] smoke test command:")
    print(f"  python {Path(__file__).name} --out_dir {out_dir} --smoke_test")
    print("[OK] full run command:")
    print(f"  python {Path(__file__).name} --out_dir {out_dir} --algos {','.join(algos)} --budgets {','.join(map(str,budgets))} --R {args.R}")

if __name__ == "__main__":
    main()
