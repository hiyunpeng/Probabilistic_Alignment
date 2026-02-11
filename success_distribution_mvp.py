#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
success_distribution_mvp.py
===========================

MVP for "Step 1 Option A" success-distribution testing:

For every problem instance i:
- run each algorithm a many times (R runs with different RNG seeds)
- define success as "hit a target value T_i within evaluation budget B"
- estimate the success probability p_{i,a} from s successes out of R

This script supports BOTH:
- continuous problems (Sphere, Rastrigin)
- combinatorial/binary problems (OneMax, 0/1 Knapsack)

It outputs:
1) runs.csv: one row per (instance, algo, run)
2) instance_algo_summary.csv: aggregated successes + Beta posterior mean/CI per (instance, algo)

Run:
  python success_distribution_mvp.py --out_dir out_succ

Notes:
- All objectives are MINIMIZATION.
- Targets are defined so the global optimum corresponds to value 0.
- For knapsack, we compute the true optimum value by DP, and set objective = opt_value - achieved_value (feasible).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Any

import numpy as np

EPS = 1e-12


# -----------------------------
# Utility: Beta posterior summary
# -----------------------------

def beta_posterior_mean_ci(successes: int, trials: int, a0: float = 1.0, b0: float = 1.0, ci: float = 0.90) -> Tuple[float, float, float]:
    """
    Beta(a0+s, b0+(n-s)) posterior.
    Returns mean and central CI (default 90% -> p05,p95).
    Uses a normal approximation when scipy isn't available.
    """
    a = a0 + successes
    b = b0 + (trials - successes)
    mean = a / (a + b)

    # Normal approx of Beta variance (ok for MVP with trials>=20)
    var = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
    sd = math.sqrt(max(var, 0.0))

    # central CI bounds for normal approx
    # For 90% CI, z ~ 1.64485; for 95% CI, z ~ 1.96
    if abs(ci - 0.90) < 1e-9:
        z = 1.6448536269514722
    elif abs(ci - 0.95) < 1e-9:
        z = 1.959963984540054
    else:
        # fallback: approximate inverse CDF of normal via erfinv
        z = math.sqrt(2.0) * _erfinv(ci)

    lo = max(0.0, mean - z * sd)
    hi = min(1.0, mean + z * sd)
    return mean, lo, hi


def _erfinv(y: float) -> float:
    # Approximation of inverse error function (Winitzki)
    a = 0.147
    sgn = 1.0 if y >= 0 else -1.0
    ln = math.log(1.0 - y * y)
    term = (2.0 / (math.pi * a)) + (ln / 2.0)
    x = sgn * math.sqrt(max(0.0, math.sqrt(term * term - (ln / a)) - term))
    return x


# -----------------------------
# Problem definitions (all minimization)
# -----------------------------

class Problem:
    name: str
    domain: str  # "cont" or "bin"
    target: float

    def evaluate(self, x: Union[np.ndarray, np.ndarray]) -> float:
        raise NotImplementedError

    def sample_init(self, rng: np.random.Generator) -> Union[np.ndarray, np.ndarray]:
        raise NotImplementedError


@dataclass
class Sphere(Problem):
    dim: int
    bounds: float = 5.0
    target: float = 1e-3
    name: str = "sphere"
    domain: str = "cont"

    def evaluate(self, x: np.ndarray) -> float:
        return float(np.sum(x * x))

    def sample_init(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-self.bounds, self.bounds, size=(self.dim,)).astype(np.float64)


@dataclass
class Rastrigin(Problem):
    dim: int
    bounds: float = 5.12
    target: float = 5.0
    name: str = "rastrigin"
    domain: str = "cont"

    def evaluate(self, x: np.ndarray) -> float:
        A = 10.0
        return float(A * self.dim + np.sum(x * x - A * np.cos(2.0 * math.pi * x)))

    def sample_init(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(-self.bounds, self.bounds, size=(self.dim,)).astype(np.float64)


@dataclass
class OneMax(Problem):
    n_bits: int
    target: float = 0.0
    name: str = "onemax"
    domain: str = "bin"

    def evaluate(self, x: np.ndarray) -> float:
        # objective is distance to all-ones => optimum 0
        return float(self.n_bits - int(np.sum(x)))

    def sample_init(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, 2, size=(self.n_bits,), dtype=np.int8)


@dataclass
class Knapsack01(Problem):
    weights: np.ndarray  # (n,)
    values: np.ndarray   # (n,)
    capacity: int
    opt_value: int
    target: float = 0.0
    name: str = "knapsack01"
    domain: str = "bin"

    def evaluate(self, x: np.ndarray) -> float:
        w = int(np.sum(self.weights * x))
        v = int(np.sum(self.values * x))
        if w <= self.capacity:
            # objective: gap to optimum value -> optimum 0
            return float(self.opt_value - v)
        # infeasible: penalize heavily (still minimization)
        overweight = w - self.capacity
        return float(self.opt_value + 10_000 + 100 * overweight)

    def sample_init(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, 2, size=(len(self.weights),), dtype=np.int8)


def knapsack_opt_dp(weights: np.ndarray, values: np.ndarray, capacity: int) -> int:
    # classic 0/1 knapsack DP for optimum value
    cap = int(capacity)
    dp = np.zeros(cap + 1, dtype=np.int64)
    for w, v in zip(weights.astype(int), values.astype(int)):
        w = int(w); v = int(v)
        for c in range(cap, w - 1, -1):
            cand = dp[c - w] + v
            if cand > dp[c]:
                dp[c] = cand
    return int(dp.max())


# -----------------------------
# Algorithms (minimal, eval-count budgeted)
# -----------------------------

class Algo:
    name: str
    domain: str  # "cont" or "bin"

    def run(self, problem: Problem, budget: int, rng: np.random.Generator) -> Tuple[float, int]:
        """Return (best_value, evals_used)."""
        raise NotImplementedError


@dataclass
class RandomSearch(Algo):
    name: str
    domain: str

    def run(self, problem: Problem, budget: int, rng: np.random.Generator) -> Tuple[float, int]:
        best = float("inf")
        evals = 0
        while evals < budget:
            x = problem.sample_init(rng)
            fx = problem.evaluate(x)
            evals += 1
            if fx < best:
                best = fx
                if best <= problem.target:
                    break
        return best, evals


@dataclass
class PSO(Algo):
    name: str = "PSO_STD"
    domain: str = "cont"
    n_particles: int = 30
    w: float = 0.72
    c1: float = 1.49
    c2: float = 1.49
    bounds: float = 5.0

    def run(self, problem: Problem, budget: int, rng: np.random.Generator) -> Tuple[float, int]:
        assert problem.domain == "cont"
        dim = getattr(problem, "dim")
        X = rng.uniform(-self.bounds, self.bounds, size=(self.n_particles, dim))
        V = rng.uniform(-1.0, 1.0, size=(self.n_particles, dim))
        P = X.copy()
        fp = np.full(self.n_particles, np.inf)
        evals = 0

        for i in range(self.n_particles):
            fp[i] = problem.evaluate(X[i])
            evals += 1
        gbest_idx = int(np.argmin(fp))
        G = P[gbest_idx].copy()
        gbest = float(fp[gbest_idx])

        while evals < budget:
            r1 = rng.random(size=(self.n_particles, dim))
            r2 = rng.random(size=(self.n_particles, dim))
            V = self.w * V + self.c1 * r1 * (P - X) + self.c2 * r2 * (G[None, :] - X)
            X = np.clip(X + V, -self.bounds, self.bounds)

            for i in range(self.n_particles):
                if evals >= budget:
                    break
                fx = problem.evaluate(X[i])
                evals += 1
                if fx < fp[i]:
                    fp[i] = fx
                    P[i] = X[i].copy()
                    if fx < gbest:
                        gbest = float(fx)
                        G = X[i].copy()

            if gbest <= problem.target:
                break

        return gbest, evals


@dataclass
class PSO_Ring(Algo):
    name: str = "PSO_RING"
    domain: str = "cont"
    n_particles: int = 30
    w: float = 0.72
    c1: float = 1.49
    c2: float = 1.49
    bounds: float = 5.0

    def run(self, problem: Problem, budget: int, rng: np.random.Generator) -> Tuple[float, int]:
        assert problem.domain == "cont"
        dim = getattr(problem, "dim")
        X = rng.uniform(-self.bounds, self.bounds, size=(self.n_particles, dim))
        V = rng.uniform(-1.0, 1.0, size=(self.n_particles, dim))
        P = X.copy()
        fp = np.full(self.n_particles, np.inf)
        evals = 0

        for i in range(self.n_particles):
            fp[i] = problem.evaluate(X[i])
            evals += 1

        def neigh_best(i: int) -> Tuple[np.ndarray, float]:
            idxs = [(i - 1) % self.n_particles, i, (i + 1) % self.n_particles]
            j = idxs[int(np.argmin(fp[idxs]))]
            return P[j], float(fp[j])

        gbest = float(np.min(fp))

        while evals < budget:
            for i in range(self.n_particles):
                if evals >= budget:
                    break
                nb, _ = neigh_best(i)
                r1 = rng.random(size=(dim,))
                r2 = rng.random(size=(dim,))
                V[i] = self.w * V[i] + self.c1 * r1 * (P[i] - X[i]) + self.c2 * r2 * (nb - X[i])
                X[i] = np.clip(X[i] + V[i], -self.bounds, self.bounds)

                fx = problem.evaluate(X[i])
                evals += 1
                if fx < fp[i]:
                    fp[i] = fx
                    P[i] = X[i].copy()
                    if fx < gbest:
                        gbest = float(fx)

            if gbest <= problem.target:
                break

        return gbest, evals


@dataclass
class DE(Algo):
    name: str = "DE"
    domain: str = "cont"
    pop: int = 30
    F: float = 0.8
    CR: float = 0.9
    bounds: float = 5.0

    def run(self, problem: Problem, budget: int, rng: np.random.Generator) -> Tuple[float, int]:
        assert problem.domain == "cont"
        dim = getattr(problem, "dim")
        X = rng.uniform(-self.bounds, self.bounds, size=(self.pop, dim))
        fx = np.full(self.pop, np.inf)
        evals = 0

        for i in range(self.pop):
            fx[i] = problem.evaluate(X[i])
            evals += 1

        best = float(np.min(fx))

        while evals < budget:
            for i in range(self.pop):
                if evals >= budget:
                    break
                idxs = [j for j in range(self.pop) if j != i]
                r1, r2, r3 = rng.choice(idxs, size=3, replace=False)
                mutant = X[r1] + self.F * (X[r2] - X[r3])
                mutant = np.clip(mutant, -self.bounds, self.bounds)

                cross = rng.random(size=(dim,)) < self.CR
                cross[rng.integers(0, dim)] = True
                trial = np.where(cross, mutant, X[i])

                ftrial = problem.evaluate(trial)
                evals += 1
                if ftrial < fx[i]:
                    X[i] = trial
                    fx[i] = ftrial
                    if ftrial < best:
                        best = float(ftrial)

            if best <= problem.target:
                break

        return best, evals


@dataclass
class GA_Binary(Algo):
    name: str = "GA"
    domain: str = "bin"
    pop: int = 50
    pc: float = 0.9
    pm: float = 0.02
    tournament_k: int = 3

    def run(self, problem: Problem, budget: int, rng: np.random.Generator) -> Tuple[float, int]:
        assert problem.domain == "bin"
        n = len(problem.sample_init(rng))

        P = rng.integers(0, 2, size=(self.pop, n), dtype=np.int8)
        fx = np.full(self.pop, np.inf)
        evals = 0

        for i in range(self.pop):
            fx[i] = problem.evaluate(P[i])
            evals += 1

        best = float(np.min(fx))

        def tournament() -> int:
            idx = rng.integers(0, self.pop, size=(self.tournament_k,))
            return int(idx[np.argmin(fx[idx])])

        while evals < budget:
            newP = np.empty_like(P)

            elite = int(np.argmin(fx))
            newP[0] = P[elite].copy()

            for i in range(1, self.pop, 2):
                if evals >= budget:
                    break
                p1 = P[tournament()].copy()
                p2 = P[tournament()].copy()

                if rng.random() < self.pc:
                    cx = int(rng.integers(1, n))
                    c1 = np.concatenate([p1[:cx], p2[cx:]])
                    c2 = np.concatenate([p2[:cx], p1[cx:]])
                else:
                    c1, c2 = p1, p2

                mut1 = rng.random(size=(n,)) < self.pm
                mut2 = rng.random(size=(n,)) < self.pm
                c1 = c1.copy(); c2 = c2.copy()
                c1[mut1] ^= 1
                c2[mut2] ^= 1

                newP[i] = c1
                if i + 1 < self.pop:
                    newP[i + 1] = c2

            P = newP

            for i in range(self.pop):
                if evals >= budget:
                    break
                fx[i] = problem.evaluate(P[i])
                evals += 1
                if fx[i] < best:
                    best = float(fx[i])

            if best <= problem.target:
                break

        return best, evals


@dataclass
class SA_Binary(Algo):
    name: str = "SA"
    domain: str = "bin"
    T0: float = 1.0
    alpha: float = 0.995
    flips: int = 1

    def run(self, problem: Problem, budget: int, rng: np.random.Generator) -> Tuple[float, int]:
        assert problem.domain == "bin"
        x = problem.sample_init(rng).copy()
        fx = problem.evaluate(x)
        evals = 1
        best = float(fx)

        T = self.T0
        n = len(x)

        while evals < budget:
            y = x.copy()
            idx = rng.choice(n, size=self.flips, replace=False)
            y[idx] ^= 1

            fy = problem.evaluate(y)
            evals += 1

            d = fy - fx
            if d <= 0 or rng.random() < math.exp(-d / max(T, 1e-12)):
                x, fx = y, fy
                if fx < best:
                    best = float(fx)

            T *= self.alpha
            if best <= problem.target:
                break

        return best, evals


# -----------------------------
# Instance generation
# -----------------------------

@dataclass
class Instance:
    instance_id: str
    problem: Problem


def make_instances(seed: int, n_cont: int, n_bin: int) -> List[Instance]:
    rng = np.random.default_rng(seed)
    instances: List[Instance] = []

    cont_dims = [5, 10, 20]
    for j in range(n_cont):
        dim = int(rng.choice(cont_dims))
        if j % 2 == 0:
            p = Sphere(dim=dim, target=1e-3)
        else:
            p = Rastrigin(dim=dim, target=5.0)
        instances.append(Instance(instance_id=f"{p.name}_d{dim}_i{j:03d}", problem=p))

    bin_sizes = [30, 60, 100]
    for j in range(n_bin):
        if j % 2 == 0:
            n_bits = int(rng.choice(bin_sizes))
            p = OneMax(n_bits=n_bits, target=0.0)
            instances.append(Instance(instance_id=f"{p.name}_n{n_bits}_i{j:03d}", problem=p))
        else:
            n_items = int(rng.choice([20, 30, 40]))
            weights = rng.integers(1, 30, size=(n_items,), dtype=np.int64)
            values = rng.integers(1, 50, size=(n_items,), dtype=np.int64)
            capacity = int(0.35 * int(np.sum(weights)))
            opt_val = knapsack_opt_dp(weights, values, capacity)
            p = Knapsack01(
                weights=weights.astype(np.int64),
                values=values.astype(np.int64),
                capacity=capacity,
                opt_value=opt_val,
                target=0.0
            )
            instances.append(Instance(instance_id=f"{p.name}_n{n_items}_cap{capacity}_i{j:03d}", problem=p))

    return instances


# -----------------------------
# Main experiment loop
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="out_succ")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--runs_per_instance", type=int, default=30)
    ap.add_argument("--budget_cont", type=int, default=2000)
    ap.add_argument("--budget_bin", type=int, default=2000)
    ap.add_argument("--n_cont_instances", type=int, default=10)
    ap.add_argument("--n_bin_instances", type=int, default=10)
    ap.add_argument("--ci", type=float, default=0.90)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    instances = make_instances(seed=args.seed, n_cont=args.n_cont_instances, n_bin=args.n_bin_instances)

    algos: List[Algo] = [
        RandomSearch(name="RS_CONT", domain="cont"),
        PSO(name="PSO_STD"),
        PSO_Ring(name="PSO_RING"),
        DE(name="DE"),
        RandomSearch(name="RS_BIN", domain="bin"),
        GA_Binary(name="GA"),
        SA_Binary(name="SA"),
    ]

    runs_path = os.path.join(args.out_dir, "runs.csv")
    summary_path = os.path.join(args.out_dir, "instance_algo_summary.csv")

    run_rows: List[Dict[str, Any]] = []
    agg: Dict[Tuple[str, str], List[Tuple[float, int]]] = {}

    base_rng = np.random.default_rng(args.seed)

    for inst in instances:
        p = inst.problem
        budget = args.budget_cont if p.domain == "cont" else args.budget_bin

        run_seeds = base_rng.integers(0, 2**32 - 1, size=(args.runs_per_instance,), dtype=np.uint32)

        for algo in algos:
            if algo.domain != p.domain:
                continue
            for r_i in range(args.runs_per_instance):
                rng = np.random.default_rng(int(run_seeds[r_i]) + (hash(algo.name) & 0xFFFF))
                best, evals = algo.run(p, budget=budget, rng=rng)
                success = 1 if best <= p.target else 0

                run_rows.append({
                    "instance_id": inst.instance_id,
                    "problem": p.name,
                    "domain": p.domain,
                    "target": p.target,
                    "budget": budget,
                    "algo": algo.name,
                    "run_idx": r_i,
                    "best": best,
                    "success": success,
                    "evals": evals,
                })
                agg.setdefault((inst.instance_id, algo.name), []).append((best, success))

    with open(runs_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["instance_id", "problem", "domain", "target", "budget", "algo", "run_idx", "best", "success", "evals"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(run_rows)
    print(f"[OK] wrote {runs_path}")

    summary_rows: List[Dict[str, Any]] = []
    for (instance_id, algo), vals in agg.items():
        bests = np.array([v[0] for v in vals], dtype=float)
        succ = np.array([v[1] for v in vals], dtype=int)

        s = int(np.sum(succ))
        n = int(len(succ))

        mean_best = float(np.mean(bests))
        median_best = float(np.median(bests))
        p_mean, p_lo, p_hi = beta_posterior_mean_ci(s, n, a0=1.0, b0=1.0, ci=args.ci)

        first = next(rr for rr in run_rows if rr["instance_id"] == instance_id and rr["algo"] == algo)
        summary_rows.append({
            "instance_id": instance_id,
            "problem": first["problem"],
            "domain": first["domain"],
            "algo": algo,
            "budget": first["budget"],
            "target": first["target"],
            "successes": s,
            "trials": n,
            "succ_rate": s / max(n, 1),
            "beta_mean": p_mean,
            f"beta_p{int((1-args.ci)/2*100):02d}": p_lo,
            f"beta_p{int((1-(1-args.ci)/2)*100):02d}": p_hi,
            "mean_best": mean_best,
            "median_best": median_best,
        })

    summary_rows.sort(key=lambda r: (r["problem"], r["instance_id"], r["algo"]))

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[OK] wrote {summary_path}")

    print("\n=== Quick aggregate view (mean of per-instance beta_mean) ===")
    by_algo: Dict[str, List[float]] = {}
    for r in summary_rows:
        by_algo.setdefault(r["algo"], []).append(float(r["beta_mean"]))
    for algo, xs in sorted(by_algo.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"{algo:10s} mean_beta_success={np.mean(xs):.3f}  n_instances={len(xs)}")


if __name__ == "__main__":
    main()
