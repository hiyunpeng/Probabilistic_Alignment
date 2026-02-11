#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
success_distribution_mvp_v2.py
==============================

Step 1 Option A MVP (success distribution) + richer experiments:
- For each problem instance i, algorithm a, budget B:
    run R times (different RNG seeds)
    success = 1 if best_value <= target_i_B within eval budget

Supports:
- Continuous: Sphere, Rastrigin (minimization; optimum ~ 0)
- Binary/combinatorial: OneMax, 0/1 Knapsack (objective = optimum gap; optimum = 0)

Experiment upgrades:
1) Multiple budgets (cost tiers) per domain: budgets_cont, budgets_bin
2) Multiple algorithm variants ("steps/strength"): different pop/particles/temps etc.
3) Optional target calibration mode:
   - fixed targets (fast, deterministic)
   - pilot-percentile targets per (instance, budget): make success rates informative & comparable

Outputs:
- out_dir/runs.csv
  one row per (instance, budget, algo_variant, run)
- out_dir/instance_algo_budget_summary.csv
  aggregated per (instance, budget, algo_variant): successes, trials, beta_mean + CI, mean/median best
- out_dir/meta.csv
  metadata + CLI params snapshot

Console prints:
- prior floor (Beta(1,1) with R runs, 0 successes -> 1/(R+2))
- summary by (domain, problem, budget)
- summary by (domain, budget)
- global summary by budget

Usage (PowerShell):
  python success_distribution_mvp_v2.py --out_dir out_succ_v2

Recommended MVP configs:
  python success_distribution_mvp_v2.py --out_dir out_succ_v2 --runs_per_instance 30 --budgets_cont 500,2000 --budgets_bin 500,2000 --n_cont_instances 20 --n_bin_instances 20
If you want the target to be "learned" per instance:
  python success_distribution_mvp_v2.py --out_dir out_succ_v2 --target_mode pilot --pilot_runs 5 --pilot_percentile 0.2
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Any, Optional

import numpy as np

EPS = 1e-12


# -----------------------------
# Beta posterior summary (fast MVP)
# -----------------------------

def beta_posterior_mean_ci(successes: int, trials: int, a0: float = 1.0, b0: float = 1.0, ci: float = 0.90) -> Tuple[float, float, float]:
    """
    Beta(a0+s, b0+(n-s)) posterior. Returns mean and central CI.
    Uses normal approximation (reasonable for trials>=20; MVP-friendly).
    """
    a = a0 + successes
    b = b0 + (trials - successes)
    mean = a / (a + b)

    var = (a * b) / (((a + b) ** 2) * (a + b + 1.0))
    sd = math.sqrt(max(var, 0.0))

    if abs(ci - 0.90) < 1e-9:
        z = 1.6448536269514722
    elif abs(ci - 0.95) < 1e-9:
        z = 1.959963984540054
    else:
        z = 1.6448536269514722

    lo = max(0.0, mean - z * sd)
    hi = min(1.0, mean + z * sd)
    return mean, lo, hi


# -----------------------------
# Problem definitions (all minimization)
# -----------------------------

class Problem:
    name: str
    domain: str  # "cont" or "bin"
    target: float  # default/fixed target if target_mode=fixed

    def evaluate(self, x: np.ndarray) -> float:
        raise NotImplementedError

    def sample_init(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def default_bounds(self) -> float:
        return 5.0


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

    def default_bounds(self) -> float:
        return float(self.bounds)


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

    def default_bounds(self) -> float:
        return float(self.bounds)


@dataclass
class OneMax(Problem):
    n_bits: int
    target: float = 0.0
    name: str = "onemax"
    domain: str = "bin"

    def evaluate(self, x: np.ndarray) -> float:
        return float(self.n_bits - int(np.sum(x)))

    def sample_init(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, 2, size=(self.n_bits,), dtype=np.int8)


@dataclass
class Knapsack01(Problem):
    weights: np.ndarray
    values: np.ndarray
    capacity: int
    opt_value: int
    target: float = 0.0
    name: str = "knapsack01"
    domain: str = "bin"

    def evaluate(self, x: np.ndarray) -> float:
        w = int(np.sum(self.weights * x))
        v = int(np.sum(self.values * x))
        if w <= self.capacity:
            return float(self.opt_value - v)
        overweight = w - self.capacity
        return float(self.opt_value + 10_000 + 100 * overweight)

    def sample_init(self, rng: np.random.Generator) -> np.ndarray:
        return rng.integers(0, 2, size=(len(self.weights),), dtype=np.int8)


def knapsack_opt_dp(weights: np.ndarray, values: np.ndarray, capacity: int) -> int:
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
# Algorithms (eval-budgeted)
# -----------------------------

class Algo:
    name: str
    domain: str  # "cont" or "bin"

    def run(self, problem: Problem, budget: int, rng: np.random.Generator) -> Tuple[float, int]:
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


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
                if best <= get_target(problem):
                    break
        return best, evals


@dataclass
class PSO(Algo):
    name: str
    domain: str = "cont"
    n_particles: int = 30
    w: float = 0.72
    c1: float = 1.49
    c2: float = 1.49

    def describe(self) -> str:
        return f"{self.name}(p={self.n_particles},w={self.w},c1={self.c1},c2={self.c2})"

    def run(self, problem: Problem, budget: int, rng: np.random.Generator) -> Tuple[float, int]:
        assert problem.domain == "cont"
        dim = int(getattr(problem, "dim"))
        bounds = float(problem.default_bounds())

        X = rng.uniform(-bounds, bounds, size=(self.n_particles, dim))
        V = rng.uniform(-1.0, 1.0, size=(self.n_particles, dim))
        P = X.copy()
        fp = np.full(self.n_particles, np.inf)
        evals = 0

        for i in range(self.n_particles):
            fp[i] = problem.evaluate(X[i])
            evals += 1
            if evals >= budget:
                break

        gbest_idx = int(np.argmin(fp))
        G = P[gbest_idx].copy()
        gbest = float(fp[gbest_idx])
        tgt = get_target(problem)

        while evals < budget:
            r1 = rng.random(size=(self.n_particles, dim))
            r2 = rng.random(size=(self.n_particles, dim))
            V = self.w * V + self.c1 * r1 * (P - X) + self.c2 * r2 * (G[None, :] - X)
            X = np.clip(X + V, -bounds, bounds)

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

            if gbest <= tgt:
                break

        return gbest, evals


@dataclass
class PSO_Ring(Algo):
    name: str
    domain: str = "cont"
    n_particles: int = 30
    w: float = 0.72
    c1: float = 1.49
    c2: float = 1.49

    def describe(self) -> str:
        return f"{self.name}(p={self.n_particles},ring,w={self.w},c1={self.c1},c2={self.c2})"

    def run(self, problem: Problem, budget: int, rng: np.random.Generator) -> Tuple[float, int]:
        assert problem.domain == "cont"
        dim = int(getattr(problem, "dim"))
        bounds = float(problem.default_bounds())

        X = rng.uniform(-bounds, bounds, size=(self.n_particles, dim))
        V = rng.uniform(-1.0, 1.0, size=(self.n_particles, dim))
        P = X.copy()
        fp = np.full(self.n_particles, np.inf)
        evals = 0

        for i in range(self.n_particles):
            fp[i] = problem.evaluate(X[i])
            evals += 1
            if evals >= budget:
                break

        tgt = get_target(problem)
        gbest = float(np.min(fp))

        def neigh_best(i: int) -> np.ndarray:
            idxs = [(i - 1) % self.n_particles, i, (i + 1) % self.n_particles]
            j = idxs[int(np.argmin(fp[idxs]))]
            return P[j]

        while evals < budget:
            for i in range(self.n_particles):
                if evals >= budget:
                    break
                nb = neigh_best(i)
                r1 = rng.random(size=(dim,))
                r2 = rng.random(size=(dim,))
                V[i] = self.w * V[i] + self.c1 * r1 * (P[i] - X[i]) + self.c2 * r2 * (nb - X[i])
                X[i] = np.clip(X[i] + V[i], -bounds, bounds)

                fx = problem.evaluate(X[i])
                evals += 1
                if fx < fp[i]:
                    fp[i] = fx
                    P[i] = X[i].copy()
                    if fx < gbest:
                        gbest = float(fx)

            if gbest <= tgt:
                break

        return gbest, evals


@dataclass
class DE(Algo):
    name: str
    domain: str = "cont"
    pop: int = 30
    F: float = 0.8
    CR: float = 0.9

    def describe(self) -> str:
        return f"{self.name}(pop={self.pop},F={self.F},CR={self.CR})"

    def run(self, problem: Problem, budget: int, rng: np.random.Generator) -> Tuple[float, int]:
        assert problem.domain == "cont"
        dim = int(getattr(problem, "dim"))
        bounds = float(problem.default_bounds())

        X = rng.uniform(-bounds, bounds, size=(self.pop, dim))
        fx = np.full(self.pop, np.inf)
        evals = 0

        for i in range(self.pop):
            fx[i] = problem.evaluate(X[i])
            evals += 1
            if evals >= budget:
                break

        best = float(np.min(fx))
        tgt = get_target(problem)

        while evals < budget:
            for i in range(self.pop):
                if evals >= budget:
                    break
                idxs = [j for j in range(self.pop) if j != i]
                if len(idxs) < 3:
                    break
                r1, r2, r3 = rng.choice(idxs, size=3, replace=False)
                mutant = X[r1] + self.F * (X[r2] - X[r3])
                mutant = np.clip(mutant, -bounds, bounds)

                cross = rng.random(size=(dim,)) < self.CR
                cross[int(rng.integers(0, dim))] = True
                trial = np.where(cross, mutant, X[i])

                ftrial = problem.evaluate(trial)
                evals += 1
                if ftrial < fx[i]:
                    X[i] = trial
                    fx[i] = ftrial
                    if ftrial < best:
                        best = float(ftrial)

            if best <= tgt:
                break

        return best, evals


@dataclass
class GA_Binary(Algo):
    name: str
    domain: str = "bin"
    pop: int = 50
    pc: float = 0.9
    pm: float = 0.02
    tournament_k: int = 3

    def describe(self) -> str:
        return f"{self.name}(pop={self.pop},pc={self.pc},pm={self.pm})"

    def run(self, problem: Problem, budget: int, rng: np.random.Generator) -> Tuple[float, int]:
        assert problem.domain == "bin"
        # derive length once
        n = len(problem.sample_init(rng))

        P = rng.integers(0, 2, size=(self.pop, n), dtype=np.int8)
        fx = np.full(self.pop, np.inf)
        evals = 0

        for i in range(self.pop):
            fx[i] = problem.evaluate(P[i])
            evals += 1
            if evals >= budget:
                break

        best = float(np.min(fx))
        tgt = get_target(problem)

        def tournament() -> int:
            idx = rng.integers(0, self.pop, size=(self.tournament_k,))
            return int(idx[np.argmin(fx[idx])])

        while evals < budget:
            newP = np.empty_like(P)
            elite = int(np.argmin(fx))
            newP[0] = P[elite].copy()

            for i in range(1, self.pop, 2):
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

            if best <= tgt:
                break

        return best, evals


@dataclass
class SA_Binary(Algo):
    name: str
    domain: str = "bin"
    T0: float = 1.0
    alpha: float = 0.995
    flips: int = 1

    def describe(self) -> str:
        return f"{self.name}(T0={self.T0},alpha={self.alpha},flips={self.flips})"

    def run(self, problem: Problem, budget: int, rng: np.random.Generator) -> Tuple[float, int]:
        assert problem.domain == "bin"
        x = problem.sample_init(rng).copy()
        fx = problem.evaluate(x)
        evals = 1
        best = float(fx)
        tgt = get_target(problem)

        T = float(self.T0)
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
            if best <= tgt:
                break

        return best, evals


@dataclass
class HC_Binary(Algo):
    """Simple hill climbing: always accept improving single-bit flips."""
    name: str
    domain: str = "bin"
    flips: int = 1

    def describe(self) -> str:
        return f"{self.name}(flips={self.flips})"

    def run(self, problem: Problem, budget: int, rng: np.random.Generator) -> Tuple[float, int]:
        assert problem.domain == "bin"
        x = problem.sample_init(rng).copy()
        fx = problem.evaluate(x)
        evals = 1
        best = float(fx)
        tgt = get_target(problem)

        n = len(x)
        while evals < budget:
            y = x.copy()
            idx = rng.choice(n, size=self.flips, replace=False)
            y[idx] ^= 1
            fy = problem.evaluate(y)
            evals += 1
            if fy <= fx:
                x, fx = y, fy
                if fx < best:
                    best = float(fx)
            if best <= tgt:
                break
        return best, evals


# -----------------------------
# Target handling (fixed vs pilot)
# -----------------------------

_TARGET_MAP: Dict[Tuple[str, int], float] = {}
# key: (instance_id, budget) -> target used in that setting


def set_target(instance_id: str, budget: int, target: float) -> None:
    _TARGET_MAP[(instance_id, int(budget))] = float(target)


def get_target_for_instance_budget(instance_id: str, budget: int, default_target: float) -> float:
    return float(_TARGET_MAP.get((instance_id, int(budget)), default_target))


def get_target(problem: Problem) -> float:
    # runtime target is stored at problem.target for current context (we update it per run)
    return float(problem.target)


# -----------------------------
# Instances
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
# Algorithm variants (more experiments)
# -----------------------------

def build_algorithms(include_variants: bool = True) -> List[Algo]:
    algos: List[Algo] = []

    # Continuous baselines + variants
    algos.append(RandomSearch(name="RS_CONT", domain="cont"))

    # PSO global variants
    algos.append(PSO(name="PSO_STD", n_particles=20))
    algos.append(PSO(name="PSO_STD", n_particles=40))

    # PSO ring variants
    algos.append(PSO_Ring(name="PSO_RING", n_particles=20))
    algos.append(PSO_Ring(name="PSO_RING", n_particles=40))

    # DE variants
    algos.append(DE(name="DE", pop=20))
    algos.append(DE(name="DE", pop=40))

    # Binary baselines + variants
    algos.append(RandomSearch(name="RS_BIN", domain="bin"))

    algos.append(GA_Binary(name="GA", pop=50, pm=0.02))
    algos.append(GA_Binary(name="GA", pop=100, pm=0.02))
    algos.append(GA_Binary(name="GA", pop=50, pm=0.05))

    algos.append(SA_Binary(name="SA", T0=1.0, alpha=0.995, flips=1))
    algos.append(SA_Binary(name="SA", T0=2.0, alpha=0.990, flips=1))
    algos.append(SA_Binary(name="SA", T0=2.0, alpha=0.995, flips=2))

    algos.append(HC_Binary(name="HC", flips=1))
    algos.append(HC_Binary(name="HC", flips=2))

    # If you want "minimal set", user can filter via CLI later
    return algos


# -----------------------------
# Pilot target calibration (optional)
# -----------------------------

def calibrate_targets_pilot(
    instances: List[Instance],
    algos: List[Algo],
    budgets_cont: List[int],
    budgets_bin: List[int],
    pilot_runs: int,
    pilot_percentile: float,
    seed: int,
) -> None:
    """
    For each (instance, budget), run quick pilot on applicable algos:
    collect final best values across algos and pilot runs
    set target to percentile of those values (lower is better).
    This makes success rates informative and avoids 0%/100% collapse.
    """
    rng_master = np.random.default_rng(seed + 777)
    for inst in instances:
        p = inst.problem
        budgets = budgets_cont if p.domain == "cont" else budgets_bin
        applicable = [a for a in algos if a.domain == p.domain]

        for B in budgets:
            vals = []
            # generate pilot seeds per algo/run
            seeds = rng_master.integers(0, 2**32 - 1, size=(len(applicable), pilot_runs), dtype=np.uint32)
            for ai, algo in enumerate(applicable):
                for r in range(pilot_runs):
                    rng = np.random.default_rng(int(seeds[ai, r]) + (hash(algo.describe()) & 0xFFFF))
                    # set a temporary target so early stopping doesn't short-circuit pilot
                    # Use the fixed default target for evaluation only (not stopping), so disable early stop:
                    # simplest: run with a very low target so it won't stop early.
                    old_t = p.target
                    p.target = -1e18  # practically never succeed => no early stop
                    best, _ = algo.run(p, budget=int(B), rng=rng)
                    p.target = old_t
                    vals.append(best)

            # percentile target (lower is easier to meet)
            if len(vals) == 0:
                continue
            t = float(np.quantile(np.array(vals, dtype=float), pilot_percentile))
            set_target(inst.instance_id, int(B), t)


# -----------------------------
# Reporting utilities
# -----------------------------

def parse_int_list(s: str) -> List[int]:
    xs = []
    for part in s.split(","):
        part = part.strip()
        if part:
            xs.append(int(part))
    return xs


def write_meta(out_dir: str, args: argparse.Namespace, algos: List[Algo]) -> None:
    meta_path = os.path.join(out_dir, "meta.csv")
    rows = [
        {"key": "seed", "value": str(args.seed)},
        {"key": "runs_per_instance", "value": str(args.runs_per_instance)},
        {"key": "budgets_cont", "value": str(args.budgets_cont)},
        {"key": "budgets_bin", "value": str(args.budgets_bin)},
        {"key": "n_cont_instances", "value": str(args.n_cont_instances)},
        {"key": "n_bin_instances", "value": str(args.n_bin_instances)},
        {"key": "target_mode", "value": str(args.target_mode)},
        {"key": "pilot_runs", "value": str(args.pilot_runs)},
        {"key": "pilot_percentile", "value": str(args.pilot_percentile)},
        {"key": "beta_prior_a0", "value": str(args.beta_a0)},
        {"key": "beta_prior_b0", "value": str(args.beta_b0)},
        {"key": "ci", "value": str(args.ci)},
        {"key": "algo_variants_count", "value": str(len(algos))},
    ]
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["key", "value"])
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] wrote {meta_path}")


def print_summaries(summary_rows: List[Dict[str, Any]], runs_per_instance: int, ci: float) -> None:
    prior_floor = 1.0 / (2.0 + runs_per_instance)
    print("\n=== Sanity ===")
    print(f"Beta(1,1) prior floor with 0 successes: 1/(R+2) = {prior_floor:.3f} (R={runs_per_instance})")

    # Index by (domain, problem, budget, algo_variant)
    bucket: Dict[Tuple[str, str, int, str], List[float]] = {}
    bucket_sr: Dict[Tuple[str, str, int, str], List[float]] = {}
    for r in summary_rows:
        key = (r["domain"], r["problem"], int(r["budget"]), r["algo_variant"])
        bucket.setdefault(key, []).append(float(r["beta_mean"]))
        bucket_sr.setdefault(key, []).append(float(r["succ_rate"]))

    print("\n=== Summary by (domain, problem, budget) ===")
    domains = sorted({r["domain"] for r in summary_rows})
    for domain in domains:
        probs = sorted({(r["problem"], int(r["budget"])) for r in summary_rows if r["domain"] == domain})
        for (problem, budget) in probs:
            print(f"\n-- {domain.upper()} / {problem} / budget={budget} --")
            items = []
            for (d, p, b, algo), vals in bucket.items():
                if d == domain and p == problem and b == budget:
                    items.append((algo, float(np.mean(vals)), float(np.mean(bucket_sr[(d,p,b,algo)])), len(vals)))
            items.sort(key=lambda x: -x[1])
            for algo, m_beta, m_sr, ninst in items:
                print(f"{algo:28s} beta_mean={m_beta:.3f}  succ_rate={m_sr:.3f}  n={ninst}")

    print("\n=== Summary by (domain, budget) ===")
    for domain in domains:
        budgets = sorted({int(r["budget"]) for r in summary_rows if r["domain"] == domain})
        for budget in budgets:
            # aggregate across problems & instances within domain
            by_algo: Dict[str, List[float]] = {}
            by_algo_sr: Dict[str, List[float]] = {}
            for r in summary_rows:
                if r["domain"] == domain and int(r["budget"]) == budget:
                    by_algo.setdefault(r["algo_variant"], []).append(float(r["beta_mean"]))
                    by_algo_sr.setdefault(r["algo_variant"], []).append(float(r["succ_rate"]))
            items = [(algo, float(np.mean(xs)), float(np.mean(by_algo_sr[algo])), len(xs)) for algo, xs in by_algo.items()]
            items.sort(key=lambda x: -x[1])
            print(f"\n-- {domain.upper()} / budget={budget} --")
            for algo, mb, msr, n in items:
                print(f"{algo:28s} beta_mean={mb:.3f}  succ_rate={msr:.3f}  n={n}")

    print("\n=== Global view by budget (domain-separated) ===")
    for domain in domains:
        budgets = sorted({int(r["budget"]) for r in summary_rows if r["domain"] == domain})
        for budget in budgets:
            # show top 5 only to keep console readable
            by_algo: Dict[str, List[float]] = {}
            for r in summary_rows:
                if r["domain"] == domain and int(r["budget"]) == budget:
                    by_algo.setdefault(r["algo_variant"], []).append(float(r["beta_mean"]))
            items = [(algo, float(np.mean(xs))) for algo, xs in by_algo.items()]
            items.sort(key=lambda x: -x[1])
            topk = items[:5]
            print(f"{domain.upper()} budget={budget} top5: " + ", ".join([f"{a}={m:.3f}" for a, m in topk]))


# -----------------------------
# Main experiment
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="out_succ_v2")
    ap.add_argument("--seed", type=int, default=2026)

    ap.add_argument("--runs_per_instance", type=int, default=30)
    ap.add_argument("--budgets_cont", type=str, default="500,2000")
    ap.add_argument("--budgets_bin", type=str, default="500,2000")

    ap.add_argument("--n_cont_instances", type=int, default=20)
    ap.add_argument("--n_bin_instances", type=int, default=20)

    ap.add_argument("--ci", type=float, default=0.90)
    ap.add_argument("--beta_a0", type=float, default=1.0)
    ap.add_argument("--beta_b0", type=float, default=1.0)

    ap.add_argument("--target_mode", choices=["fixed", "pilot"], default="fixed",
                    help="fixed: use problem.target; pilot: set target per (instance,budget) using percentile of pilot outcomes")
    ap.add_argument("--pilot_runs", type=int, default=5)
    ap.add_argument("--pilot_percentile", type=float, default=0.2)

    args = ap.parse_args()

    budgets_cont = parse_int_list(args.budgets_cont)
    budgets_bin = parse_int_list(args.budgets_bin)

    os.makedirs(args.out_dir, exist_ok=True)

    instances = make_instances(seed=args.seed, n_cont=args.n_cont_instances, n_bin=args.n_bin_instances)
    algos = build_algorithms(include_variants=True)

    write_meta(args.out_dir, args, algos)

    # optional pilot targets
    if args.target_mode == "pilot":
        print("[INFO] calibrating targets via pilot runs ...")
        calibrate_targets_pilot(
            instances=instances,
            algos=algos,
            budgets_cont=budgets_cont,
            budgets_bin=budgets_bin,
            pilot_runs=args.pilot_runs,
            pilot_percentile=args.pilot_percentile,
            seed=args.seed
        )
        print("[OK] pilot target calibration done.")

    runs_path = os.path.join(args.out_dir, "runs.csv")
    summary_path = os.path.join(args.out_dir, "instance_algo_budget_summary.csv")

    run_rows: List[Dict[str, Any]] = []
    agg: Dict[Tuple[str, int, str], List[Tuple[float, int]]] = {}  # (instance_id, budget, algo_variant) -> list[(best, success)]

    rng_master = np.random.default_rng(args.seed)
    # Pre-generate seeds for reproducibility: per instance, per budget, per algo, per run
    # We'll just generate on the fly deterministically.

    for inst in instances:
        p0 = inst.problem
        budgets = budgets_cont if p0.domain == "cont" else budgets_bin
        applicable = [a for a in algos if a.domain == p0.domain]

        for B in budgets:
            # determine target for this (instance,budget)
            if args.target_mode == "pilot":
                target = get_target_for_instance_budget(inst.instance_id, int(B), p0.target)
            else:
                target = float(p0.target)

            # per run seeds (shared across algos for fair-ish noise)
            run_seeds = rng_master.integers(0, 2**32 - 1, size=(args.runs_per_instance,), dtype=np.uint32)

            for algo in applicable:
                algo_variant = algo.describe()

                for r_i in range(args.runs_per_instance):
                    # clone a "working problem" to safely set target without global side effects
                    # simplest: reuse p0 but restore target immediately
                    old_target = p0.target
                    p0.target = target

                    rng = np.random.default_rng(int(run_seeds[r_i]) + (hash(algo_variant) & 0xFFFF))
                    best, evals = algo.run(p0, budget=int(B), rng=rng)

                    p0.target = old_target

                    success = 1 if best <= target else 0

                    run_rows.append({
                        "instance_id": inst.instance_id,
                        "problem": p0.name,
                        "domain": p0.domain,
                        "budget": int(B),
                        "target": float(target),
                        "algo_variant": algo_variant,
                        "algo_base": algo.name,
                        "run_idx": int(r_i),
                        "best": float(best),
                        "success": int(success),
                        "evals": int(evals),
                    })

                    agg.setdefault((inst.instance_id, int(B), algo_variant), []).append((float(best), int(success)))

    # Write runs.csv
    with open(runs_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["instance_id", "problem", "domain", "budget", "target", "algo_variant", "algo_base", "run_idx", "best", "success", "evals"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(run_rows)
    print(f"[OK] wrote {runs_path}")

    # Build summary rows
    summary_rows: List[Dict[str, Any]] = []
    # For quick lookup of problem/domain by instance_id & algo_variant
    # We'll pick the first run row match.
    first_by_key: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for rr in run_rows:
        k = (rr["instance_id"], int(rr["budget"]), rr["algo_variant"])
        if k not in first_by_key:
            first_by_key[k] = rr

    for (instance_id, budget, algo_variant), vals in agg.items():
        bests = np.array([v[0] for v in vals], dtype=float)
        succ = np.array([v[1] for v in vals], dtype=int)

        s = int(np.sum(succ))
        n = int(len(succ))
        succ_rate = float(s / max(n, 1))

        p_mean, p_lo, p_hi = beta_posterior_mean_ci(s, n, a0=args.beta_a0, b0=args.beta_b0, ci=args.ci)

        rr0 = first_by_key[(instance_id, budget, algo_variant)]
        summary_rows.append({
            "instance_id": instance_id,
            "problem": rr0["problem"],
            "domain": rr0["domain"],
            "budget": int(budget),
            "target": float(rr0["target"]),
            "algo_variant": algo_variant,
            "algo_base": rr0["algo_base"],
            "successes": int(s),
            "trials": int(n),
            "succ_rate": succ_rate,
            "beta_mean": float(p_mean),
            "beta_p05": float(p_lo) if args.ci == 0.90 else float(p_lo),
            "beta_p95": float(p_hi) if args.ci == 0.90 else float(p_hi),
            "mean_best": float(np.mean(bests)),
            "median_best": float(np.median(bests)),
            "min_best": float(np.min(bests)),
            "max_best": float(np.max(bests)),
        })

    summary_rows.sort(key=lambda r: (r["domain"], r["problem"], int(r["budget"]), r["instance_id"], r["algo_variant"]))

    # Write summary CSV
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[OK] wrote {summary_path}")

    # Print rich summaries
    print_summaries(summary_rows, runs_per_instance=args.runs_per_instance, ci=args.ci)


if __name__ == "__main__":
    main()
