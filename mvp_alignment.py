#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MVP: Probabilistic alignment of optimisation problems & metaheuristics
Domains: continuous + combinatorial (TSP)
Outputs:
  - results.csv: per-run outcomes (instance, algorithm, seed, budget, best_value, best_eval)
  - alignment_summary.csv: Dirichlet-posterior win probs per group (problem features)
  - regret_summary.csv: expected regret vs best-per-group and vs best-single-alg baseline

Run:
  python mvp_alignment.py run --out results.csv
  python mvp_alignment.py analyze --in results.csv --out_prefix mvp

No heavy deps. Requires: numpy
"""

from __future__ import annotations
import argparse, csv, math, os, time, json
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Any
import numpy as np


# -----------------------------
# Utilities
# -----------------------------
def now_ms() -> int:
    return int(time.time() * 1000)

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def set_global_seed(seed: int):
    np.random.seed(seed)

def dirichlet_posterior_summary(counts: np.ndarray, alpha0: float = 1.0, n_mc: int = 5000, q=(0.05, 0.95), rng=None):
    """
    counts: [K] win counts
    Returns mean probs + credible interval via MC sampling.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    alpha = counts.astype(float) + alpha0
    samples = rng.dirichlet(alpha, size=n_mc)  # [n_mc, K]
    mean = samples.mean(axis=0)
    lo = np.quantile(samples, q[0], axis=0)
    hi = np.quantile(samples, q[1], axis=0)
    return mean, lo, hi


# -----------------------------
# Continuous benchmark functions
# -----------------------------
def f_sphere(x):      # min at 0
    return float(np.sum(x * x))

def f_rosenbrock(x):  # min at 0 with shift
    # classic Rosenbrock with optimum at ones; shift to 0 for consistency
    y = x + 1.0
    return float(np.sum(100.0 * (y[1:] - y[:-1] ** 2) ** 2 + (1.0 - y[:-1]) ** 2))

def f_rastrigin(x):
    A = 10.0
    return float(A * x.size + np.sum(x * x - A * np.cos(2.0 * np.pi * x)))

def f_ackley(x):
    d = x.size
    a, b, c = 20.0, 0.2, 2.0 * np.pi
    s1 = np.sum(x * x)
    s2 = np.sum(np.cos(c * x))
    return float(-a * np.exp(-b * np.sqrt(s1 / d)) - np.exp(s2 / d) + a + math.e)

CONT_FUNCS = {
    "sphere": f_sphere,
    "rosenbrock": f_rosenbrock,
    "rastrigin": f_rastrigin,
    "ackley": f_ackley,
}


# -----------------------------
# Continuous metaheuristics (budget = function evals)
# -----------------------------
def cont_random_search(obj, bounds, budget, rng):
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    best_x = None
    best_y = float("inf")
    evals = 0
    while evals < budget:
        x = rng.uniform(lo, hi)
        y = obj(x); evals += 1
        if y < best_y:
            best_y, best_x = y, x.copy()
    return best_y, evals

def cont_hillclimb_gaussian(obj, bounds, budget, rng):
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    x = rng.uniform(lo, hi)
    y = obj(x); evals = 1
    best_y = y
    sigma = 0.2 * np.mean(hi - lo)
    stall = 0
    while evals < budget:
        cand = x + rng.normal(0, sigma, size=d)
        cand = clamp(cand, lo, hi)
        yc = obj(cand); evals += 1
        if yc < y:
            x, y = cand, yc
            best_y = min(best_y, y)
            stall = 0
            sigma *= 1.02
        else:
            stall += 1
            sigma *= 0.999
        if stall > 200:
            # mild restart
            x = rng.uniform(lo, hi)
            y = obj(x); evals += 1
            stall = 0
            sigma = 0.2 * np.mean(hi - lo)
            best_y = min(best_y, y)
    return best_y, evals

def cont_simulated_annealing(obj, bounds, budget, rng):
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    x = rng.uniform(lo, hi)
    y = obj(x); evals = 1
    best_y = y
    step0 = 0.25 * np.mean(hi - lo)
    T0 = 1.0
    while evals < budget:
        t = evals / max(1, budget - 1)
        T = T0 * (0.01 ** t)
        step = step0 * (0.1 + 0.9 * (1 - t))
        cand = x + rng.normal(0, step, size=d)
        cand = clamp(cand, lo, hi)
        yc = obj(cand); evals += 1
        if yc < y or rng.random() < math.exp(-(yc - y) / max(1e-12, T)):
            x, y = cand, yc
            best_y = min(best_y, y)
    return best_y, evals

def cont_differential_evolution(obj, bounds, budget, rng):
    # Minimal DE/rand/1/bin
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    pop_size = max(10, 5 * d)
    F, CR = 0.8, 0.9
    pop = rng.uniform(lo, hi, size=(pop_size, d))
    fit = np.array([obj(pop[i]) for i in range(pop_size)], dtype=float)
    evals = pop_size
    best_y = float(np.min(fit))
    while evals < budget:
        for i in range(pop_size):
            if evals >= budget:
                break
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = rng.choice(idxs, size=3, replace=False)
            mutant = pop[a] + F * (pop[b] - pop[c])
            mutant = clamp(mutant, lo, hi)
            cross = pop[i].copy()
            jrand = rng.integers(0, d)
            for j in range(d):
                if rng.random() < CR or j == jrand:
                    cross[j] = mutant[j]
            yc = obj(cross); evals += 1
            if yc < fit[i]:
                pop[i] = cross
                fit[i] = yc
                if yc < best_y:
                    best_y = yc
    return best_y, evals


CONT_ALGOS = {
    "RS": cont_random_search,
    "HC": cont_hillclimb_gaussian,
    "SA": cont_simulated_annealing,
    "DE": cont_differential_evolution,
}


# -----------------------------
# Combinatorial benchmark: TSP instances
# -----------------------------

def tsp_generate(n: int, kind: str, rng) -> np.ndarray:
    """
    Returns coords [n,2] in [0,1]^2
    kind: 'uniform' | 'clustered'
    """
    if kind == "uniform":
        return rng.random((n, 2))
    if kind == "clustered":
        k = max(2, n // 15)
        centers = rng.random((k, 2))
        assign = rng.integers(0, k, size=n)
        pts = centers[assign] + 0.06 * rng.normal(size=(n, 2))
        return np.clip(pts, 0.0, 1.0)
    raise ValueError(f"unknown tsp kind: {kind}")

def tsp_tour_length(coords: np.ndarray, tour: np.ndarray) -> float:
    pts = coords[tour]
    shifted = np.roll(pts, -1, axis=0)
    d = np.linalg.norm(pts - shifted, axis=1)
    return float(np.sum(d))

def tsp_nn_tour(coords: np.ndarray, start: int = 0) -> np.ndarray:
    n = coords.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = np.empty(n, dtype=int)
    cur = start
    for i in range(n):
        tour[i] = cur
        visited[cur] = True
        if i == n - 1:
            break
        # choose nearest unvisited
        d = np.linalg.norm(coords - coords[cur], axis=1)
        d[visited] = np.inf
        cur = int(np.argmin(d))
    return tour

def tsp_two_opt_swap(tour: np.ndarray, i: int, k: int) -> np.ndarray:
    # reverse segment [i,k]
    new = tour.copy()
    new[i:k+1] = new[i:k+1][::-1]
    return new

def tsp_move_random_two_opt(n: int, rng) -> Tuple[int, int]:
    i = int(rng.integers(0, n - 1))
    k = int(rng.integers(i + 1, n))
    return i, k

def tsp_clusteriness(coords: np.ndarray, rng) -> float:
    # Proxy: mean NN distance / mean pair distance (lower => more clustered)
    n = coords.shape[0]
    # NN
    nn = []
    for i in range(n):
        d = np.linalg.norm(coords - coords[i], axis=1)
        d[i] = np.inf
        nn.append(float(np.min(d)))
    nn_mean = float(np.mean(nn))
    # approximate mean pair via sampling
    m = min(2000, n * (n - 1) // 2)
    idx = rng.integers(0, n, size=(m, 2))
    mask = idx[:, 0] != idx[:, 1]
    idx = idx[mask]
    if idx.shape[0] == 0:
        return 1.0
    pd = np.linalg.norm(coords[idx[:, 0]] - coords[idx[:, 1]], axis=1)
    pair_mean = float(np.mean(pd))
    return float(nn_mean / max(1e-12, pair_mean))


# -----------------------------
# TSP metaheuristics (budget = move proposals / evaluations)
# -----------------------------
def tsp_random_tour(coords, budget, rng):
    n = coords.shape[0]
    tour = rng.permutation(n)
    best = tsp_tour_length(coords, tour)
    evals = 1
    while evals < budget:
        cand = rng.permutation(n)
        yc = tsp_tour_length(coords, cand); evals += 1
        if yc < best:
            best = yc
    return best, evals

def tsp_two_opt_hillclimb(coords, budget, rng):
    n = coords.shape[0]
    tour = tsp_nn_tour(coords, start=int(rng.integers(0, n)))
    y = tsp_tour_length(coords, tour); evals = 1
    best = y
    stall = 0
    while evals < budget:
        i, k = tsp_move_random_two_opt(n, rng)
        cand = tsp_two_opt_swap(tour, i, k)
        yc = tsp_tour_length(coords, cand); evals += 1
        if yc < y:
            tour, y = cand, yc
            best = min(best, y)
            stall = 0
        else:
            stall += 1
        if stall > 500:
            tour = tsp_nn_tour(coords, start=int(rng.integers(0, n)))
            y = tsp_tour_length(coords, tour); evals += 1
            best = min(best, y)
            stall = 0
    return best, evals

def tsp_simulated_annealing(coords, budget, rng):
    n = coords.shape[0]
    tour = tsp_nn_tour(coords, start=int(rng.integers(0, n)))
    y = tsp_tour_length(coords, tour); evals = 1
    best = y
    T0 = 0.1 * y / n
    while evals < budget:
        t = evals / max(1, budget - 1)
        T = T0 * (0.01 ** t)
        i, k = tsp_move_random_two_opt(n, rng)
        cand = tsp_two_opt_swap(tour, i, k)
        yc = tsp_tour_length(coords, cand); evals += 1
        if yc < y or rng.random() < math.exp(-(yc - y) / max(1e-12, T)):
            tour, y = cand, yc
            best = min(best, y)
    return best, evals

def tsp_iterated_local_search(coords, budget, rng):
    n = coords.shape[0]
    # local search primitive: small 2-opt run
    def local_search(tour, y, steps):
        nonlocal evals
        for _ in range(steps):
            if evals >= budget:
                break
            i, k = tsp_move_random_two_opt(n, rng)
            cand = tsp_two_opt_swap(tour, i, k)
            yc = tsp_tour_length(coords, cand); evals += 1
            if yc < y:
                tour, y = cand, yc
        return tour, y

    tour = tsp_nn_tour(coords, start=int(rng.integers(0, n)))
    y = tsp_tour_length(coords, tour); evals = 1
    best = y
    best_tour = tour.copy()

    while evals < budget:
        # intensify
        tour, y = local_search(tour, y, steps=200)
        if y < best:
            best, best_tour = y, tour.copy()
        # perturb: reverse a random segment (cheap "kick")
        i, k = tsp_move_random_two_opt(n, rng)
        tour = tsp_two_opt_swap(best_tour, i, k)
        y = tsp_tour_length(coords, tour); evals += 1

    return best, evals


TSP_ALGOS = {
    "RT": tsp_random_tour,
    "2OPT": tsp_two_opt_hillclimb,
    "SA": tsp_simulated_annealing,
    "ILS": tsp_iterated_local_search,
}


# -----------------------------
# Instance definitions
# -----------------------------
@dataclass
class Instance:
    instance_id: str
    domain: str  # "continuous" | "tsp"
    meta: Dict[str, Any]
    payload: Any  # callable+bounds for continuous; coords for TSP


def make_instances_continuous(rng, n_instances: int = 40) -> List[Instance]:
    instances = []
    func_names = list(CONT_FUNCS.keys())
    dims = [2, 5, 10, 20]
    for idx in range(n_instances):
        fn = func_names[idx % len(func_names)]
        d = dims[(idx // len(func_names)) % len(dims)]
        bounds = np.array([[-5.0, 5.0]] * d, dtype=float)
        obj = CONT_FUNCS[fn]
        iid = f"cont_{fn}_d{d}_{idx:04d}"
        instances.append(Instance(
            instance_id=iid,
            domain="continuous",
            meta={
                "func": fn,
                "dim": d,
            },
            payload={"obj": obj, "bounds": bounds},
        ))
    return instances

def make_instances_tsp(rng, n_instances: int = 40) -> List[Instance]:
    instances = []
    kinds = ["uniform", "clustered"]
    sizes = [20, 40, 80]
    for idx in range(n_instances):
        kind = kinds[idx % len(kinds)]
        n = sizes[(idx // len(kinds)) % len(sizes)]
        coords = tsp_generate(n, kind=kind, rng=rng)
        cid = float(tsp_clusteriness(coords, rng))
        iid = f"tsp_{kind}_n{n}_{idx:04d}"
        instances.append(Instance(
            instance_id=iid,
            domain="tsp",
            meta={
                "kind": kind,
                "n": n,
                "clusteriness": cid,
            },
            payload={"coords": coords},
        ))
    return instances


# -----------------------------
# Experiment runner
# -----------------------------
def run_experiment(out_csv: str,
                   n_instances_each: int = 40,
                   seeds: List[int] = [0, 1, 2, 3, 4],
                   budgets: List[int] = [2000, 10000, 50000],
                   master_seed: int = 123):
    rng_master = np.random.default_rng(master_seed)
    insts = []
    insts += make_instances_continuous(rng_master, n_instances_each)
    insts += make_instances_tsp(rng_master, n_instances_each)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    fieldnames = [
        "ts_ms", "domain", "instance_id", "budget",
        "algo", "seed", "best_value", "evals",
        "meta_json"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for inst in insts:
            domain = inst.domain
            for budget in budgets:
                for algo_name, algo_fn in (CONT_ALGOS.items() if domain == "continuous" else TSP_ALGOS.items()):
                    for s in seeds:
                        rng = np.random.default_rng(hash((master_seed, inst.instance_id, budget, algo_name, s)) & 0xFFFFFFFF)
                        if domain == "continuous":
                            obj = inst.payload["obj"]
                            bounds = inst.payload["bounds"]
                            best, evals = algo_fn(obj=obj, bounds=bounds, budget=budget, rng=rng)
                        else:
                            coords = inst.payload["coords"]
                            best, evals = algo_fn(coords=coords, budget=budget, rng=rng)

                        w.writerow({
                            "ts_ms": now_ms(),
                            "domain": domain,
                            "instance_id": inst.instance_id,
                            "budget": budget,
                            "algo": algo_name,
                            "seed": s,
                            "best_value": best,
                            "evals": evals,
                            "meta_json": json.dumps(inst.meta, ensure_ascii=False),
                        })

    print(f"[OK] wrote: {out_csv}")


# -----------------------------
# Analysis: probabilistic alignment via Dirichlet posterior
# -----------------------------
def size_bin_cont(dim: int) -> str:
    if dim <= 5: return "d<=5"
    if dim <= 10: return "d<=10"
    return "d>10"

def size_bin_tsp(n: int) -> str:
    if n <= 30: return "n<=30"
    if n <= 60: return "n<=60"
    return "n>60"

def parse_results(in_csv: str) -> List[Dict[str, Any]]:
    rows = []
    with open(in_csv, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["budget"] = int(row["budget"])
            row["seed"] = int(row["seed"])
            row["best_value"] = float(row["best_value"])
            row["evals"] = int(row["evals"])
            row["meta"] = json.loads(row["meta_json"])
            rows.append(row)
    return rows

def analyze_alignment(in_csv: str, out_prefix: str, alpha0: float = 1.0, mc: int = 5000):
    rows = parse_results(in_csv)

    # Aggregate per (domain, instance_id, budget, algo): mean across seeds
    key = lambda r: (r["domain"], r["instance_id"], r["budget"], r["algo"])
    agg: Dict[Tuple, List[float]] = {}
    meta_by_inst: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        agg.setdefault(key(r), []).append(r["best_value"])
        meta_by_inst[(r["domain"], r["instance_id"])] = r["meta"]

    mean_perf = {k: float(np.mean(v)) for k, v in agg.items()}

    # Determine winner per instance/budget (lowest mean objective)
    winners = []  # dict entries: domain, budget, group_key, winner_algo, best_value, meta
    # Build list of algorithms per domain
    algos_by_domain = {
        "continuous": list(CONT_ALGOS.keys()),
        "tsp": list(TSP_ALGOS.keys()),
    }

    # Instance list
    inst_keys = sorted(set((d, iid, b) for (d, iid, b, a) in mean_perf.keys()))
    for domain, iid, budget in inst_keys:
        algos = algos_by_domain[domain]
        vals = [(a, mean_perf[(domain, iid, budget, a)]) for a in algos]
        vals.sort(key=lambda x: x[1])
        winner_a, winner_v = vals[0]
        meta = meta_by_inst[(domain, iid)]
        if domain == "continuous":
            g = (domain, meta["func"], size_bin_cont(int(meta["dim"])), f"B{budget}")
        else:
            g = (domain, meta["kind"], size_bin_tsp(int(meta["n"])), f"B{budget}")
        winners.append({
            "domain": domain,
            "instance_id": iid,
            "budget": budget,
            "group": g,
            "winner": winner_a,
            "best_value": winner_v,
            "meta": meta,
        })

    # Build probabilistic alignment table: Dirichlet posterior over winners per group
    summary_rows = []
    rng = np.random.default_rng(2026)

    for domain in ["continuous", "tsp"]:
        algos = algos_by_domain[domain]
        # group -> counts
        groups = sorted(set(w["group"] for w in winners if w["domain"] == domain))
        for g in groups:
            wins = [w for w in winners if w["group"] == g]
            counts = np.zeros(len(algos), dtype=int)
            for w in wins:
                counts[algos.index(w["winner"])] += 1
            meanp, lo, hi = dirichlet_posterior_summary(counts, alpha0=alpha0, n_mc=mc, rng=rng)
            for j, a in enumerate(algos):
                summary_rows.append({
                    "domain": g[0],
                    "group_type": g[1],
                    "size_bin": g[2],
                    "budget_tier": g[3],
                    "algo": a,
                    "win_count": int(counts[j]),
                    "n_instances": int(np.sum(counts)),
                    "post_mean_p_best": float(meanp[j]),
                    "post_p05": float(lo[j]),
                    "post_p95": float(hi[j]),
                })

    out_align = f"{out_prefix}_alignment_summary.csv"
    with open(out_align, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Regret analysis: compare (a) per-group oracle, (b) best single algorithm per domain+budget
    # Build per instance: true best algo value among algos (from mean_perf)
    regret_rows = []
    for domain in ["continuous", "tsp"]:
        algos = algos_by_domain[domain]
        # best single algorithm per (domain, budget): minimize average best_value
        budgets = sorted(set(w["budget"] for w in winners if w["domain"] == domain))
        best_single = {}
        for b in budgets:
            # average over instances for each algo
            insts_b = sorted(set(iid for (d, iid, bb, a) in mean_perf.keys() if d == domain and bb == b))
            scores = []
            for a in algos:
                vals = [mean_perf[(domain, iid, b, a)] for iid in insts_b]
                scores.append((a, float(np.mean(vals))))
            scores.sort(key=lambda x: x[1])
            best_single[b] = scores[0][0]

        # For each group, recommend algo = argmax posterior mean prob-best
        # Build posterior mean lookup
        post_mean = {}
        for r in summary_rows:
            if r["domain"] == domain:
                g = (r["domain"], r["group_type"], r["size_bin"], r["budget_tier"])
                post_mean.setdefault(g, {})
                post_mean[g][r["algo"]] = r["post_mean_p_best"]

        # Compute regret per instance under:
        #  - policy: choose algo with highest posterior mean p_best in group
        #  - baseline: best single algo for domain+budget
        for w in winners:
            if w["domain"] != domain:
                continue
            meta = w["meta"]
            if domain == "continuous":
                g = (domain, meta["func"], size_bin_cont(int(meta["dim"])), f"B{w['budget']}")
            else:
                g = (domain, meta["kind"], size_bin_tsp(int(meta["n"])), f"B{w['budget']}")

            iid = w["instance_id"]
            b = w["budget"]

            # oracle value (best among algos)
            vals = [mean_perf[(domain, iid, b, a)] for a in algos]
            oracle = float(np.min(vals))

            # policy value
            rec = max(post_mean[g].items(), key=lambda kv: kv[1])[0]
            y_rec = float(mean_perf[(domain, iid, b, rec)])

            # best-single baseline
            base_a = best_single[b]
            y_base = float(mean_perf[(domain, iid, b, base_a)])

            regret_rows.append({
                "domain": domain,
                "group_type": g[1],
                "size_bin": g[2],
                "budget": b,
                "instance_id": iid,
                "policy_algo": rec,
                "baseline_algo": base_a,
                "oracle_value": oracle,
                "policy_value": y_rec,
                "baseline_value": y_base,
                "policy_regret": y_rec - oracle,
                "baseline_regret": y_base - oracle,
            })

    out_reg = f"{out_prefix}_regret_summary.csv"
    with open(out_reg, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(regret_rows[0].keys()))
        w.writeheader()
        w.writerows(regret_rows)

    # Quick console readout (management-friendly)
    def mean_by(rows, keys, valkey):
        agg = {}
        for r in rows:
            k = tuple(r[kx] for kx in keys)
            agg.setdefault(k, []).append(float(r[valkey]))
        return {k: float(np.mean(v)) for k, v in agg.items()}

    m_policy = mean_by(regret_rows, ["domain", "budget"], "policy_regret")
    m_base = mean_by(regret_rows, ["domain", "budget"], "baseline_regret")

    print(f"[OK] wrote: {out_align}")
    print(f"[OK] wrote: {out_reg}")
    print("\n=== Expected regret (lower is better) ===")
    for k in sorted(m_policy.keys()):
        print(f"{k}: policy={m_policy[k]:.4g} | best-single={m_base[k]:.4g} | lift={(m_base[k]-m_policy[k]):.4g}")

    print("\nTip: open *_alignment_summary.csv to see probabilistic alignment map per group.")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_run = sub.add_parser("run")
    ap_run.add_argument("--out", type=str, default="results.csv")
    ap_run.add_argument("--n_instances_each", type=int, default=30)   # keep MVP quick
    ap_run.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap_run.add_argument("--budgets", type=str, default="2000,10000")  # MVP default; add 50000 if you want
    ap_run.add_argument("--master_seed", type=int, default=123)

    ap_an = sub.add_parser("analyze")
    ap_an.add_argument("--in", dest="inp", type=str, default="results.csv")
    ap_an.add_argument("--out_prefix", type=str, default="mvp")
    ap_an.add_argument("--alpha0", type=float, default=1.0)
    ap_an.add_argument("--mc", type=int, default=5000)

    args = ap.parse_args()

    if args.cmd == "run":
        seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
        budgets = [int(x) for x in args.budgets.split(",") if x.strip() != ""]
        run_experiment(
            out_csv=args.out,
            n_instances_each=args.n_instances_each,
            seeds=seeds,
            budgets=budgets,
            master_seed=args.master_seed,
        )
    elif args.cmd == "analyze":
        analyze_alignment(
            in_csv=args.inp,
            out_prefix=args.out_prefix,
            alpha0=args.alpha0,
            mc=args.mc,
        )

if __name__ == "__main__":
    main()

