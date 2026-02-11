#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
family_alignment_3family_prob.py
================================
3-family probabilistic classification using Beta-Binomial evidence model.

Input (from your existing pipeline):
- wins_by_group_top1.csv
- hits_by_group_top3.csv

Families:
- PSO_global = {PSO_STD, PSO_CONSTR}
- PSO_local  = {PSO_RING}
- Evolution  = {DE, ES_ML, GA}

Output:
- membership_3family.csv (posterior mean + [p05, p95] via bootstrap)

Run example:
python family_alignment_3family_prob.py ^
  --wins out\\wins_by_group_top1.csv ^
  --hits out\\hits_by_group_top3.csv ^
  --out_dir out_3fam ^
  --boot 2000
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


EPS = 1e-12


def log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -1e100
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def log_betabinom(k: int, n: int, alpha: float, beta: float) -> float:
    # log [ C(n,k) * B(k+alpha, n-k+beta) / B(alpha,beta) ]
    return log_comb(n, k) + log_beta(k + alpha, (n - k) + beta) - log_beta(alpha, beta)


@dataclass
class GroupKey:
    func: str
    dim_bin: str
    budget_tier: str

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.func, self.dim_bin, self.budget_tier)


def read_top1_wins(path: str):
    # columns: func, dim_bin, budget_tier, algo, win_count, n_instances_in_group
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def read_top3_hits(path: str):
    # columns: func, dim_bin, budget_tier, algo, hit_count_top3, n_instances_in_group
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def build_tables(wins_rows, hits_rows):
    # Collect groups + algos
    groups = {}
    algos = set()

    # (group, algo) -> k_top1, n
    top1 = {}
    for rr in wins_rows:
        g = GroupKey(rr["func"], rr["dim_bin"], rr["budget_tier"]).as_tuple()
        a = rr["algo"]
        k = int(rr["win_count"])
        n = int(rr["n_instances_in_group"])
        top1[(g, a)] = (k, n)
        groups[g] = n
        algos.add(a)

    # (group, algo) -> h_top3, n
    top3 = {}
    for rr in hits_rows:
        g = GroupKey(rr["func"], rr["dim_bin"], rr["budget_tier"]).as_tuple()
        a = rr["algo"]
        h = int(rr["hit_count_top3"])
        n = int(rr["n_instances_in_group"])
        top3[(g, a)] = (h, n)
        groups[g] = n
        algos.add(a)

    # Ensure common group set
    group_list = sorted(groups.keys())

    # Fill missing (group, algo) with 0 counts but keep n
    for g in group_list:
        n = groups[g]
        for a in algos:
            if (g, a) not in top1:
                top1[(g, a)] = (0, n)
            if (g, a) not in top3:
                top3[(g, a)] = (0, n)

    return group_list, sorted(algos), top1, top3, groups


def pooled_family_counts(group_list, anchors, top1, top3, groups):
    # For each group g, pool across anchor algos:
    # K = sum successes, N = sum trials
    # trials per anchor is n_g
    K1 = np.zeros(len(group_list), dtype=int)
    N1 = np.zeros(len(group_list), dtype=int)
    K3 = np.zeros(len(group_list), dtype=int)
    N3 = np.zeros(len(group_list), dtype=int)

    for gi, g in enumerate(group_list):
        n = groups[g]
        N1[gi] = n * len(anchors)
        N3[gi] = n * len(anchors)
        ksum = 0
        hsum = 0
        for a in anchors:
            ksum += top1[(g, a)][0]
            hsum += top3[(g, a)][0]
        K1[gi] = ksum
        K3[gi] = hsum

    return K1, N1, K3, N3


def posterior_probs_for_algo(
    algo: str,
    group_list,
    top1,
    top3,
    groups,
    family_stats,
    alpha0: float = 1.0,
    beta0: float = 1.0
) -> np.ndarray:
    """
    family_stats: dict family -> (K1,N1,K3,N3) arrays over groups
    returns posterior probabilities over families (softmax of log evidence)
    """
    fams = list(family_stats.keys())
    log_e = np.zeros(len(fams), dtype=float)

    for fi, fam in enumerate(fams):
        K1, N1, K3, N3 = family_stats[fam]

        s = 0.0
        for gi, g in enumerate(group_list):
            n = groups[g]
            k = top1[(g, algo)][0]
            h = top3[(g, algo)][0]

            # family posterior over p at this group (via pooled anchors)
            a1 = alpha0 + float(K1[gi])
            b1 = beta0 + float(N1[gi] - K1[gi])

            a3 = alpha0 + float(K3[gi])
            b3 = beta0 + float(N3[gi] - K3[gi])

            s += log_betabinom(k, n, a1, b1)
            s += log_betabinom(h, n, a3, b3)

        log_e[fi] = s

    # softmax
    m = float(np.max(log_e))
    p = np.exp(log_e - m)
    p = p / (np.sum(p) + EPS)
    return p


def bootstrap_posterior(
    algo: str,
    group_list,
    top1,
    top3,
    groups,
    family_stats,
    boot: int,
    rng: np.random.Generator,
    alpha0: float = 1.0,
    beta0: float = 1.0
):
    fams = list(family_stats.keys())
    G = len(group_list)
    samples = np.zeros((boot, len(fams)), dtype=float)

    # Pre-pack per group counts for speed
    k_top1 = np.array([top1[(g, algo)][0] for g in group_list], dtype=int)
    h_top3 = np.array([top3[(g, algo)][0] for g in group_list], dtype=int)
    n_g = np.array([groups[g] for g in group_list], dtype=int)

    fam_arrays = {}
    for fam in fams:
        K1, N1, K3, N3 = family_stats[fam]
        fam_arrays[fam] = (K1.astype(float), N1.astype(float), K3.astype(float), N3.astype(float))

    for b in range(boot):
        idx = rng.integers(0, G, size=G)  # resample groups with replacement
        log_e = np.zeros(len(fams), dtype=float)

        for fi, fam in enumerate(fams):
            K1, N1, K3, N3 = fam_arrays[fam]

            # compute log evidence on resampled groups
            s = 0.0
            for gi in idx:
                k = int(k_top1[gi]); h = int(h_top3[gi]); n = int(n_g[gi])

                a1 = alpha0 + K1[gi]
                b1 = beta0 + (N1[gi] - K1[gi])

                a3 = alpha0 + K3[gi]
                b3 = beta0 + (N3[gi] - K3[gi])

                s += log_betabinom(k, n, a1, b1)
                s += log_betabinom(h, n, a3, b3)

            log_e[fi] = s

        m = float(np.max(log_e))
        p = np.exp(log_e - m)
        p = p / (np.sum(p) + EPS)
        samples[b] = p

    mean = np.mean(samples, axis=0)
    p05 = np.quantile(samples, 0.05, axis=0)
    p95 = np.quantile(samples, 0.95, axis=0)
    return fams, mean, p05, p95


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wins", type=str, required=True, help="wins_by_group_top1.csv")
    ap.add_argument("--hits", type=str, required=True, help="hits_by_group_top3.csv")
    ap.add_argument("--out_dir", type=str, default="out_3fam")

    ap.add_argument("--pso_global", type=str, default="PSO_STD,PSO_CONSTR")
    ap.add_argument("--pso_local", type=str, default="PSO_RING")
    ap.add_argument("--evolution", type=str, default="DE,ES_ML,GA")

    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)

    ap.add_argument("--min_evidence_hits", type=int, default=5)
    ap.add_argument("--min_evidence_wins", type=int, default=1)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    wins_rows = read_top1_wins(args.wins)
    hits_rows = read_top3_hits(args.hits)

    group_list, algos, top1, top3, groups = build_tables(wins_rows, hits_rows)

    pso_global = [x.strip() for x in args.pso_global.split(",") if x.strip()]
    pso_local = [x.strip() for x in args.pso_local.split(",") if x.strip()]
    evolution = [x.strip() for x in args.evolution.split(",") if x.strip()]

    # Sanity: anchors exist in algos
    for a in pso_global + pso_local + evolution:
        if a not in algos:
            raise ValueError(f"Anchor algo '{a}' not found in CSV. Available: {algos}")

    family_stats = {
        "PSO_global": pooled_family_counts(group_list, pso_global, top1, top3, groups),
        "PSO_local":  pooled_family_counts(group_list, pso_local,  top1, top3, groups),
        "Evolution":  pooled_family_counts(group_list, evolution,  top1, top3, groups),
    }

    rng = np.random.default_rng(args.seed)

    out_path = os.path.join(args.out_dir, "membership_3family.csv")
    fieldnames = [
        "algo",
        "status",
        "total_top1_wins",
        "total_top3_hits",
        "P_PSO_global", "P_PSO_global_p05", "P_PSO_global_p95",
        "P_PSO_local",  "P_PSO_local_p05",  "P_PSO_local_p95",
        "P_Evolution",  "P_Evolution_p05",  "P_Evolution_p95",
    ]

    rows = []
    for a in algos:
        total_wins = sum(top1[(g, a)][0] for g in group_list)
        total_hits = sum(top3[(g, a)][0] for g in group_list)

        status = "OK"
        if total_hits < args.min_evidence_hits and total_wins < args.min_evidence_wins:
            status = "INSUFFICIENT"

        fams, mean, p05, p95 = bootstrap_posterior(
            algo=a,
            group_list=group_list,
            top1=top1,
            top3=top3,
            groups=groups,
            family_stats=family_stats,
            boot=args.boot,
            rng=rng,
        )

        # map order
        idx = {f: i for i, f in enumerate(fams)}
        row = {
            "algo": a,
            "status": status,
            "total_top1_wins": total_wins,
            "total_top3_hits": total_hits,
            "P_PSO_global": float(mean[idx["PSO_global"]]),
            "P_PSO_global_p05": float(p05[idx["PSO_global"]]),
            "P_PSO_global_p95": float(p95[idx["PSO_global"]]),
            "P_PSO_local": float(mean[idx["PSO_local"]]),
            "P_PSO_local_p05": float(p05[idx["PSO_local"]]),
            "P_PSO_local_p95": float(p95[idx["PSO_local"]]),
            "P_Evolution": float(mean[idx["Evolution"]]),
            "P_Evolution_p05": float(p05[idx["Evolution"]]),
            "P_Evolution_p95": float(p95[idx["Evolution"]]),
        }
        rows.append(row)

    # Sort: most PSO-ish first (global+local)
    rows.sort(key=lambda r: (r["P_PSO_global"] + r["P_PSO_local"]), reverse=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] wrote {out_path}")

    print("\n=== 3-family membership (posterior mean; bootstrap p05/p95) ===")
    for r in rows:
        print(
            f"{r['algo']:10s} status={r['status']:<12s} "
            f"PSO_global={r['P_PSO_global']:.3f} [{r['P_PSO_global_p05']:.3f},{r['P_PSO_global_p95']:.3f}]  "
            f"PSO_local={r['P_PSO_local']:.3f} [{r['P_PSO_local_p05']:.3f},{r['P_PSO_local_p95']:.3f}]  "
            f"Evolution={r['P_Evolution']:.3f} [{r['P_Evolution_p05']:.3f},{r['P_Evolution_p95']:.3f}]  "
            f"(wins={r['total_top1_wins']}, hits={r['total_top3_hits']})"
        )

    print("\nInterpretation:")
    print("- PSO_global high => aligns with STD/CONSTR pattern")
    print("- PSO_local high  => aligns with RING pattern (often Top1-sharp but Top3-shape differs)")
    print("- Evolution high  => aligns with DE/ES_ML/GA pattern")


if __name__ == "__main__":
    main()
