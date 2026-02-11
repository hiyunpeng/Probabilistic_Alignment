#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
family_alignment_3family_dm.py
==============================
3-family probabilistic classification using Dirichlet–Multinomial on rank buckets:
  c1   = Top1 count
  c23  = Top3 count - Top1 count
  c0   = n - Top3 count

This avoids double-counting (Top1 is subset of Top3).

Input (from your pipeline, under out/):
- out/wins_by_group_top1.csv
- out/hits_by_group_top3.csv

Families (anchors):
- PSO_global = {PSO_STD, PSO_CONSTR}
- PSO_local  = {PSO_RING}
- Evolution  = {DE, ES_ML, GA}  (GA optional; can be evidence-gated)

Outputs (out_dir):
- membership_3family_dm.csv

Run:
python family_alignment_3family_dm.py --in_dir out --out_dir out_3fam_dm --boot 2000
"""

import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np

EPS = 1e-12


# ----------------------------
# math helpers
# ----------------------------

def log_multinomial_coeff(c: np.ndarray) -> float:
    # log( n! / prod(c_i!) )
    n = int(np.sum(c))
    s = math.lgamma(n + 1)
    for x in c:
        s -= math.lgamma(int(x) + 1)
    return s


def log_dirichlet_multinomial(c: np.ndarray, alpha: np.ndarray) -> float:
    """
    DM(c | alpha) proportional to:
      Multinomial coeff * Γ(A)/Γ(A+n) * ∏ Γ(alpha_i + c_i)/Γ(alpha_i)
    """
    c = np.asarray(c, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    n = float(np.sum(c))
    A = float(np.sum(alpha))

    s = log_multinomial_coeff(c)
    s += math.lgamma(A) - math.lgamma(A + n)
    for i in range(len(alpha)):
        s += math.lgamma(alpha[i] + c[i]) - math.lgamma(alpha[i])
    return float(s)


def softmax(logv: np.ndarray) -> np.ndarray:
    m = float(np.max(logv))
    p = np.exp(logv - m)
    return p / (np.sum(p) + EPS)


# ----------------------------
# IO + tables
# ----------------------------

@dataclass(frozen=True)
class GroupKey:
    func: str
    dim_bin: str
    budget_tier: str

    def as_tuple(self) -> Tuple[str, str, str]:
        return (self.func, self.dim_bin, self.budget_tier)


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def build_tables(wins_rows, hits_rows):
    # (group, algo) -> k_top1, n
    top1: Dict[Tuple[Tuple[str, str, str], str], Tuple[int, int]] = {}
    # (group, algo) -> h_top3, n
    top3: Dict[Tuple[Tuple[str, str, str], str], Tuple[int, int]] = {}

    groups: Dict[Tuple[str, str, str], int] = {}
    algos = set()

    for rr in wins_rows:
        g = GroupKey(rr["func"], rr["dim_bin"], rr["budget_tier"]).as_tuple()
        a = rr["algo"]
        k = int(rr["win_count"])
        n = int(rr["n_instances_in_group"])
        top1[(g, a)] = (k, n)
        groups[g] = n
        algos.add(a)

    for rr in hits_rows:
        g = GroupKey(rr["func"], rr["dim_bin"], rr["budget_tier"]).as_tuple()
        a = rr["algo"]
        h = int(rr["hit_count_top3"])
        n = int(rr["n_instances_in_group"])
        top3[(g, a)] = (h, n)
        groups[g] = n
        algos.add(a)

    group_list = sorted(groups.keys())
    algos = sorted(algos)

    # Fill missing entries with 0 (same n)
    for g in group_list:
        n = groups[g]
        for a in algos:
            if (g, a) not in top1:
                top1[(g, a)] = (0, n)
            if (g, a) not in top3:
                top3[(g, a)] = (0, n)

    return group_list, algos, top1, top3, groups


def counts_3bucket(k_top1: int, h_top3: int, n: int) -> np.ndarray:
    # c1 = top1, c23 = top3-top1, c0 = rest
    c1 = int(k_top1)
    c23 = int(max(0, h_top3 - k_top1))
    c0 = int(max(0, n - h_top3))
    return np.array([c1, c23, c0], dtype=int)


# ----------------------------
# family model
# ----------------------------

def family_posterior_alpha_per_group(
    group_list,
    anchors: List[str],
    top1,
    top3,
    groups,
    alpha0: np.ndarray
) -> List[np.ndarray]:
    """
    For each group, build Dirichlet posterior alpha = alpha0 + pooled anchor counts.
    alpha0 is 3-dim prior (symmetric recommended).
    """
    post = []
    for g in group_list:
        pooled = np.zeros(3, dtype=int)
        n = groups[g]
        for a in anchors:
            k, _ = top1[(g, a)]
            h, _ = top3[(g, a)]
            pooled += counts_3bucket(k, h, n)
        post.append(alpha0 + pooled.astype(float))
    return post


def log_evidence_algo_given_family(
    algo: str,
    group_list,
    top1,
    top3,
    groups,
    family_post_alpha: List[np.ndarray]
) -> float:
    """
    Evidence = sum_g log DM(c_algo_g | alpha_family_g)
    """
    s = 0.0
    for gi, g in enumerate(group_list):
        n = groups[g]
        k, _ = top1[(g, algo)]
        h, _ = top3[(g, algo)]
        c = counts_3bucket(k, h, n)
        alpha = family_post_alpha[gi]
        s += log_dirichlet_multinomial(c.astype(float), alpha)
    return float(s)


def bootstrap_groups_posterior(
    algo: str,
    fam_names: List[str],
    fam_post_alpha: Dict[str, List[np.ndarray]],
    group_list,
    top1,
    top3,
    groups,
    boot: int,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample groups with replacement, compute posterior over families each bootstrap.
    Returns mean, p05, p95 over boot samples.
    """
    G = len(group_list)
    samples = np.zeros((boot, len(fam_names)), dtype=float)

    # Precompute algo counts per group
    c_algo = []
    for g in group_list:
        n = groups[g]
        k, _ = top1[(g, algo)]
        h, _ = top3[(g, algo)]
        c_algo.append(counts_3bucket(k, h, n).astype(float))
    c_algo = np.stack(c_algo, axis=0)  # (G,3)

    for b in range(boot):
        idx = rng.integers(0, G, size=G)
        loge = np.zeros(len(fam_names), dtype=float)

        for fi, fam in enumerate(fam_names):
            s = 0.0
            alphas = fam_post_alpha[fam]
            for gi in idx:
                s += log_dirichlet_multinomial(c_algo[gi], alphas[gi])
            loge[fi] = s

        samples[b] = softmax(loge)

    mean = np.mean(samples, axis=0)
    p05 = np.quantile(samples, 0.05, axis=0)
    p95 = np.quantile(samples, 0.95, axis=0)
    return mean, p05, p95


# ----------------------------
# main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="out", help="directory containing wins_by_group_top1.csv and hits_by_group_top3.csv")
    ap.add_argument("--out_dir", type=str, default="out_3fam_dm")

    ap.add_argument("--pso_global", type=str, default="PSO_STD,PSO_CONSTR")
    ap.add_argument("--pso_local", type=str, default="PSO_RING")
    ap.add_argument("--evolution", type=str, default="DE,ES_ML,GA")

    ap.add_argument("--alpha_prior", type=float, default=1.0, help="symmetric Dirichlet prior strength per bucket")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)

    ap.add_argument("--min_evidence_hits", type=int, default=5)
    ap.add_argument("--min_evidence_wins", type=int, default=1)

    args = ap.parse_args()

    wins_path = os.path.join(args.in_dir, "wins_by_group_top1.csv")
    hits_path = os.path.join(args.in_dir, "hits_by_group_top3.csv")

    wins_rows = read_csv(wins_path)
    hits_rows = read_csv(hits_path)

    group_list, algos, top1, top3, groups = build_tables(wins_rows, hits_rows)

    pso_global = [x.strip() for x in args.pso_global.split(",") if x.strip()]
    pso_local = [x.strip() for x in args.pso_local.split(",") if x.strip()]
    evolution = [x.strip() for x in args.evolution.split(",") if x.strip()]

    for a in pso_global + pso_local + evolution:
        if a not in algos:
            raise ValueError(f"Anchor algo '{a}' not found. Available algos: {algos}")

    alpha0 = np.array([args.alpha_prior] * 3, dtype=float)  # (Top1, Top2-3, Others)

    fam_names = ["PSO_global", "PSO_local", "Evolution"]
    fam_anchors = {
        "PSO_global": pso_global,
        "PSO_local": pso_local,
        "Evolution": evolution,
    }

    # Build per-group family posterior alpha
    fam_post_alpha = {}
    for fam in fam_names:
        fam_post_alpha[fam] = family_posterior_alpha_per_group(
            group_list=group_list,
            anchors=fam_anchors[fam],
            top1=top1,
            top3=top3,
            groups=groups,
            alpha0=alpha0
        )

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "membership_3family_dm.csv")

    rng = np.random.default_rng(args.seed)

    fieldnames = [
        "algo", "status",
        "total_top1_wins", "total_top3_hits",
        "P_PSO_global", "P_PSO_global_p05", "P_PSO_global_p95",
        "P_PSO_local",  "P_PSO_local_p05",  "P_PSO_local_p95",
        "P_Evolution",  "P_Evolution_p05",  "P_Evolution_p95",
    ]

    rows_out: List[Dict[str, Any]] = []
    for a in algos:
        total_wins = sum(top1[(g, a)][0] for g in group_list)
        total_hits = sum(top3[(g, a)][0] for g in group_list)

        status = "OK"
        if total_hits < args.min_evidence_hits and total_wins < args.min_evidence_wins:
            status = "INSUFFICIENT"

        mean, p05, p95 = bootstrap_groups_posterior(
            algo=a,
            fam_names=fam_names,
            fam_post_alpha=fam_post_alpha,
            group_list=group_list,
            top1=top1,
            top3=top3,
            groups=groups,
            boot=args.boot,
            rng=rng
        )

        row = {
            "algo": a,
            "status": status,
            "total_top1_wins": int(total_wins),
            "total_top3_hits": int(total_hits),
            "P_PSO_global": float(mean[0]),
            "P_PSO_global_p05": float(p05[0]),
            "P_PSO_global_p95": float(p95[0]),
            "P_PSO_local": float(mean[1]),
            "P_PSO_local_p05": float(p05[1]),
            "P_PSO_local_p95": float(p95[1]),
            "P_Evolution": float(mean[2]),
            "P_Evolution_p05": float(p05[2]),
            "P_Evolution_p95": float(p95[2]),
        }
        rows_out.append(row)

    # sort by PSO-ishness
    rows_out.sort(key=lambda r: (r["P_PSO_global"] + r["P_PSO_local"]), reverse=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)

    print(f"[OK] wrote {out_path}")

    print("\n=== 3-family membership (Dirichlet–Multinomial; posterior mean; bootstrap p05/p95) ===")
    for r in rows_out:
        print(
            f"{r['algo']:10s} status={r['status']:<12s} "
            f"PSO_global={r['P_PSO_global']:.3f} [{r['P_PSO_global_p05']:.3f},{r['P_PSO_global_p95']:.3f}]  "
            f"PSO_local={r['P_PSO_local']:.3f} [{r['P_PSO_local_p05']:.3f},{r['P_PSO_local_p95']:.3f}]  "
            f"Evolution={r['P_Evolution']:.3f} [{r['P_Evolution_p05']:.3f},{r['P_Evolution_p95']:.3f}]  "
            f"(wins={r['total_top1_wins']}, hits={r['total_top3_hits']})"
        )

    print("\nNotes:")
    print("- Buckets are (Top1, Top2-3, Others). This avoids double counting Top1 inside Top3.")
    print("- If an algorithm has low wins/hits, posteriors will remain wide/unstable (as they should).")


if __name__ == "__main__":
    main()
