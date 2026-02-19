#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
success_profile_analysis_v2_2.py  (Windows-safe, no self-writing)

Works with your CSV schema (no 'view' column required). It auto-creates:
- ABS view: raw per-instance values (default: beta_mean)
- REL view: per-instance normalized across algorithms (default: zscore)

Membership is variant-level (algo_variant) and uncertainty is via instance-bootstrap.

Example (PowerShell, single line):
python success_profile_analysis_v2_2.py --in_csv out_succ_v2/instance_algo_budget_summary.csv --out_dir out_succ_v2_analysis_v2_2 --value_col beta_mean --make_views auto --rel_norm zscore --anchor_select rel_medoid --tau 6.0 --n_boot 400
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Distances / transforms
# ----------------------------

def cosine_distance_matrix(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(norms, eps, None)
    sim = Xn @ Xn.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def l2_distance_matrix(X: np.ndarray) -> np.ndarray:
    G = X @ X.T
    sq = np.sum(X * X, axis=1, keepdims=True)
    D2 = np.clip(sq + sq.T - 2 * G, 0.0, None)
    D = np.sqrt(D2)
    np.fill_diagonal(D, 0.0)
    return D


def transform_values(X: np.ndarray, mode: str, clip_eps: float = 1e-3) -> np.ndarray:
    if mode == "none":
        return X
    if mode == "logit_clip":
        Xc = np.clip(X, clip_eps, 1 - clip_eps)
        return np.log(Xc / (1 - Xc))
    if mode == "zscore_row":
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-12
        return (X - mu) / sd
    raise ValueError(f"Unknown transform: {mode}")


def make_rel_view(X: np.ndarray, rel_norm: str) -> np.ndarray:
    """
    REL view = normalize per instance (per column) across algorithms.
    """
    if rel_norm == "zscore":
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True) + 1e-12
        return (X - mu) / sd
    if rel_norm == "minmax":
        lo = X.min(axis=0, keepdims=True)
        hi = X.max(axis=0, keepdims=True)
        return (X - lo) / (hi - lo + 1e-12)
    if rel_norm == "rank":
        # higher is better -> rank descending; map to [0,1] where 1=best
        ranks = np.zeros_like(X)
        for j in range(X.shape[1]):
            order = np.argsort(-X[:, j])
            r = np.empty(X.shape[0], dtype=float)
            r[order] = np.arange(X.shape[0], dtype=float)
            ranks[:, j] = 1.0 - r / max(1, X.shape[0] - 1)
        return ranks
    raise ValueError(f"Unknown rel_norm: {rel_norm}")


def instance_weights(Z: np.ndarray, weighting: str, floor: float = 0.05) -> np.ndarray:
    if weighting == "none":
        return np.ones(Z.shape[1], dtype=float)
    if weighting == "variance":
        v = np.var(Z, axis=0)
        if np.mean(v) > 0:
            v = v / (np.mean(v) + 1e-12)
        else:
            v = np.ones_like(v)
        return np.clip(v, floor, None)
    raise ValueError(f"Unknown instance_weighting: {weighting}")


# ----------------------------
# Clustering (dependency-free)
# ----------------------------

def hierarchical_clusters(dist: np.ndarray, k: int) -> List[List[int]]:
    n = dist.shape[0]
    clusters: List[List[int]] = [[i] for i in range(n)]
    if k >= n:
        return clusters

    def cdist(a: List[int], b: List[int]) -> float:
        return float(np.mean(dist[np.ix_(a, b)]))

    while len(clusters) > k:
        best_pair = None
        best_d = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = cdist(clusters[i], clusters[j])
                if d < best_d:
                    best_d = d
                    best_pair = (i, j)
        i, j = best_pair
        merged = clusters[i] + clusters[j]
        clusters.pop(j)
        clusters.pop(i)
        clusters.append(merged)

    return sorted(clusters, key=lambda c: min(c))


def nearest_neighbors(dist: np.ndarray, names: List[str], topk: int = 2) -> pd.DataFrame:
    rows = []
    n = dist.shape[0]
    for i in range(n):
        idx = np.argsort(dist[i])
        idx = [j for j in idx if j != i][:topk]
        for rank, j in enumerate(idx, 1):
            rows.append({
                "algo_variant": names[i],
                "rank": rank,
                "neighbor": names[j],
                "distance": float(dist[i, j]),
            })
    return pd.DataFrame(rows)


# ----------------------------
# Family / anchor logic
# ----------------------------

def default_family_patterns() -> Dict[str, Dict[str, List[str]]]:
    return {
        "bin": {
            "Evolution": [r"^GA\("],
            "LocalSearch": [r"^HC\(", r"^SA\("],
            "Random": [r"^RS_BIN$"],
        },
        "cont": {
            "Evolution": [r"^DE\("],
            "PSO_global": [r"^PSO_STD\("],
            "PSO_local": [r"^PSO_RING\("],
            "Random": [r"^RS_CONT$"],
        }
    }


def match_variants(algos: List[str], patterns: List[str]) -> List[int]:
    idxs = []
    for i, a in enumerate(algos):
        for p in patterns:
            if re.search(p, a):
                idxs.append(i)
                break
    return sorted(set(idxs))


def medoid_index(dist: np.ndarray, idxs: List[int]) -> int:
    if len(idxs) == 1:
        return idxs[0]
    sub = dist[np.ix_(idxs, idxs)]
    s = np.sum(sub, axis=1)
    return idxs[int(np.argmin(s))]


def choose_anchors(
    algos: List[str],
    dist_rel: np.ndarray,
    family_patterns: Dict[str, List[str]],
    anchor_select: str,
    abs_scores: Optional[np.ndarray],
    min_abs_score: float,
) -> Dict[str, str]:
    anchors: Dict[str, str] = {}
    for fam, pats in family_patterns.items():
        cand = match_variants(algos, pats)
        if not cand:
            continue

        if anchor_select == "rel_medoid":
            anchors[fam] = algos[medoid_index(dist_rel, cand)]
        elif anchor_select == "abs_top":
            if abs_scores is None:
                raise ValueError("abs_top requires ABS scores available.")
            best = max(cand, key=lambda i: abs_scores[i])
            anchors[fam] = algos[best]
        elif anchor_select == "hybrid":
            if abs_scores is None:
                raise ValueError("hybrid requires ABS scores available.")
            best = max(cand, key=lambda i: abs_scores[i])
            if abs_scores[best] < min_abs_score:
                best = medoid_index(dist_rel, cand)
            anchors[fam] = algos[best]
        else:
            raise ValueError(f"Unknown anchor_select: {anchor_select}")

    return anchors


# ----------------------------
# Membership (bootstrap on instances)
# ----------------------------

def softmax_negdist(d: np.ndarray, tau: float) -> np.ndarray:
    tau = max(1e-12, float(tau))
    x = -d / tau
    x = x - np.max(x)
    w = np.exp(x)
    s = np.sum(w)
    return w / s if s > 0 else np.ones_like(w) / len(w)


def membership_point(dist: np.ndarray, algos: List[str], anchors: Dict[str, str], tau: float) -> Tuple[np.ndarray, List[str]]:
    fams = list(anchors.keys())
    aidx = [algos.index(anchors[f]) for f in fams]
    M = np.zeros((len(algos), len(fams)), dtype=float)
    for i in range(len(algos)):
        d = np.array([dist[i, j] for j in aidx], dtype=float)
        M[i] = softmax_negdist(d, tau)
    return M, fams


def membership_bootstrap_instances(
    X: np.ndarray,
    metric: str,
    algos: List[str],
    anchors: Dict[str, str],
    tau: float,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    dist_full = cosine_distance_matrix(X) if metric == "cosine" else l2_distance_matrix(X)
    _, fams = membership_point(dist_full, algos, anchors, tau=tau)

    if n_boot <= 0:
        M0, _ = membership_point(dist_full, algos, anchors, tau=tau)
        return M0, M0, M0, fams

    n_algos, d = X.shape
    boots = np.zeros((n_boot, n_algos, len(fams)), dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, d, size=d)
        Xb = X[:, idx]
        dist_b = cosine_distance_matrix(Xb) if metric == "cosine" else l2_distance_matrix(Xb)
        Mb, _ = membership_point(dist_b, algos, anchors, tau=tau)
        boots[b] = Mb

    mean = boots.mean(axis=0)
    p05 = np.quantile(boots, 0.05, axis=0)
    p95 = np.quantile(boots, 0.95, axis=0)
    return mean, p05, p95, fams


# ----------------------------
# Data preparation
# ----------------------------

def build_pivot(df: pd.DataFrame, instance_col: str, algo_col: str, value_col: str) -> Tuple[np.ndarray, List[str], List[str]]:
    piv = df.pivot_table(index=algo_col, columns=instance_col, values=value_col, aggfunc="mean")
    piv = piv.sort_index(axis=0).sort_index(axis=1)
    X = piv.values.astype(float)
    if np.isnan(X).any():
        cm = np.nanmean(X, axis=0)
        cm = np.where(np.isnan(cm), 0.5, cm)
        ii = np.where(np.isnan(X))
        X[ii] = np.take(cm, ii[1])
    return X, list(piv.index), list(piv.columns)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--value_col", type=str, default="beta_mean")
    ap.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    ap.add_argument("--value_transform", choices=["logit_clip", "zscore_row", "none"], default="logit_clip")
    ap.add_argument("--instance_weighting", choices=["variance", "none"], default="variance")
    ap.add_argument("--rel_norm", choices=["zscore", "minmax", "rank"], default="zscore")
    ap.add_argument("--make_views", choices=["auto", "both", "abs_only", "rel_only", "none"], default="auto")
    ap.add_argument("--k_clusters", type=int, default=3)
    ap.add_argument("--tau", type=float, default=6.0)

    ap.add_argument("--n_boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print_top_neighbors", type=int, default=2)

    ap.add_argument("--family_json", type=str, default="")
    ap.add_argument("--anchors_json", type=str, default="")
    ap.add_argument("--anchor_select", choices=["rel_medoid", "abs_top", "hybrid"], default="rel_medoid")
    ap.add_argument("--anchor_min_abs", type=float, default=0.10)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    if args.value_col not in df.columns:
        raise ValueError(f"--value_col {args.value_col!r} not found. Available: {list(df.columns)}")

    required = ["instance_id", "domain", "budget", "algo_variant"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found: {list(df.columns)}")

    family_map = default_family_patterns()
    if args.family_json.strip():
        family_map = json.loads(args.family_json)

    anchors_override: Dict[str, Dict[str, str]] = {}
    if args.anchors_json.strip():
        anchors_override = json.loads(args.anchors_json)

    slices = sorted(df[["domain", "budget"]].drop_duplicates().itertuples(index=False, name=None))
    print(f"[INFO] loaded {len(df)} rows from {in_csv}")
    print(f"[INFO] found {len(slices)} slices (domain,budget)")

    overview_rows = []

    for domain, budget in slices:
        df_db = df[(df["domain"] == domain) & (df["budget"] == budget)].copy()

        # Build ABS matrix from value_col
        X_abs_raw, algos, insts = build_pivot(df_db, "instance_id", "algo_variant", args.value_col)

        # Decide views (since your file has no 'view', we synthesize by default)
        make = args.make_views
        if make in ("auto", "both"):
            views_to_run = ["ABS", "REL"]
        elif make == "abs_only":
            views_to_run = ["ABS"]
        elif make == "rel_only":
            views_to_run = ["REL"]
        else:
            views_to_run = ["REL"]

        X_by_view = {"ABS": X_abs_raw, "REL": make_rel_view(X_abs_raw, args.rel_norm)}

        # REL distance used for anchor selection
        Z_rel = transform_values(X_by_view["REL"], args.value_transform)
        w_rel = instance_weights(Z_rel, args.instance_weighting)
        Z_rel_w = Z_rel * w_rel[None, :]
        dist_rel = cosine_distance_matrix(Z_rel_w) if args.metric == "cosine" else l2_distance_matrix(Z_rel_w)

        abs_scores = np.mean(X_by_view["ABS"], axis=1)

        fam_patterns = family_map.get(domain, {})
        if domain in anchors_override and anchors_override[domain]:
            anchors = anchors_override[domain]
            anchor_mode = "override"
        else:
            anchors = choose_anchors(
                algos=algos,
                dist_rel=dist_rel,
                family_patterns=fam_patterns,
                anchor_select=args.anchor_select,
                abs_scores=abs_scores,
                min_abs_score=args.anchor_min_abs,
            )
            anchor_mode = args.anchor_select

        for view in views_to_run:
            X_raw = X_by_view[view]
            Z = transform_values(X_raw, args.value_transform)
            w = instance_weights(Z, args.instance_weighting)
            Z_w = Z * w[None, :]

            dist = cosine_distance_matrix(Z_w) if args.metric == "cosine" else l2_distance_matrix(Z_w)
            clusters = hierarchical_clusters(dist, k=args.k_clusters)
            nn = nearest_neighbors(dist, algos, topk=args.print_top_neighbors)

            meanM, p05M, p95M, fams = membership_bootstrap_instances(
                X=Z_w,
                metric=args.metric,
                algos=algos,
                anchors=anchors,
                tau=args.tau,
                n_boot=args.n_boot,
                rng=rng,
            )

            print("\n=== Slice ===")
            print(f"view={view}  domain={domain} budget={budget}  algos={len(algos)} instances={len(insts)}")
            print(f"anchor_mode={anchor_mode}  value_col={args.value_col}  rel_norm={args.rel_norm}  transform={args.value_transform}")
            print("Anchors (variant):")
            for f, a in anchors.items():
                print(f"  {f}: {a}")

            print("Cluster summary:")
            for ci, cidx in enumerate(clusters):
                names = ", ".join(algos[i] for i in cidx)
                print(f"  Cluster {ci}: {names}")

            tag = f"{domain}_B{int(budget)}_{view}"
            pd.DataFrame(dist, index=algos, columns=algos).to_csv(out_dir / f"algo_distance_{tag}.csv", index=True)
            nn.to_csv(out_dir / f"neighbors_{tag}.csv", index=False)

            cl_rows = []
            for ci, cidx in enumerate(clusters):
                for i in cidx:
                    cl_rows.append({"cluster": ci, "algo_variant": algos[i]})
            pd.DataFrame(cl_rows).to_csv(out_dir / f"clusters_{tag}.csv", index=False)

            mem_rows = []
            for i, a in enumerate(algos):
                rec = {
                    "domain": domain,
                    "budget": budget,
                    "view": view,
                    "algo_variant": a,
                    "anchor_mode": anchor_mode,
                    "value_col": args.value_col,
                    "rel_norm": args.rel_norm,
                    "value_transform": args.value_transform,
                    "instance_weighting": args.instance_weighting,
                    "metric": args.metric,
                    "tau": args.tau,
                    "n_boot": args.n_boot,
                    "anchors_json": json.dumps(anchors, ensure_ascii=False),
                }
                for k, f in enumerate(fams):
                    rec[f"{f}_mean"] = float(meanM[i, k])
                    rec[f"{f}_p05"] = float(p05M[i, k])
                    rec[f"{f}_p95"] = float(p95M[i, k])
                mem_rows.append(rec)

            pd.DataFrame(mem_rows).to_csv(out_dir / f"membership_{tag}.csv", index=False)

            overview_rows.append({
                "domain": domain,
                "budget": budget,
                "view": view,
                "n_algos": len(algos),
                "n_instances": len(insts),
                "anchor_mode": anchor_mode,
                "anchors_json": json.dumps(anchors, ensure_ascii=False),
                "value_col": args.value_col,
                "rel_norm": args.rel_norm,
                "value_transform": args.value_transform,
                "instance_weighting": args.instance_weighting,
                "metric": args.metric,
                "tau": args.tau,
                "n_boot": args.n_boot,
            })

    pd.DataFrame(overview_rows).to_csv(out_dir / "summary_overview.csv", index=False)
    print(f"\n[OK] wrote {out_dir / 'summary_overview.csv'}")
    print("[DONE] v2.2 analysis complete.")


if __name__ == "__main__":
    main()
