#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
success_profile_analysis_v2_2.py

Analyze algorithm behavioral similarity from per-instance success profiles and compute
family membership using variant-level anchors.

Key change vs v2:
- Anchors are selected by default as REL-view medoids (behavioral representatives),
  not by ABS best performance. This stabilizes family semantics and removes the
  BIN@500 "HC/SA looks Random/Evolution" artifact you observed.

Input
-----
CSV: <out_dir>/instance_algo_budget_summary.csv
Required columns (names can vary; script tries to infer):
  - domain        (bin/cont)
  - budget        (int)
  - instance_id   (string or int)
  - algo_variant  (string; algorithm with hyperparams, i.e., "variant")
  - view          ("ABS" or "REL")
  - value         (numeric; e.g., beta_mean or succ_rate)

Outputs
-------
Writes into --out_dir:
  - summary_overview.csv
  - algo_distance_<domain>_B<budget>_<view>.csv
  - neighbors_<domain>_B<budget>_<view>.csv
  - clusters_<domain>_B<budget>_<view>.csv
  - membership_<domain>_B<budget>_<view>.csv

Example
-------
python success_profile_analysis_v2_2.py \
  --in_csv out_succ_v2/instance_algo_budget_summary.csv \
  --out_dir out_succ_v2_analysis_v2_1 \
  --anchor_select rel_medoid \
  --tau 6.0
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
# Core utilities
# ----------------------------

def softmax_negdist(d: np.ndarray, tau: float) -> np.ndarray:
    """Convert distances to weights via softmax(-d/tau)."""
    tau = max(1e-12, float(tau))
    x = -d / tau
    x = x - np.max(x)
    w = np.exp(x)
    s = np.sum(w)
    return w / s if s > 0 else np.ones_like(w) / len(w)


def cosine_distance_matrix(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Cosine distance (1 - cosine similarity). X: [n, d]."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(norms, eps, None)
    sim = Xn @ Xn.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    return dist


def l2_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance. X: [n, d]."""
    G = X @ X.T
    sq = np.sum(X * X, axis=1, keepdims=True)
    D2 = np.clip(sq + sq.T - 2 * G, 0.0, None)
    D = np.sqrt(D2)
    np.fill_diagonal(D, 0.0)
    return D


def hierarchical_clusters(dist: np.ndarray, k: int) -> List[List[int]]:
    """
    Dependency-free hierarchical clustering (average linkage greedy merge).
    Good enough for MVP (use scipy if you want stronger clustering).
    """
    n = dist.shape[0]
    clusters: List[List[int]] = [[i] for i in range(n)]
    if k >= n:
        return clusters

    def cdist(a: List[int], b: List[int]) -> float:
        return float(np.mean(dist[np.ix_(a, b)]))

    while len(clusters) > k:
        best = None
        best_d = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = cdist(clusters[i], clusters[j])
                if d < best_d:
                    best_d = d
                    best = (i, j)
        i, j = best
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
# Data handling
# ----------------------------

def infer_cols(df: pd.DataFrame) -> Dict[str, str]:
    cols = {c.lower(): c for c in df.columns}

    def pick(*cands):
        for c in cands:
            if c in cols:
                return cols[c]
        return None

    out = {
        "domain": pick("domain"),
        "budget": pick("budget"),
        "instance": pick("instance_id", "instance", "problem_instance", "pid"),
        "algo": pick("algo_variant", "variant", "algo", "algorithm"),
        "view": pick("view"),
        "value": pick("value", "beta_mean", "succ_rate", "success_rate", "p_success"),
    }
    missing = [k for k in ("domain", "budget", "instance", "algo", "value") if out.get(k) is None]
    if missing:
        raise ValueError(f"Missing required columns {missing}. Found: {list(df.columns)}")
    if out["view"] is None:
        # allow missing view -> treat all as REL
        df["view"] = "REL"
        out["view"] = "view"
    return out


def make_feature_matrix(
    sdf: pd.DataFrame,
    c_inst: str,
    c_algo: str,
    c_val: str,
    value_transform: str = "logit_clip",
    clip_eps: float = 1e-3,
    instance_weighting: str = "variance",
    weight_floor: float = 0.05,
) -> Tuple[np.ndarray, List[str], List[str], np.ndarray]:
    """
    Pivot to X[algo, instance], transform, and optionally weight instances.
    Returns: Xw, algos, insts, inst_weights
    """
    piv = sdf.pivot_table(index=c_algo, columns=c_inst, values=c_val, aggfunc="mean")
    piv = piv.sort_index(axis=0).sort_index(axis=1)

    algos = list(piv.index)
    insts = list(piv.columns)
    X = piv.values.astype(float)

    # fill missing with column mean; fallback 0.5
    if np.isnan(X).any():
        cm = np.nanmean(X, axis=0)
        cm = np.where(np.isnan(cm), 0.5, cm)
        ii = np.where(np.isnan(X))
        X[ii] = np.take(cm, ii[1])

    if value_transform == "none":
        Z = X
    elif value_transform == "zscore":
        mu = X.mean(axis=1, keepdims=True)
        sd = X.std(axis=1, keepdims=True) + 1e-12
        Z = (X - mu) / sd
    elif value_transform == "logit_clip":
        Xc = np.clip(X, clip_eps, 1 - clip_eps)
        Z = np.log(Xc / (1 - Xc))
    else:
        raise ValueError(f"Unknown value_transform: {value_transform}")

    w = np.ones(Z.shape[1], dtype=float)
    if instance_weighting == "none":
        pass
    elif instance_weighting == "variance":
        v = np.var(Z, axis=0)
        if np.mean(v) > 0:
            v = v / (np.mean(v) + 1e-12)
        else:
            v = np.ones_like(v)
        w = np.clip(v, weight_floor, None)
    else:
        raise ValueError(f"Unknown instance_weighting: {instance_weighting}")

    return Z * w[None, :], algos, insts, w


# ----------------------------
# Family / anchor logic
# ----------------------------

def default_family_patterns() -> Dict[str, Dict[str, List[str]]]:
    """
    Family definitions by regex on algo_variant.
    Override via --family_json if needed.
    """
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
    """
    Returns {family: anchor_algo_variant}.
    rel_medoid: pick within-family medoid in REL space
    abs_top: pick max ABS mean within family
    hybrid: abs_top if >= min_abs_score else rel_medoid
    """
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


def membership_from_anchors(
    dist: np.ndarray,
    algos: List[str],
    anchors: Dict[str, str],
    tau: float,
    n_boot: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Membership via softmax(-dist/tau) over anchor points.
    Uncertainty: pragmatic noise-bootstrap on distances (keeps MVP dependency-free).
    """
    fams = list(anchors.keys())
    aidx = [algos.index(anchors[f]) for f in fams]

    rows = []
    for i, a in enumerate(algos):
        d = np.array([dist[i, j] for j in aidx], dtype=float)
        m = softmax_negdist(d, tau)

        boot = []
        for _ in range(n_boot):
            noise = rng.normal(0.0, 0.02, size=d.shape)
            mb = softmax_negdist(np.clip(d + noise, 0.0, None), tau)
            boot.append(mb)
        boot = np.stack(boot, axis=0)
        lo = np.quantile(boot, 0.05, axis=0)
        hi = np.quantile(boot, 0.95, axis=0)

        rec = {"algo_variant": a}
        for k, f in enumerate(fams):
            rec[f"{f}_mean"] = float(m[k])
            rec[f"{f}_p05"] = float(lo[k])
            rec[f"{f}_p95"] = float(hi[k])
        rows.append(rec)

    return pd.DataFrame(rows)


# ----------------------------
# Runner
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--k_clusters", type=int, default=3)
    ap.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    ap.add_argument("--value_transform", choices=["logit_clip", "zscore", "none"], default="logit_clip")
    ap.add_argument("--instance_weighting", choices=["variance", "none"], default="variance")
    ap.add_argument("--tau", type=float, default=6.0)
    ap.add_argument("--n_boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--print_top_neighbors", type=int, default=2)

    ap.add_argument("--family_json", type=str, default="",
                    help='JSON override: {"bin":{"Evolution":["^GA\\("],...},"cont":{...}}')
    ap.add_argument("--anchors_json", type=str, default="",
                    help='Explicit anchors: {"bin":{"Evolution":"GA(...)",...},"cont":{...}}')
    ap.add_argument("--anchor_select", choices=["rel_medoid", "abs_top", "hybrid"], default="rel_medoid")
    ap.add_argument("--anchor_min_abs", type=float, default=0.10)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    print(f"[INFO] loaded {len(df)} rows from {in_csv}")

    col = infer_cols(df)
    c_domain, c_budget, c_inst, c_algo, c_view, c_val = (
        col["domain"], col["budget"], col["instance"], col["algo"], col["view"], col["value"]
    )

    family_map = default_family_patterns()
    if args.family_json.strip():
        family_map = json.loads(args.family_json)

    anchors_override: Dict[str, Dict[str, str]] = {}
    if args.anchors_json.strip():
        anchors_override = json.loads(args.anchors_json)

    slices = sorted(df[[c_domain, c_budget]].drop_duplicates().itertuples(index=False, name=None))
    print(f"[INFO] found {len(slices)} slices (domain,budget)")

    overview_rows = []

    for domain, budget in slices:
        df_db = df[(df[c_domain] == domain) & (df[c_budget] == budget)].copy()

        # Build REL distance for anchor selection (if REL not present, fallback to current view)
        df_rel = df_db[df_db[c_view] == "REL"].copy()
        if df_rel.empty:
            df_rel = df_db.copy()

        Xrel, algos_rel, insts_rel, _ = make_feature_matrix(
            df_rel, c_inst, c_algo, c_val,
            value_transform=args.value_transform,
            instance_weighting=args.instance_weighting,
        )
        dist_rel = cosine_distance_matrix(Xrel) if args.metric == "cosine" else l2_distance_matrix(Xrel)

        # ABS scores for abs_top/hybrid
        abs_scores = None
        df_abs = df_db[df_db[c_view] == "ABS"].copy()
        if not df_abs.empty:
            piv_abs = df_abs.pivot_table(index=c_algo, columns=c_inst, values=c_val, aggfunc="mean")
            piv_abs = piv_abs.reindex(index=algos_rel)
            abs_scores = np.nanmean(piv_abs.values.astype(float), axis=1)

        # Anchors
        fam_patterns = family_map.get(domain, {})
        if domain in anchors_override and anchors_override[domain]:
            anchors = anchors_override[domain]
            anchor_mode = "override"
        else:
            anchors = choose_anchors(
                algos=algos_rel,
                dist_rel=dist_rel,
                family_patterns=fam_patterns,
                anchor_select=args.anchor_select,
                abs_scores=abs_scores,
                min_abs_score=args.anchor_min_abs,
            )
            anchor_mode = args.anchor_select

        # Process each view separately for distances/clusters/membership (but same anchors)
        for view in ["ABS", "REL"]:
            sdf = df_db[df_db[c_view] == view].copy()
            if sdf.empty:
                continue

            X, algos, insts, _ = make_feature_matrix(
                sdf, c_inst, c_algo, c_val,
                value_transform=args.value_transform,
                instance_weighting=args.instance_weighting,
            )

            # align order to algos_rel
            if algos != algos_rel:
                # reindex pivot to algos_rel
                piv = sdf.pivot_table(index=c_algo, columns=c_inst, values=c_val, aggfunc="mean")
                piv = piv.reindex(index=algos_rel).sort_index(axis=1)
                X = piv.values.astype(float)
                if np.isnan(X).any():
                    cm = np.nanmean(X, axis=0)
                    cm = np.where(np.isnan(cm), 0.5, cm)
                    ii = np.where(np.isnan(X))
                    X[ii] = np.take(cm, ii[1])
                if args.value_transform == "logit_clip":
                    Xc = np.clip(X, 1e-3, 1 - 1e-3)
                    X = np.log(Xc / (1 - Xc))
                elif args.value_transform == "zscore":
                    mu = X.mean(axis=1, keepdims=True)
                    sd = X.std(axis=1, keepdims=True) + 1e-12
                    X = (X - mu) / sd
                algos = algos_rel

            dist = cosine_distance_matrix(X) if args.metric == "cosine" else l2_distance_matrix(X)
            clusters = hierarchical_clusters(dist, k=args.k_clusters)
            nn = nearest_neighbors(dist, algos, topk=args.print_top_neighbors)
            mem = membership_from_anchors(dist, algos, anchors, tau=args.tau, n_boot=args.n_boot, rng=rng)

            # Print slice summary
            print("\n=== Slice ===")
            print(f"view={view}  domain={domain} budget={budget}  algos={len(algos)} instances={len(insts)}")
            print("Cluster summary:")
            for ci, cidx in enumerate(clusters):
                names = ", ".join(algos[i] for i in cidx)
                print(f"  Cluster {ci}: {names}")
            print("Anchors (variant):")
            for f, a in anchors.items():
                print(f"  {f}: {a}")

            # Save artifacts
            tag = f"{domain}_B{int(budget)}_{view}"
            pd.DataFrame(dist, index=algos, columns=algos).to_csv(out_dir / f"algo_distance_{tag}.csv", index=True)
            nn.to_csv(out_dir / f"neighbors_{tag}.csv", index=False)

            cl_rows = []
            for ci, cidx in enumerate(clusters):
                for i in cidx:
                    cl_rows.append({"cluster": ci, "algo_variant": algos[i]})
            pd.DataFrame(cl_rows).to_csv(out_dir / f"clusters_{tag}.csv", index=False)

            mem["domain"] = domain
            mem["budget"] = budget
            mem["view"] = view
            mem.to_csv(out_dir / f"membership_{tag}.csv", index=False)

            overview_rows.append({
                "domain": domain,
                "budget": budget,
                "view": view,
                "n_algos": len(algos),
                "n_instances": len(insts),
                "anchor_mode": anchor_mode,
                "anchors": json.dumps(anchors, ensure_ascii=False),
            })

    pd.DataFrame(overview_rows).to_csv(out_dir / "summary_overview.csv", index=False)
    print(f"\n[OK] wrote {out_dir / 'summary_overview.csv'}")
    print("[DONE] v2.1 analysis complete.")


if __name__ == "__main__":
    main()
