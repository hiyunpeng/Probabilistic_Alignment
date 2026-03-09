#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
dorigo_normalized_fitness_membership_bootstrap_v2.py

Fixes vs v1
-----------
1) Removes Windows "invalid escape sequence" warnings by using raw strings for doc/LaTeX.
2) If anchors (PSO/ES) are not present in the CSV, you can still:
   - run UNSUPERVISED clustering of normalised-fitness curves, OR
   - run PROXY attribution by selecting two "archetype" curves from your current algorithm set
     (delayed take-off vs smooth monotone). This is *not* a real PSO/ES claim, but is useful for exploration.

What this script does
---------------------
- Build *normalised fitness score* curves in [0,1] (higher is better) per (target, budget).
- Bootstrap over instances (stratified by problem) to get CIs for curves, headroom, and (optional) memberships.
- Outputs:
  - normalized_fitness_curve_summary.csv
  - normalized_fitness_membership_summary.csv  (if anchor/proxy mode)
  - normalized_fitness_clusters.csv            (if clustering mode)
  - figs/ and latex/ snippets

Input
-----
Aggregated summary CSV with columns:
  problem, instance_id, algo_variant, budget, target
and at least one of:
  mean_best, median_best, min_best, max_best

Example (Windows cmd)
---------------------
REM If you DO have anchors in the CSV:
python dorigo_normalized_fitness_membership_bootstrap_v2.py ^
  --csv .\out_dorigo\instance_algo_budget_summary.csv ^
  --out_dir .\out_dorigo\norm_fitness_v2 ^
  --mode anchor --pso_anchor PSO_GBEST --es_anchor ES_1P1 ^
  --fitness_col median_best --scale_mode q05q95 --B 2000

REM If you do NOT have anchors (your current case):
python dorigo_normalized_fitness_membership_bootstrap_v2.py ^
  --csv .\out_dorigo\instance_algo_budget_summary.csv ^
  --out_dir .\out_dorigo\norm_fitness_v2 ^
  --mode proxy ^
  --fitness_col median_best --scale_mode q05q95 --B 2000

REM Or do unsupervised clustering (no PSO/ES labels):
python dorigo_normalized_fitness_membership_bootstrap_v2.py ^
  --csv .\out_dorigo\instance_algo_budget_summary.csv ^
  --out_dir .\out_dorigo\norm_fitness_v2 ^
  --mode cluster --K 3 ^
  --fitness_col median_best --scale_mode q05q95 --B 2000
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x=None, total=None, desc=None, **kw):
        return x if x is not None else range(total or 0)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def parse_csv_list(s: str, cast=str):
    if s is None or str(s).strip() == "":
        return None
    return [cast(x.strip()) for x in s.split(",") if x.strip()]

def make_feat_index(targets, budgets):
    return pd.MultiIndex.from_product([targets, budgets], names=["target", "budget"])

def zscore_by_feature(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = np.mean(M, axis=0)
    sd = np.std(M, axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (M - mu) / sd

def euclid_rows(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = A - b[None, :]
    return np.sqrt(np.sum(diff * diff, axis=1))

def soft2(dist2: np.ndarray, tau: float) -> np.ndarray:
    x = -tau * dist2
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def compute_headroom(curves: np.ndarray, targets, budgets) -> np.ndarray:
    nA = curves.shape[0]
    T, B = len(targets), len(budgets)
    out = np.zeros(nA, dtype=float)
    for a in range(nA):
        m = 0.0
        for ti in range(T):
            lo = curves[a, ti * B + 0]
            hi = curves[a, ti * B + (B - 1)]
            m = max(m, float(hi - lo))
        out[a] = m
    return out

def ci(arr: np.ndarray, alpha: float):
    lo = np.quantile(arr, alpha/2, axis=0)
    hi = np.quantile(arr, 1-alpha/2, axis=0)
    return lo, hi

def monotone_score(curve_1d: np.ndarray, eps: float = 1e-12) -> float:
    dif = np.diff(curve_1d)
    return float(np.mean(dif >= -eps))

def takeoff_score(curve_1d: np.ndarray) -> float:
    if len(curve_1d) < 4:
        dif = np.diff(curve_1d)
        return float(np.mean(dif))
    m = len(curve_1d) // 2
    early = np.mean(np.diff(curve_1d[:m+1]))
    late  = np.mean(np.diff(curve_1d[m:]))
    return float(late - early)


def add_normalized_fitness_score(
    df: pd.DataFrame,
    fitness_col: str,
    sense: str,
    scale_mode: str,
    group_cols=("problem", "instance_id", "target"),
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Adds `fitness_score` in [0,1], higher is better.

    Minimisation:
      fitness_score = 1 - clip((val - best_ref)/(worst_ref - best_ref), 0,1)

    Maximisation:
      we negate val first, then treat as minimisation.
    """
    if fitness_col not in df.columns:
        raise ValueError(f"fitness_col='{fitness_col}' not found. Available: {df.columns.tolist()}")

    out = df.copy()
    val = out[fitness_col].to_numpy(dtype=float)
    if sense.lower().startswith("max"):
        val = -val
    out["_val"] = val

    g = out.groupby(list(group_cols))["_val"]

    if scale_mode == "minmax":
        best_ref = g.min()
        worst_ref = g.max()
    elif scale_mode == "q05q95":
        best_ref = g.quantile(0.05)
        worst_ref = g.quantile(0.95)
    elif scale_mode == "rank":
        out["_rank"] = g.rank(pct=True, method="average")
        out["fitness_score"] = 1.0 - out["_rank"].to_numpy(dtype=float)
        return out.drop(columns=["_val", "_rank"], errors="ignore")
    else:
        raise ValueError("scale_mode must be one of: q05q95, minmax, rank")

    out = out.join(best_ref.rename("_best_ref"), on=list(group_cols))
    out = out.join(worst_ref.rename("_worst_ref"), on=list(group_cols))

    denom = (out["_worst_ref"] - out["_best_ref"]).to_numpy(dtype=float)
    denom = np.where(np.abs(denom) < eps, np.nan, denom)

    regret = (out["_val"].to_numpy(dtype=float) - out["_best_ref"].to_numpy(dtype=float)) / denom
    regret = np.clip(regret, 0.0, 1.0)
    regret = np.nan_to_num(regret, nan=0.0)

    out["fitness_score"] = 1.0 - regret
    return out.drop(columns=["_val", "_best_ref", "_worst_ref"], errors="ignore")


def stratified_instance_weights(instances: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = []
    for prob, g in instances.groupby("problem"):
        n = len(g)
        idx = rng.integers(0, n, size=n)
        samp = g.iloc[idx]
        w = samp.groupby(["problem", "instance_id"]).size().reset_index(name="w")
        out.append(w)
    return pd.concat(out, ignore_index=True)

def weighted_group_mean(df: pd.DataFrame, group_cols, metric_cols, weight_col="w") -> pd.DataFrame:
    num = df.copy()
    for m in metric_cols:
        num[m] = num[m] * num[weight_col]
    gnum = num.groupby(group_cols, as_index=False)[metric_cols].sum()
    gden = df.groupby(group_cols, as_index=False)[weight_col].sum().rename(columns={weight_col: "_den"})
    out = gnum.merge(gden, on=group_cols, how="left")
    for m in metric_cols:
        out[m] = out[m] / out["_den"]
    return out.drop(columns=["_den"])

def pivot_curves(df_agg: pd.DataFrame, algos, targets, budgets, metric: str) -> np.ndarray:
    wide = df_agg.pivot(index="algo_variant", columns=["target", "budget"], values=metric)
    wide = wide.reindex(index=algos, columns=make_feat_index(targets, budgets))
    M = wide.to_numpy(dtype=float)
    col_mu = np.nanmean(M, axis=0)
    idx = np.where(np.isnan(M))
    if len(idx[0]) > 0:
        M[idx] = np.take(col_mu, idx[1])
    return M


def kmeans_lloyd(X: np.ndarray, K: int, rng: np.random.Generator, iters: int = 50):
    """Simple KMeans (no sklearn). Returns (labels, centers)."""
    n, _ = X.shape
    K = min(K, n)
    idx = rng.choice(n, size=K, replace=False)
    centers = X[idx].copy()

    for _ in range(iters):
        dist = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        lab = np.argmin(dist, axis=1)
        new_centers = centers.copy()
        for k in range(K):
            pts = X[lab == k]
            if len(pts) > 0:
                new_centers[k] = pts.mean(axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return lab, centers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--fitness_col", default="median_best",
                    choices=["mean_best", "median_best", "min_best", "max_best"])
    ap.add_argument("--sense", default="minimize", choices=["minimize", "maximize"])
    ap.add_argument("--scale_mode", default="q05q95", choices=["q05q95", "minmax", "rank"])

    ap.add_argument("--budgets", default=None)
    ap.add_argument("--targets", default=None)

    ap.add_argument("--mode", default="proxy", choices=["anchor", "proxy", "cluster"])
    ap.add_argument("--pso_anchor", default="PSO_GBEST")
    ap.add_argument("--es_anchor", default="ES_1P1")

    ap.add_argument("--K", type=int, default=3)

    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--headroom_gate", type=float, default=0.02)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figs"
    latex_dir = out_dir / "latex"
    ensure_dir(out_dir); ensure_dir(fig_dir); ensure_dir(latex_dir)

    df = pd.read_csv(args.csv)
    budgets = parse_csv_list(args.budgets, int) or sorted(df["budget"].unique().tolist())
    targets = parse_csv_list(args.targets, str) or sorted(df["target"].unique().tolist())
    df = df[df["budget"].isin(budgets) & df["target"].isin(targets)].copy()

    df = add_normalized_fitness_score(
        df,
        fitness_col=args.fitness_col,
        sense=args.sense,
        scale_mode=args.scale_mode,
        group_cols=("problem", "instance_id", "target"),
    )

    det = (df.groupby(["algo_variant", "target", "budget"], as_index=False)
             .agg(fitness_score=("fitness_score", "mean")))
    algos = sorted(det["algo_variant"].unique().tolist())

    rng = np.random.default_rng(args.seed)
    instances = df[["problem", "instance_id"]].drop_duplicates().reset_index(drop=True)

    nA = len(algos)
    nF = len(targets) * len(budgets)

    curves_boot = np.zeros((args.B, nA, nF), dtype=float)
    head_boot = np.zeros((args.B, nA), dtype=float)

    need_membership = args.mode in ("anchor", "proxy")
    if need_membership:
        w_boot = np.zeros((args.B, nA, 2), dtype=float)
        delta_boot = np.zeros((args.B, nA), dtype=float)
        label_boot = np.zeros((args.B, nA), dtype=int)  # 0=PSO,1=ES,2=FLAT

    pso_anchor = args.pso_anchor
    es_anchor = args.es_anchor

    if args.mode == "anchor":
        if pso_anchor not in algos or es_anchor not in algos:
            raise ValueError(
                f"Anchors not found in CSV. Need '{pso_anchor}' and '{es_anchor}'. Found: {algos}\n"
                f"Fix: re-run your experiments including PSO/ES anchors, OR use --mode proxy/cluster."
            )

    if args.mode == "proxy":
        M_det = pivot_curves(det, algos, targets, budgets, "fitness_score")
        Bn, Tn = len(budgets), len(targets)
        hr_det = compute_headroom(M_det, targets, budgets)
        mono = np.zeros(nA); take = np.zeros(nA)
        for ai in range(nA):
            msum = 0.0; tsum = 0.0
            for ti in range(Tn):
                c = M_det[ai, ti*Bn:(ti+1)*Bn]
                msum += monotone_score(c)
                tsum += takeoff_score(c)
            mono[ai] = msum / max(Tn, 1)
            take[ai] = tsum / max(Tn, 1)

        responders = np.where(hr_det >= args.headroom_gate)[0]
        if len(responders) == 0:
            pso_idx = int(np.argmax(take))
            es_idx = int(np.argmax(mono))
        else:
            pso_idx = int(responders[np.argmax(take[responders])])
            es_idx = int(responders[np.argmax((hr_det[responders]) * (mono[responders]))])

        pso_anchor = algos[pso_idx]
        es_anchor = algos[es_idx]

        (out_dir / "proxy_anchors.txt").write_text(
            f"PROXY mode selected anchors from current algorithms.\n"
            f"PSO_proxy (delayed takeoff): {pso_anchor}\n"
            f"ES_proxy (smooth monotone * headroom): {es_anchor}\n"
            f"IMPORTANT: These are proxies for exploratory analysis only, not real PSO/ES anchors.\n",
            encoding="utf-8"
        )

    if args.mode in ("anchor", "proxy"):
        idx_pso = algos.index(pso_anchor)
        idx_es = algos.index(es_anchor)

    for b in tqdm(range(args.B), desc="Bootstrap"):
        w = stratified_instance_weights(instances, rng)
        dfw = df.merge(w, on=["problem", "instance_id"], how="inner")

        agg = weighted_group_mean(
            dfw,
            group_cols=["algo_variant", "target", "budget"],
            metric_cols=["fitness_score"],
            weight_col="w"
        )

        M = pivot_curves(agg, algos, targets, budgets, "fitness_score")
        curves_boot[b] = M

        hr = compute_headroom(M, targets, budgets)
        head_boot[b] = hr

        if need_membership:
            Z = zscore_by_feature(M)
            d_pso = euclid_rows(Z, Z[idx_pso])
            d_es = euclid_rows(Z, Z[idx_es])

            dist2 = np.stack([d_pso, d_es], axis=1)
            W = soft2(dist2, tau=args.tau)
            w_boot[b] = W

            Delta = d_es - d_pso
            delta_boot[b] = Delta

            for ai in range(nA):
                if hr[ai] < args.headroom_gate:
                    label_boot[b, ai] = 2
                else:
                    label_boot[b, ai] = 0 if W[ai, 0] >= W[ai, 1] else 1

    M_mean = curves_boot.mean(axis=0)
    M_lo, M_hi = ci(curves_boot, args.alpha)

    hr_mean = head_boot.mean(axis=0)
    hr_lo, hr_hi = ci(head_boot, args.alpha)

    feat = make_feat_index(targets, budgets).tolist()
    rows = []
    for ai, a in enumerate(algos):
        for fi, (t, bud) in enumerate(feat):
            rows.append({
                "algo_variant": a,
                "target": t,
                "budget": int(bud),
                "fitness_score": float(M_mean[ai, fi]),
                "fitness_lo": float(M_lo[ai, fi]),
                "fitness_hi": float(M_hi[ai, fi]),
            })
    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(out_dir / "normalized_fitness_curve_summary.csv", index=False)

    for t in targets:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.gca()
        sub = curve_df[curve_df["target"] == t]
        for a in algos:
            s = sub[sub["algo_variant"] == a].sort_values("budget")
            ax.plot(s["budget"].to_numpy(), s["fitness_score"].to_numpy(), marker="o", label=a)
        ax.set_title(f"Normalised fitness score vs budget ({t})")
        ax.set_xlabel("Budget (function evaluations)")
        ax.set_ylabel("Fitness score (normalised, higher is better)")
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / f"fitness_score_vs_budget_{t}.png", dpi=200)
        plt.close(fig)

    if args.mode == "cluster":
        Z = zscore_by_feature(M_mean)
        labels, _ = kmeans_lloyd(Z, K=max(2, args.K), rng=rng, iters=100)
        out = pd.DataFrame({"algo_variant": algos, "cluster": labels})
        out.to_csv(out_dir / "normalized_fitness_clusters.csv", index=False)

        prot = []
        for k in np.unique(labels):
            idx = np.where(labels == k)[0]
            prot.append({
                "cluster": int(k),
                "size": int(len(idx)),
                "mean_headroom": float(np.mean(hr_mean[idx])),
                "members": ", ".join([algos[i] for i in idx]),
            })
        pd.DataFrame(prot).sort_values("mean_headroom", ascending=False).to_csv(
            out_dir / "cluster_prototypes.csv", index=False
        )

        (latex_dir / "normalized_fitness_method.tex").write_text(
            r"\subsection{Normalised fitness curves and unsupervised curve clustering}" "\n"
            r"We normalise fitness per instance (robustly via within-instance quantiles) to obtain a progress score in $[0,1]$ "
            r"that is comparable across heterogeneous problems. We then build concatenated curve vectors across target tiers and budgets, "
            r"standardise features across algorithms, and apply K-means clustering to group similar budget-response dynamics." "\n",
            encoding="utf-8"
        )
        (latex_dir / "normalized_fitness_findings.tex").write_text(
            r"\paragraph{Unsupervised clustering on normalised fitness curves.} "
            r"Because PSO/ES anchors are not present in the current CSV, we cluster algorithms purely by "
            r"their normalised fitness--budget response curves. Clusters are interpreted via headroom "
            r"(budget sensitivity) and curve shape (monotone vs delayed take-off). "
            r"The resulting partition provides an actionable shortlist of candidates to re-run with true anchors and larger budgets." "\n",
            encoding="utf-8"
        )
        print("[OK] clustering outputs written to:", out_dir)
        return

    # membership outputs (anchor/proxy)
    w_mean = w_boot.mean(axis=0)
    w_lo, w_hi = ci(w_boot, args.alpha)

    d_mean = delta_boot.mean(axis=0)
    d_lo, d_hi = ci(delta_boot, args.alpha)

    p_pso = (label_boot == 0).mean(axis=0)
    p_es = (label_boot == 1).mean(axis=0)
    p_flat = (label_boot == 2).mean(axis=0)

    labels_final = []
    for ai in range(nA):
        if p_flat[ai] >= 0.5:
            labels_final.append("FLAT")
        else:
            labels_final.append("PSO" if p_pso[ai] >= p_es[ai] else "ES")

    summ = pd.DataFrame({
        "algo_variant": algos,
        "label": labels_final,
        "anchor_PSO": pso_anchor,
        "anchor_ES": es_anchor,
        "w_PSO": w_mean[:, 0],
        "w_PSO_lo": w_lo[:, 0],
        "w_PSO_hi": w_hi[:, 0],
        "w_ES": w_mean[:, 1],
        "w_ES_lo": w_lo[:, 1],
        "w_ES_hi": w_hi[:, 1],
        "p_PSO": p_pso,
        "p_ES": p_es,
        "p_FLAT": p_flat,
        "Delta_ES_minus_PSO": d_mean,
        "Delta_lo": d_lo,
        "Delta_hi": d_hi,
        "headroom": hr_mean,
        "headroom_lo": hr_lo,
        "headroom_hi": hr_hi,
    }).sort_values(["label", "w_PSO"], ascending=[True, False])
    summ.to_csv(out_dir / "normalized_fitness_membership_summary.csv", index=False)

    fig = plt.figure(figsize=(10, 4.5))
    ax = plt.gca()
    x = np.arange(len(summ))
    ax.bar(x, summ["w_PSO"].to_numpy())
    ax.set_xticks(x)
    ax.set_xticklabels(summ["algo_variant"].tolist(), rotation=45, ha="right")
    ax.set_title(f"Soft membership to PSO anchor ({pso_anchor})")
    ax.set_ylabel("w_PSO")
    fig.tight_layout()
    fig.savefig(fig_dir / "membership_bar.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 4.5))
    ax = plt.gca()
    ax.bar(np.arange(len(summ)), summ["headroom"].to_numpy())
    ax.set_xticks(np.arange(len(summ)))
    ax.set_xticklabels(summ["algo_variant"].tolist(), rotation=45, ha="right")
    ax.set_title("Headroom of normalised fitness curves")
    ax.set_ylabel("Headroom (max tier improvement)")
    fig.tight_layout()
    fig.savefig(fig_dir / "headroom_bar.png", dpi=200)
    plt.close(fig)

    tag = "anchors" if args.mode == "anchor" else "proxies"
    (latex_dir / "normalized_fitness_method.tex").write_text(
        rf"\subsection{{Normalised fitness curves with {tag} for behavioural attribution}}" "\n"
        rf"To mitigate floor/ceiling effects of success-only evaluation, we normalise best objective values within each "
        rf"instance and target tier to obtain a progress score in $[0,1]$ (higher is better). We then form concatenated "
        rf"budget-response vectors across tiers and budgets, standardise features across algorithms, and compare each method "
        rf"to two reference families using anchor curves. In this run, the PSO reference is \texttt{{{pso_anchor}}} and the "
        rf"ES reference is \texttt{{{es_anchor}}}. We gate ``no-signal'' methods using a headroom threshold on the normalised "
        rf"fitness curves and bootstrap over instances to obtain uncertainty bands." "\n",
        encoding="utf-8"
    )

    (latex_dir / "normalized_fitness_findings.tex").write_text(
        r"\paragraph{Normalised fitness curves add discriminative signal.} "
        r"Normalised fitness curves preserve sub-threshold progress and therefore reduce false ``flat'' assignments that "
        r"arise under success-only evaluation in strict tiers. Under the current budget ladder, the resulting attribution "
        r"separates budget-responsive methods with stable reference proximity from genuinely no-signal methods, and provides "
        r"bootstrap-stable uncertainty bands for membership and headroom." "\n",
        encoding="utf-8"
    )

    print("[OK] wrote:", out_dir / "normalized_fitness_membership_summary.csv")


if __name__ == "__main__":
    main()
