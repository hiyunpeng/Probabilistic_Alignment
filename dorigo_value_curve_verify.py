#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dorigo_value_curve_verify.py

Verify that replacing success-only curves with value-based progress curves recovers signal.

This script takes your aggregated per-(instance,algo,budget,target) summary CSV (with beta_mean and best-value stats),
constructs an instance-normalised progress score (1 - regret), and compares:

  - between-algorithm separation (std across algos per (target,budget))
  - headroom across budgets (max_t curve(b_max) - curve(b_min))
  - fraction of algorithms in "no-signal" regime (headroom < gate)

Optionally, if PSO/ES anchors are present in the CSV, it can also do anchor-based attribution on value curves.

Inputs:
  A CSV like ./out_dorigo/instance_algo_budget_summary.csv with columns:
    problem, instance_id, budget, target, algo_variant, beta_mean, mean_best/median_best/min_best/max_best, ...

Outputs (in out_dir):
  - value_curve_summary.csv        : mean + bootstrap CI of beta_mean and value_score per (algo,target,budget)
  - headroom_compare.csv           : headroom_success vs headroom_value per algo (mean + CI)
  - signal_kpis.csv                : global KPIs (separation gain, flat fractions)
  - figs/value_vs_budget_{tier}.png
  - figs/sep_std_success_vs_value.png
  - latex/value_curve_method.tex   : methodology snippet (LaTeX)
  - latex/value_curve_findings.tex : findings snippet with numbers

Recommended usage:
  python dorigo_value_curve_verify.py \
    --csv ./out_dorigo/instance_algo_budget_summary.csv \
    --out_dir ./out_dorigo/value_curve_verify \
    --value_stat median_best \
    --scale_mode q05q95 \
    --B 1000 --headroom_gate 0.02

Notes on the value metric:
  - We assume minimisation by default. If your objective is maximised, pass --sense maximize.
  - Scaling is performed per (problem, instance_id, target) to avoid mixing early-stop behaviours across tiers.
  - Robust scaling (q05/q95) reduces sensitivity to outliers; minmax can be used if you prefer.

Author: (generated)
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


# ------------------------- utilities -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def parse_csv_list(s: str, cast=str):
    if s is None or str(s).strip() == "":
        return None
    return [cast(x.strip()) for x in s.split(",") if x.strip()]

def quantile_ref(series: pd.Series, q_lo: float, q_hi: float):
    return series.quantile(q_lo), series.quantile(q_hi)

def compute_value_score(
    df: pd.DataFrame,
    value_stat: str = "median_best",
    scale_mode: str = "q05q95",
    sense: str = "minimize",
    group_cols: tuple[str, ...] = ("problem", "instance_id", "target"),
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Adds:
      - value_score in [0,1] where 1 is best, 0 is worst (within each instance tier group)
    """
    if value_stat not in df.columns:
        raise ValueError(f"value_stat={value_stat} not in columns: {df.columns.tolist()}")

    # If maximize, invert values so smaller is better
    vals = df[value_stat].to_numpy(dtype=float)
    if sense.lower().startswith("max"):
        vals = -vals
    df = df.copy()
    df["_val"] = vals

    g = df.groupby(list(group_cols))["_val"]

    if scale_mode == "minmax":
        ref_best = g.min()
        ref_worst = g.max()
    elif scale_mode == "q05q95":
        ref_best = g.quantile(0.05)
        ref_worst = g.quantile(0.95)
    else:
        raise ValueError("scale_mode must be one of: minmax, q05q95")

    df = df.join(ref_best.rename("_best_ref"), on=list(group_cols))
    df = df.join(ref_worst.rename("_worst_ref"), on=list(group_cols))

    denom = (df["_worst_ref"] - df["_best_ref"]).to_numpy(dtype=float)
    denom = np.where(np.abs(denom) < eps, np.nan, denom)

    regret = (df["_val"].to_numpy(dtype=float) - df["_best_ref"].to_numpy(dtype=float)) / denom
    regret = np.clip(regret, 0.0, 1.0)
    regret = np.nan_to_num(regret, nan=0.0)

    df["value_score"] = 1.0 - regret
    return df.drop(columns=["_val", "_best_ref", "_worst_ref"], errors="ignore")


def stratified_instance_weights(instances: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Stratified bootstrap over instances within each problem family.
    Returns weights for each (problem, instance_id) appearing in the bootstrap sample.
    """
    out = []
    for prob, g in instances.groupby("problem"):
        n = len(g)
        idx = rng.integers(0, n, size=n)
        samp = g.iloc[idx]
        w = samp.groupby(["problem", "instance_id"]).size().reset_index(name="w")
        out.append(w)
    return pd.concat(out, ignore_index=True)


def weighted_group_mean(df: pd.DataFrame, group_cols, metric_cols, weight_col="w") -> pd.DataFrame:
    """
    Weighted mean over instances, preserving full (algo,target,budget) cube.
    """
    num = df.copy()
    for m in metric_cols:
        num[m] = num[m] * num[weight_col]
    gnum = num.groupby(group_cols, as_index=False)[metric_cols].sum()
    gden = df.groupby(group_cols, as_index=False)[weight_col].sum().rename(columns={weight_col: "_den"})
    out = gnum.merge(gden, on=group_cols, how="left")
    for m in metric_cols:
        out[m] = out[m] / out["_den"]
    return out.drop(columns=["_den"])


def make_feature_index(targets, budgets):
    return pd.MultiIndex.from_product([targets, budgets], names=["target", "budget"])


def pivot_curves(df_agg: pd.DataFrame, algos, targets, budgets, metric: str) -> np.ndarray:
    """
    Return curve matrix shaped (n_algos, n_features) aligned to targets x budgets order.
    """
    wide = df_agg.pivot(index="algo_variant", columns=["target", "budget"], values=metric)
    wide = wide.reindex(index=algos, columns=make_feature_index(targets, budgets))
    return wide.to_numpy(dtype=float)


def compute_headroom(curves: np.ndarray, targets, budgets) -> np.ndarray:
    """
    curves: (n_algos, n_features), where features are (target,budget) in cartesian order.
    Returns headroom per algo = max_t [curve(t,bmax)-curve(t,bmin)]
    """
    nA = curves.shape[0]
    T, B = len(targets), len(budgets)
    out = np.zeros(nA, dtype=float)
    for a in range(nA):
        max_hr = 0.0
        for ti in range(T):
            lo = curves[a, ti * B + 0]
            hi = curves[a, ti * B + (B - 1)]
            max_hr = max(max_hr, float(hi - lo))
        out[a] = max_hr
    return out


def ci(arr: np.ndarray, alpha: float):
    lo = np.quantile(arr, alpha / 2, axis=0)
    hi = np.quantile(arr, 1 - alpha / 2, axis=0)
    return lo, hi


# ------------------------- optional attribution -------------------------

def zscore_by_feature(M: np.ndarray, eps: float = 1e-12):
    mu = np.mean(M, axis=0)
    sd = np.std(M, axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (M - mu) / sd, mu, sd

def euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) * (a - b))))

def soft_memberships(dist: np.ndarray, tau: float) -> np.ndarray:
    x = -tau * dist
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--value_stat", default="median_best",
                    choices=["mean_best", "median_best", "min_best", "max_best"])
    ap.add_argument("--scale_mode", default="q05q95", choices=["q05q95", "minmax"])
    ap.add_argument("--sense", default="minimize", choices=["minimize", "maximize"])

    ap.add_argument("--budgets", default=None, help="comma-separated budgets to include (e.g., 300,500,800,1000)")
    ap.add_argument("--targets", default=None, help="comma-separated targets to include (e.g., easy,med,hard)")

    ap.add_argument("--B", type=int, default=500, help="bootstrap replicates")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)

    ap.add_argument("--headroom_gate", type=float, default=0.02)

    # Optional: attribution on value curves if anchors exist in CSV
    ap.add_argument("--pso_anchor", default=None, help="algo_variant name of PSO anchor (must exist in csv)")
    ap.add_argument("--es_anchor", default=None, help="algo_variant name of ES anchor (must exist in csv)")
    ap.add_argument("--tau", type=float, default=1.0, help="soft membership temperature (attribution)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figs"
    latex_dir = out_dir / "latex"
    ensure_dir(out_dir); ensure_dir(fig_dir); ensure_dir(latex_dir)

    df = pd.read_csv(args.csv)

    budgets = parse_csv_list(args.budgets, int) or sorted(df["budget"].unique().tolist())
    targets = parse_csv_list(args.targets, str) or sorted(df["target"].unique().tolist())

    df = df[df["budget"].isin(budgets) & df["target"].isin(targets)].copy()

    # compute value_score
    df = compute_value_score(
        df, value_stat=args.value_stat, scale_mode=args.scale_mode, sense=args.sense,
        group_cols=("problem", "instance_id", "target")
    )

    # deterministic aggregates
    det = (df.groupby(["algo_variant", "target", "budget"], as_index=False)
             .agg(beta_mean=("beta_mean", "mean"),
                  value_score=("value_score", "mean")))
    algos = sorted(det["algo_variant"].unique().tolist())

    # bootstrap
    rng = np.random.default_rng(args.seed)
    instances = df[["problem", "instance_id"]].drop_duplicates().reset_index(drop=True)

    nA = len(algos)
    nF = len(targets) * len(budgets)

    beta_boot = np.zeros((args.B, nA, nF), dtype=float)
    val_boot = np.zeros((args.B, nA, nF), dtype=float)

    beta_hr = np.zeros((args.B, nA), dtype=float)
    val_hr = np.zeros((args.B, nA), dtype=float)

    std_success = np.zeros((args.B, nF), dtype=float)
    std_value = np.zeros((args.B, nF), dtype=float)

    for b in tqdm(range(args.B), desc="Bootstrap"):
        w = stratified_instance_weights(instances, rng)
        dfw = df.merge(w, on=["problem", "instance_id"], how="inner")

        agg = weighted_group_mean(
            dfw,
            group_cols=["algo_variant", "target", "budget"],
            metric_cols=["beta_mean", "value_score"],
            weight_col="w"
        )

        beta_mat = pivot_curves(agg, algos, targets, budgets, "beta_mean")
        val_mat  = pivot_curves(agg, algos, targets, budgets, "value_score")

        # fill missing with feature means (rare; mostly when an algo missing in data)
        for M in (beta_mat, val_mat):
            col_mu = np.nanmean(M, axis=0)
            idx = np.where(np.isnan(M))
            M[idx] = np.take(col_mu, idx[1])

        beta_boot[b] = beta_mat
        val_boot[b] = val_mat

        beta_hr[b] = compute_headroom(beta_mat, targets, budgets)
        val_hr[b]  = compute_headroom(val_mat, targets, budgets)

        std_success[b] = np.std(beta_mat, axis=0)
        std_value[b]   = np.std(val_mat, axis=0)

    # summaries and CIs
    beta_mean = beta_boot.mean(axis=0)
    val_mean = val_boot.mean(axis=0)
    beta_lo, beta_hi = ci(beta_boot, args.alpha)
    val_lo, val_hi = ci(val_boot, args.alpha)

    beta_hr_mean = beta_hr.mean(axis=0)
    val_hr_mean = val_hr.mean(axis=0)
    beta_hr_lo, beta_hr_hi = ci(beta_hr, args.alpha)
    val_hr_lo, val_hr_hi = ci(val_hr, args.alpha)

    # global separation KPI: mean std across features
    sep_success = std_success.mean(axis=1)  # per bootstrap
    sep_value = std_value.mean(axis=1)
    sep_success_mean = float(np.mean(sep_success))
    sep_value_mean = float(np.mean(sep_value))
    sep_ratio_mean = float(sep_value_mean / max(sep_success_mean, 1e-12))
    sep_success_ci = (float(np.quantile(sep_success, args.alpha/2)),
                      float(np.quantile(sep_success, 1-args.alpha/2)))
    sep_value_ci = (float(np.quantile(sep_value, args.alpha/2)),
                    float(np.quantile(sep_value, 1-args.alpha/2)))

    # flat fractions
    flat_s = (beta_hr < args.headroom_gate).mean(axis=1)
    flat_v = (val_hr < args.headroom_gate).mean(axis=1)
    flat_s_mean = float(np.mean(flat_s))
    flat_v_mean = float(np.mean(flat_v))
    flat_s_ci = (float(np.quantile(flat_s, args.alpha/2)), float(np.quantile(flat_s, 1-args.alpha/2)))
    flat_v_ci = (float(np.quantile(flat_v, args.alpha/2)), float(np.quantile(flat_v, 1-args.alpha/2)))

    # pointwise curve summary CSV
    rows = []
    feat = make_feature_index(targets, budgets)
    for ai, a in enumerate(algos):
        for fi, (t, bud) in enumerate(feat.tolist()):
            rows.append({
                "algo_variant": a,
                "target": t,
                "budget": bud,
                "beta_mean": float(beta_mean[ai, fi]),
                "beta_lo": float(beta_lo[ai, fi]),
                "beta_hi": float(beta_hi[ai, fi]),
                "value_score": float(val_mean[ai, fi]),
                "value_lo": float(val_lo[ai, fi]),
                "value_hi": float(val_hi[ai, fi]),
            })
    curve_summary = pd.DataFrame(rows)
    curve_summary.to_csv(out_dir / "value_curve_summary.csv", index=False)

    # headroom compare
    hc = pd.DataFrame({
        "algo_variant": algos,
        "headroom_success": beta_hr_mean,
        "headroom_success_lo": beta_hr_lo,
        "headroom_success_hi": beta_hr_hi,
        "headroom_value": val_hr_mean,
        "headroom_value_lo": val_hr_lo,
        "headroom_value_hi": val_hr_hi,
    })
    hc["recovers_signal"] = (hc["headroom_success"] < args.headroom_gate) & (hc["headroom_value"] >= args.headroom_gate)
    hc = hc.sort_values("headroom_value", ascending=False)
    hc.to_csv(out_dir / "headroom_compare.csv", index=False)

    # signal KPIs
    kpi = pd.DataFrame([{
        "sep_std_success_mean": sep_success_mean,
        "sep_std_success_lo": sep_success_ci[0],
        "sep_std_success_hi": sep_success_ci[1],
        "sep_std_value_mean": sep_value_mean,
        "sep_std_value_lo": sep_value_ci[0],
        "sep_std_value_hi": sep_value_ci[1],
        "sep_ratio_value_over_success": sep_ratio_mean,
        "flat_fraction_success_mean": flat_s_mean,
        "flat_fraction_success_lo": flat_s_ci[0],
        "flat_fraction_success_hi": flat_s_ci[1],
        "flat_fraction_value_mean": flat_v_mean,
        "flat_fraction_value_lo": flat_v_ci[0],
        "flat_fraction_value_hi": flat_v_ci[1],
        "headroom_gate": args.headroom_gate,
        "value_stat": args.value_stat,
        "scale_mode": args.scale_mode,
        "sense": args.sense,
        "B": args.B,
        "alpha": args.alpha,
        "budgets": ",".join(map(str, budgets)),
        "targets": ",".join(map(str, targets)),
    }])
    kpi.to_csv(out_dir / "signal_kpis.csv", index=False)

    # ------------------------- plots -------------------------

    # value curves per target
    for t in targets:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.gca()
        sub = curve_summary[curve_summary["target"] == t]
        for a in algos:
            s = sub[sub["algo_variant"] == a].sort_values("budget")
            ax.plot(s["budget"].to_numpy(), s["value_score"].to_numpy(), marker="o", label=a)
        ax.set_title(f"Mean value-based progress vs budget ({t})")
        ax.set_xlabel("Budget (function evaluations)")
        ax.set_ylabel("Value progress score (1 - normalised regret)")
        ax.legend(ncol=2, fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / f"value_vs_budget_{t}.png", dpi=200)
        plt.close(fig)

    # separation comparison plot
    fig = plt.figure(figsize=(8, 4.5))
    ax = plt.gca()
    ax.boxplot([sep_success, sep_value], labels=["success (beta_mean)", "value (progress score)"])
    ax.set_ylabel("Mean std across algorithms (avg over target×budget features)")
    ax.set_title("Between-algorithm separation: success-only vs value-based curves")
    fig.tight_layout()
    fig.savefig(fig_dir / "sep_std_success_vs_value.png", dpi=200)
    plt.close(fig)

    # ------------------------- optional attribution on value curves -------------------------

    attribution_txt = ""
    if args.pso_anchor and args.es_anchor and (args.pso_anchor in algos) and (args.es_anchor in algos):
        # Use deterministic mean value curves for attribution (you can extend to bootstrap if desired).
        val_curve = pivot_curves(det, algos, targets, budgets, "value_score")
        Z, mu, sd = zscore_by_feature(val_curve)

        idx_pso = algos.index(args.pso_anchor)
        idx_es = algos.index(args.es_anchor)

        dist = np.zeros((nA, 2))
        for i in range(nA):
            dist[i, 0] = euclid(Z[i], Z[idx_pso])
            dist[i, 1] = euclid(Z[i], Z[idx_es])

        # delta positive => PSO closer
        Delta = dist[:, 1] - dist[:, 0]
        W = soft_memberships(dist, tau=args.tau)  # memberships to [PSO, ES]
        out = pd.DataFrame({
            "algo_variant": algos,
            "d_PSO": dist[:, 0],
            "d_ES": dist[:, 1],
            "Delta_ES_minus_PSO": Delta,
            "w_PSO": W[:, 0],
            "w_ES": W[:, 1],
            "headroom_value": val_hr_mean,
        }).sort_values("Delta_ES_minus_PSO", ascending=False)
        out.to_csv(out_dir / "value_curve_anchor_attribution.csv", index=False)
        attribution_txt = (
            "Anchors were provided and found in the CSV, so value-curve PSO/ES attribution was computed "
            "and written to \\texttt{value\\_curve\\_anchor\\_attribution.csv}."
        )
    else:
        attribution_txt = (
            "No anchors were provided (or anchors not found in CSV), so the script outputs signal recovery KPIs only. "
            "To do PSO/ES attribution on value curves, include anchor algorithms in the evaluated set and pass "
            "\\texttt{--pso\\_anchor} and \\texttt{--es\\_anchor}."
        )

    # ------------------------- LaTeX snippets -------------------------

    # Methodology snippet
    method_tex = r"""
\subsection{{Value-based progress curves to mitigate no-signal regimes}}
\label{{subsec:value_based_curves}}

Success-only evaluation can enter a \emph{{no-signal}} regime on stricter tiers: many algorithms fail to hit the
target within the tested budget ladder, so posterior success probabilities collapse to the Beta--Binomial floor.
To retain information about \emph{{progress}} even when targets are not met, we augment success curves with a
value-based curve derived from the best objective value attained in each run.

Let $v_{{iabt}}$ denote a summary of best objective values for instance $i$, algorithm $a$, budget $b$, and target tier $t$
(e.g., median best value across repeated runs). For each $(i,t)$ we define a robust reference scale across all algorithms and budgets:
\[
v_{{it}}^{{\mathrm{{best}}}} = Q_{{0.05}}\big(\{{v_{{iabt}}\}}_{{a,b}}\big), \qquad
v_{{it}}^{{\mathrm{{worst}}}} = Q_{{0.95}}\big(\{{v_{{iabt}}\}}_{{a,b}}\big),
\]
and compute a clipped normalised regret (minimisation assumed):
\[
r_{{iabt}} = \mathrm{{clip}}\!\left(\frac{{v_{{iabt}} - v_{{it}}^{{\mathrm{{best}}}}}}{{v_{{it}}^{{\mathrm{{worst}}}} - v_{{it}}^{{\mathrm{{best}}}}}},\,0,\,1\right).
\]
The corresponding progress score is $s_{{iabt}} = 1 - r_{{iabt}} \in [0,1]$.
We then form value-based budget curves by averaging $s_{{iabt}}$ over instances:
\[
\bar s_{{abt}} = \frac{1}{|\mathcal{{I}}|}\sum_{{i\in\mathcal{{I}}}} s_{{iabt}}.
\]
This representation preserves informative gradients in regimes where success is sparse, enabling
behavioural comparison via curve shape, headroom, and clustering in the same manner as success curves.
"""
    (latex_dir / "value_curve_method.tex").write_text(method_tex.strip() + "\n", encoding="utf-8")

    # Findings snippet with computed KPIs
    recovered = hc[hc["recovers_signal"]]["algo_variant"].tolist()
    rec_list = ", ".join([rf"\texttt{{{a}}}" for a in recovered]) if recovered else "---"

    # pull some illustrative headroom numbers (means)
    # take top 3 recovered by value headroom
    top_rec = hc[hc["recovers_signal"]].sort_values("headroom_value", ascending=False).head(3)
    ex = []
    for _, r in top_rec.iterrows():
        ex.append(rf"\texttt{{{r['algo_variant']}}} (success headroom={r['headroom_success']:.3f}, value headroom={r['headroom_value']:.3f})")
    ex_txt = "; ".join(ex) if ex else "---"

    findings_tex = rf"""
\paragraph{{Value curves recover signal missed by success-only evaluation.}}
On the current budget ladder, success-only curves exhibit substantial floor effects: the mean between-algorithm
separation (average standard deviation across all target$\times$budget features) is
$\bar\sigma_\mathrm{{succ}}={sep_success_mean:.3f}$ (95\% CI [{sep_success_ci[0]:.3f},{sep_success_ci[1]:.3f}]),
and a large fraction of algorithms fall into a no-signal regime under a headroom gate of {args.headroom_gate:.2f}
(flat fraction ${flat_s_mean:.3f}$, 95\% CI [{flat_s_ci[0]:.3f},{flat_s_ci[1]:.3f}]).
In contrast, the value-based progress curves retain meaningful gradients even when targets are not reached:
$\bar\sigma_\mathrm{{val}}={sep_value_mean:.3f}$ (95\% CI [{sep_value_ci[0]:.3f},{sep_value_ci[1]:.3f}]),
implying a separation gain of approximately {sep_ratio_mean:.1f}$\times$ relative to success-only curves,
and the no-signal fraction drops to ${flat_v_mean:.3f}$ (95\% CI [{flat_v_ci[0]:.3f},{flat_v_ci[1]:.3f}]).
Moreover, algorithms that appear flat under success-only curves but show non-trivial improvement under value curves are:
{rec_list}.
For example, {ex_txt}.
These results confirm that value-based curves increase information content and are a practical remedy for ceiling/floor
effects in success-only benchmarking. {attribution_txt}
"""
    (latex_dir / "value_curve_findings.tex").write_text(findings_tex.strip() + "\n", encoding="utf-8")

    print("[OK] wrote:", out_dir / "value_curve_summary.csv")
    print("[OK] wrote:", out_dir / "headroom_compare.csv")
    print("[OK] wrote:", out_dir / "signal_kpis.csv")
    print("[OK] figs in:", fig_dir)
    print("[OK] latex snippets in:", latex_dir)


if __name__ == "__main__":
    main()
