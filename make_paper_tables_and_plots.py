#!/usr/bin/env python3
"""make_paper_tables_and_plots.py

Creates:
- Paper-ready CSV + LaTeX tables (top-k per slice, winners, portfolio)
- Plots (bar charts and heatmaps) from instance_algo_budget_summary.csv

Usage example:
  python make_paper_tables_and_plots.py \
    --in_csv out_succ_small_v3/instance_algo_budget_summary.csv \
    --out_dir paper_artifacts \
    --metric beta_mean \
    --topk 10 \
    --save_plots

Notes:
- No seaborn; matplotlib only.
- Works on laptop; does only aggregation + plotting.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def try_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def sort_targets(target_series: pd.Series) -> List:
    vals = []
    for t in target_series.unique():
        tf = try_float(t)
        if tf is None:
            vals.append((1, str(t), t))
        else:
            vals.append((0, tf, t))
    vals.sort(key=lambda z: (z[0], z[1]))
    return [v[2] for v in vals]


def sanitize_filename(s: str) -> str:
    s = str(s).replace(" ", "")
    s = re.sub(r"[^A-Za-z0-9\.\-\_\=\,\(\)]", "_", s)
    return s[:120] if len(s) > 120 else s


def compute_algo_stats(df_slice: pd.DataFrame, metric: str) -> pd.DataFrame:
    grp = df_slice.groupby("algo_variant", dropna=False)
    out = grp.agg(
        n_instances=("instance_id", "nunique"),
        mean_metric=(metric, "mean"),
        median_metric=(metric, "median"),
        std_metric=(metric, "std"),
    ).reset_index()

    if (metric == "beta_mean") and ("beta_p05" in df_slice.columns) and ("beta_p95" in df_slice.columns):
        out["mean_p05"] = grp["beta_p05"].mean().values
        out["mean_p95"] = grp["beta_p95"].mean().values
    else:
        out["mean_p05"] = np.nan
        out["mean_p95"] = np.nan

    out = out.sort_values(["mean_metric", "n_instances"], ascending=[False, False]).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def latex_table_topk_by_slice(rows: pd.DataFrame, out_tex: Path, metric: str) -> None:
    df = rows.copy()
    rename = {
        "domain": "Domain",
        "budget": "Budget",
        "target": "Target",
        "algo_variant": "Algorithm",
        "mean_metric": "Mean",
        "mean_p05": "Mean p05",
        "mean_p95": "Mean p95",
        "n_instances": "N",
        "rank": "Rank",
    }
    keep_cols = ["domain", "budget", "target", "rank", "algo_variant", "mean_metric", "mean_p05", "mean_p95", "n_instances"]
    df = df[keep_cols].rename(columns=rename)

    for c in ["Mean", "Mean p05", "Mean p95"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(3)

    latex = df.to_latex(
        index=False,
        longtable=True,
        escape=True,
        caption=f"Top algorithms by slice (domain, budget, target) using mean {metric}.",
        label="tab:top_algos_by_slice",
    )
    out_tex.write_text(latex, encoding="utf-8")


def plot_topk_bar(stats: pd.DataFrame, domain: str, budget: int, target, metric: str, out_png: Path, topk: int) -> None:
    df = stats.head(topk).copy()
    if df.empty:
        return
    labels = df["algo_variant"].astype(str).tolist()
    y = df["mean_metric"].astype(float).values

    fig = plt.figure(figsize=(12, max(3.5, 0.35 * len(labels) + 1.0)))
    ax = fig.add_subplot(111)
    ax.barh(range(len(labels))[::-1], y[::-1])
    ax.set_yticks(range(len(labels))[::-1])
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel(metric)
    ax.set_title(f"Top-{min(topk, len(labels))} algorithms | domain={domain} budget={budget} target={target}")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_heatmap_domain_budget(
    df: pd.DataFrame,
    domain: str,
    budget: int,
    metric: str,
    out_png: Path,
    min_instances: int,
    top_algos: Optional[int],
) -> None:
    dfb = df[(df["domain"] == domain) & (df["budget"] == budget)].copy()
    if dfb.empty:
        return

    targets_sorted = sort_targets(dfb["target"])
    pivot = dfb.pivot_table(index="algo_variant", columns="target", values=metric, aggfunc="mean")
    pivot = pivot.reindex(columns=targets_sorted)

    ninst = dfb.groupby("algo_variant")["instance_id"].nunique()
    pivot = pivot.loc[pivot.index.intersection(ninst[ninst >= min_instances].index)]
    if pivot.empty:
        return

    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    if top_algos and top_algos > 0:
        pivot = pivot.head(top_algos)

    data = pivot.values.astype(float)

    fig = plt.figure(figsize=(max(8, 1.2 + 0.9 * pivot.shape[1]), max(5, 1.2 + 0.35 * pivot.shape[0])))
    ax = fig.add_subplot(111)
    im = ax.imshow(data, aspect="auto")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns.tolist()], rotation=45, ha="right", fontsize=8)
    ax.set_title(f"Heatmap of mean {metric} | domain={domain} budget={budget}")
    ax.set_xlabel("target")
    ax.set_ylabel("algorithm")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label(metric)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_portfolio(domain_port: pd.DataFrame, domain: str, out_png: Path) -> None:
    if domain_port.empty:
        return
    df = domain_port.sort_values(["wins", "mean_of_wins"], ascending=[False, False]).copy()
    labels = df["algo_variant"].astype(str).tolist()
    wins = df["wins"].astype(int).values

    fig = plt.figure(figsize=(12, max(4, 0.35 * len(labels) + 1.0)))
    ax = fig.add_subplot(111)
    ax.barh(range(len(labels))[::-1], wins[::-1])
    ax.set_yticks(range(len(labels))[::-1])
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_xlabel("#slices where algorithm is top-1")
    ax.set_title(f"Portfolio stability (top-1 frequency) | domain={domain}")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--metric", default="beta_mean")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--min_instances", type=int, default=1)
    ap.add_argument("--only_domain", default="")
    ap.add_argument("--only_budget", default="")
    ap.add_argument("--save_plots", action="store_true")
    ap.add_argument("--top_algos_heatmap", type=int, default=0)
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_dir = Path(args.out_dir)
    safe_mkdir(out_dir)

    df = pd.read_csv(in_csv)

    required = ["domain", "budget", "target", "algo_variant", "instance_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] missing required columns: {missing}")

    if args.metric not in df.columns:
        raise SystemExit(f"[ERROR] metric column not found: {args.metric}. Available: {list(df.columns)}")

    if args.only_domain:
        df = df[df["domain"].astype(str) == args.only_domain]
    if args.only_budget:
        df = df[df["budget"].astype(str) == str(args.only_budget)]

    if df.empty:
        raise SystemExit("[ERROR] empty after filtering")

    df["budget"] = pd.to_numeric(df["budget"], errors="coerce")
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")

    # Per-slice top-k + winners
    rows_topk = []
    winners = []

    slice_keys = ["domain", "budget", "target"]
    slices = df[slice_keys].drop_duplicates().sort_values(slice_keys).values.tolist()

    for domain, budget, target in slices:
        dfs = df[(df["domain"] == domain) & (df["budget"] == budget) & (df["target"] == target)].copy()
        if dfs.empty:
            continue
        stats = compute_algo_stats(dfs, metric=args.metric)

        topk = stats.head(args.topk).copy()
        topk.insert(0, "target", target)
        topk.insert(0, "budget", int(budget) if not pd.isna(budget) else budget)
        topk.insert(0, "domain", domain)
        rows_topk.append(topk)

        win = stats.head(1).copy()
        win.insert(0, "target", target)
        win.insert(0, "budget", int(budget) if not pd.isna(budget) else budget)
        win.insert(0, "domain", domain)
        winners.append(win[["domain", "budget", "target", "algo_variant", "mean_metric", "n_instances", "mean_p05", "mean_p95"]])

        if args.save_plots:
            out_png = out_dir / f"fig_topk_bar__{sanitize_filename(domain)}_B{int(budget)}_T{sanitize_filename(target)}.png"
            plot_topk_bar(stats, str(domain), int(budget), target, args.metric, out_png, args.topk)

    if not rows_topk:
        raise SystemExit("[ERROR] no slices found")

    df_topk = pd.concat(rows_topk, ignore_index=True)
    df_topk.to_csv(out_dir / "paper_table_top_algos_by_slice.csv", index=False)
    latex_table_topk_by_slice(df_topk, out_dir / "paper_table_top_algos_by_slice.tex", metric=args.metric)

    df_winners = pd.concat(winners, ignore_index=True) if winners else pd.DataFrame()
    df_winners.to_csv(out_dir / "paper_table_winners_by_slice.csv", index=False)

    # Portfolio stability
    if not df_winners.empty:
        port = df_winners.groupby(["domain", "algo_variant"], dropna=False).agg(
            wins=("algo_variant", "count"),
            mean_of_wins=("mean_metric", "mean"),
        ).reset_index()
        port.sort_values(["domain", "wins", "mean_of_wins"], ascending=[True, False, False]).to_csv(
            out_dir / "paper_table_portfolio.csv", index=False
        )

        if args.save_plots:
            for dom in port["domain"].unique():
                out_png = out_dir / f"fig_portfolio__{sanitize_filename(dom)}.png"
                plot_portfolio(port[port["domain"] == dom].copy(), str(dom), out_png)

    # Heatmaps per (domain,budget)
    if args.save_plots:
        for dom in sorted(df["domain"].unique()):
            for bud in sorted(df[df["domain"] == dom]["budget"].dropna().unique()):
                out_png = out_dir / f"fig_heatmap__{sanitize_filename(dom)}_B{int(bud)}.png"
                plot_heatmap_domain_budget(
                    df=df,
                    domain=str(dom),
                    budget=int(bud),
                    metric=args.metric,
                    out_png=out_png,
                    min_instances=args.min_instances,
                    top_algos=(args.top_algos_heatmap if args.top_algos_heatmap > 0 else None),
                )

    # Overview slices (for the paper “Table 1” style)
    overview = df.groupby(["domain", "budget", "target"], dropna=False).agg(
        n_rows=("algo_variant", "count"),
        n_instances=("instance_id", "nunique"),
        n_algos=("algo_variant", "nunique"),
        mean_metric=(args.metric, "mean"),
    ).reset_index()
    overview["budget"] = overview["budget"].astype("Int64")
    overview.to_csv(out_dir / "paper_table_overview_slices.csv", index=False)

    print("[OK] wrote:")
    print(" - paper_table_overview_slices.csv")
    print(" - paper_table_top_algos_by_slice.csv / .tex")
    print(" - paper_table_winners_by_slice.csv")
    print(" - paper_table_portfolio.csv")
    if args.save_plots:
        print("[OK] plots: fig_topk_bar__*.png, fig_heatmap__*.png, fig_portfolio__*.png")


if __name__ == "__main__":
    main()
