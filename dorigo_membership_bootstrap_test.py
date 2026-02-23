#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dorigo_membership_bootstrap_test.py

Bootstrap-based statistical attribution test for "PSO-like vs ES-like" (plus FLAT/no-signal).

Input:
  Aggregated CSV with per-(instance, algo, budget, target) statistics:
    - domain, problem, instance_id, algo_variant, budget, target, beta_mean, ...

Output:
  - membership_bootstrap_summary.csv: per-algo probabilities & CIs:
        p_win_PSO, p_win_ES, p_win_FLAT,
        soft weights w_PSO, w_ES, w_FLAT,
        distance margin Δ = d(ES) - d(PSO),
        headroom.
  - latex_tables/membership_table.tex (booktabs-ready)
  - latex_tables/bootstrap_subsection.tex (paper-ready LaTeX subsection)
  - figs/pso_prob_bar.png (and optional per-algo margin histograms)

Method:
  Stratified bootstrap over instances (by problem family). For each replicate:
    1) recompute mean success curves for each algo across (targets,budgets)
    2) z-score features across algorithms
    3) compute Euclidean distances to anchors (PSO, ES, FLAT)
    4) record winners and margins; apply headroom gating to label no-signal

Usage:
  python dorigo_membership_bootstrap_test.py \
    --csv ./out_dorigo/instance_algo_budget_summary.csv \
    --out_dir ./out_dorigo/membership_bootstrap \
    --pso_anchor PSO_GBEST --es_anchor ES_1P1 \
    --B 2000 --flat_gate 0.70 --headroom_gate 0.02 --alpha 0.05
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple

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

def zscore_by_feature(M: np.ndarray, eps: float = 1e-12):
    mu = np.mean(M, axis=0)
    sd = np.std(M, axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    Z = (M - mu) / sd
    return Z, mu, sd

def euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.sum((a - b) * (a - b))))

def infer_floor(df: pd.DataFrame, budgets: List[int], targets: List[str]) -> float:
    sub = df[df["budget"].isin(budgets) & df["target"].isin(targets)]
    m = float(np.nanmin(sub["beta_mean"].to_numpy()))
    return float(np.round(m, 6))

def stratified_bootstrap_instances(instances: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = []
    for _, g in instances.groupby("problem"):
        idx = rng.integers(0, len(g), size=len(g))
        out.append(g.iloc[idx])
    return pd.concat(out, ignore_index=True)

def build_feature_matrix(df: pd.DataFrame,
                         algos: List[str],
                         budgets: List[int],
                         targets: List[str],
                         instances_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    key = instances_df[["problem", "instance_id"]].drop_duplicates()
    df2 = df.merge(key, on=["problem", "instance_id"], how="inner")

    g = (df2.groupby(["algo_variant", "target", "budget"], as_index=False)["beta_mean"]
           .mean())

    feat_cols = [f"{t}|{b}" for t in targets for b in budgets]

    rows = []
    for a in algos:
        for t in targets:
            for b in budgets:
                rows.append((a, t, b))
    grid = pd.DataFrame(rows, columns=["algo_variant", "target", "budget"])
    g = grid.merge(g, on=["algo_variant", "target", "budget"], how="left")

    wide = g.pivot(index="algo_variant", columns=["target", "budget"], values="beta_mean")
    wide = wide.reindex(index=algos,
                        columns=pd.MultiIndex.from_product([targets, budgets], names=["target", "budget"]))
    wide.columns = feat_cols
    M = wide.to_numpy(dtype=float)
    return M, feat_cols

def compute_soft_memberships(dist: np.ndarray, tau: float) -> np.ndarray:
    x = -tau * dist
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)

def headroom_metric(curve_vec: np.ndarray, targets: List[str], budgets: List[int]) -> float:
    T, B = len(targets), len(budgets)
    max_hr = 0.0
    for ti in range(T):
        lo = curve_vec[ti * B + 0]
        hi = curve_vec[ti * B + (B - 1)]
        max_hr = max(max_hr, float(hi - lo))
    return max_hr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--pso_anchor", default="PSO_GBEST")
    ap.add_argument("--es_anchor", default="ES_1P1")
    ap.add_argument("--budgets", default="300,500,800,1000")
    ap.add_argument("--targets", default="easy,med,hard")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--flat_gate", type=float, default=0.70)
    ap.add_argument("--headroom_gate", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--no_hist", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figs"
    tex_dir = out_dir / "latex_tables"
    ensure_dir(out_dir); ensure_dir(fig_dir); ensure_dir(tex_dir)

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    targets = [x.strip() for x in args.targets.split(",") if x.strip()]

    df = pd.read_csv(args.csv)
    algos = sorted(df["algo_variant"].unique().tolist())

    if args.pso_anchor not in algos:
        raise SystemExit(f"pso_anchor {args.pso_anchor} not in csv algos")
    if args.es_anchor not in algos:
        raise SystemExit(f"es_anchor {args.es_anchor} not in csv algos")
    idx_pso = algos.index(args.pso_anchor)
    idx_es = algos.index(args.es_anchor)

    instances = df[["problem", "instance_id"]].drop_duplicates().reset_index(drop=True)
    floor_val = infer_floor(df, budgets, targets)

    rng = np.random.default_rng(args.seed)

    nA = len(algos)
    win_pso = np.zeros((args.B, nA), dtype=float)
    win_es  = np.zeros((args.B, nA), dtype=float)
    win_flat= np.zeros((args.B, nA), dtype=float)
    w_pso   = np.zeros((args.B, nA), dtype=float)
    w_es    = np.zeros((args.B, nA), dtype=float)
    w_flat  = np.zeros((args.B, nA), dtype=float)
    dist_margin = np.zeros((args.B, nA), dtype=float)
    headroom = np.zeros((args.B, nA), dtype=float)

    for bb in tqdm(range(args.B), desc="Bootstrap"):
        samp = stratified_bootstrap_instances(instances, rng)
        M, feat_cols = build_feature_matrix(df, algos, budgets, targets, samp)

        # impute missing with floor
        M = np.where(np.isnan(M), floor_val, M)

        Z, mu, sd = zscore_by_feature(M)
        Z_pso = Z[idx_pso].copy()
        Z_es  = Z[idx_es].copy()
        flat_vec = np.full_like(mu, floor_val)
        Z_flat = (flat_vec - mu) / sd

        dist = np.zeros((nA, 3), dtype=float)
        for i in range(nA):
            dist[i, 0] = euclid(Z[i], Z_pso)
            dist[i, 1] = euclid(Z[i], Z_es)
            dist[i, 2] = euclid(Z[i], Z_flat)

        W = compute_soft_memberships(dist, tau=args.tau)
        w_pso[bb] = W[:, 0]; w_es[bb] = W[:, 1]; w_flat[bb] = W[:, 2]

        for i in range(nA):
            headroom[bb, i] = headroom_metric(M[i], targets, budgets)

        for i in range(nA):
            # headroom gating => no-signal
            if headroom[bb, i] < args.headroom_gate:
                win_flat[bb, i] = 1.0
            else:
                j = int(np.argmin(dist[i]))
                if j == 0: win_pso[bb, i] = 1.0
                elif j == 1: win_es[bb, i] = 1.0
                else: win_flat[bb, i] = 1.0

            dist_margin[bb, i] = dist[i, 1] - dist[i, 0]  # positive => PSO closer

    def summarise(P: np.ndarray):
        mean = np.mean(P, axis=0)
        lo = np.quantile(P, args.alpha / 2, axis=0)
        hi = np.quantile(P, 1 - args.alpha / 2, axis=0)
        return mean, lo, hi

    pso_m, pso_lo, pso_hi = summarise(win_pso)
    es_m, es_lo, es_hi = summarise(win_es)
    fl_m, fl_lo, fl_hi = summarise(win_flat)

    wp_m, wp_lo, wp_hi = summarise(w_pso)
    we_m, we_lo, we_hi = summarise(w_es)
    wf_m, wf_lo, wf_hi = summarise(w_flat)

    dm_m, dm_lo, dm_hi = summarise(dist_margin)
    hr_m, hr_lo, hr_hi = summarise(headroom)

    out = pd.DataFrame({
        "algo": algos,

        "p_win_PSO": pso_m, "p_win_PSO_lo": pso_lo, "p_win_PSO_hi": pso_hi,
        "p_win_ES": es_m, "p_win_ES_lo": es_lo, "p_win_ES_hi": es_hi,
        "p_win_FLAT": fl_m, "p_win_FLAT_lo": fl_lo, "p_win_FLAT_hi": fl_hi,

        "w_PSO": wp_m, "w_PSO_lo": wp_lo, "w_PSO_hi": wp_hi,
        "w_ES": we_m, "w_ES_lo": we_lo, "w_ES_hi": we_hi,
        "w_FLAT": wf_m, "w_FLAT_lo": wf_lo, "w_FLAT_hi": wf_hi,

        "dist_margin_ES_minus_PSO": dm_m, "dist_margin_lo": dm_lo, "dist_margin_hi": dm_hi,
        "headroom": hr_m, "headroom_lo": hr_lo, "headroom_hi": hr_hi,
    })

    # label rule:
    labels = []
    for _, r in out.iterrows():
        if (r["w_FLAT"] >= args.flat_gate) or (r["headroom"] < args.headroom_gate):
            labels.append("FLAT")
        else:
            if r["dist_margin_lo"] > 0:
                labels.append("PSO")
            elif r["dist_margin_hi"] < 0:
                labels.append("ES")
            else:
                labels.append("UNCERTAIN")
    out["label"] = labels

    out = out.sort_values(["label", "w_FLAT", "p_win_PSO"], ascending=[True, False, False])
    out.to_csv(out_dir / "membership_bootstrap_summary.csv", index=False)

    # bar plot: P(closer to PSO)
    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    x = np.arange(len(out))
    y = out["p_win_PSO"].to_numpy()
    lo = out["p_win_PSO_lo"].to_numpy()
    hi = out["p_win_PSO_hi"].to_numpy()
    ax.bar(x, y)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(out["algo"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("P(closer to PSO) under bootstrap")
    ax.set_title("Bootstrap attribution: PSO closeness probability (with CI)")
    fig.tight_layout()
    fig.savefig(fig_dir / "pso_prob_bar.png", dpi=200)
    plt.close(fig)

    # optional histograms
    if not args.no_hist:
        for algo in out["algo"].tolist():
            i = algos.index(algo)
            vals = dist_margin[:, i]
            fig = plt.figure(figsize=(6, 3))
            ax = plt.gca()
            ax.hist(vals, bins=40)
            ax.axvline(0.0, linestyle="--")
            ax.set_title(f"Δ = d(ES) - d(PSO) bootstrap | {algo}")
            ax.set_xlabel("Δ (positive => PSO closer)")
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.savefig(fig_dir / f"dist_margin_hist_{algo}.png", dpi=200)
            plt.close(fig)

    # LaTeX table
    tab = out[["algo","label","w_PSO","w_ES","w_FLAT","p_win_PSO","p_win_ES","p_win_FLAT",
               "dist_margin_ES_minus_PSO","headroom"]].copy()
    tab.columns = ["Algorithm","Label","$\\bar w_{\\mathrm{PSO}}$","$\\bar w_{\\mathrm{ES}}$","$\\bar w_{\\mathrm{Flat}}$",
                   "$P(\\mathrm{PSO})$","$P(\\mathrm{ES})$","$P(\\mathrm{Flat})$","$\\overline{\\Delta}$","Headroom"]
    for c in tab.columns[2:]:
        tab[c] = tab[c].map(lambda v: f"{float(v):.3f}")
    latex = tab.to_latex(index=False, escape=False, column_format="llrrrrrrrr",
                         caption="Bootstrap curve-attribution summary (higher $\\overline{\\Delta}=d(ES)-d(PSO)$ indicates PSO-side similarity). ",
                         label="tab:bootstrap_membership")
    (tex_dir / "membership_table.tex").write_text(latex, encoding="utf-8")

    # LaTeX subsection text
    pso_list = out[out["label"]=="PSO"]["algo"].tolist()
    es_list = out[out["label"]=="ES"]["algo"].tolist()
    flat_list = out[out["label"]=="FLAT"]["algo"].tolist()
    unc_list = out[out["label"]=="UNCERTAIN"]["algo"].tolist()

    para = (
        "\\subsection{Bootstrap attribution test for PSO vs ES}\n"
        "\\label{subsec:bootstrap_attribution}\n\n"
        "To move beyond descriptive clustering, we quantify uncertainty in the curve-based attribution using a stratified bootstrap over instances. "
        "For each replicate, we resample instances within each problem family, recompute the success--budget curves, standardise curve features across algorithms, "
        "and compute Euclidean distances to three anchors: "
        f"\\texttt{{{args.pso_anchor}}} (PSO), \\texttt{{{args.es_anchor}}} (ES), and a \\texttt{{Flat}} baseline corresponding to the Bayesian floor. "
        "We record the distance margin $\\Delta = d(\\mathrm{ES}) - d(\\mathrm{PSO})$, where $\\Delta>0$ indicates PSO-side similarity.\n\n"
        "We classify an algorithm as \\texttt{PSO-like} if the $(1-\\alpha)$ bootstrap interval for $\\Delta$ lies strictly above zero; "
        "as \\texttt{ES-like} if the interval lies strictly below zero; and as \\texttt{UNCERTAIN} otherwise. "
        "Algorithms with negligible headroom across the tested budgets or with high Flat membership are labelled \\texttt{Flat/no-signal}, "
        "reflecting insufficient resolution under success-only metrics on strict tiers.\n\n"
        f"In the present evaluation, the bootstrap test yields PSO-like algorithms: \\texttt{{{', '.join(pso_list) if pso_list else '---'}}}; "
        f"ES-like: \\texttt{{{', '.join(es_list) if es_list else '---'}}}; "
        f"Flat/no-signal: \\texttt{{{', '.join(flat_list) if flat_list else '---'}}}; "
        f"and Uncertain: \\texttt{{{', '.join(unc_list) if unc_list else '---'}}}. "
        "Table~\\ref{tab:bootstrap_membership} reports mean membership weights and bootstrap probabilities.\n"
    )
    (tex_dir / "bootstrap_subsection.tex").write_text(para, encoding="utf-8")

    print("[OK] wrote", out_dir / "membership_bootstrap_summary.csv")
    print("[OK] wrote", tex_dir / "membership_table.tex")
    print("[OK] wrote", tex_dir / "bootstrap_subsection.tex")
    print("[OK] fig", fig_dir / "pso_prob_bar.png")

if __name__ == "__main__":
    main()
