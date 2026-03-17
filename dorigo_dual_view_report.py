#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
dorigo_dual_view_report.py

Create a paper-ready "dual-view" report that compares:
  (A) success-only curve membership (PSO/ES/FLAT) from membership_bootstrap_summary.csv
  (B) normalised-fitness curve membership (PSO/ES/FLAT) from normalized_fitness_membership_summary.csv

Inputs:
  --succ_summary  membership_bootstrap_summary.csv
  --fit_summary   normalized_fitness_membership_summary.csv

Outputs (in out_dir):
  - dual_view_membership.csv
  - dual_view_disagreement.csv
  - latex/dual_view_table.tex
  - latex/dual_view_paragraph.tex
  - figs/dual_view_confusion.png

Usage:
  python dorigo_dual_view_report.py ^
    --succ_summary .\out_dorigo_new\membership_bootstrap_success\membership_bootstrap_summary.csv ^
    --fit_summary  .\out_dorigo_new\norm_fitness\normalized_fitness_membership_summary.csv ^
    --out_dir .\out_dorigo_new\dual_view_report
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--succ_summary", required=True)
    ap.add_argument("--fit_summary", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    latex_dir = out_dir / "latex"
    fig_dir = out_dir / "figs"
    ensure_dir(out_dir); ensure_dir(latex_dir); ensure_dir(fig_dir)

    succ = pd.read_csv(args.succ_summary).rename(columns={"algo":"algo_variant"})
    fit = pd.read_csv(args.fit_summary)

    # minimal columns
    succ_cols = ["algo_variant","label","w_PSO","w_ES","w_FLAT",
                "dist_margin_ES_minus_PSO","dist_margin_lo","dist_margin_hi",
                "headroom","headroom_lo","headroom_hi",
                "p_win_PSO","p_win_ES","p_win_FLAT"]
    fit_cols = ["algo_variant","label","w_PSO","w_PSO_lo","w_PSO_hi","w_ES","w_ES_lo","w_ES_hi",
                "Delta_ES_minus_PSO","Delta_lo","Delta_hi",
                "headroom","headroom_lo","headroom_hi"]

    succ = succ[succ_cols].rename(columns={
        "label":"label_succ",
        "w_PSO":"w_PSO_succ","w_ES":"w_ES_succ","w_FLAT":"w_FLAT_succ",
        "headroom":"headroom_succ","headroom_lo":"headroom_lo_succ","headroom_hi":"headroom_hi_succ",
        "p_win_PSO":"p_win_PSO_succ","p_win_ES":"p_win_ES_succ","p_win_FLAT":"p_win_FLAT_succ",
        "dist_margin_ES_minus_PSO":"Delta_succ","dist_margin_lo":"Delta_succ_lo","dist_margin_hi":"Delta_succ_hi",
    })

    fit = fit[fit_cols].rename(columns={
        "label":"label_fit",
        "w_PSO":"w_PSO_fit","w_ES":"w_ES_fit",
        "headroom":"headroom_fit","headroom_lo":"headroom_lo_fit","headroom_hi":"headroom_hi_fit",
        "Delta_ES_minus_PSO":"Delta_fit","Delta_lo":"Delta_fit_lo","Delta_hi":"Delta_fit_hi",
    })

    df = succ.merge(fit, on="algo_variant", how="inner")
    df.to_csv(out_dir / "dual_view_membership.csv", index=False)

    # Disagreements
    df["disagree"] = (df["label_succ"] != df["label_fit"])
    dis = df[df["disagree"]].copy()
    dis.to_csv(out_dir / "dual_view_disagreement.csv", index=False)

    # Confusion matrix plot
    order = ["PSO","ES","FLAT","UNCERTAIN"]
    lab_s = [l if l in order else "UNCERTAIN" for l in df["label_succ"].tolist()]
    lab_f = [l if l in order else "UNCERTAIN" for l in df["label_fit"].tolist()]
    cm = pd.crosstab(pd.Series(lab_s, name="success"), pd.Series(lab_f, name="fitness"), dropna=False)
    for o in order:
        if o not in cm.index: cm.loc[o] = 0
        if o not in cm.columns: cm[o] = 0
    cm = cm.loc[order, order]
    fig = plt.figure(figsize=(5.2,4.2))
    ax = plt.gca()
    im = ax.imshow(cm.values)
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order); ax.set_yticklabels(order)
    ax.set_title("Dual-view label confusion (success vs fitness)")
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, str(int(cm.values[i,j])), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(fig_dir / "dual_view_confusion.png", dpi=200)
    plt.close(fig)

    # LaTeX table (compact)
    def fmt_ci(m, lo, hi, nd=3):
        return f"{m:.{nd}f} [{lo:.{nd}f},{hi:.{nd}f}]"

    tab = df.copy()
    tab["HR_succ"] = tab.apply(lambda r: fmt_ci(r["headroom_succ"], r["headroom_lo_succ"], r["headroom_hi_succ"], 3), axis=1)
    tab["HR_fit"]  = tab.apply(lambda r: fmt_ci(r["headroom_fit"], r["headroom_lo_fit"], r["headroom_hi_fit"], 3), axis=1)

    tab["wPSO_succ"] = tab["w_PSO_succ"].map(lambda x: f"{x:.3f}")
    tab["wFLAT_succ"] = tab["w_FLAT_succ"].map(lambda x: f"{x:.3f}")
    tab["wPSO_fit"]  = tab["w_PSO_fit"].map(lambda x: f"{x:.3f}")
    tab["wES_fit"]  = tab["w_ES_fit"].map(lambda x: f"{x:.3f}")

    out_tab = tab[["algo_variant","label_succ","label_fit","HR_succ","HR_fit","wPSO_succ","wFLAT_succ","wPSO_fit","wES_fit"]]
    out_tab = out_tab.sort_values("algo_variant")

    # write latex manually to avoid heavy deps
    lines = []
    lines.append(r"\begin{tabular}{l ll cc cccc}")
    lines.append(r"\hline")
    lines.append(r"Algorithm & Success label & Fitness label & HR$_{\text{succ}}$ & HR$_{\text{fit}}$ & $w^{\text{succ}}_{\text{PSO}}$ & $w^{\text{succ}}_{\text{FLAT}}$ & $w^{\text{fit}}_{\text{PSO}}$ & $w^{\text{fit}}_{\text{ES}}$ \\")
    lines.append(r"\hline")
    for _, r in out_tab.iterrows():
        lines.append(
            rf"\texttt{{{r['algo_variant']}}} & {r['label_succ']} & {r['label_fit']} & {r['HR_succ']} & {r['HR_fit']} & {r['wPSO_succ']} & {r['wFLAT_succ']} & {r['wPSO_fit']} & {r['wES_fit']} \\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    (latex_dir / "dual_view_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # LaTeX paragraph (auto)
    n = len(df)
    n_dis = int(df["disagree"].sum())
    # count patterns
    a = df[(df["label_succ"]=="FLAT") & (df["label_fit"]!="FLAT")]["algo_variant"].tolist()
    b = df[(df["label_succ"]!="FLAT") & (df["label_fit"]=="FLAT")]["algo_variant"].tolist()
    c = df[(df["label_succ"]=="PSO") & (df["label_fit"]=="ES")]["algo_variant"].tolist()
    d = df[(df["label_succ"]=="ES") & (df["label_fit"]=="PSO")]["algo_variant"].tolist()

    def tt(lst):
        return ", ".join([rf"\texttt{{{x}}}" for x in lst]) if lst else "---"

    para = (
        r"\paragraph{Dual-view attribution: attainment vs progress.}" "\n"
        rf"We compare success-curve attribution (threshold attainment) with normalised-fitness attribution (sub-threshold progress) "
        rf"across {n} evaluated algorithms. The two views disagree on {n_dis}/{n} methods, "
        r"indicating that binary success can collapse information in strict regimes while value-based curves preserve progress dynamics. "
        rf"In particular, methods labelled \texttt{{FLAT}} under success-only but non-\texttt{{FLAT}} under fitness (progress without attainment) are: {tt(a)}. "
        rf"A notable cross-family disagreement is \texttt{{PSO}} in success but \texttt{{ES}} in fitness: {tt(c)}. "
        r"This motivates reporting both views: success captures whether a method crosses a deployment-relevant threshold, "
        r"while normalised fitness captures how efficiently a method converts additional budget into optimisation progress below that threshold."
        "\n"
    )
    (latex_dir / "dual_view_paragraph.tex").write_text(para, encoding="utf-8")

    print("[OK] wrote:", out_dir / "dual_view_membership.csv")
    print("[OK] wrote:", latex_dir / "dual_view_table.tex")
    print("[OK] wrote:", latex_dir / "dual_view_paragraph.tex")
    print("[OK] fig:", fig_dir / "dual_view_confusion.png")

if __name__ == "__main__":
    main()
