#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
dorigo_interpolation_crossover.py

Compute the "crossover lambda" between PSO and ES on interpolated benchmark families:
  Interp_<A>_<B>_lam{lambda}

Goal
----
Given an interpolation experiment, estimate (with 95% bootstrap CI):
  lambda*  where  mean_delta(lambda) crosses 0
where delta = mean(beta_mean_PSO - beta_mean_ES) across instances
(or you can use median_best / normalised fitness with a flag).

This directly supports a strong conclusion:
"Beyond lambda*, PSO-like dynamics dominate; below lambda*, ES-like dominates"
and you can report this per (budget, target) slice.

Inputs
------
- instance_algo_budget_summary.csv produced by interpolation runner
  Must include:
    problem names with "lam{xx.xx}"
    algo_variant includes both anchors (default PSO_GBEST and ES_1P1)
    columns: problem, instance_id, algo_variant, budget, target, beta_mean

Outputs
-------
- crossover_summary.csv  (lambda* mean + 95% CI per budget×target)
- delta_curves.csv       (mean delta vs lambda per budget×target)
- figs/delta_vs_lambda_<target>_b<budget>.png
- latex/crossover_paragraph.tex

Usage (Windows CMD)
-------------------
python dorigo_interpolation_crossover.py ^
  --csv .\out_dorigo_track3_full\instance_algo_budget_summary.csv ^
  --out_dir .\out_dorigo_track3_full\crossover ^
  --pso PSO_GBEST --es ES_1P1 ^
  --B 2000 --seed 0

Notes
-----
- If delta never crosses 0 in [min_lambda, max_lambda], lambda* is reported as NaN and the sign is recorded.
"""
from __future__ import annotations
import argparse
import re
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

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

LAM_RE = re.compile(r"lam([0-9]+\.[0-9]+)")

def parse_lambda(problem: str) -> float:
    m = LAM_RE.search(problem)
    if not m:
        raise ValueError(f"Cannot parse lambda from problem='{problem}'")
    return float(m.group(1))

def stratified_bootstrap(inst_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    # stratify by problem (which corresponds to lambda)
    out=[]
    for prob, g in inst_df.groupby("problem"):
        n=len(g)
        idx = rng.integers(0, n, size=n)
        samp=g.iloc[idx]
        w = samp.groupby(["problem","instance_id"]).size().reset_index(name="w")
        out.append(w)
    return pd.concat(out, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pso", default="PSO_GBEST")
    ap.add_argument("--es", default="ES_1P1")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = out_dir/"figs"
    latex_dir = out_dir/"latex"
    ensure_dir(out_dir); ensure_dir(fig_dir); ensure_dir(latex_dir)

    df = pd.read_csv(args.csv)
    if args.pso not in df["algo_variant"].unique():
        raise ValueError(f"PSO anchor {args.pso} not found in CSV.")
    if args.es not in df["algo_variant"].unique():
        raise ValueError(f"ES anchor {args.es} not found in CSV.")

    # keep only anchors
    df = df[df["algo_variant"].isin([args.pso, args.es])].copy()

    # attach lambda
    df["lambda"] = df["problem"].map(parse_lambda)

    # per instance, we want delta = beta_mean(pso) - beta_mean(es)
    piv = df.pivot_table(index=["problem","lambda","instance_id","budget","target"], columns="algo_variant", values="beta_mean")
    piv = piv.reset_index()
    piv["delta"] = piv[args.pso] - piv[args.es]

    # deterministic mean delta curve
    mean_curve = (piv.groupby(["lambda","budget","target"], as_index=False)
                    .agg(delta=("delta","mean")))
    mean_curve.to_csv(out_dir/"delta_curves.csv", index=False)

    budgets = sorted(mean_curve["budget"].unique().tolist())
    targets = sorted(mean_curve["target"].unique().tolist())
    lambdas = sorted(mean_curve["lambda"].unique().tolist())

    # helper: find crossing by linear interpolation
    def crossing(lams, vals):
        # find first sign change
        s = np.sign(vals)
        for i in range(len(vals)-1):
            if s[i] == 0:
                return float(lams[i])
            if s[i] * s[i+1] < 0:
                # interpolate between i and i+1
                x0,x1 = lams[i], lams[i+1]
                y0,y1 = vals[i], vals[i+1]
                # y = y0 + (y1-y0)*t, solve y=0
                t = -y0 / (y1 - y0 + 1e-12)
                return float(x0 + (x1-x0)*t)
        return float("nan")

    # bootstrap lambda*
    rng = np.random.default_rng(args.seed)
    inst_keys = piv[["problem","instance_id"]].drop_duplicates()

    boot_rows=[]
    cross_rows=[]
    for t in targets:
        for b in budgets:
            sub = piv[(piv["target"]==t) & (piv["budget"]==b)].copy()
            if sub.empty:
                continue
            # deterministic
            det = sub.groupby("lambda")["delta"].mean().reindex(lambdas).to_numpy()
            lam_star_det = crossing(lambdas, det)

            stars=[]
            for k in tqdm(range(args.B), desc=f"bootstrap t={t} b={b}", leave=False):
                w = stratified_bootstrap(inst_keys, rng)
                subw = sub.merge(w, on=["problem","instance_id"], how="inner")
                # weighted mean delta per lambda
                g = (subw.assign(d=subw["delta"]*subw["w"])
                        .groupby("lambda", as_index=False)
                        .agg(num=("d","sum"), den=("w","sum")))
                g["delta"] = g["num"]/g["den"]
                vals = g.set_index("lambda")["delta"].reindex(lambdas).to_numpy()
                # fill missing with 0 (rare)
                vals = np.nan_to_num(vals, nan=0.0)
                stars.append(crossing(lambdas, vals))

            stars = np.array(stars, dtype=float)
            stars_valid = stars[np.isfinite(stars)]
            if len(stars_valid) == 0:
                lam_mean = float("nan")
                lam_lo = float("nan")
                lam_hi = float("nan")
            else:
                lam_mean = float(np.mean(stars_valid))
                lam_lo = float(np.quantile(stars_valid, args.alpha/2))
                lam_hi = float(np.quantile(stars_valid, 1-args.alpha/2))

            cross_rows.append({
                "budget": b,
                "target": t,
                "lambda_star_det": lam_star_det,
                "lambda_star_mean": lam_mean,
                "lambda_star_lo": lam_lo,
                "lambda_star_hi": lam_hi,
            })

            # plot
            plt.figure(figsize=(5.6,4.2))
            plt.plot(lambdas, det, marker="o")
            plt.axhline(0.0, linewidth=1)
            if np.isfinite(lam_star_det):
                plt.axvline(lam_star_det, linestyle="--", linewidth=1)
            plt.title(f"Delta vs lambda ({t}, budget={b})")
            plt.xlabel("lambda (mixing weight)")
            plt.ylabel(f"mean(beta_mean[{args.pso}] - beta_mean[{args.es}])")
            plt.tight_layout()
            plt.savefig(fig_dir/f"delta_vs_lambda_{t}_b{b}.png", dpi=200)
            plt.close()

    cross = pd.DataFrame(cross_rows)
    cross.to_csv(out_dir/"crossover_summary.csv", index=False)

    # latex paragraph (generic)
    para = r"""\paragraph{Interpolation crossover between anchor families.}
We estimate a crossover mixing coefficient $\lambda^\*$ on the interpolated benchmark family
$f_\lambda=(1-\lambda)\,f_A+\lambda\,f_B$ by tracking the sign of the mean performance gap
$\Delta(\lambda)=\mathbb{E}[\hat p_{\mathrm{PSO}}-\hat p_{\mathrm{ES}}]$ across instances.
For each (budget, target) slice, we compute $\lambda^\*$ by linear interpolation at the first sign change of $\Delta(\lambda)$
and quantify uncertainty via bootstrap over instances (95\% percentile bands).
Slices with no sign change within the tested $\lambda$ grid are reported as non-crossing regimes.
"""
    (latex_dir/"crossover_paragraph.tex").write_text(para+"\n", encoding="utf-8")

    print("[OK] wrote:", out_dir/"crossover_summary.csv")

if __name__ == "__main__":
    main()
