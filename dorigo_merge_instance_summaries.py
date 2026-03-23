#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
dorigo_merge_instance_summaries.py

Merge multiple `instance_algo_budget_summary.csv` files into ONE consistent summary.

Why you need this
-----------------
When you run extra budgets (e.g., 12000 and 15000) or Phase-2/Phase-3 runs in batches,
you often end up with multiple summary CSVs that cover different budget ranges.
This script merges them *correctly* for the success statistics:
  - successes and trials are SUMMED per cell
  - Beta(1,1) posterior mean + p05/p95 are RECOMPUTED from the merged counts

It also merges value statistics conservatively:
  - mean_best is trial-weighted mean of per-file mean_best
  - min_best is global min across files
  - max_best is global max across files
  - median_best is approximated as trial-weighted mean of per-file median_best
    (True pooled median is not recoverable from summary-only data; for an exact median,
     you must merge run-level logs.)

Inputs must share these columns:
  domain, problem, instance_id, algo_variant, budget, target,
  successes, trials, beta_mean, beta_p05, beta_p95,
  mean_best, median_best, min_best, max_best

Usage (Windows CMD)
-------------------
python dorigo_merge_instance_summaries.py ^
  --inputs .\out_dorigo_phase2\instance_algo_budget_summary.csv .\out_dorigo_extra\instance_algo_budget_summary.csv ^
  --out .\out_dorigo_phase2\instance_algo_budget_summary_merged.csv

Optionally enforce schema checks:
  --strict

Outputs
-------
- merged summary CSV at --out
- a small merge_report.txt next to it

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.stats import beta as sp_beta

REQ_COLS = [
    "domain","problem","instance_id","algo_variant","budget","target",
    "successes","trials",
    "mean_best","median_best","min_best","max_best",
]

GROUP_KEYS = ["domain","problem","instance_id","algo_variant","budget","target"]

def beta_posterior(succ: int, trials: int):
    a = 1 + succ
    b = 1 + trials - succ
    mean = a / (a + b)
    p05 = float(sp_beta.ppf(0.05, a, b))
    p95 = float(sp_beta.ppf(0.95, a, b))
    return float(mean), p05, p95

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="List of summary CSVs to merge (order does not matter)")
    ap.add_argument("--out", required=True, help="Output merged CSV path")
    ap.add_argument("--strict", action="store_true", help="Fail if any input is missing required columns")
    args = ap.parse_args()

    inputs = [Path(p) for p in args.inputs]
    out_path = Path(args.out)

    frames = []
    missing_any = False
    for p in inputs:
        df = pd.read_csv(p)
        miss = [c for c in REQ_COLS if c not in df.columns]
        if miss:
            missing_any = True
            msg = f"[WARN] {p} missing cols: {miss}"
            if args.strict:
                raise ValueError(msg)
            else:
                print(msg)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    # Ensure numeric types
    for c in ["successes","trials","mean_best","median_best","min_best","max_best","budget","instance_id"]:
        if c in all_df.columns:
            all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

    # Weighted merge for means
    def agg_group(g):
        succ = int(np.nansum(g["successes"].to_numpy()))
        trials = int(np.nansum(g["trials"].to_numpy()))
        # weighted mean by trials (fallback weight=1 if trials missing)
        w = g["trials"].to_numpy(dtype=float)
        w = np.where(np.isfinite(w) & (w>0), w, 1.0)
        def wmean(col):
            x = g[col].to_numpy(dtype=float)
            m = np.nansum(w * x) / max(1e-12, np.nansum(w))
            return float(m)
        mean_best = wmean("mean_best")
        median_best = wmean("median_best")  # approximation
        min_best = float(np.nanmin(g["min_best"].to_numpy(dtype=float)))
        max_best = float(np.nanmax(g["max_best"].to_numpy(dtype=float)))

        beta_mean, beta_p05, beta_p95 = beta_posterior(succ, trials) if trials>0 else (np.nan,np.nan,np.nan)

        out = {
            "successes": succ,
            "trials": trials,
            "beta_mean": beta_mean,
            "beta_p05": beta_p05,
            "beta_p95": beta_p95,
            "mean_best": mean_best,
            "median_best": median_best,
            "min_best": min_best,
            "max_best": max_best,
        }
        return pd.Series(out)

    merged = all_df.groupby(GROUP_KEYS, as_index=False).apply(agg_group).reset_index()

    # groupby+apply adds an extra index column in some pandas versions; drop if present
    if "level_0" in merged.columns:
        merged = merged.drop(columns=["level_0"])
    if "level_1" in merged.columns:
        merged = merged.drop(columns=["level_1"])

    # Restore key columns in front
    merged = merged[GROUP_KEYS + ["successes","trials","beta_mean","beta_p05","beta_p95","mean_best","median_best","min_best","max_best"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    # Report
    report = out_path.with_suffix(".merge_report.txt")
    report.write_text(
        "Merged instance_algo_budget_summary files\n"
        f"inputs:\n" + "\n".join([f"  - {p}" for p in inputs]) + "\n\n"
        f"rows_out={len(merged)}\n"
        f"budgets={sorted(merged['budget'].unique().tolist())}\n"
        f"algos={sorted(merged['algo_variant'].unique().tolist())}\n"
        f"targets={sorted(merged['target'].unique().tolist())}\n"
        "NOTE: median_best is an approximation unless run-level logs are merged.\n",
        encoding="utf-8"
    )

    print("[OK] wrote:", out_path)
    print("[OK] wrote:", report)

if __name__ == "__main__":
    main()
