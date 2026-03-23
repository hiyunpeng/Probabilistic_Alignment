#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
dorigo_combine_and_calibrate.py

Combine:
  - out_anchor_calibration (4 anchors only)
  - out_final_dim5 (anchors + metaheuristics)
into one unified dataset, then calibrate epsilon using within-family anchor distances.

What it does
------------
1) Merge instance summaries:
   - sums successes/trials per (domain,problem,instance_id,algo_variant,budget,target)
   - recomputes Beta(1,1) posterior stats (beta_mean, beta_p05, beta_p95)
   - merges value stats conservatively (trial-weighted mean_best/median_best; min/min; max/max)

2) Merge run-level logs (if present):
   - concatenates runs_detail.csv files
   - drops duplicates using a stable key

3) Calibrate epsilon:
   - epsilon_success = max( p95 distance(PSO_GBEST vs PSO_RING), p95 distance(ES_1P1 vs ES_MULAMBDA) )
     where distance is computed on SUCCESS curve vectors (targets × budgets) in feature-zscored space,
     with bootstrap over instances.
   - If runs_detail is available: also computes epsilon_fitness using fitness curve vectors (budgets only).

Outputs
-------
out_dir/
  instance_algo_budget_summary.csv
  runs_detail.csv                 (if any input had it)
  epsilon_calibration.json
  epsilon_calibration.txt

Usage (Windows CMD)
-------------------
python dorigo_combine_and_calibrate.py ^
  --anchor_dir .\out_anchor_calibration ^
  --main_dir   .\out_final_dim5 ^
  --out_dir    .\out_final_dim5_combined ^
  --B 2000 --seed 0

Notes
-----
- This assumes both dirs are from the SAME benchmark configuration (same problems and dimension).
- If the budget ladders differ, the script will use the union; missing cells are filled by column mean
  during distance computation (acceptable, but best practice is to keep the same ladder).
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import beta as sp_beta

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x=None, total=None, desc=None, **kw):
        return x if x is not None else range(total or 0)

GROUP_KEYS = ["domain","problem","instance_id","algo_variant","budget","target"]

ANCHORS = {
    "PSO": ("PSO_GBEST", "PSO_RING"),
    "ES":  ("ES_1P1", "ES_MULAMBDA"),
}

def beta_posterior(succ: int, trials: int):
    a = 1 + succ
    b = 1 + trials - succ
    mean = a / (a + b)
    p05 = float(sp_beta.ppf(0.05, a, b))
    p95 = float(sp_beta.ppf(0.95, a, b))
    return float(mean), p05, p95

def merge_summaries(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    all_df = pd.concat(dfs, ignore_index=True)
    # numeric
    for c in ["successes","trials","mean_best","median_best","min_best","max_best","budget","instance_id"]:
        if c in all_df.columns:
            all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

    def agg_group(g: pd.DataFrame) -> pd.Series:
        succ = int(np.nansum(g["successes"].to_numpy()))
        trials = int(np.nansum(g["trials"].to_numpy()))
        w = g["trials"].to_numpy(dtype=float)
        w = np.where(np.isfinite(w) & (w>0), w, 1.0)

        def wmean(col):
            x = g[col].to_numpy(dtype=float)
            return float(np.nansum(w*x) / max(1e-12, np.nansum(w)))

        mean_best = wmean("mean_best")
        median_best = wmean("median_best")  # approximation at summary level
        min_best = float(np.nanmin(g["min_best"].to_numpy(dtype=float)))
        max_best = float(np.nanmax(g["max_best"].to_numpy(dtype=float)))

        beta_mean, beta_p05, beta_p95 = beta_posterior(succ, trials) if trials>0 else (np.nan,np.nan,np.nan)
        return pd.Series({
            "successes": succ,
            "trials": trials,
            "beta_mean": beta_mean,
            "beta_p05": beta_p05,
            "beta_p95": beta_p95,
            "mean_best": mean_best,
            "median_best": median_best,
            "min_best": min_best,
            "max_best": max_best,
        })

    merged = all_df.groupby(GROUP_KEYS, as_index=False).apply(agg_group).reset_index(drop=True)
    merged = merged[GROUP_KEYS + ["successes","trials","beta_mean","beta_p05","beta_p95","mean_best","median_best","min_best","max_best"]]
    return merged

def zscore(M: np.ndarray) -> np.ndarray:
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (M - mu) / sd

def dist(a: np.ndarray, b: np.ndarray) -> float:
    d = a-b
    return float(np.sqrt(np.dot(d,d)))

def stratified_weights(instances: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out=[]
    for prob, g in instances.groupby("problem"):
        n=len(g)
        idx = rng.integers(0, n, size=n)
        samp=g.iloc[idx]
        w = samp.groupby(["problem","instance_id"]).size().reset_index(name="w")
        out.append(w)
    return pd.concat(out, ignore_index=True)

def curve_matrix_success(summary: pd.DataFrame, algos: list[str], targets: list[str], budgets: list[int], w: pd.DataFrame) -> np.ndarray:
    base = summary[["problem","instance_id","algo_variant","budget","target","beta_mean"]].rename(columns={"beta_mean":"metric"})
    m = base.merge(w, on=["problem","instance_id"], how="inner")
    m["wm"] = m["metric"] * m["w"]
    g = (m.groupby(["algo_variant","target","budget"], as_index=False)
           .agg(num=("wm","sum"), den=("w","sum")))
    g["metric"] = g["num"]/g["den"]
    cols = pd.MultiIndex.from_product([targets, budgets], names=["target","budget"])
    wide = (g.pivot(index="algo_variant", columns=["target","budget"], values="metric")
              .reindex(index=algos).reindex(columns=cols))
    M = wide.to_numpy(float)
    col_mu = np.nanmean(M, axis=0)
    idx = np.where(np.isnan(M))
    if len(idx[0])>0:
        M[idx] = np.take(col_mu, idx[1])
    return M

def curve_matrix_fitness(runs: pd.DataFrame, algos: list[str], budgets: list[int], w: pd.DataFrame) -> np.ndarray:
    rm = (runs.groupby(["problem","instance_id","algo_variant","budget"], as_index=False)
            .agg(best=("best","mean")))
    rm = rm.merge(w, on=["problem","instance_id"], how="inner")

    def add_fit(g):
        x = g["best"].to_numpy(float)
        q05 = np.quantile(x,0.05); q95=np.quantile(x,0.95)
        denom=q95-q05
        if abs(denom) < 1e-12:
            score = np.ones_like(x)
        else:
            regret = np.clip((x-q05)/denom, 0, 1)
            score = 1-regret
        g = g.copy()
        g["metric"] = score
        return g
    rm = rm.groupby(["problem","instance_id"], group_keys=False).apply(add_fit)

    rm["wm"] = rm["metric"] * rm["w"]
    g = (rm.groupby(["algo_variant","budget"], as_index=False)
           .agg(num=("wm","sum"), den=("w","sum")))
    g["metric"] = g["num"]/g["den"]

    wide = g.pivot(index="algo_variant", columns="budget", values="metric").reindex(index=algos).reindex(columns=budgets)
    M = wide.to_numpy(float)
    col_mu = np.nanmean(M, axis=0)
    idx = np.where(np.isnan(M))
    if len(idx[0])>0:
        M[idx] = np.take(col_mu, idx[1])
    return M

def bootstrap_anchor_distance(summary: pd.DataFrame, runs: pd.DataFrame|None, view: str, B: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    algos = sorted(summary["algo_variant"].unique().tolist())
    for fam,(a1,a2) in ANCHORS.items():
        if a1 not in algos or a2 not in algos:
            raise ValueError(f"Missing anchors for {fam}: need {a1},{a2}. Found algos={algos}")

    budgets = sorted(summary["budget"].unique().tolist())
    targets = sorted(summary["target"].unique().tolist())
    instances = summary[["problem","instance_id"]].drop_duplicates().reset_index(drop=True)

    results = {}
    for fam,(a1,a2) in ANCHORS.items():
        dists=[]
        for _ in tqdm(range(B), desc=f"bootstrap {view} {fam}", leave=False):
            w = stratified_weights(instances, rng)
            if view == "success":
                M = curve_matrix_success(summary, algos, targets, budgets, w)
            else:
                if runs is None:
                    raise ValueError("Need runs_detail.csv for fitness calibration.")
                M = curve_matrix_fitness(runs, algos, budgets, w)
            Z = zscore(M)
            i1 = algos.index(a1); i2 = algos.index(a2)
            dists.append(dist(Z[i1], Z[i2]))
        dists = np.array(dists, float)
        results[f"{fam}_mean"] = float(np.mean(dists))
        results[f"{fam}_p95"]  = float(np.quantile(dists, 0.95))
    eps = max(results["PSO_p95"], results["ES_p95"])
    return {"epsilon": float(eps), "within": results, "budgets": budgets, "targets": targets}

def merge_runs(run_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    allr = pd.concat(run_dfs, ignore_index=True)
    key_cols = [c for c in ["problem","instance_id","algo_variant","budget","rep","seed","best","evals"] if c in allr.columns]
    if key_cols:
        allr = allr.drop_duplicates(subset=key_cols)
    return allr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor_dir", required=True)
    ap.add_argument("--main_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    anchor_dir = Path(args.anchor_dir)
    main_dir = Path(args.main_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s1 = pd.read_csv(anchor_dir/"instance_algo_budget_summary.csv"); s1.columns=[c.strip() for c in s1.columns]
    s2 = pd.read_csv(main_dir/"instance_algo_budget_summary.csv");   s2.columns=[c.strip() for c in s2.columns]
    merged_summary = merge_summaries([s1,s2])
    merged_summary.to_csv(out_dir/"instance_algo_budget_summary.csv", index=False)

    run_dfs=[]
    for d in (anchor_dir, main_dir):
        rp = d/"runs_detail.csv"
        if rp.exists():
            r = pd.read_csv(rp); r.columns=[c.strip() for c in r.columns]
            run_dfs.append(r)
    merged_runs=None
    if run_dfs:
        merged_runs=merge_runs(run_dfs)
        merged_runs.to_csv(out_dir/"runs_detail.csv", index=False)

    succ = bootstrap_anchor_distance(merged_summary, merged_runs, view="success", B=args.B, seed=args.seed+1)
    out = {"epsilon_success": succ["epsilon"], "success_within": succ["within"], "budgets": succ["budgets"], "targets": succ["targets"]}

    if merged_runs is not None:
        fit = bootstrap_anchor_distance(merged_summary, merged_runs, view="fitness", B=args.B, seed=args.seed+7)
        out["epsilon_fitness"] = fit["epsilon"]
        out["fitness_within"] = fit["within"]

    (out_dir/"epsilon_calibration.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (out_dir/"epsilon_calibration.txt").write_text(
        "Combined calibration\n"
        f"epsilon_success={out['epsilon_success']:.4f}\n"
        + (f"epsilon_fitness={out.get('epsilon_fitness', float('nan')):.4f}\n" if "epsilon_fitness" in out else "")
        + "Within-family (p95):\n"
        + "\n".join([f"  {k}={v:.4f}" for k,v in out["success_within"].items() if k.endswith("p95")]) + "\n",
        encoding="utf-8"
    )

    print("[OK] wrote:", out_dir)

if __name__ == "__main__":
    main()
