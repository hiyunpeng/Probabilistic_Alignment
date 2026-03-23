#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
dorigo_calibrate_epsilon_within_family.py

Calibrate an equivalence margin epsilon for "95% similar" claims using within-family anchor distances.

Why
----
Using epsilon = eps_frac * dist(PSO_anchor, ES_anchor) can be arbitrary with only one anchor per family.
With TWO anchors per family, we can define epsilon from within-family variability:

  epsilon_success = max( d95(PSO_GBEST vs PSO_RING), d95(ES_1P1 vs ES_MULAMBDA) )
  epsilon_fitness = same but computed on fitness curves

This yields a defensible 95% similarity boundary:
two algorithms are behaviourally similar if their distance is <= epsilon (with bootstrap 95% quantile).

Inputs
------
- instance_algo_budget_summary.csv (must contain all four anchors)
- runs_detail.csv (optional but recommended for fitness calibration)

Outputs
-------
- epsilon_calibration.json
- a short text report

Usage
-----
python dorigo_calibrate_epsilon_within_family.py ^
  --summary .\out_final_with_anchors\instance_algo_budget_summary.csv ^
  --runs    .\out_final_with_anchors\runs_detail.csv ^
  --out     .\out_final_with_anchors\epsilon_calibration.json ^
  --B 2000 --seed 0
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x=None, total=None, desc=None, **kw):
        return x if x is not None else range(total or 0)

ANCHORS = {
    "PSO": ["PSO_GBEST", "PSO_RING"],
    "ES":  ["ES_1P1", "ES_MULAMBDA"],
}

def zscore_features(M: np.ndarray) -> np.ndarray:
    mu = M.mean(axis=0); sd = M.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (M - mu) / sd

def stratified_instance_weights(instances: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out=[]
    for prob, g in instances.groupby("problem"):
        n=len(g)
        idx = rng.integers(0, n, size=n)
        samp=g.iloc[idx]
        w = samp.groupby(["problem","instance_id"]).size().reset_index(name="w")
        out.append(w)
    return pd.concat(out, ignore_index=True)

def build_success_inst_table(df: pd.DataFrame) -> pd.DataFrame:
    return df[["problem","instance_id","algo_variant","budget","target","beta_mean"]].rename(columns={"beta_mean":"metric"})

def build_fitness_inst_table(runs: pd.DataFrame) -> pd.DataFrame:
    runs_mean = (runs.groupby(["problem","instance_id","algo_variant","budget"], as_index=False)
                 .agg(best=("best","mean")))
    def add_fit(g):
        x = g["best"].to_numpy(float)
        q05 = np.quantile(x,0.05); q95=np.quantile(x,0.95)
        denom=q95-q05
        if abs(denom) < 1e-12:
            score=np.ones_like(x)
        else:
            regret=np.clip((x-q05)/denom,0,1)
            score=1-regret
        g=g.copy(); g["metric"]=score
        return g
    inst = runs_mean.groupby(["problem","instance_id"], group_keys=False).apply(add_fit)
    inst["target"]="all"
    return inst[["problem","instance_id","algo_variant","budget","target","metric"]]

def curve_matrix(inst_tbl: pd.DataFrame, algos: list[str], targets: list[str], budgets: list[int], w: pd.DataFrame|None) -> np.ndarray:
    base = inst_tbl.copy()
    if w is not None:
        base = base.merge(w, on=["problem","instance_id"], how="inner")
        base["wm"] = base["metric"] * base["w"]
        g = (base.groupby(["algo_variant","target","budget"], as_index=False)
                .agg(num=("wm","sum"), den=("w","sum")))
        g["metric"] = g["num"]/g["den"]
        g = g.drop(columns=["num","den"])
    else:
        g = (base.groupby(["algo_variant","target","budget"], as_index=False)
                .agg(metric=("metric","mean")))
    cols = pd.MultiIndex.from_product([targets, budgets], names=["target","budget"])
    wide = g.pivot(index="algo_variant", columns=["target","budget"], values="metric").reindex(index=algos).reindex(columns=cols)
    M = wide.to_numpy(float)
    col_mu = np.nanmean(M, axis=0)
    idx = np.where(np.isnan(M))
    if len(idx[0])>0:
        M[idx] = np.take(col_mu, idx[1])
    return M

def dist(a: np.ndarray, b: np.ndarray) -> float:
    d = a-b
    return float(np.sqrt(np.dot(d,d)))

def bootstrap_dist(inst_tbl, algos, targets, budgets, a1, a2, instances, B, seed):
    rng = np.random.default_rng(seed)
    dists=[]
    for k in tqdm(range(B), desc=f"bootstrap {a1} vs {a2}"):
        w = stratified_instance_weights(instances, rng)
        M = curve_matrix(inst_tbl, algos, targets, budgets, w)
        Z = zscore_features(M)
        i1 = algos.index(a1); i2 = algos.index(a2)
        dists.append(dist(Z[i1], Z[i2]))
    dists=np.array(dists,float)
    return float(np.mean(dists)), float(np.quantile(dists,0.95))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--runs", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    df.columns=[c.strip() for c in df.columns]
    budgets = sorted(df.budget.unique().tolist())
    targets = sorted(df.target.unique().tolist())

    # anchors must exist
    for fam, lst in ANCHORS.items():
        for a in lst:
            if a not in df.algo_variant.unique():
                raise ValueError(f"Missing anchor {a} in summary.")
    all_algos = sorted(df.algo_variant.unique().tolist())

    instances = df[["problem","instance_id"]].drop_duplicates().reset_index(drop=True)

    # Success calibration
    inst_succ = build_success_inst_table(df)
    succ = {}
    for fam, (a1,a2) in ANCHORS.items():
        m, p95 = bootstrap_dist(inst_succ, all_algos, targets, budgets, a1, a2, instances, args.B, args.seed+1)
        succ[f"{fam}_mean"] = m
        succ[f"{fam}_p95"] = p95
    epsilon_success = max(succ["PSO_p95"], succ["ES_p95"])

    out = {"epsilon_success": epsilon_success, "success_within": succ, "budgets": budgets, "targets": targets}

    # Fitness calibration (optional)
    if args.runs:
        runs = pd.read_csv(args.runs)
        runs.columns=[c.strip() for c in runs.columns]
        inst_fit = build_fitness_inst_table(runs)
        fit = {}
        for fam, (a1,a2) in ANCHORS.items():
            m, p95 = bootstrap_dist(inst_fit, all_algos, ["all"], budgets, a1, a2, instances, args.B, args.seed+9)
            fit[f"{fam}_mean"] = m
            fit[f"{fam}_p95"] = p95
        epsilon_fitness = max(fit["PSO_p95"], fit["ES_p95"])
        out["epsilon_fitness"] = epsilon_fitness
        out["fitness_within"] = fit

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    report = out_path.with_suffix(".txt")
    report.write_text(
        "Within-family epsilon calibration\n"
        f"epsilon_success={out['epsilon_success']:.4f}\n"
        + (f"epsilon_fitness={out.get('epsilon_fitness', float('nan')):.4f}\n" if 'epsilon_fitness' in out else "")
        + "Within-family distances (p95):\n"
        + "\n".join([f"  {k}={v:.4f}" for k,v in out["success_within"].items() if k.endswith("p95")]) + "\n",
        encoding="utf-8"
    )
    print("[OK] wrote:", out_path)
    print("[OK] wrote:", report)

if __name__ == "__main__":
    main()
