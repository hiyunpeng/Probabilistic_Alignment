#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dorigo_make_all_figures_tables.py

One-shot generator for ALL dissertation artifacts:
  - Figures (PNG) under <out_dir>/figs
  - Tables (LaTeX .tex) under <out_dir>/tables
  - Useful CSVs (clusters, portfolios, distance matrices) under <out_dir>/csv

Inputs (required)
-----------------
--summary   instance_algo_budget_summary.csv
--runs      runs_detail.csv
--eps       epsilon_calibration.json

Optional
--------
--pairs     track1_similarity_pairs.csv
           If omitted, the script will compute bootstrap pairwise distances for BOTH success and fitness views.
           If provided, it will use that file as the "fitness-view pairwise distances" (dist_mean/dist_p95),
           and still compute success-view distances from scratch.

Key outputs match dissertation placeholders:
  FIG: success_vs_budget_easy/med/hard.png
  FIG: fitness_vs_budget.png
  FIG: dual_view_confusion.png
  FIG: dendrogram_fitness.png, heatmap_fitness.png
  FIG: dendrogram_success.png, heatmap_success.png
  TAB: coverage_overview.tex
  TAB: final_max_budget_success.tex
  TAB: final_headroom.tex
  TAB: epsilon_calibration.tex
  TAB: clusters_success_pso_p95.tex
  TAB: clusters_fitness_epsilon.tex
  TAB: portfolio_comparison.tex
  TAB: nearest_neighbours_fitness.tex, nearest_neighbours_success.tex
  TAB: dual_view_labels.tex

Usage (Windows CMD)
-------------------
python dorigo_make_all_figures_tables.py ^
  --summary .\\instance_algo_budget_summary.csv ^
  --runs    .\\runs_detail.csv ^
  --pairs   .\\track1_similarity_pairs.csv ^
  --eps     .\\epsilon_calibration.json ^
  --out_dir .\\dissertation_artifacts ^
  --B 2000 --seed 0 ^
  --use_pairs_as_fitness

Usage (Linux/macOS)
-------------------
python dorigo_make_all_figures_tables.py \\
  --summary ./instance_algo_budget_summary.csv \\
  --runs    ./runs_detail.csv \\
  --pairs   ./track1_similarity_pairs.csv \\
  --eps     ./epsilon_calibration.json \\
  --out_dir ./dissertation_artifacts \\
  --B 2000 --seed 0 \\
  --use_pairs_as_fitness

Notes
-----
- Minimisation is assumed for fitness normalisation.
- Bootstrapping is stratified by problem family.
- For strict "95% similar" language, prefer dist_p95 pair tables.
"""

from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x=None, total=None, desc=None, **kw):
        return x if x is not None else range(total or 0)

# -------------------------
# Utilities
# -------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "--"
    return f"{x:.3f}"

def latex_escape(s: str) -> str:
    return s.replace("&", "\\&").replace("%", "\\%")

def write_tex_table(path: Path, header: List[str], rows: List[List[str]], caption: str, label: str) -> None:
    lines = []
    lines.append(r"\\begin{table}[t]")
    lines.append(r"\\centering")
    lines.append(rf"\\caption{{{caption}}}")
    lines.append(rf"\\label{{{label}}}")
    lines.append(r"\\vspace{2mm}")
    lines.append(r"\\begin{tabular}{" + "l" + "c"*(len(header)-1) + r"}")
    lines.append(r"\\toprule")
    lines.append(" & ".join(header) + r" \\\\")
    lines.append(r"\\midrule")
    for r in rows:
        lines.append(" & ".join(r) + r" \\\\")
    lines.append(r"\\bottomrule")
    lines.append(r"\\end{tabular}")
    lines.append(r"\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def zscore_features(M: np.ndarray) -> np.ndarray:
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (M - mu) / sd

def stratified_instance_weights(instances: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = []
    for prob, g in instances.groupby("problem"):
        n = len(g)
        idx = rng.integers(0, n, size=n)
        samp = g.iloc[idx]
        w = samp.groupby(["problem","instance_id"]).size().reset_index(name="w")
        out.append(w)
    return pd.concat(out, ignore_index=True)

# -------------------------
# Input / coverage
# -------------------------
def load_inputs(summary_path: Path, runs_path: Path, eps_path: Path, pairs_path: Optional[Path]):
    df = pd.read_csv(summary_path)
    df.columns = [str(c).strip().lstrip('\ufeff').lower() for c in df.columns]
    runs = pd.read_csv(runs_path)
    runs.columns = [str(c).strip().lstrip('\ufeff').lower() for c in runs.columns]
    eps = json.loads(eps_path.read_text(encoding="utf-8"))
    pairs = None
    if pairs_path is not None and pairs_path.exists():
        pairs = pd.read_csv(pairs_path)
        pairs.columns = [str(c).strip().lstrip('\ufeff').lower() for c in pairs.columns]
    return df, runs, eps, pairs

def compute_coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    budgets = df[["domain","budget"]].drop_duplicates().groupby("domain")["budget"].apply(lambda x: ",".join(map(str, sorted(x.unique()))))
    algos = df[["domain","algo_variant"]].drop_duplicates().groupby("domain")["algo_variant"].nunique()
    probs = df[["domain","problem"]].drop_duplicates().groupby("domain")["problem"].nunique()
    insts = df[["domain","problem","instance_id"]].drop_duplicates().groupby("domain").size()
    targets = df[["domain","target"]].drop_duplicates().groupby("domain")["target"].nunique()
    R = df.groupby("domain")["trials"].agg(lambda x: ",".join(map(str, sorted(pd.unique(x)))))
    out = pd.DataFrame({
        "Domain": budgets.index,
        "#Problems": probs.values,
        "#Instances": insts.values,
        "Budgets": budgets.values,
        "#Algos": algos.values,
        "#Targets": targets.values,
        "R (trials/cell)": R.values
    })
    return out

# -------------------------
# Success curves
# -------------------------
def mean_success_curves(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["algo_variant","budget","target"], as_index=False)
              .agg(beta_mean=("beta_mean","mean")))

def plot_success_vs_budget(curves: pd.DataFrame, out_dir: Path, title_prefix: str = "Success vs budget"):
    targets = sorted(curves["target"].unique().tolist())
    algos = sorted(curves["algo_variant"].unique().tolist())
    for t in targets:
        plt.figure(figsize=(9.5,4.4))
        for a in algos:
            sub = curves[(curves.algo_variant==a) & (curves.target==t)].sort_values("budget")
            if len(sub)==0:
                continue
            plt.plot(sub["budget"], sub["beta_mean"], marker="o", label=a)
        plt.xlabel("Budget (function evaluations)")
        plt.ylabel("Mean posterior success (beta_mean)")
        plt.title(f"{title_prefix} ({t})")
        plt.legend(ncol=4, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir/f"success_vs_budget_{t}.png", dpi=220)
        plt.close()

# -------------------------
# Fitness curves (progress)
# -------------------------
def build_instance_fitness(runs: pd.DataFrame) -> pd.DataFrame:
    rm = (runs.groupby(["problem","instance_id","algo_variant","budget"], as_index=False)
            .agg(best=("best","mean")))
    def add_score(g: pd.DataFrame) -> pd.DataFrame:
        x = g["best"].to_numpy(float)
        q05 = np.quantile(x, 0.05)
        q95 = np.quantile(x, 0.95)
        denom = q95 - q05
        if abs(denom) < 1e-12:
            score = np.ones_like(x)
        else:
            regret = np.clip((x - q05)/denom, 0, 1)
            score = 1 - regret
        gg = g.copy()
        gg["fitness_score"] = score
        return gg
    return rm.groupby(["problem","instance_id"], group_keys=False).apply(add_score)

def mean_fitness_curves(scored: pd.DataFrame) -> pd.DataFrame:
    return (scored.groupby(["algo_variant","budget"], as_index=False)
                  .agg(fitness=("fitness_score","mean")))

def plot_fitness_vs_budget(curves: pd.DataFrame, out_dir: Path):
    algos = sorted(curves["algo_variant"].unique().tolist())
    plt.figure(figsize=(9.5,4.4))
    for a in algos:
        sub = curves[curves.algo_variant==a].sort_values("budget")
        if len(sub)==0:
            continue
        plt.plot(sub["budget"], sub["fitness"], marker="o", label=a)
    plt.xlabel("Budget (function evaluations)")
    plt.ylabel("Mean normalised fitness score")
    plt.title("Normalised fitness progress vs budget")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir/"fitness_vs_budget.png", dpi=220)
    plt.close()

# -------------------------
# Feature matrices for distances
# -------------------------
def feature_matrix_success(df: pd.DataFrame, algos: List[str], budgets: List[int], targets: List[str], w: Optional[pd.DataFrame]=None) -> np.ndarray:
    base = df[["problem","instance_id","algo_variant","budget","target","beta_mean"]].rename(columns={"beta_mean":"metric"})
    if w is not None:
        m = base.merge(w, on=["problem","instance_id"], how="inner")
        m["wm"] = m["metric"] * m["w"]
        g = (m.groupby(["algo_variant","target","budget"], as_index=False)
               .agg(num=("wm","sum"), den=("w","sum")))
        g["metric"] = g["num"]/g["den"]
    else:
        g = (base.groupby(["algo_variant","target","budget"], as_index=False)
               .agg(metric=("metric","mean")))
    cols = pd.MultiIndex.from_product([targets, budgets], names=["target","budget"])
    wide = (g.pivot(index="algo_variant", columns=["target","budget"], values="metric")
              .reindex(index=algos).reindex(columns=cols))
    M = wide.to_numpy(float)
    col_mu = np.nanmean(M, axis=0)
    idx = np.where(np.isnan(M))
    if len(idx[0])>0:
        M[idx] = np.take(col_mu, idx[1])
    return M

def feature_matrix_fitness(scored: pd.DataFrame, algos: List[str], budgets: List[int], w: Optional[pd.DataFrame]=None) -> np.ndarray:
    base = scored[["problem","instance_id","algo_variant","budget","fitness_score"]].rename(columns={"fitness_score":"metric"})
    if w is not None:
        m = base.merge(w, on=["problem","instance_id"], how="inner")
        m["wm"] = m["metric"] * m["w"]
        g = (m.groupby(["algo_variant","budget"], as_index=False)
               .agg(num=("wm","sum"), den=("w","sum")))
        g["metric"] = g["num"]/g["den"]
    else:
        g = (base.groupby(["algo_variant","budget"], as_index=False)
               .agg(metric=("metric","mean")))
    wide = (g.pivot(index="algo_variant", columns="budget", values="metric")
              .reindex(index=algos).reindex(columns=budgets))
    M = wide.to_numpy(float)
    col_mu = np.nanmean(M, axis=0)
    idx = np.where(np.isnan(M))
    if len(idx[0])>0:
        M[idx] = np.take(col_mu, idx[1])
    return M

def bootstrap_pairwise_distances(df: pd.DataFrame,
                                 scored: pd.DataFrame,
                                 view: str,
                                 B: int,
                                 seed: int,
                                 out_csv: Path) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    algos = sorted(df["algo_variant"].unique().tolist())
    budgets_success = sorted(df["budget"].unique().tolist())
    targets = sorted(df["target"].unique().tolist())
    budgets_fitness = sorted(scored["budget"].unique().tolist())
    budgets_common = sorted(set(budgets_success).intersection(set(budgets_fitness)))
    if len(budgets_common) < 2:
        budgets_common = budgets_success

    instances = df[["problem","instance_id"]].drop_duplicates().reset_index(drop=True)

    pairs=[]
    for i,a in enumerate(algos):
        for j,b in enumerate(algos):
            if j<=i:
                continue
            pairs.append((a,b,i,j))
    dist_boot = np.zeros((B, len(pairs)), dtype=float)

    for k in tqdm(range(B), desc=f"bootstrap distances ({view})"):
        w = stratified_instance_weights(instances, rng)
        if view == "success":
            M = feature_matrix_success(df, algos, budgets_success, targets, w=w)
        else:
            M = feature_matrix_fitness(scored, algos, budgets_common, w=w)
        Z = zscore_features(M)
        for p_idx,(au,av,i,j) in enumerate(pairs):
            d = Z[i]-Z[j]
            dist_boot[k, p_idx] = float(np.sqrt(np.dot(d,d)))

    out = pd.DataFrame({
        "algo_u":[p[0] for p in pairs],
        "algo_v":[p[1] for p in pairs],
        "dist_mean":dist_boot.mean(axis=0),
        "dist_p95":np.quantile(dist_boot, 0.95, axis=0),
    })
    out.to_csv(out_csv, index=False)
    return out

def pairs_to_matrix(pairs_df: pd.DataFrame, algos: List[str], col: str="dist_p95") -> pd.DataFrame:
    D = pd.DataFrame(np.zeros((len(algos), len(algos))), index=algos, columns=algos, dtype=float)
    for r in pairs_df.itertuples(index=False):
        u = r.algo_u; v = r.algo_v
        d = float(getattr(r, col))
        D.loc[u,v]=d; D.loc[v,u]=d
    np.fill_diagonal(D.values, 0.0)
    return D

# -------------------------
# Clustering + plots
# -------------------------
def plot_dendrogram_and_heatmap(dist_df: pd.DataFrame, out_fig_dir: Path, prefix: str):
    algos = list(dist_df.index)
    D = dist_df.loc[algos, algos].to_numpy()
    D = (D + D.T)/2.0
    np.fill_diagonal(D, 0.0)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")

    plt.figure(figsize=(9.5,4.2))
    dn = dendrogram(Z, labels=algos, leaf_rotation=90)
    plt.title(f"Dendrogram ({prefix})")
    plt.tight_layout()
    plt.savefig(out_fig_dir/f"dendrogram_{prefix}.png", dpi=220)
    plt.close()

    order = dn["ivl"]
    D_ord = dist_df.loc[order, order].to_numpy()
    plt.figure(figsize=(7.2,6.2))
    plt.imshow(D_ord)
    plt.xticks(range(len(order)), order, rotation=90, fontsize=8)
    plt.yticks(range(len(order)), order, fontsize=8)
    plt.title(f"Distance heatmap ({prefix})")
    plt.tight_layout()
    plt.savefig(out_fig_dir/f"heatmap_{prefix}.png", dpi=220)
    plt.close()
    return order

# -------------------------
# Clusters / portfolios
# -------------------------
def build_components(nodes: List[str], edges: List[Tuple[str,str]]) -> List[List[str]]:
    g={n:set() for n in nodes}
    for u,v in edges:
        if u not in g: g[u]=set()
        if v not in g: g[v]=set()
        g[u].add(v); g[v].add(u)
    seen=set(); comps=[]
    for n in nodes:
        if n in seen: continue
        stack=[n]; seen.add(n); comp=[]
        while stack:
            x=stack.pop(); comp.append(x)
            for y in g[x]:
                if y not in seen:
                    seen.add(y); stack.append(y)
        comps.append(sorted(comp))
    comps.sort(key=lambda c:(-len(c), c))
    return comps

def clusters_from_pairs(pairs: pd.DataFrame, nodes: List[str], threshold: float, use_col: str="dist_p95") -> pd.DataFrame:
    edges = [(r.algo_u, r.algo_v) for r in pairs.itertuples(index=False) if float(getattr(r, use_col)) <= threshold]
    comps = build_components(nodes, edges)
    rows=[]
    for i,c in enumerate(comps):
        rows.append({"cluster_id":i, "cluster_size":len(c), "representative":sorted(c)[0], "members":",".join(c)})
    return pd.DataFrame(rows).sort_values(["cluster_size","cluster_id"], ascending=[False, True])

def write_clusters_tex(path: Path, clusters: pd.DataFrame, caption: str, label: str):
    header = ["Cluster", "Size", "Representative", "Members"]
    rows=[]
    for r in clusters.itertuples(index=False):
        rows.append([str(r.cluster_id), str(r.cluster_size),
                     rf"\\texttt{{{latex_escape(r.representative)}}}",
                     rf"\\texttt{{{latex_escape(r.members)}}}"])
    write_tex_table(path, header, rows, caption, label)

# -------------------------
# Nearest neighbours
# -------------------------
def nearest_neighbours_table_from_matrix(D: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    algos = list(D.index)
    out=[]
    for a in algos:
        s = D.loc[a].copy().drop(index=a, errors="ignore")
        nn = s.sort_values().head(k)
        out.append({
            "algo": a,
            "nn1": nn.index[0], "d1": float(nn.iloc[0]),
            "nn2": nn.index[1], "d2": float(nn.iloc[1]),
            "nn3": nn.index[2], "d3": float(nn.iloc[2]),
        })
    return pd.DataFrame(out)

def write_nearest_neighbours_tex(path: Path, nn: pd.DataFrame, caption: str, label: str):
    header=["Algorithm","NN1","d","NN2","d","NN3","d"]
    rows=[]
    for r in nn.itertuples(index=False):
        rows.append([
            rf"\\texttt{{{latex_escape(r.algo)}}}",
            rf"\\texttt{{{latex_escape(r.nn1)}}}", fmt(r.d1),
            rf"\\texttt{{{latex_escape(r.nn2)}}}", fmt(r.d2),
            rf"\\texttt{{{latex_escape(r.nn3)}}}", fmt(r.d3),
        ])
    write_tex_table(path, header, rows, caption, label)

# -------------------------
# Dual view labels + confusion
# -------------------------
def family_label_from_features(M: np.ndarray,
                               algos: List[str],
                               fam_anchors: Dict[str, List[str]],
                               headroom: np.ndarray,
                               headroom_gate: float = 0.02) -> List[str]:
    Z = zscore_features(M)
    fam_d = {}
    for fam, anchors in fam_anchors.items():
        idx = [algos.index(a) for a in anchors if a in algos]
        if len(idx)==0:
            raise ValueError(f"Missing anchors for family {fam}: {anchors}")
        d = np.min([np.sqrt(((Z - Z[i])**2).sum(axis=1)) for i in idx], axis=0)
        fam_d[fam] = d
    dP = fam_d["PSO"]; dE = fam_d["ES"]
    labels=[]
    for hr, dp, de in zip(headroom, dP, dE):
        if hr < headroom_gate:
            labels.append("FLAT")
        else:
            labels.append("PSO" if dp <= de else "ES")
    return labels

def plot_confusion(labels_s: List[str], labels_f: List[str], out_path: Path, title: str):
    order = ["PSO","ES","FLAT"]
    idx = {k:i for i,k in enumerate(order)}
    cm = np.zeros((3,3), dtype=int)
    for a,b in zip(labels_s, labels_f):
        if a not in idx: a="FLAT"
        if b not in idx: b="FLAT"
        cm[idx[a], idx[b]] += 1
    plt.figure(figsize=(5.2,4.2))
    ax = plt.gca()
    ax.imshow(cm)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(order); ax.set_yticklabels(order)
    ax.set_title(title)
    for i in range(3):
        for j in range(3):
            ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

# -------------------------
# Tables
# -------------------------
def build_max_budget_success_table(curves: pd.DataFrame, max_budget: int) -> pd.DataFrame:
    return (curves[curves.budget==max_budget]
            .pivot(index="algo_variant", columns="target", values="beta_mean")
            .reset_index())

def build_headroom_bootstrap(df: pd.DataFrame, B: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    algos = sorted(df["algo_variant"].unique().tolist())
    budgets = sorted(df["budget"].unique().tolist())
    targets = sorted(df["target"].unique().tolist())
    bmin, bmax = budgets[0], budgets[-1]

    instances = df[["problem","instance_id"]].drop_duplicates().reset_index(drop=True)
    base = df[["problem","instance_id","algo_variant","budget","target","beta_mean"]].rename(columns={"beta_mean":"metric"})

    hr_boot = np.zeros((B, len(algos)), dtype=float)
    for k in tqdm(range(B), desc="bootstrap headroom"):
        w = stratified_instance_weights(instances, rng)
        m = base.merge(w, on=["problem","instance_id"], how="inner")
        m["wm"] = m["metric"] * m["w"]
        g = (m.groupby(["algo_variant","target","budget"], as_index=False)
               .agg(num=("wm","sum"), den=("w","sum")))
        g["metric"] = g["num"]/g["den"]
        for ai,a in enumerate(algos):
            sub = g[g.algo_variant==a].pivot(index="target", columns="budget", values="metric")
            hr = float((sub[bmax]-sub[bmin]).max()) if (bmin in sub.columns and bmax in sub.columns) else float("nan")
            hr_boot[k, ai] = hr

    out=[]
    for ai,a in enumerate(algos):
        x = hr_boot[:,ai]
        out.append({
            "algo_variant": a,
            "headroom_mean": float(np.nanmean(x)),
            "headroom_p05": float(np.nanquantile(x,0.05)),
            "headroom_p95": float(np.nanquantile(x,0.95)),
        })
    return pd.DataFrame(out).sort_values("headroom_mean", ascending=False)

def write_headroom_tex(path: Path, head: pd.DataFrame, caption: str, label: str):
    header=["Algorithm","Headroom mean","p05","p95"]
    rows=[]
    for r in head.itertuples(index=False):
        rows.append([rf"\\texttt{{{latex_escape(r.algo_variant)}}}", fmt(r.headroom_mean), fmt(r.headroom_p05), fmt(r.headroom_p95)])
    write_tex_table(path, header, rows, caption, label)

def write_epsilon_tex(path: Path, eps: Dict, caption: str, label: str):
    header=["Metric","Global $\\epsilon$","PSO p95","ES p95"]
    rows=[]
    rows.append(["Success (attainment)",
                 fmt(float(eps["epsilon_success"])),
                 fmt(float(eps["success_within"]["PSO_p95"])),
                 fmt(float(eps["success_within"]["ES_p95"]))])
    rows.append(["Fitness (progress)",
                 fmt(float(eps["epsilon_fitness"])),
                 fmt(float(eps["fitness_within"]["PSO_p95"])),
                 fmt(float(eps["fitness_within"]["ES_p95"]))])
    write_tex_table(path, header, rows, caption, label)

def write_portfolio_tex(path: Path, clusters_s: pd.DataFrame, clusters_f: pd.DataFrame, caption: str, label: str):
    header=["View","Cluster ID","Representative","Size","Members"]
    rows=[]
    for r in clusters_s.itertuples(index=False):
        rows.append(["Success (PSO p95)", str(r.cluster_id), rf"\\texttt{{{latex_escape(r.representative)}}}", str(r.cluster_size), rf"\\texttt{{{latex_escape(r.members)}}}"])
    for r in clusters_f.itertuples(index=False):
        rows.append(["Fitness (global $\\epsilon$)", str(r.cluster_id), rf"\\texttt{{{latex_escape(r.representative)}}}", str(r.cluster_size), rf"\\texttt{{{latex_escape(r.members)}}}"])
    write_tex_table(path, header, rows, caption, label)

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--eps", required=True)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_pairs_as_fitness", action="store_true")
    args = ap.parse_args()

    out_dir = ensure_dir(Path(args.out_dir))
    fig_dir = ensure_dir(out_dir/"figs")
    tab_dir = ensure_dir(out_dir/"tables")
    csv_dir = ensure_dir(out_dir/"csv")

    df, runs, eps, pairs = load_inputs(Path(args.summary), Path(args.runs), Path(args.eps),
                                       Path(args.pairs) if args.pairs else None)

    # Coverage
    cov = compute_coverage_table(df)
    cov.to_csv(csv_dir/"coverage_overview.csv", index=False)
    cov_rows=[]
    for r in cov.itertuples(index=False):
        cov_rows.append([str(r.Domain), str(r._1), str(r._2), latex_escape(str(r.Budgets)), str(r._4), str(r._5), latex_escape(str(r._6))])
    write_tex_table(tab_dir/"coverage_overview.tex",
                    ["Domain","#Problems","#Instances","Budgets","#Algos","#Targets","R"],
                    cov_rows,
                    caption="Coverage overview of experiments (per domain).",
                    label="tab:coverage_overview")

    # Fitness coverage from runs_detail (may have a different ladder than success summary)
    runs_cov = runs[["problem","instance_id","budget","algo_variant"]].drop_duplicates()
    runs_budgets = ",".join(map(str, sorted(runs_cov["budget"].unique().tolist())))
    runs_probs = runs_cov["problem"].nunique()
    runs_insts = runs_cov[["problem","instance_id"]].drop_duplicates().shape[0]
    runs_algos = runs_cov["algo_variant"].nunique()
    # approximate R as median number of seeds per (problem,instance,algo,budget)
    if "seed" in runs.columns:
        r_counts = (runs.groupby(["problem","instance_id","algo_variant","budget"])["seed"].nunique())
        runs_R = int(r_counts.median())
    else:
        runs_R = int(runs.groupby(["problem","instance_id","algo_variant","budget"]).size().median())
    runs_cov_tex_rows = [["cont", str(runs_probs), str(runs_insts), latex_escape(runs_budgets), str(runs_algos), "--", str(runs_R)]]
    write_tex_table(tab_dir/"coverage_overview_fitness.tex",
                    ["Domain","#Problems","#Instances","Budgets","#Algos","#Targets","R"],
                    runs_cov_tex_rows,
                    caption="Coverage overview of progress (fitness) runs derived from run-level logs.",
                    label="tab:coverage_overview_fitness")

    # Success curves
    succ_curves = mean_success_curves(df)
    succ_curves.to_csv(csv_dir/"success_curves.csv", index=False)
    plot_success_vs_budget(succ_curves, fig_dir)

    budgets = sorted(df["budget"].unique().tolist())
    max_b = budgets[-1]

    # Max budget table
    max_tbl = build_max_budget_success_table(succ_curves, max_b)
    max_tbl.to_csv(csv_dir/"max_budget_success.csv", index=False)
    header=["Algorithm","easy","med","hard"]
    rows=[]
    if "hard" in max_tbl.columns:
        max_tbl = max_tbl.sort_values("hard", ascending=False)
    for r in max_tbl.itertuples(index=False):
        rows.append([rf"\\texttt{{{latex_escape(r.algo_variant)}}}",
                     fmt(getattr(r, "easy", float("nan"))),
                     fmt(getattr(r, "med", float("nan"))),
                     fmt(getattr(r, "hard", float("nan")))])
    write_tex_table(tab_dir/"final_max_budget_success.tex",
                    header, rows,
                    caption=f"Mean posterior success (beta\\_mean) at max budget {max_b}.",
                    label="tab:max_budget_success")

    # Headroom
    head = build_headroom_bootstrap(df, B=args.B, seed=args.seed+1)
    head.to_csv(csv_dir/"headroom_bootstrap.csv", index=False)
    write_headroom_tex(tab_dir/"final_headroom.tex",
                       head,
                       caption=f"Headroom across budget ladder (bootstrap over instances, B={args.B}).",
                       label="tab:headroom")

    # Fitness curves
    scored = build_instance_fitness(runs)
    scored.to_csv(csv_dir/"instance_fitness_scores.csv", index=False)
    fit_curves = mean_fitness_curves(scored)
    fit_curves.to_csv(csv_dir/"fitness_curves.csv", index=False)
    plot_fitness_vs_budget(fit_curves, fig_dir)

    # Epsilon table
    write_epsilon_tex(tab_dir/"epsilon_calibration.tex",
                      eps,
                      caption="Within-family calibrated similarity margins (bootstrap 95\\% quantile distances).",
                      label="tab:epsilon_calibration")

    # Pairwise distances
    pairs_success = bootstrap_pairwise_distances(df, scored, view="success",
                                                 B=args.B, seed=args.seed+11,
                                                 out_csv=csv_dir/"pairs_success_bootstrap.csv")

    if pairs is not None and args.use_pairs_as_fitness:
        # Use provided as fitness-view pairs
        pairs_fitness = pairs[["algo_u","algo_v","dist_mean","dist_p95"]].copy()
        pairs_fitness.to_csv(csv_dir/"pairs_fitness_from_input.csv", index=False)
    else:
        pairs_fitness = bootstrap_pairwise_distances(df, scored, view="fitness",
                                                     B=args.B, seed=args.seed+21,
                                                     out_csv=csv_dir/"pairs_fitness_bootstrap.csv")

    algos_success = sorted(df["algo_variant"].unique().tolist())
    algos_fitness = sorted(set(pairs_fitness["algo_u"]).union(set(pairs_fitness["algo_v"]))) if (pairs is not None and args.use_pairs_as_fitness) else sorted(df["algo_variant"].unique().tolist())
    algos_all = algos_success  # for success-view artifacts

    D_succ = pairs_to_matrix(pairs_success, algos_success, "dist_p95")
    D_fit  = pairs_to_matrix(pairs_fitness, algos_fitness, "dist_p95")
    D_succ.to_csv(csv_dir/"dist_matrix_success_p95.csv")
    D_fit.to_csv(csv_dir/"dist_matrix_fitness_p95.csv")

    plot_dendrogram_and_heatmap(D_succ, fig_dir, "success")
    plot_dendrogram_and_heatmap(D_fit, fig_dir, "fitness")

    nn_s = nearest_neighbours_table_from_matrix(D_succ, k=3)
    nn_f = nearest_neighbours_table_from_matrix(D_fit, k=3)
    nn_s.to_csv(csv_dir/"nearest_neighbours_success.csv", index=False)
    nn_f.to_csv(csv_dir/"nearest_neighbours_fitness.csv", index=False)
    write_nearest_neighbours_tex(tab_dir/"nearest_neighbours_success.tex", nn_s,
                                 caption="Nearest neighbours under success-view distances (dist\\_p95).",
                                 label="tab:nn_success")
    write_nearest_neighbours_tex(tab_dir/"nearest_neighbours_fitness.tex", nn_f,
                                 caption="Nearest neighbours under fitness-view distances (dist\\_p95).",
                                 label="tab:nn_fitness")

    eps_fitness = float(eps["epsilon_fitness"])
    pso_p95_success = float(eps["success_within"]["PSO_p95"])
    nodes_s = sorted(set(pairs_success["algo_u"]).union(set(pairs_success["algo_v"])))
    nodes_f = sorted(set(pairs_fitness["algo_u"]).union(set(pairs_fitness["algo_v"])))
    clusters_s = clusters_from_pairs(pairs_success, nodes_s, threshold=pso_p95_success, use_col="dist_p95")
    clusters_f = clusters_from_pairs(pairs_fitness, nodes_f, threshold=eps_fitness, use_col="dist_p95")
    clusters_s.to_csv(csv_dir/"clusters_success_pso_p95.csv", index=False)
    clusters_f.to_csv(csv_dir/"clusters_fitness_epsilon.csv", index=False)

    write_clusters_tex(tab_dir/"clusters_success_pso_p95.tex",
                       clusters_s,
                       caption=f"Redundancy clusters under success-view threshold PSO\\_p95={pso_p95_success:.3f}.",
                       label="tab:clusters_success")
    write_clusters_tex(tab_dir/"clusters_fitness_epsilon.tex",
                       clusters_f,
                       caption=f"Redundancy clusters under fitness-view global threshold $\\epsilon$={eps_fitness:.3f}.",
                       label="tab:clusters_fitness")

    write_portfolio_tex(tab_dir/"portfolio_comparison.tex",
                        clusters_s, clusters_f,
                        caption="Minimal portfolios by choosing one representative per redundancy cluster (success vs fitness).",
                        label="tab:portfolio")

    # Dual view confusion (point estimate labels)
    budgets_success = sorted(df["budget"].unique().tolist())
    targets = sorted(df["target"].unique().tolist())
    budgets_fitness = sorted(scored["budget"].unique().tolist())
    budgets_common = sorted(set(budgets_success).intersection(set(budgets_fitness)))
    if len(budgets_common) < 2:
        budgets_common = budgets_success

    M_s = feature_matrix_success(df, algos_success, budgets_common, targets, w=None)
    M_f = feature_matrix_fitness(scored, algos_success, budgets_common, w=None)

    bmin, bmax = budgets_common[0], budgets_common[-1]
    succ_mean = mean_success_curves(df)
    head_s=[]
    for a in algos_success:
        sub = succ_mean[succ_mean.algo_variant==a].pivot(index="target", columns="budget", values="beta_mean")
        hr = float((sub[bmax]-sub[bmin]).max()) if (bmin in sub.columns and bmax in sub.columns) else 0.0
        head_s.append(hr)
    head_s = np.array(head_s, float)
    fit_mean = mean_fitness_curves(scored)
    head_f=[]
    for a in algos_success:
        sub = fit_mean[fit_mean.algo_variant==a].set_index("budget")["fitness"]
        hr = float(sub.loc[bmax]-sub.loc[bmin]) if (bmin in sub.index and bmax in sub.index) else 0.0
        head_f.append(hr)
    head_f = np.array(head_f, float)

    pso_anchors = [a for a in ["PSO_GBEST","PSO_RING"] if a in algos_success]
    es_anchors  = [a for a in ["ES_1P1","ES_MULAMBDA"] if a in algos_success]
    if len(pso_anchors)==0 or len(es_anchors)==0:
        raise ValueError(f"Missing required anchors in success-view data. Have PSO={pso_anchors}, ES={es_anchors}.")
    fam_anchors = {"PSO":pso_anchors, "ES":es_anchors}
    labels_s = family_label_from_features(M_s, algos_all, fam_anchors, head_s, headroom_gate=0.02)
    labels_f = family_label_from_features(M_f, algos_all, fam_anchors, head_f, headroom_gate=0.02)

    plot_confusion(labels_s, labels_f, fig_dir/"dual_view_confusion.png",
                   title="Dual-view confusion (success labels vs fitness labels)")

    dual = pd.DataFrame({"algo_variant":algos_success, "label_success":labels_s, "label_fitness":labels_f})
    dual["disagree"] = dual["label_success"] != dual["label_fitness"]
    dual.to_csv(csv_dir/"dual_view_labels.csv", index=False)

    header=["Algorithm","Success label","Fitness label","Disagree"]
    rows=[]
    for r in dual.sort_values(["disagree","algo_variant"], ascending=[False, True]).itertuples(index=False):
        rows.append([rf"\\texttt{{{latex_escape(r.algo_variant)}}}", r.label_success, r.label_fitness, "Yes" if r.disagree else "No"])
    write_tex_table(tab_dir/"dual_view_labels.tex",
                    header, rows,
                    caption="Dual-view label comparison (success vs fitness).",
                    label="tab:dual_view")

    (out_dir/"README_ARTIFACTS.txt").write_text(
        "Generated dissertation artifacts.\n\n"
        "FIGURES (include via \\\\includegraphics):\n"
        "  figs/success_vs_budget_easy.png\n"
        "  figs/success_vs_budget_med.png\n"
        "  figs/success_vs_budget_hard.png\n"
        "  figs/fitness_vs_budget.png\n"
        "  figs/dual_view_confusion.png\n"
        "  figs/dendrogram_success.png\n"
        "  figs/heatmap_success.png\n"
        "  figs/dendrogram_fitness.png\n"
        "  figs/heatmap_fitness.png\n\n"
        "TABLES (include via \\\\input{tables/...}):\n"
        "  tables/coverage_overview.tex\n"
        f"  tables/final_max_budget_success.tex (max budget={max_b})\n"
        "  tables/final_headroom.tex\n"
        "  tables/epsilon_calibration.tex\n"
        "  tables/clusters_success_pso_p95.tex\n"
        "  tables/clusters_fitness_epsilon.tex\n"
        "  tables/portfolio_comparison.tex\n"
        "  tables/nearest_neighbours_success.tex\n"
        "  tables/nearest_neighbours_fitness.tex\n"
        "  tables/dual_view_labels.tex\n\n"
        "CSVs saved under csv/ for debugging.\n",
        encoding="utf-8"
    )

    print("[OK] All figures/tables written to:", out_dir)

if __name__ == "__main__":
    main()
