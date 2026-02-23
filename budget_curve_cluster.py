#!/usr/bin/env python3
"""
Budget-response curve analysis + clustering (shape-first).

Input:
  instance_algo_budget_summary.csv
  (produced by dorigo_variant_budget_cluster*.py)

Core idea:
  For each algorithm, build a success curve p(b) across budgets (optionally per target),
  then cluster algorithms by curve SHAPE (correlation / z-normalized Euclidean),
  with optional bootstrap stability over instances.

Outputs (under --out_dir):
  - curves_long.csv                (algo, target, budget, mean_beta, ...)
  - curve_kpis.csv                 (algo, target, AUC, slopes, gain, ...)
  - curve_vectors.csv              (algo, features...)
  - curve_distance.csv             (pairwise distances)
  - curve_clusters.csv             (algo -> cluster label)
  - cocluster_matrix.csv           (if bootstrap>0)
  - figs/curves.png
  - figs/heatmap_zcurves.png
  - figs/dendrogram.png
  - figs/cocluster.png             (if bootstrap>0)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", type=str, required=True,
                    help="Path to instance_algo_budget_summary.csv")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory for analysis artifacts")
    ap.add_argument("--domain", type=str, default="cont", choices=["cont", "bin", "all"],
                    help="Filter by domain (default: cont)")
    ap.add_argument("--targets", type=str, default="all",
                    help="Comma list like easy,med,hard or 'all'")
    ap.add_argument("--algos", type=str, default="all",
                    help="Comma list of algo names to include or 'all'")
    ap.add_argument("--view", type=str, default="shape", choices=["abs", "shape", "delta"],
                    help="abs: raw curve; shape: z-normalize curve; delta: differences across budgets")
    ap.add_argument("--dist", type=str, default="corr", choices=["corr", "euclid"],
                    help="corr: 1-corr distance; euclid: Euclidean in chosen view space")
    ap.add_argument("--cluster_k", type=int, default=2,
                    help="Number of clusters to cut dendrogram into (default: 2)")
    ap.add_argument("--linkage", type=str, default="average",
                    choices=["single", "complete", "average", "ward"],
                    help="Hierarchical clustering linkage")
    ap.add_argument("--combine_targets", type=str, default="concat",
                    choices=["concat", "separate"],
                    help="concat: concatenate targets into one vector per algo; separate: cluster per target")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="Bootstrap replicates over instances for cluster stability (0 disables)")
    ap.add_argument("--bootstrap_seed", type=int, default=0,
                    help="RNG seed for bootstrap")
    return ap.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd < eps:
        return (x - mu) * 0.0
    return (x - mu) / sd


def build_curves(df: pd.DataFrame, algos: list, targets: list, budgets: list):
    """
    Returns:
      curves_long: long df (algo,target,budget,mean_beta,mean_best, ...)
      curve_mat: dict[target] -> matrix (n_algos, n_budgets) of mean_beta
    """
    curves = []
    curve_mat = {}

    for tgt in targets:
        sub = df[df["target"] == tgt].copy()
        # mean over instances (instance-level estimates already in beta_mean)
        g = sub.groupby(["algo_variant", "budget"], as_index=False).agg(
            mean_beta=("beta_mean", "mean"),
            mean_best=("mean_best", "mean"),
            mean_trials=("trials", "mean"),
        )
        # ensure full grid
        pivot = g.pivot(index="algo_variant", columns="budget", values="mean_beta")
        pivot = pivot.reindex(index=algos, columns=budgets)

        curve_mat[tgt] = pivot.to_numpy(dtype=float)

        # long
        for a in algos:
            for b in budgets:
                val = pivot.loc[a, b]
                curves.append(
                    {"algo": a, "target": tgt, "budget": int(b), "mean_beta": float(val) if pd.notna(val) else np.nan})

    curves_long = pd.DataFrame(curves)
    return curves_long, curve_mat


def curve_kpis_from_vector(budgets: np.ndarray, y: np.ndarray):
    # y may contain NaNs
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return dict(auc=np.nan, gain=np.nan, slope_early=np.nan, slope_late=np.nan, p_first=np.nan, p_last=np.nan)
    b = budgets[mask].astype(float)
    yy = y[mask].astype(float)
    # normalized AUC in [min_b,max_b]
    auc = np.trapz(yy, b) / (b.max() - b.min())
    gain = yy[-1] - yy[0]
    slope_early = (yy[1] - yy[0]) / (b[1] - b[0])
    slope_late = (yy[-1] - yy[-2]) / (b[-1] - b[-2])
    return dict(auc=float(auc), gain=float(gain), slope_early=float(slope_early), slope_late=float(slope_late),
                p_first=float(yy[0]), p_last=float(yy[-1]))


def make_vectors(curve_mat: dict, budgets: list, targets: list, view: str, combine_targets: str):
    """
    Returns:
      X: (n_algos, n_features)
      feature_names: list[str]
      algo_order: list[str]
    """
    budgets_arr = np.array(budgets, dtype=float)
    n_algos = next(iter(curve_mat.values())).shape[0]
    algo_order = None

    # build per-target transformed vectors
    per_t = {}
    for tgt in targets:
        M = curve_mat[tgt]  # (n_algos, n_budgets)
        if view == "abs":
            V = M.copy()
        elif view == "shape":
            V = np.vstack([zscore(M[i, :]) for i in range(M.shape[0])])
        elif view == "delta":
            # successive differences; shape-first over deltas
            D = np.diff(M, axis=1)
            V = np.vstack([zscore(D[i, :]) for i in range(D.shape[0])])
        else:
            raise ValueError(view)
        per_t[tgt] = V

    # combine targets
    if combine_targets == "concat":
        X = np.concatenate([per_t[t] for t in targets], axis=1)
        feature_names = []
        if view == "delta":
            bnames = [f"{budgets[i + 1]}-{budgets[i]}" for i in range(len(budgets) - 1)]
        else:
            bnames = [str(b) for b in budgets]
        for t in targets:
            for bn in bnames:
                feature_names.append(f"{t}:{bn}")
        return X, feature_names
    else:
        # separate handled by caller
        raise RuntimeError("use combine_targets='separate' path")


def pairwise_distance(X: np.ndarray, metric: str):
    # handle NaNs: simple impute with 0 (safe when using z-scored shape vectors)
    X2 = np.nan_to_num(X, nan=0.0)
    if metric == "euclid":
        d = pdist(X2, metric="euclidean")
    elif metric == "corr":
        # correlation distance = 1 - corr
        d = pdist(X2, metric="correlation")
        # pdist correlation already handles mean-centering; for constant vectors it returns 0/NaN -> fix
        d = np.nan_to_num(d, nan=1.0)
    else:
        raise ValueError(metric)
    return d


def do_cluster(X: np.ndarray, algos: list, dist: str, linkage_method: str, k: int):
    d = pairwise_distance(X, dist)
    Z = linkage(d, method=linkage_method)
    labels = fcluster(Z, t=k, criterion="maxclust")
    clus = pd.DataFrame({"algo": algos, "cluster": labels})
    D = squareform(d)
    dist_df = pd.DataFrame(D, index=algos, columns=algos)
    return Z, clus, dist_df


def plot_curves(curves_long: pd.DataFrame, out_fig: Path):
    # one plot per target, but compact into one figure (no seaborn)
    targets = list(curves_long["target"].unique())
    algos = list(curves_long["algo"].unique())
    budgets = sorted(curves_long["budget"].unique())

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    for tgt in targets:
        sub = curves_long[curves_long["target"] == tgt]
        for algo in algos:
            s2 = sub[sub["algo"] == algo].sort_values("budget")
            ax.plot(s2["budget"].to_numpy(), s2["mean_beta"].to_numpy(),
                    marker="o", linewidth=1, label=f"{algo}-{tgt}")

    ax.set_xlabel("Budget (function evaluations)")
    ax.set_ylabel("Mean posterior success (beta_mean)")
    ax.set_title("Success vs budget (by algorithm and target tier)")
    ax.legend(ncol=2, fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)


def plot_heatmap_z(curve_mat: dict, algos: list, budgets: list, targets: list, out_fig: Path):
    # show concatenated z-scored curves (shape view)
    mats = []
    col_labels = []
    for tgt in targets:
        M = curve_mat[tgt]
        Zm = np.vstack([zscore(M[i, :]) for i in range(M.shape[0])])
        mats.append(Zm)
        col_labels += [f"{b}-{tgt}" for b in budgets]
    H = np.concatenate(mats, axis=1)

    fig = plt.figure(figsize=(12, max(2.5, 0.35 * len(algos))))
    ax = plt.gca()
    im = ax.imshow(H, aspect="auto", cmap="RdBu_r")
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_title("Heatmap (z-scored success curves; shape view)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)


def plot_dendro(Z, algos: list, out_fig: Path):
    fig = plt.figure(figsize=(10, 4))
    ax = plt.gca()
    dendrogram(Z, labels=algos, leaf_rotation=45, ax=ax)
    ax.set_title("Hierarchical clustering dendrogram (budget-response vectors)")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)


def bootstrap_cocluster(df: pd.DataFrame, algos: list, budgets: list, targets: list,
                        view: str, dist: str, linkage_method: str, k: int,
                        combine_targets: str, B: int, seed: int):
    """
    Bootstrap over instances:
      sample instance keys with replacement, recompute curves, re-cluster, accumulate co-clustering frequencies.
    """
    rng = np.random.default_rng(seed)

    # instance key
    df = df.copy()
    df["inst_key"] = df["problem"].astype(str) + "|" + df["instance_id"].astype(str)
    inst_keys = df["inst_key"].unique().tolist()

    n = len(algos)
    co = np.zeros((n, n), dtype=float)

    for b in range(B):
        sample_keys = rng.choice(inst_keys, size=len(inst_keys), replace=True)
        sub = df[df["inst_key"].isin(sample_keys)].copy()

        # recompute mean over sampled instances (duplicates weight naturally)
        # we do this by merging weights: count multiplicities
        counts = pd.Series(sample_keys).value_counts()
        sub["w"] = sub["inst_key"].map(counts).fillna(0).astype(float)

        curve_mat = {}
        for tgt in targets:
            ss = sub[sub["target"] == tgt]

            # Calculate weighted average properly
            # First, create weighted values
            ss = ss.copy()
            ss['weighted_beta'] = ss['beta_mean'] * ss['w']

            # Group by algo_variant and budget, sum the weighted values and weights
            g = ss.groupby(["algo_variant", "budget"], as_index=False).agg(
                weighted_sum=('weighted_beta', 'sum'),
                total_weight=('w', 'sum')
            )

            # Calculate weighted average
            g['mean_beta'] = g['weighted_sum'] / g['total_weight']

            # Pivot to create the matrix
            piv = g.pivot(index="algo_variant", columns="budget", values="mean_beta")
            piv = piv.reindex(index=algos, columns=budgets)
            curve_mat[tgt] = piv.to_numpy(dtype=float)

        if combine_targets != "concat":
            raise NotImplementedError("bootstrap only implemented for concat mode")

        X, _ = make_vectors(curve_mat, budgets, targets, view=view, combine_targets="concat")
        _, clus, _ = do_cluster(X, algos, dist=dist, linkage_method=linkage_method, k=k)
        labels = clus["cluster"].to_numpy()

        # accumulate co-cluster
        for i in range(n):
            for j in range(n):
                co[i, j] += 1.0 if labels[i] == labels[j] else 0.0

    co /= float(B)
    return co


def plot_cocluster(co: np.ndarray, algos: list, out_fig: Path):
    fig = plt.figure(figsize=(7, 6))
    ax = plt.gca()
    im = ax.imshow(co, vmin=0.0, vmax=1.0, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels(algos, rotation=45, ha="right")
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos)
    ax.set_title("Bootstrap co-clustering probability")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    figs_dir = out_dir / "figs"
    ensure_dir(out_dir)
    ensure_dir(figs_dir)

    df = pd.read_csv(args.summary_csv)

    if args.domain != "all":
        df = df[df["domain"] == args.domain].copy()

    # select targets
    if args.targets != "all":
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    else:
        targets = sorted(df["target"].unique().tolist())

    # select algos
    if args.algos != "all":
        algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    else:
        algos = sorted(df["algo_variant"].unique().tolist())

    budgets = sorted(df["budget"].unique().tolist())

    # sanity: keep only rows in algos/targets/budgets
    df = df[df["algo_variant"].isin(algos) & df["target"].isin(targets) & df["budget"].isin(budgets)].copy()

    # build curves
    curves_long, curve_mat = build_curves(df, algos=algos, targets=targets, budgets=budgets)
    curves_long.to_csv(out_dir / "curves_long.csv", index=False)

    # KPIs
    budgets_arr = np.array(budgets, dtype=float)
    kpi_rows = []
    for tgt in targets:
        M = curve_mat[tgt]
        for i, algo in enumerate(algos):
            y = M[i, :]
            kpis = curve_kpis_from_vector(budgets_arr, y)
            kpi_rows.append({"algo": algo, "target": tgt, **kpis})
    kpi_df = pd.DataFrame(kpi_rows)
    kpi_df.to_csv(out_dir / "curve_kpis.csv", index=False)

    # plot curves + heatmap
    plot_curves(curves_long, figs_dir / "curves.png")
    plot_heatmap_z(curve_mat, algos, budgets, targets, figs_dir / "heatmap_zcurves.png")

    if args.combine_targets == "concat":
        X, feat_names = make_vectors(curve_mat, budgets, targets, view=args.view, combine_targets="concat")
        vec_df = pd.DataFrame(X, columns=feat_names)
        vec_df.insert(0, "algo", algos)
        vec_df.to_csv(out_dir / "curve_vectors.csv", index=False)

        Z, clus, dist_df = do_cluster(X, algos, dist=args.dist, linkage_method=args.linkage, k=args.cluster_k)
        clus.to_csv(out_dir / "curve_clusters.csv", index=False)
        dist_df.to_csv(out_dir / "curve_distance.csv", index=True)

        plot_dendro(Z, algos, figs_dir / "dendrogram.png")

        if args.bootstrap and args.bootstrap > 0:
            co = bootstrap_cocluster(df, algos, budgets, targets, view=args.view, dist=args.dist,
                                     linkage_method=args.linkage, k=args.cluster_k,
                                     combine_targets="concat", B=args.bootstrap, seed=args.bootstrap_seed)
            co_df = pd.DataFrame(co, index=algos, columns=algos)
            co_df.to_csv(out_dir / "cocluster_matrix.csv", index=True)
            plot_cocluster(co, algos, figs_dir / "cocluster.png")

    else:
        # separate per target clustering
        rows = []
        for tgt in targets:
            M = curve_mat[tgt]
            # transform
            if args.view == "abs":
                V = M.copy()
            elif args.view == "shape":
                V = np.vstack([zscore(M[i, :]) for i in range(M.shape[0])])
            else:
                D = np.diff(M, axis=1)
                V = np.vstack([zscore(D[i, :]) for i in range(D.shape[0])])

            Z, clus, dist_df = do_cluster(V, algos, dist=args.dist, linkage_method=args.linkage, k=args.cluster_k)
            clus["target"] = tgt
            rows.append(clus)
            # write per-target
            dist_df.to_csv(out_dir / f"curve_distance_{tgt}.csv", index=True)
            plot_dendro(Z, algos, figs_dir / f"dendrogram_{tgt}.png")

        pd.concat(rows, ignore_index=True).to_csv(out_dir / "curve_clusters.csv", index=False)

    print(f"[OK] wrote analysis to: {out_dir}")


if __name__ == "__main__":
    main()