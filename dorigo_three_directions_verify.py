#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
dorigo_three_directions_verify.py

Implements 3 verification tracks:

Track 1) 95% curve-similarity (equivalence) test on your existing summary CSV
- view=success: uses beta_mean over target×budget
- view=fitness: builds normalised fitness scores from median_best (or other) and uses budget curve

Decision rule (similar @95%):
- bootstrap over instances => distance distribution
- similar if 95th percentile of distance < epsilon
- epsilon defaults to eps_frac * dist(PSO_anchor, ES_anchor), scale-free

Track 2) Segment/budget-regime similarity
- split budgets into segments, classify each algo per segment as closer-to-PSO / closer-to-ES / FLAT
- bootstrap label stability (p(label) per segment)

Track 3) Interpolated benchmarks between two base functions
- f_lambda(x) = (1-lam)*fA_norm(x) + lam*fB_norm(x)
- run algorithms across lambdas+budgets to see where attribution flips
(Note: this script keeps Track3 minimal (anchors only) by default; see message in error for extension.)

Dependencies: numpy pandas scipy tqdm matplotlib

Windows CMD examples:
  python dorigo_three_directions_verify.py curvesim ^
    --csv .\out_dorigo_new\instance_algo_budget_summary.csv ^
    --out_dir .\out_dorigo_new\verify_tracks ^
    --view success --B 2000 --eps_frac 0.25 ^
    --pso_anchor PSO_GBEST --es_anchor ES_1P1

  python dorigo_three_directions_verify.py segments ^
    --csv .\out_dorigo_new\instance_algo_budget_summary.csv ^
    --out_dir .\out_dorigo_new\verify_tracks ^
    --view success --B 2000 --headroom_gate 0.02 ^
    --segments "300,500,800|1000,2000|5000,10000" ^
    --pso_anchor PSO_GBEST --es_anchor ES_1P1

  python dorigo_three_directions_verify.py interpolate ^
    --out_dir .\out_dorigo_new\verify_tracks\track3_interpolation ^
    --algos PSO_GBEST,ES_1P1 ^
    --budgets 300,500,800,1000,2000,5000,10000 ^
    --dim 10 --instances_per_problem 10 --R 10 ^
    --A Sphere --Bfunc Rastrigin ^
    --lambdas 0,0.25,0.5,0.75,1.0 ^
    --seed 0
"""

from __future__ import annotations
import argparse
import json
import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import beta as sp_beta

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x=None, total=None, desc=None, **kw):
        return x if x is not None else range(total or 0)


# =============================
# Common helpers
# =============================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def parse_list(s: str, cast=str, sep=",") -> List:
    return [cast(x.strip()) for x in str(s).split(sep) if x.strip()]

def parse_segments(s: str) -> List[List[int]]:
    segs = []
    for part in s.split("|"):
        segs.append([int(x.strip()) for x in part.split(",") if x.strip()])
    return segs

def make_feat_index(targets, budgets):
    return pd.MultiIndex.from_product([targets, budgets], names=["target", "budget"])

def zscore_by_feature(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mu = np.mean(M, axis=0)
    sd = np.std(M, axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (M - mu) / sd

def euclid(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float(np.sqrt(np.dot(d, d)))

def stratified_instance_weights(instances: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = []
    for prob, g in instances.groupby("problem"):
        n = len(g)
        idx = rng.integers(0, n, size=n)
        samp = g.iloc[idx]
        w = samp.groupby(["problem","instance_id"]).size().reset_index(name="w")
        out.append(w)
    return pd.concat(out, ignore_index=True)

def weighted_group_mean(df: pd.DataFrame, group_cols, metric_cols, weight_col="w") -> pd.DataFrame:
    num = df.copy()
    for m in metric_cols:
        num[m] = num[m] * num[weight_col]
    gnum = num.groupby(group_cols, as_index=False)[metric_cols].sum()
    gden = df.groupby(group_cols, as_index=False)[weight_col].sum().rename(columns={weight_col:"_den"})
    out = gnum.merge(gden, on=group_cols, how="left")
    for m in metric_cols:
        out[m] = out[m] / out["_den"]
    return out.drop(columns=["_den"])

def pivot_curves(df_agg: pd.DataFrame, algos, targets, budgets, metric: str) -> np.ndarray:
    wide = df_agg.pivot(index="algo_variant", columns=["target","budget"], values=metric)
    wide = wide.reindex(index=algos, columns=make_feat_index(targets, budgets))
    M = wide.to_numpy(dtype=float)
    col_mu = np.nanmean(M, axis=0)
    idx = np.where(np.isnan(M))
    if len(idx[0]) > 0:
        M[idx] = np.take(col_mu, idx[1])
    return M


# =============================
# Build view matrices
# =============================

def add_fitness_score(df: pd.DataFrame, value_col: str = "median_best", scale_mode: str = "q05q95", sense: str = "minimize") -> pd.DataFrame:
    if value_col not in df.columns:
        raise ValueError(f"value_col={value_col} not found in CSV columns.")
    out = df.copy()
    val = out[value_col].to_numpy(dtype=float)
    if sense.lower().startswith("max"):
        val = -val
    out["_val"] = val

    g = out.groupby(["problem","instance_id"])["_val"]

    if scale_mode == "q05q95":
        best_ref = g.quantile(0.05)
        worst_ref = g.quantile(0.95)
    elif scale_mode == "minmax":
        best_ref = g.min()
        worst_ref = g.max()
    else:
        raise ValueError("scale_mode must be q05q95 or minmax")

    out = out.join(best_ref.rename("_best_ref"), on=["problem","instance_id"])
    out = out.join(worst_ref.rename("_worst_ref"), on=["problem","instance_id"])

    denom = (out["_worst_ref"] - out["_best_ref"]).to_numpy(dtype=float)
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)

    regret = (out["_val"] - out["_best_ref"]) / denom
    regret = np.clip(regret, 0.0, 1.0)
    regret = np.nan_to_num(regret, nan=0.0)
    out["fitness_score"] = 1.0 - regret
    return out.drop(columns=["_val","_best_ref","_worst_ref"], errors="ignore")

def build_view_table(
    df: pd.DataFrame,
    view: str,
    algos: List[str],
    targets: List[str],
    budgets: List[int],
    value_col: str,
    scale_mode: str,
    sense: str,
) -> Tuple[pd.DataFrame, List[str]]:
    if view == "success":
        needed = {"problem","instance_id","algo_variant","target","budget","beta_mean"}
        miss = needed - set(df.columns)
        if miss:
            raise ValueError(f"Missing columns for success view: {sorted(miss)}")
        out = df[df["algo_variant"].isin(algos) & df["target"].isin(targets) & df["budget"].isin(budgets)].copy()
        out = out.rename(columns={"beta_mean":"metric"})
        view_targets = targets
        return out[["problem","instance_id","algo_variant","target","budget","metric"]], view_targets

    if view == "fitness":
        needed = {"problem","instance_id","algo_variant","budget", value_col}
        miss = needed - set(df.columns)
        if miss:
            raise ValueError(f"Missing columns for fitness view: {sorted(miss)}")
        out = df[df["algo_variant"].isin(algos) & df["budget"].isin(budgets)].copy()
        out = add_fitness_score(out, value_col=value_col, scale_mode=scale_mode, sense=sense)
        out["target"] = "all"
        out = out.rename(columns={"fitness_score":"metric"})
        view_targets = ["all"]
        return out[["problem","instance_id","algo_variant","target","budget","metric"]], view_targets

    raise ValueError("view must be success or fitness")


# =============================
# Track 1: 95% similarity
# =============================

def curvesim(args):
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figs"
    ensure_dir(out_dir); ensure_dir(fig_dir)

    df = pd.read_csv(args.csv)

    algos = parse_list(args.algos, str) if args.algos else sorted(df["algo_variant"].unique().tolist())
    budgets = parse_list(args.budgets, int) if args.budgets else sorted(df["budget"].unique().tolist())
    targets = parse_list(args.targets, str) if (args.view == "success" and args.targets) else sorted(df["target"].unique().tolist())

    inst_tbl, view_targets = build_view_table(
        df, view=args.view, algos=algos, targets=targets, budgets=budgets,
        value_col=args.value_col, scale_mode=args.scale_mode, sense=args.sense
    )

    instances = inst_tbl[["problem","instance_id"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(args.seed)

    if args.pso_anchor not in algos or args.es_anchor not in algos:
        raise ValueError(f"Anchors must be present. pso={args.pso_anchor}, es={args.es_anchor}, algos={algos}")

    A = args.pso_anchor
    E = args.es_anchor
    iA = algos.index(A)
    iE = algos.index(E)

    pairs = [(algos[i], algos[j]) for i in range(len(algos)) for j in range(i+1, len(algos))]
    D_boot = np.zeros((args.B, len(pairs)), dtype=float)
    anchor_boot = np.zeros(args.B, dtype=float)

    for b in tqdm(range(args.B), desc="Track1 bootstrap"):
        w = stratified_instance_weights(instances, rng)
        dfw = inst_tbl.merge(w, on=["problem","instance_id"], how="inner")

        agg = weighted_group_mean(
            dfw,
            group_cols=["algo_variant","target","budget"],
            metric_cols=["metric"],
            weight_col="w"
        )
        M = pivot_curves(agg.rename(columns={"metric":"metric"}), algos, view_targets, budgets, "metric")
        Z = zscore_by_feature(M)

        anchor_boot[b] = euclid(Z[iA], Z[iE])

        for k,(u,v) in enumerate(pairs):
            iu = algos.index(u); iv = algos.index(v)
            D_boot[b,k] = euclid(Z[iu], Z[iv])

    eps = float(np.mean(anchor_boot) * args.eps_frac)
    D_mean = D_boot.mean(axis=0)
    D_p95 = np.quantile(D_boot, 0.95, axis=0)
    similar = D_p95 < eps

    out = pd.DataFrame({
        "algo_u":[u for u,v in pairs],
        "algo_v":[v for u,v in pairs],
        "dist_mean":D_mean,
        "dist_p95":D_p95,
        "epsilon":eps,
        "similar_95":similar
    }).sort_values(["similar_95","dist_p95"], ascending=[False, True])

    out.to_csv(out_dir/"track1_similarity_pairs.csv", index=False)

    mat = pd.DataFrame(np.nan, index=algos, columns=algos)
    for (u,v),m in zip(pairs, D_mean):
        mat.loc[u,v]=m; mat.loc[v,u]=m
    np.fill_diagonal(mat.values, 0.0)
    mat.to_csv(out_dir/"track1_distance_matrix.csv", index=True)

    fig = plt.figure(figsize=(7,6))
    ax = plt.gca()
    ax.imshow(mat.values.astype(float))
    ax.set_xticks(range(len(algos))); ax.set_yticks(range(len(algos)))
    ax.set_xticklabels(algos, rotation=45, ha="right"); ax.set_yticklabels(algos)
    ax.set_title(f"Track1 mean distances ({args.view}), eps={eps:.3f}")
    fig.tight_layout()
    fig.savefig(fig_dir/"track1_distance_heatmap.png", dpi=200)
    plt.close(fig)

    print("[OK] Track1 outputs:", out_dir)


# =============================
# Track 2: segments
# =============================

def segments(args):
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    df = pd.read_csv(args.csv)

    algos = parse_list(args.algos, str) if args.algos else sorted(df["algo_variant"].unique().tolist())
    budgets_all = parse_list(args.budgets, int) if args.budgets else sorted(df["budget"].unique().tolist())
    targets = parse_list(args.targets, str) if (args.view == "success" and args.targets) else sorted(df["target"].unique().tolist())
    segs = parse_segments(args.segments)

    for seg in segs:
        for b in seg:
            if b not in budgets_all:
                raise ValueError(f"Budget {b} in --segments not present in CSV budgets.")

    inst_tbl, view_targets = build_view_table(
        df, view=args.view, algos=algos, targets=targets, budgets=budgets_all,
        value_col=args.value_col, scale_mode=args.scale_mode, sense=args.sense
    )

    instances = inst_tbl[["problem","instance_id"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(args.seed)

    if args.pso_anchor not in algos or args.es_anchor not in algos:
        raise ValueError("Anchors must be in algos list for segment attribution.")
    iA = algos.index(args.pso_anchor)
    iE = algos.index(args.es_anchor)

    labels_boot = np.zeros((args.B, len(segs), len(algos)), dtype=int)  # 0=PSO,1=ES,2=FLAT
    for b in tqdm(range(args.B), desc="Track2 bootstrap"):
        w = stratified_instance_weights(instances, rng)
        dfw = inst_tbl.merge(w, on=["problem","instance_id"], how="inner")
        agg = weighted_group_mean(
            dfw,
            group_cols=["algo_variant","target","budget"],
            metric_cols=["metric"],
            weight_col="w"
        )
        for si, seg_budgets in enumerate(segs):
            M = pivot_curves(agg.rename(columns={"metric":"metric"}), algos, view_targets, seg_budgets, "metric")
            Z = zscore_by_feature(M)

            dP = np.array([euclid(Z[i], Z[iA]) for i in range(len(algos))], dtype=float)
            dE = np.array([euclid(Z[i], Z[iE]) for i in range(len(algos))], dtype=float)

            hr = np.max(M, axis=1) - np.min(M, axis=1)
            for ai in range(len(algos)):
                if hr[ai] < args.headroom_gate:
                    labels_boot[b, si, ai] = 2
                else:
                    labels_boot[b, si, ai] = 0 if dP[ai] <= dE[ai] else 1

    rows=[]
    for si, seg_budgets in enumerate(segs):
        for ai, algo in enumerate(algos):
            p_pso = float(np.mean(labels_boot[:,si,ai]==0))
            p_es  = float(np.mean(labels_boot[:,si,ai]==1))
            p_flat= float(np.mean(labels_boot[:,si,ai]==2))
            if p_flat >= 0.5: lab="FLAT"
            else: lab = "PSO" if p_pso >= p_es else "ES"
            rows.append({
                "segment": si,
                "segment_budgets": ",".join(map(str, seg_budgets)),
                "algo_variant": algo,
                "label": lab,
                "p_PSO": p_pso,
                "p_ES": p_es,
                "p_FLAT": p_flat,
            })
    pd.DataFrame(rows).to_csv(out_dir/"track2_segment_labels.csv", index=False)
    print("[OK] Track2 outputs:", out_dir)


# =============================
# Track 3: interpolated benchmark runner (minimal)
# =============================

@dataclass(frozen=True)
class ProblemSpec:
    name: str
    bounds: Tuple[float, float]
    fn: Callable[[np.ndarray], float]

def sphere(x: np.ndarray) -> float:
    return float(np.sum(x * x))

def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100.0*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2))

def rastrigin(x: np.ndarray) -> float:
    A = 10.0
    return float(A * x.size + np.sum(x * x - A * np.cos(2 * math.pi * x)))

def ackley(x: np.ndarray) -> float:
    a, b, c = 20.0, 0.2, 2*math.pi
    d = x.size
    s1 = np.sum(x*x)
    s2 = np.sum(np.cos(c*x))
    return float(-a*np.exp(-b*np.sqrt(s1/d)) - np.exp(s2/d) + a + math.e)

def griewank(x: np.ndarray) -> float:
    i = np.arange(1, x.size+1, dtype=float)
    return float(1.0 + np.sum(x*x)/4000.0 - np.prod(np.cos(x/np.sqrt(i))))

BASE_FUNCS = {
    "Sphere": ProblemSpec("Sphere", (-5.0, 5.0), sphere),
    "Rosenbrock": ProblemSpec("Rosenbrock", (-5.0, 5.0), rosenbrock),
    "Rastrigin": ProblemSpec("Rastrigin", (-5.12, 5.12), rastrigin),
    "Ackley": ProblemSpec("Ackley", (-5.0, 5.0), ackley),
    "Griewank": ProblemSpec("Griewank", (-6.0, 6.0), griewank),
}

def rng_from(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))

def stable_hash_mod(s: str, mod: int) -> int:
    h = zlib.adler32(s.encode("utf-8")) & 0xffffffff
    return int(h % mod)

def stable_run_seed(algo: str, params: Dict[str, Any], problem: str, instance_id: int, budget: int, r: int, base_seed: int) -> int:
    blob = json.dumps(params, sort_keys=True)
    key = f"{algo}|{blob}|{problem}|{instance_id}|{budget}|{r}|{base_seed}"
    return 1_000_000 * stable_hash_mod(key, 1_000_000_000)

def clip_bounds(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)

def eval_pop(f: Callable[[np.ndarray], float], X: np.ndarray) -> np.ndarray:
    return np.array([f(x) for x in X], dtype=float)

def make_shifted(f: Callable[[np.ndarray], float], bounds: Tuple[float,float], dim: int, shift_seed: int):
    rg = rng_from(shift_seed)
    lo, hi = bounds
    shift = rg.uniform(-1.0, 1.0, size=dim) * 0.3 * (hi - lo)
    def g(x: np.ndarray) -> float:
        return f(x - shift)
    return g

def normalise_scale(f: Callable[[np.ndarray], float], bounds: Tuple[float,float], dim: int, seed: int, nprobe: int = 2048) -> float:
    rg = rng_from(seed)
    lo, hi = bounds
    X = rg.uniform(lo, hi, size=(nprobe, dim))
    vals = eval_pop(f, X)
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) + 1e-12
    return float(mad)

def make_interpolated(A: ProblemSpec, B: ProblemSpec, lam: float, dim: int, seed: int) -> Tuple[Callable[[np.ndarray], float], Tuple[float,float]]:
    lo = max(A.bounds[0], B.bounds[0])
    hi = min(A.bounds[1], B.bounds[1])
    bounds = (lo, hi)
    sA = normalise_scale(A.fn, bounds, dim, seed=seed+11)
    sB = normalise_scale(B.fn, bounds, dim, seed=seed+17)
    def f(x: np.ndarray) -> float:
        return (1.0-lam) * (A.fn(x)/sA) + lam * (B.fn(x)/sB)
    return f, bounds

# Minimal runners: anchors only (extend if you want full set)
def run_pso_gbest(f, dim, bounds, budget, seed, pop=30, w=0.72, c1=1.49, c2=1.49):
    rg = rng_from(seed)
    lo, hi = bounds
    X = rg.uniform(lo, hi, size=(pop, dim))
    v_scale = 0.1 * (hi - lo)
    V = rg.uniform(-v_scale, v_scale, size=(pop, dim))
    fit = eval_pop(f, X)
    evals = pop
    pbest = X.copy()
    pbest_fit = fit.copy()
    gbest_idx = int(np.argmin(pbest_fit))
    gbest = pbest[gbest_idx].copy()
    gbest_fit = float(pbest_fit[gbest_idx])
    while evals + pop <= budget:
        r1 = rg.random((pop, dim))
        r2 = rg.random((pop, dim))
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest[None, :] - X)
        X = X + V
        X = clip_bounds(X, lo, hi)
        fit = eval_pop(f, X)
        evals += pop
        improved = fit < pbest_fit
        pbest[improved] = X[improved]
        pbest_fit[improved] = fit[improved]
        gbest_idx = int(np.argmin(pbest_fit))
        if float(pbest_fit[gbest_idx]) < gbest_fit:
            gbest_fit = float(pbest_fit[gbest_idx])
            gbest = pbest[gbest_idx].copy()
    return float(gbest_fit), evals

def run_es_1p1(f, dim, bounds, budget, seed, pop=30, sigma0: float = None):
    rg = rng_from(seed)
    lo, hi = bounds
    X0 = rg.uniform(lo, hi, size=(pop, dim))
    f0 = eval_pop(f, X0)
    evals = pop
    x = X0[int(np.argmin(f0))].copy()
    fx = float(np.min(f0))
    sigma = 0.2 * (hi - lo) if sigma0 is None else float(sigma0)
    success_window = 20
    succ = 0
    steps = 0
    while evals < budget:
        x2 = x + sigma * rg.normal(0.0, 1.0, size=dim)
        x2 = clip_bounds(x2, lo, hi)
        f2 = float(f(x2))
        evals += 1
        steps += 1
        if f2 < fx:
            x, fx = x2, f2
            succ += 1
        if steps % success_window == 0:
            rate = succ / success_window
            sigma *= 1.2 if rate > 0.2 else 0.82
            sigma = float(np.clip(sigma, 1e-12, 0.5 * (hi - lo)))
            succ = 0
    return float(fx), evals

ALGO_RUNNERS_3 = {"PSO_GBEST": run_pso_gbest, "ES_1P1": run_es_1p1}
DEFAULT_PARAMS_3 = {"PSO_GBEST": {"pop":30, "w":0.72, "c1":1.49, "c2":1.49}, "ES_1P1": {"pop":30, "sigma0":None}}

def beta_posterior(succ: int, trials: int):
    a = 1 + succ
    b = 1 + trials - succ
    mean = a / (a + b)
    p05 = float(sp_beta.ppf(0.05, a, b))
    p95 = float(sp_beta.ppf(0.95, a, b))
    return float(mean), p05, p95

def build_instance_summary(runs_df: pd.DataFrame, target_tols: Dict[str, float]) -> pd.DataFrame:
    out_rows = []
    for (problem, instance_id, algo, budget), g in runs_df.groupby(
        ["problem","instance_id","algo_variant","budget"], as_index=False
    ):
        trials = int(g.shape[0])
        mean_best = float(g["best"].mean())
        median_best = float(g["best"].median())
        min_best = float(g["best"].min())
        max_best = float(g["best"].max())
        for tgt, tol in target_tols.items():
            succ = int(np.sum(g["best"].to_numpy() <= float(tol)))
            beta_mean, beta_p05, beta_p95 = beta_posterior(succ, trials)
            out_rows.append({
                "domain":"cont",
                "problem":problem,
                "instance_id":int(instance_id),
                "algo_variant":algo,
                "budget":int(budget),
                "target":tgt,
                "successes":succ,
                "trials":trials,
                "beta_mean":beta_mean,
                "beta_p05":beta_p05,
                "beta_p95":beta_p95,
                "mean_best":mean_best,
                "median_best":median_best,
                "min_best":min_best,
                "max_best":max_best,
            })
    return pd.DataFrame(out_rows)

def parse_target_tols(s: str) -> Dict[str, float]:
    out={}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k,v = part.split("=")
        out[k.strip()] = float(v.strip())
    return out

def interpolate(args):
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    A = BASE_FUNCS[args.A]
    B = BASE_FUNCS[args.Bfunc]
    lambdas = parse_list(args.lambdas, float)
    budgets = parse_list(args.budgets, int)
    algos = parse_list(args.algos, str)

    missing = [a for a in algos if a not in ALGO_RUNNERS_3]
    if missing:
        raise ValueError(
            "Track3 runner currently implements only PSO_GBEST and ES_1P1 to keep it light.\n"
            f"Missing runners for: {missing}\n"
            "If you want Track3 for all 8 algorithms, ask and I'll generate the full runner (copying your full runner set)."
        )

    target_tols = parse_target_tols(args.targets)

    total = len(lambdas) * args.instances_per_problem * len(algos) * len(budgets) * args.R
    pb = tqdm(total=total, desc="Track3 interpolate runs")

    rows=[]
    for lam in lambdas:
        base_f, bounds = make_interpolated(A, B, lam, dim=args.dim, seed=args.seed+123)
        for inst_id in range(args.instances_per_problem):
            shift_seed = 100_000 + int(10_000*lam) + inst_id
            f = make_shifted(base_f, bounds, args.dim, shift_seed)
            prob_name = f"Interp_{A.name}_{B.name}_lam{lam:.2f}"
            for algo in algos:
                params = DEFAULT_PARAMS_3.get(algo, {})
                runner = ALGO_RUNNERS_3[algo]
                for bud in budgets:
                    for r in range(args.R):
                        seed = stable_run_seed(algo, params, prob_name, inst_id, bud, r, base_seed=args.seed+999)
                        best, evals = runner(f, args.dim, bounds, bud, seed, **params)
                        rows.append({
                            "domain":"cont",
                            "problem":prob_name,
                            "instance_id":inst_id,
                            "lambda":lam,
                            "algo_variant":algo,
                            "budget":bud,
                            "rep":r,
                            "seed":seed,
                            "evals":evals,
                            "best":best,
                        })
                        pb.update(1)
    pb.close()

    runs_df = pd.DataFrame(rows)
    runs_df.to_csv(out_dir/"runs_detail.csv", index=False)
    summ = build_instance_summary(runs_df, target_tols)
    summ.to_csv(out_dir/"instance_algo_budget_summary.csv", index=False)
    print("[OK] Track3 outputs:", out_dir)


# =============================
# CLI
# =============================

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("curvesim")
    p1.add_argument("--csv", required=True)
    p1.add_argument("--out_dir", required=True)
    p1.add_argument("--view", choices=["success","fitness"], default="success")
    p1.add_argument("--algos", default=None)
    p1.add_argument("--budgets", default=None)
    p1.add_argument("--targets", default=None)
    p1.add_argument("--pso_anchor", default="PSO_GBEST")
    p1.add_argument("--es_anchor", default="ES_1P1")
    p1.add_argument("--B", type=int, default=2000)
    p1.add_argument("--seed", type=int, default=0)
    p1.add_argument("--eps_frac", type=float, default=0.25)
    p1.add_argument("--value_col", default="median_best")
    p1.add_argument("--scale_mode", default="q05q95", choices=["q05q95","minmax"])
    p1.add_argument("--sense", default="minimize", choices=["minimize","maximize"])
    p1.set_defaults(func=curvesim)

    p2 = sub.add_parser("segments")
    p2.add_argument("--csv", required=True)
    p2.add_argument("--out_dir", required=True)
    p2.add_argument("--view", choices=["success","fitness"], default="success")
    p2.add_argument("--algos", default=None)
    p2.add_argument("--budgets", default=None)
    p2.add_argument("--targets", default=None)
    p2.add_argument("--segments", required=True)
    p2.add_argument("--pso_anchor", default="PSO_GBEST")
    p2.add_argument("--es_anchor", default="ES_1P1")
    p2.add_argument("--B", type=int, default=2000)
    p2.add_argument("--seed", type=int, default=0)
    p2.add_argument("--headroom_gate", type=float, default=0.02)
    p2.add_argument("--value_col", default="median_best")
    p2.add_argument("--scale_mode", default="q05q95", choices=["q05q95","minmax"])
    p2.add_argument("--sense", default="minimize", choices=["minimize","maximize"])
    p2.set_defaults(func=segments)

    p3 = sub.add_parser("interpolate")
    p3.add_argument("--out_dir", required=True)
    p3.add_argument("--A", required=True, choices=list(BASE_FUNCS.keys()))
    p3.add_argument("--Bfunc", required=True, choices=list(BASE_FUNCS.keys()))
    p3.add_argument("--lambdas", required=True)
    p3.add_argument("--algos", required=True)
    p3.add_argument("--budgets", required=True)
    p3.add_argument("--dim", type=int, default=10)
    p3.add_argument("--instances_per_problem", type=int, default=10)
    p3.add_argument("--R", type=int, default=10)
    p3.add_argument("--seed", type=int, default=0)
    p3.add_argument("--targets", type=str, default="easy=1e-1,med=1e-2,hard=1e-3")
    p3.set_defaults(func=interpolate)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
