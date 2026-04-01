#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
dorigo_apply_epsilons.py

Apply calibrated epsilons (within-family) to your Track-1 pairwise distance results,
and optionally build a minimal "portfolio" via graph connected components.

Inputs
------
1) track1_similarity_pairs.csv
   Must contain:
     algo_u, algo_v, dist_p95   (and optionally dist_mean)
2) epsilon_calibration.json
   Must contain keys:
     epsilon_success
     epsilon_fitness
     success_within (PSO_p95, ES_p95)
     fitness_within (PSO_p95, ES_p95)  [optional]

What it does
------------
A) Adds calibrated similarity decisions:
   - similar_success_global: dist_p95 <= epsilon_success
   - similar_fitness_global: dist_p95 <= epsilon_fitness  (if present)
   - similar_success_PSO: dist_p95 <= success_within.PSO_p95
   - similar_success_ES : dist_p95 <= success_within.ES_p95
   - similar_fitness_PSO / similar_fitness_ES if fitness_within exists

B) Constructs portfolios (representative sets) for each epsilon:
   - Build an undirected graph connecting pairs deemed "similar"
   - Each connected component is a redundancy cluster
   - Choose representative per component:
        if you provide a score table (--scores), pick highest-score;
        else pick lexicographically smallest algo.

Optional: Score-based representatives
------------------------------------
If you pass --scores with columns:
  algo_variant, score
the representative for each component is the max-score node.

Outputs
-------
out_dir/
  pairs_with_epsilons.csv
  clusters_success_global.csv
  clusters_fitness_global.csv (if epsilon_fitness exists)
  portfolio_success_global.csv
  portfolio_fitness_global.csv
  README.txt

Usage
-----
python dorigo_apply_epsilons.py ^
  --pairs .\track1_similarity_pairs.csv ^
  --eps   .\epsilon_calibration.json ^
  --out_dir .\epsilon_applied

With score-based reps (recommended if you have one):
python dorigo_apply_epsilons.py ^
  --pairs .\track1_similarity_pairs.csv ^
  --eps   .\epsilon_calibration.json ^
  --scores .\phase2_algo_scores.csv ^
  --out_dir .\epsilon_applied

"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

def load_eps(path: Path) -> Dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Required
    if "epsilon_success" not in data:
        raise ValueError("epsilon_calibration.json missing epsilon_success")
    if "success_within" not in data:
        raise ValueError("epsilon_calibration.json missing success_within")
    return data

def build_graph(edges: List[Tuple[str,str]], nodes: List[str]) -> Dict[str, List[str]]:
    g = {n: [] for n in nodes}
    for u,v in edges:
        if u not in g: g[u]=[]
        if v not in g: g[v]=[]
        g[u].append(v)
        g[v].append(u)
    return g

def connected_components(g: Dict[str, List[str]]) -> List[List[str]]:
    seen=set()
    comps=[]
    for n in g.keys():
        if n in seen:
            continue
        stack=[n]
        seen.add(n)
        comp=[]
        while stack:
            x=stack.pop()
            comp.append(x)
            for y in g.get(x, []):
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(sorted(comp))
    return comps

def choose_rep(comp: List[str], scores: Optional[Dict[str,float]]) -> str:
    if not scores:
        return comp[0]
    # choose highest score; tie-break lexicographically
    best = None
    best_s = None
    for a in comp:
        s = scores.get(a, float("-inf"))
        if best is None or s > best_s or (s == best_s and a < best):
            best = a
            best_s = s
    return best

def components_to_df(comps: List[List[str]], scores: Optional[Dict[str,float]], label: str) -> pd.DataFrame:
    rows=[]
    for i, comp in enumerate(comps):
        rep = choose_rep(comp, scores)
        rows.append({
            "cluster_id": i,
            "cluster_size": len(comp),
            "representative": rep,
            "members": ",".join(comp),
            "label": label
        })
    return pd.DataFrame(rows).sort_values(["cluster_size","cluster_id"], ascending=[False, True])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="track1_similarity_pairs.csv")
    ap.add_argument("--eps", required=True, help="epsilon_calibration.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scores", default=None, help="Optional CSV: algo_variant, score")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pairs)
    # Standardize column names
    pairs.columns = [c.strip() for c in pairs.columns]
    need = {"algo_u","algo_v","dist_p95"}
    if not need.issubset(set(pairs.columns)):
        raise ValueError(f"--pairs missing columns {sorted(need - set(pairs.columns))}")

    eps = load_eps(Path(args.eps))
    eps_success = float(eps["epsilon_success"])
    eps_fit = float(eps.get("epsilon_fitness", float("nan")))

    pso_succ = float(eps["success_within"].get("PSO_p95"))
    es_succ  = float(eps["success_within"].get("ES_p95"))

    pso_fit = None
    es_fit = None
    if "fitness_within" in eps:
        pso_fit = float(eps["fitness_within"].get("PSO_p95"))
        es_fit  = float(eps["fitness_within"].get("ES_p95"))

    # Optional scores
    scores = None
    if args.scores:
        sc = pd.read_csv(args.scores)
        sc.columns=[c.strip() for c in sc.columns]
        if not {"algo_variant","score"}.issubset(sc.columns):
            raise ValueError("--scores must contain columns: algo_variant, score")
        scores = dict(zip(sc["algo_variant"].astype(str), sc["score"].astype(float)))

    # Apply decisions
    pairs["similar_success_global"] = pairs["dist_p95"] <= eps_success
    pairs["similar_success_PSO"] = pairs["dist_p95"] <= pso_succ
    pairs["similar_success_ES"]  = pairs["dist_p95"] <= es_succ

    if pd.notna(eps_fit):
        pairs["similar_fitness_global"] = pairs["dist_p95"] <= eps_fit

    if pso_fit is not None and es_fit is not None:
        pairs["similar_fitness_PSO"] = pairs["dist_p95"] <= pso_fit
        pairs["similar_fitness_ES"]  = pairs["dist_p95"] <= es_fit

    pairs.to_csv(out_dir/"pairs_with_epsilons.csv", index=False)

    # Build portfolios (components) for success_global and fitness_global
    nodes = sorted(set(pairs["algo_u"]).union(set(pairs["algo_v"])))
    # success global
    edges_s = list(zip(pairs.loc[pairs["similar_success_global"],"algo_u"], pairs.loc[pairs["similar_success_global"],"algo_v"]))
    g_s = build_graph(edges_s, nodes)
    comps_s = connected_components(g_s)
    clusters_s = components_to_df(comps_s, scores, "success_global")
    clusters_s.to_csv(out_dir/"clusters_success_global.csv", index=False)
    # portfolio reps
    port_s = clusters_s[["cluster_id","representative","cluster_size"]].copy()
    port_s.to_csv(out_dir/"portfolio_success_global.csv", index=False)

    # fitness global (if available)
    if "similar_fitness_global" in pairs.columns:
        edges_f = list(zip(pairs.loc[pairs["similar_fitness_global"],"algo_u"], pairs.loc[pairs["similar_fitness_global"],"algo_v"]))
        g_f = build_graph(edges_f, nodes)
        comps_f = connected_components(g_f)
        clusters_f = components_to_df(comps_f, scores, "fitness_global")
        clusters_f.to_csv(out_dir/"clusters_fitness_global.csv", index=False)
        clusters_f[["cluster_id","representative","cluster_size"]].to_csv(out_dir/"portfolio_fitness_global.csv", index=False)

    (out_dir/"README.txt").write_text(
        "Applied calibrated epsilons to pairwise distances.\n"
        f"epsilon_success_global={eps_success}\n"
        f"epsilon_success_PSO_within={pso_succ}\n"
        f"epsilon_success_ES_within={es_succ}\n"
        + (f"epsilon_fitness_global={eps_fit}\n" if pd.notna(eps_fit) else "")
        + (f"epsilon_fitness_PSO_within={pso_fit}\n" if pso_fit is not None else "")
        + (f"epsilon_fitness_ES_within={es_fit}\n" if es_fit is not None else "")
        + "\nOutputs:\n"
        "  pairs_with_epsilons.csv: original pairs with similarity flags\n"
        "  clusters_*.csv: connected components under each epsilon\n"
        "  portfolio_*.csv: one representative per cluster\n",
        encoding="utf-8"
    )

    print("[OK] wrote outputs to:", out_dir)

if __name__ == "__main__":
    main()
