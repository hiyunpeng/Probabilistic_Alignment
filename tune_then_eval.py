import argparse, json, os, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

def auc_over_budgets(budgets, vals):
    budgets = np.asarray(budgets, dtype=float)
    vals = np.asarray(vals, dtype=float)
    mask = np.isfinite(vals)
    if mask.sum() < 2:
        return np.nan
    b = budgets[mask]
    v = vals[mask]
    return float(np.trapz(v, b) / (b.max() - b.min()))

def score_from_summary(summary_csv: Path, budgets, targets, target_weights):
    df = pd.read_csv(summary_csv)
    # expects columns: algo_variant, budget, target, beta_mean
    # aggregate over instances first
    g = df.groupby(["algo_variant", "budget", "target"], as_index=False)["beta_mean"].mean()

    # if multiple algos exist in file, take first (or filter upstream)
    algo = g["algo_variant"].iloc[0]

    total = 0.0
    for t in targets:
        gt = g[g["target"] == t].sort_values("budget")
        # align budgets
        m = {int(b): float(v) for b, v in zip(gt["budget"], gt["beta_mean"])}
        vals = [m.get(int(b), np.nan) for b in budgets]
        auc = auc_over_budgets(budgets, vals)
        if np.isfinite(auc):
            total += target_weights.get(t, 1.0) * auc
    return algo, float(total)

def sample_params(rng, space):
    params = {}
    for k, spec in space.items():
        typ = spec["type"]
        if typ == "float_log":
            lo, hi = spec["lo"], spec["hi"]
            x = 10 ** rng.uniform(np.log10(lo), np.log10(hi))
            params[k] = float(x)
        elif typ == "float":
            lo, hi = spec["lo"], spec["hi"]
            params[k] = float(rng.uniform(lo, hi))
        elif typ == "int":
            lo, hi = spec["lo"], spec["hi"]
            params[k] = int(rng.integers(lo, hi + 1))
        elif typ == "cat":
            params[k] = rng.choice(spec["values"]).item()
        else:
            raise ValueError(f"Unknown type: {typ}")
    return params

def run_one(cmd_template: str, params: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    params_json = out_dir / "params.json"
    params_json.write_text(json.dumps(params, indent=2))

    # template can use {params_json} and {out_dir}
    cmd = cmd_template.format(params_json=str(params_json), out_dir=str(out_dir))
    subprocess.run(cmd, shell=True, check=True)

    # you must make the runner write this file:
    summary_csv = out_dir / "instance_algo_budget_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Expected output not found: {summary_csv}")
    return summary_csv

def successive_halving(candidates, rung_specs, run_fn, score_fn):
    # candidates: list[params]
    # rung_specs: list of dict with "name" and runner resource baked into cmd_template
    alive = candidates
    history = []

    for rung in rung_specs:
        scored = []
        for i, p in enumerate(alive):
            summary_csv = run_fn(p, rung)
            algo, s = score_fn(summary_csv)
            scored.append((s, p, rung["name"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        history += scored

        keep = max(1, int(np.ceil(len(scored) * rung["keep_frac"])))
        alive = [p for _, p, _ in scored[:keep]]

    best_score, best_params, _ = max(history, key=lambda x: x[0])
    return best_score, best_params, history

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space_json", required=True, help="Parameter space JSON file")
    ap.add_argument("--cmd_template", required=True,
                    help="Command template to run one candidate. Must output instance_algo_budget_summary.csv into {out_dir}. "
                         "Use placeholders {params_json} and {out_dir}.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_init", type=int, default=60)
    ap.add_argument("--budgets", default="300,500,800,1000")
    ap.add_argument("--targets", default="easy,med,hard")
    ap.add_argument("--target_weights", default="easy:1,med:2,hard:3")
    ap.add_argument("--keep_fracs", default="0.33,0.33")  # two rungs -> keep 1/3 then 1/3
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]
    targets = [x.strip() for x in args.targets.split(",")]
    tw = {}
    for kv in args.target_weights.split(","):
        k, v = kv.split(":")
        tw[k.strip()] = float(v)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    space = json.loads(Path(args.space_json).read_text())
    rng = np.random.default_rng(args.seed)
    candidates = [sample_params(rng, space) for _ in range(args.n_init)]

    keep_fracs = [float(x) for x in args.keep_fracs.split(",")]
    rung_specs = []
    for i, kf in enumerate(keep_fracs, start=1):
        rung_specs.append({"name": f"rung{i}", "keep_frac": kf})

    def run_fn(params, rung):
        run_dir = out_root / rung["name"] / f"cand_{abs(hash(json.dumps(params, sort_keys=True)))%10**9}"
        # You can encode rung resources via env vars if your runner supports them
        env = os.environ.copy()
        env["RUNG_NAME"] = rung["name"]
        # For simplicity we just call cmd_template and let it read params_json
        # If you want rung-specific resources, bake them into cmd_template or read RUNG_NAME in runner.
        cmd = args.cmd_template.format(params_json=str(run_dir/"params.json"), out_dir=str(run_dir))
        # write params before run
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir/"params.json").write_text(json.dumps(params, indent=2))
        subprocess.run(cmd, shell=True, check=True, env=env)
        return run_dir / "instance_algo_budget_summary.csv"

    def score_fn(summary_csv):
        return score_from_summary(summary_csv, budgets, targets, tw)

    best_score, best_params, history = successive_halving(candidates, rung_specs, run_fn, score_fn)

    Path(out_root/"best_params.json").write_text(json.dumps(best_params, indent=2))
    hist_df = pd.DataFrame([{"score": s, "rung": rung, **p} for s,p,rung in history])
    hist_df.to_csv(out_root/"tuning_history.csv", index=False)
    print("[OK] best_score =", best_score)
    print("[OK] wrote:", out_root/"best_params.json")

if __name__ == "__main__":
    main()