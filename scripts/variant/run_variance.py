#!/usr/bin/env python3
"""Run-to-run variance for a single real model.

WHY THIS EXISTS
---------------
The main sweep runs each model ONCE. Reviewers ask whether an agent's numbers
are stable across repeated runs. For a reasoning model (e.g. gpt-5.x) the
sampler runs at temperature=1 (these models reject a pinned temperature=0), so
repeated runs are genuinely stochastic -- the only thing that would otherwise
mask that is the on-disk prompt cache, whose key is (model_id, system, user)
and does NOT include the run index. This script therefore gives EACH repeat its
OWN cache file, so every repeat re-queries the API and the spread across repeats
is real run-to-run variance, not a cache echo.

It re-runs the FULL benchmark (real cohort, all strategies) R times for the
chosen model slot, collects the agent row's A(k) / PE / rho each time, and
reports mean +/- SD across the R runs.

USAGE
-----
    export OPENAI_API_KEY=...            # (or the vendor key your slot needs)
    pip install -e ".[openai]"           # (or [anthropic]/[gemini])
    python scripts/variant/run_variance.py \
        --cohort data/sample/cohort_full_real.parquet \
        --config configs/models.yaml --models real \
        --slot frontier_B --runs 3 --outdir results/variant/variance

    # smoke first (cheaper): add --max-variants 40

Output: results/variant/variance/variance_<slot>.csv  (one row per run + a mean/SD row)
and a console summary. Nothing else in the repo is touched.
"""
import argparse, os, sys, copy
import pandas as pd
import yaml

# Scripts live one level deeper than the repo root (scripts/<block>/x.py),
# so the root is three dirnames up, not two.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from pwkbench.domains.variant import VARIANT_DOMAIN
from pwkbench.domains.base import load_real_cohort
from pwkbench import harness              # harness.run() is the runner's entry point


def _agent_metrics(results_df, slot):
    """Pull the agent row for `slot` from a results DataFrame -> flat dict."""
    row = results_df[results_df.strategy.str.contains(slot, na=False)]
    if len(row) == 0:
        raise SystemExit(f"no agent row matching slot '{slot}' in results; "
                         f"strategies were: {list(results_df.strategy)}")
    r = row.iloc[0]
    out = {"strategy": r.strategy}
    for k in (1, 2, 3, 4):
        out[f"A_k{k}"] = float(r[f"A_k{k}"])
    for c in ("PE", "rho_vs_relevance", "rho_vs_oracle", "masked_auc"):
        if c in r:
            out[c] = float(r[c])
    # over-acquisition drop = peak - final (the headline efficiency signal)
    aks = [out[f"A_k{k}"] for k in (1, 2, 3, 4)]
    out["stopping_drop"] = float(max(aks) - aks[-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--models", default="real")
    ap.add_argument("--slot", required=True,
                    help="slot substring to track, e.g. frontier_B (gpt-5.5)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-variants", type=int, default=None)
    ap.add_argument("--outdir", default="results/variant/variance")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    if args.models not in cfg:
        raise SystemExit(f"'{args.models}' not in {args.config}")
    # keep ONLY the requested slot so we pay for just that model
    rows = [m for m in cfg[args.models] if m.get("slot") == args.slot]
    if not rows:
        raise SystemExit(f"slot '{args.slot}' not found under '{args.models}:' "
                         f"in {args.config}")

    df = pd.read_parquet(args.cohort)
    if args.max_variants:
        # Gene-stratified subsample. Written as an explicit concat rather than
        # `groupby(...).apply(...)`: that form is deprecated in pandas and, once
        # the deprecation lands, the grouping column is dropped from the result --
        # here that silently removes `gene`, which `load_real_cohort` requires, so
        # the failure would surface far from its cause. Verified to produce
        # row-identical output to the old form at several --max-variants values.
        df = pd.concat([
            g.sample(max(1, round(len(g) * args.max_variants / len(df))),
                     random_state=0)
            for _, g in df.groupby("gene")])
    cohort = load_real_cohort(df, VARIANT_DOMAIN)

    per_run = []
    for run_i in range(args.runs):
        model_rows = copy.deepcopy(rows)
        # give THIS repeat its own cache + its own seed so it genuinely re-queries
        for m in model_rows:
            m["seed"] = 20260101 + run_i
            m["cache_path"] = os.path.join(
                args.outdir, f".cache_{args.slot}_run{run_i}.json")
        print(f"[variance] run {run_i+1}/{args.runs} for slot={args.slot} "
              f"(seed {20260101+run_i}, fresh cache)", flush=True)
        res = harness.run(cohort, model_matrix=model_rows, with_ci=False)
        m = _agent_metrics(res, args.slot)
        m["run"] = run_i
        per_run.append(m)
        pd.DataFrame(per_run).to_csv(
            os.path.join(args.outdir, f"variance_{args.slot}.csv"), index=False)

    runs_df = pd.DataFrame(per_run)
    num = runs_df.drop(columns=["strategy", "run"])
    summary = pd.concat([
        runs_df,
        pd.DataFrame([{"run": "MEAN", "strategy": args.slot, **num.mean().to_dict()}]),
        pd.DataFrame([{"run": "SD",   "strategy": args.slot, **num.std(ddof=1).to_dict()}]),
    ], ignore_index=True)
    out_csv = os.path.join(args.outdir, f"variance_{args.slot}.csv")
    summary.to_csv(out_csv, index=False)

    print("\n=== run-to-run variance (", args.slot, ",", args.runs, "runs ) ===")
    cols = [c for c in ("A_k1", "A_k2", "A_k3", "A_k4", "PE",
                        "rho_vs_oracle", "stopping_drop") if c in num.columns]
    for c in cols:
        print(f"  {c:16s}  mean {num[c].mean():.4f}   SD {num[c].std(ddof=1):.4f}")
    print("written:", out_csv)


if __name__ == "__main__":
    main()
