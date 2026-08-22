#!/usr/bin/env python
"""
P1(b) CONTEXT ABLATION — the control for the paper's headline order-optimality
claim.

The no-context harness gives every variant the IDENTICAL step-1 prompt (the
channel list carries no variant identity), so an agent CANNOT plan
instance-adaptively at k=1 even in principle. A low rho_vs_oracle in that
condition is therefore ambiguous:

  (a) the model cannot plan                      -> capability limit, or
  (b) we never told it what it was planning for  -> information limit.

This script runs ONE model_id under both prompt conditions on the SAME cohort
and the SAME scorer, so the only thing that differs is whether the variant's
gene / consequence / disease domain appears in the prompt. Read it as:

  with_context >> no_context  =>  (b): planning is limited by context supply.
  with_context ~= no_context  =>  (a): the paper's strong claim, now controlled.

Outputs (results/<outdir>/):
  results.csv          full harness table for both conditions + references
  ablation.csv         per-condition trajectory diversity, order-optimality,
                       reliability, and the paired contrast vs RelMax
  context_contrast.csv PAIRED gene-clustered bootstrap of with_context minus
                       no_context at each budget k -- the ablation's own test

Usage:
    python scripts/variant/run_context_ablation.py \
        --cohort data/sample/cohort_full_real.parquet --models ablation_real
"""

import os as _os, sys as _sys
# Make the package importable when the script is run directly from a
# checkout (scripts/<block>/x.py -> repo root is three dirnames up).
# Without this the script only works under `PYTHONPATH=.` or an
# installed package, which silently looks like the layout is broken.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import json
import os
import pandas as pd
import yaml
from pwkbench.utils import set_seed
from pwkbench import harness, metrics as M, strategies as S
from pwkbench.domains.base import load_real_cohort
from pwkbench.domains.variant import VARIANT_DOMAIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, help="parquet cohort")
    ap.add_argument("--models", default="ablation_real", help="key in the config")
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--max-variants", type=int, default=None)
    ap.add_argument("--outdir", default="results/variant/ablation")
    ap.add_argument("--with-ci", action="store_true")
    args = ap.parse_args()
    set_seed()

    from scripts.run_benchmark import _stratified_subsample
    df = _stratified_subsample(pd.read_parquet(args.cohort), args.max_variants)
    cohort = load_real_cohort(df, VARIANT_DOMAIN)
    matrix = yaml.safe_load(open(args.config))[args.models]
    if len(matrix) != 2:
        ap.error(f"'{args.models}' must hold exactly 2 rows (context off / on)")
    ids = {m.get("model_id") for m in matrix}
    if len(ids) != 1:
        ap.error(f"the ablation must hold ONE model_id under two conditions, got {ids}")

    os.makedirs(args.outdir, exist_ok=True)
    res = harness.run(cohort, model_matrix=matrix, with_ci=args.with_ci)
    res.to_csv(f"{args.outdir}/results.csv", index=False)

    # per-condition detail. agent_and_order hits the harness cache populated by
    # harness.run above, so nothing is re-elicited (no duplicate API spend).
    rel = S.relmax_order(cohort)
    orders, rows = {}, []
    for m in matrix:
        ag, order = harness.agent_and_order(cohort, m)
        orders[m["slot"]] = order
        td = M.trajectory_diversity(order, cohort.domain.channels)
        oo = M.order_optimality(cohort, order)
        pc = M.paired_contrast_ci(cohort, order, rel, k=1)
        tot, ff, fs = ag.total_steps, ag.forced_parse_failures, ag.forced_steps
        rows.append({
            "slot": m["slot"], "context": bool(m.get("context", False)),
            "model_id": ag.model_id,
            "n_distinct_orders": td["n_distinct_orders"],
            "first_query_counts": json.dumps(td["first_query_counts"]),
            "po_vs_oracle": oo["po_vs_oracle"],
            "po_vs_oracle_adj": oo["po_vs_oracle_adj"],
            "rho_vs_oracle": oo["rho_vs_oracle"],
            "po_vs_relevance": oo["po_vs_relevance"],
            "rho_vs_relevance": oo["rho_vs_relevance"],
            "parse_failure_rate": ag.parse_failures / tot if tot else 0.0,
            "parse_failure_rate_effective":
                (ag.parse_failures - ff) / (tot - fs) if tot - fs > 0 else 0.0,
            "cache_hits": ag.cache_hits, "total_steps": tot,
            "diff_vs_relmax_k1": pc["diff"], "k1_lo": pc["lo"], "k1_hi": pc["hi"],
            "k1_p": pc["p_value"],
        })
    pd.DataFrame(rows).to_csv(f"{args.outdir}/ablation.csv", index=False)

    # The ablation's own test: paired bootstrap of with_context - no_context.
    on = next(m["slot"] for m in matrix if m.get("context"))
    off = next(m["slot"] for m in matrix if not m.get("context"))
    contrast = [dict(budget=k, **M.paired_contrast_ci(cohort, orders[on], orders[off], k))
                for k in range(1, cohort.domain.K + 1)]
    pd.DataFrame(contrast).to_csv(f"{args.outdir}/context_contrast.csv", index=False)

    print(f"wrote -> {args.outdir}/ (results.csv, ablation.csv, context_contrast.csv)")
    print(pd.DataFrame(rows)[["slot", "context", "n_distinct_orders", "po_vs_oracle",
                              "rho_vs_oracle", "parse_failure_rate_effective"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
