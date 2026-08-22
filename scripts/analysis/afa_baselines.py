#!/usr/bin/env python3
"""E1 -- AFA baselines in the frozen paradigm.

Reviewer concern: "include AFA baselines (discriminative greedy, simple RL)";
"observed [agent] behaviors may be just prompt heuristics" -- i.e. without a
standard active-feature-acquisition reference, a reader cannot tell whether
the agent's order is doing anything beyond textbook myopic relevance-greedy.

Adds ONE new row to `results.csv`:
  - DiscriminativeGreedy: `pwkbench.strategies.discriminative_greedy_order`,
    the standard AFA myopic-greedy policy, chosen out-of-fold (gene-grouped
    CV) so it cannot overfit to which variants are in this cohort.

`BestFixed` (already in every results.csv) is relabelled here as the
label-conditioned AFA STATIC baseline for the paper table -- it needs no new
run, it is already the tightest FIXED order under the benchmark's own metric.
The "(optional) simple RL / tabular policy" item from the experiment plan is
deliberately NOT added: with K=4 the only well-posed per-instance-adaptive
exhaustive policy already IS the Oracle (argmax over per-budget subsets), so a
separate "RL-style" reference would either duplicate Oracle exactly or need an
arbitrary reward-shaping choice that the reviewer did not ask for -- adding it
would be a distinct baseline in name only.

USAGE
    python scripts/analysis/afa_baselines.py \
        --cohort data/sample/cohort_full_real.parquet --domain variant \
        --results results/variant/real_trusted/results.csv
"""
import argparse, os, sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench import strategies as S, metrics as M, harness as H  # noqa: E402
from pwkbench.domains.base import load_real_cohort                 # noqa: E402
from pwkbench.domains.variant import VARIANT_DOMAIN                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/sample/cohort_full_real.parquet")
    ap.add_argument("--domain", choices=["variant"], default="variant")
    ap.add_argument("--results", default="results/variant/real_trusted/results.csv",
                     help="existing results.csv to add the DiscriminativeGreedy "
                          "row to (in place); every other row is left untouched")
    args = ap.parse_args()
    os.chdir(_ROOT)

    cohort = load_real_cohort(pd.read_parquet(args.cohort), VARIANT_DOMAIN)

    order = S.discriminative_greedy_order(cohort)
    row = H.evaluate_order(cohort, order, "DiscriminativeGreedy")
    row["masked_auc"] = M.masked_auc(cohort)

    d = pd.read_csv(args.results)
    pe_oracle = float(d.loc[d.strategy == "Oracle", "PE"].iloc[0])
    row["delta_oracle"] = pe_oracle - row["PE"]

    d = d[d.strategy != "DiscriminativeGreedy"]  # idempotent re-run
    d = pd.concat([d, pd.DataFrame([row])], ignore_index=True)
    d.to_csv(args.results, index=False)
    print(f"wrote DiscriminativeGreedy row -> {args.results}")

    order_names = [cohort.domain.channels[j] for j in order[0]]
    print(f"  order (global, same for every variant): {order_names}")
    for k in ("PE", "A_k1", "A_k2", "A_k3", "A_k4", "po_vs_relevance"):
        print(f"  {k:16s} {row[k]:.4f}")

    agent_rows = d[d.strategy.str.startswith("agent:", na=False)]
    if len(agent_rows):
        print("\nagent PE vs. DiscriminativeGreedy PE:")
        for _, r in agent_rows.iterrows():
            side = "above" if r.PE > row["PE"] else ("below" if r.PE < row["PE"] else "equal to")
            print(f"  {r.strategy:20s} PE={r.PE:.4f}  ({side} discriminative-greedy "
                  f"PE={row['PE']:.4f}, diff={r.PE - row['PE']:+.4f})")


if __name__ == "__main__":
    main()
