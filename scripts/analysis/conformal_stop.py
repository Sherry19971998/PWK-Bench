#!/usr/bin/env python3
"""E5 -- reference stopping rule vs the agent's own
stop, replayed on already-recorded live trajectories. See
pwkbench/live/conformal_stop.py for exactly what this rule is (and is not).

Runs on the already-recorded 3-run gpt-5.5 full-acquisition pool -- no new
API calls.

USAGE
    python scripts/analysis/conformal_stop.py \
        --cohort data/sample/cohort_full_real.parquet \
        --runs results/live/trajectories_full.csv results/live/trajectories_full_r1.csv results/live/trajectories_full_r2.csv \
        --out results/live/conformal_stop.csv
"""
import argparse, os, sys

import pandas as pd
from scipy.stats import wilcoxon

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench.domains.base import load_real_cohort            # noqa: E402
from pwkbench.domains.variant import VARIANT_DOMAIN            # noqa: E402
from pwkbench.live.conformal_stop import reference_stop        # noqa: E402

DEFAULT_RUNS = ["results/live/trajectories_full.csv",
                "results/live/trajectories_full_r1.csv",
                "results/live/trajectories_full_r2.csv"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/sample/cohort_full_real.parquet")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--out", default="results/live/conformal_stop.csv")
    args = ap.parse_args()
    os.chdir(_ROOT)

    cohort = load_real_cohort(pd.read_parquet(args.cohort), VARIANT_DOMAIN)
    pooled = []
    for p in args.runs:
        d = reference_stop(cohort, pd.read_csv(p))
        d["run"] = p
        pooled.append(d)
    d = pd.concat(pooled, ignore_index=True)

    print(f"total (case, run) rows: {len(d)}")
    n_resolved = int(d["rule_resolved"].sum())
    print(f"rule reaches a definitive call on {n_resolved}/{len(d)} "
          f"({n_resolved/len(d):.1%}) -- should match E3's resolvable-population "
          f"rate (62.0% context aside, this is a DIFFERENT population: only the "
          f"agent's own acquired subset, not the retrospective best-subset k*)")

    r = d[d["rule_resolved"]].copy()
    print(f"\nOn the {len(r)} rule-resolvable (case, run) rows:")
    print(f"  agent mean tools/case : {r['agent_n_tools'].mean():.3f}")
    print(f"  rule  mean tools/case : {r['rule_n_tools'].mean():.3f}")
    print(f"  tools saved by the rule (mean): "
          f"{(r['agent_n_tools'] - r['rule_n_tools']).mean():.3f}")
    print(f"  agent accuracy (this subset)  : {r['agent_correct'].mean():.3f}")
    print(f"  rule  accuracy (this subset)  : {r['rule_correct'].mean():.3f}")

    diff = r["agent_n_tools"] - r["rule_n_tools"]
    stat, p = wilcoxon(diff)
    print(f"\nWilcoxon signed-rank (agent_n_tools vs rule_n_tools, paired per "
          f"case/run): statistic={stat:.1f}  p={p:.2e}")
    print(f"  {(diff > 0).mean():.1%} of rows: rule stops STRICTLY earlier than the agent")
    print(f"  {(diff < 0).mean():.1%} of rows: rule stops LATER than the agent")
    print(f"  {(diff == 0).mean():.1%} of rows: same stop point")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    d.to_csv(args.out, index=False)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
