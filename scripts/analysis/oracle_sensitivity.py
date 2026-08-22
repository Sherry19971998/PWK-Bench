#!/usr/bin/env python3
"""E2 -- oracle approximation sensitivity.

Reviewer concern: "how sensitive are the frozen conclusions to the greedy
oracle? deeper lookahead / local-swap?" -- i.e. is the paper's "Oracle"
reference actually a myopic greedy that might be far from the true per-budget
optimum, leaving the reported planning gap unbounded.

It is not myopic: `oracle_subset_curve` / `_best_subset_auc`
(pwkbench/metrics.py) already enumerates every channel SUBSET of size <=k
exhaustively and takes the cohort-AUC argmax -- exact by construction. This
script makes that claim falsifiable rather than asserted: it also computes a
genuinely myopic-greedy oracle and a 1-step-lookahead oracle over the same
subset lattice, and reports whether all three agree at every budget. At this
benchmark's K=4 (15 non-empty subsets total) they are expected to coincide,
because the lattice is small enough that a greedy search cannot get trapped
short of the exact optimum -- turning a reviewer weakness into a stated,
verified property of the benchmark's action size, not a promise about
general K.

USAGE
    python scripts/analysis/oracle_sensitivity.py \
        --cohort data/sample/cohort_full_real.parquet --domain variant
"""
import argparse, os, sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench import metrics as M                              # noqa: E402
from pwkbench.domains.base import load_real_cohort              # noqa: E402
from pwkbench.domains.variant import VARIANT_DOMAIN, make_synthetic_cohort  # noqa: E402
from pwkbench.spec import BUDGETS                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=None, help="real cohort parquet; omit for the offline demo")
    ap.add_argument("--domain", choices=["variant"], default="variant")
    ap.add_argument("--out", default="results/variant/real_trusted/oracle_sensitivity.csv")
    args = ap.parse_args()
    os.chdir(_ROOT)

    if args.cohort:
        cohort = load_real_cohort(pd.read_parquet(args.cohort), VARIANT_DOMAIN)
    else:
        cohort = make_synthetic_cohort()

    rows = M.oracle_approximation_sensitivity(cohort, budgets=BUDGETS)
    all_coincide = rows.pop("all_coincide")

    df = pd.DataFrame(rows).T
    df.index.name = "k"
    print(df.round(6).to_string())
    print()
    if all_coincide:
        print(f"ALL COINCIDE at every budget k in {BUDGETS}: exact global-subset "
              f"argmax == myopic-greedy == 1-step-lookahead. At K={cohort.domain.K} "
              f"there is no greedy-approximation gap in the oracle reference.")
    else:
        bad = [k for k in BUDGETS if abs(rows[k]["greedy_gap"]) > 1e-9
               or abs(rows[k]["lookahead_gap"]) > 1e-9]
        print(f"WARNING: exact/greedy/lookahead DISAGREE at budgets {bad}. "
              f"The 'no approximation gap' claim does NOT hold on this cohort "
              f"-- do not state it in the paper without this caveat.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
