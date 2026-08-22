#!/usr/bin/env python3
"""Standalone, citable CSV for the guideline-native ACMG-points clinical
endpoint -- the reference-strategy rows only.

WHY THIS EXISTS
----------------
`clinical_yield.csv` already carries `acmg_resolved` / `acmg_call_acc`
columns (`M.acmg_points_curve`, run inside `run_benchmark.py`), but they sit
folded into a wider table dominated by the logistic-posterior columns, which
makes the guideline-native result hard to cite on its own. `acmg_points_curve`
is the more clinically credible of the two clinical-endpoint views: it applies
the published ACMG/ClinGen Bayesian point table (Tavtigian et al. 2020)
directly to the acquired evidence, with no fitted model and no data-driven
threshold -- unlike `clinical_yield_curve`, which trains an out-of-fold
logistic posterior. A reviewer with a clinical-genetics background is far more
likely to trust "we applied the published point table" than "we fit a
calibrated classifier and called anything >=90% confident."

SCOPE OF THIS FILE: REFERENCE STRATEGIES ONLY (RelMax, BestFixed). Both are
deterministic, label-free-or-cohort-fixed orderings computable straight from
the cohort -- no LLM call, no cost, no risk. The real per-agent rows
(frontier_B = gpt-5.5, frontier_D = gpt-5.4, budget_C = gemini-lite) that
appear in `clinical_yield.csv` require that agent's ACTUAL elicited order,
which is NOT reproducible from this repo's on-disk state alone: real_trusted's
agent rows carry no `cache_path` (see `configs/models.yaml`, intentional -- a
fresh independent sample every invocation), and an attempt to replay the
`results/variant/variance/.cache_frontier_B_run0.json` cache against today's
`_prompt`/_SYS` text found ZERO matching keys -- the prompt-building code has
almost certainly drifted since that cache was written, so a hash-keyed replay
against it is unsound (confirmed 2026-08-07; do not retry this without either
a live `OPENAI_API_KEY` or the exact historical code revision). Getting the
agent rows into this same guideline-native form requires a fresh, cheap
(~27 real calls under context:false) sweep with a live key -- out of scope
for this script.

USAGE
    python scripts/variant/make_acmg_points_csv.py
    (seconds -- pure rule application over the cohort, no network, no LLM)
"""
import os, sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench.domains.base import load_real_cohort              # noqa: E402
from pwkbench.domains.variant import VARIANT_DOMAIN             # noqa: E402
from pwkbench.strategies import relmax_order, best_fixed_order  # noqa: E402
import pwkbench.metrics as M                                    # noqa: E402

OUT = "results/variant/real_trusted/acmg_points.csv"


def main():
    os.chdir(_ROOT)
    coh = load_real_cohort(
        pd.read_parquet("data/sample/cohort_full_real.parquet"), VARIANT_DOMAIN)

    strategies = {
        "RelMax": relmax_order(coh),
        "BestFixed": best_fixed_order(coh),
    }
    rows = []
    for name, order in strategies.items():
        apc = M.acmg_points_curve(coh, order)
        for k in sorted(apc):
            v = apc[k]
            rows.append(dict(strategy=name, budget=k,
                             resolved_frac=v["resolved_frac"],
                             vus_frac=v["vus_frac"],
                             call_accuracy=v["call_accuracy"],
                             balanced_call_accuracy=v["balanced_call_accuracy"],
                             n_called=v["n_called"]))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print(f"\nwritten: {OUT}")
    print("\nNOTE: agent rows (frontier_B/D, budget_C) are NOT in this file --")
    print("see module docstring for why they cannot be safely reconstructed")
    print("from this repo's current on-disk state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
