#!/usr/bin/env python
"""
Paper entry point for RQ3 (Table tab:closed): the closed-book two-mask
memorization control. This is the script the paper cites as
"run_closed_book.py".

It runs pwkbench.metrics.memorization_probe on the cohort and prints / writes
the full validity panel: coordinate mask, full-HGVS mask, PVS1-alone ceiling,
full-acquisition ceiling, and the acquisition-attributable gain
G_acq = full_acquisition - full_hgvs (the paper's +0.077 axis-C quantity).

Offline by default (synthetic demo cohort, no API key).

Usage:
    python scripts/variant/run_closed_book.py --demo --domain variant
    python scripts/variant/run_closed_book.py --cohort data/real/cohort_full.parquet --domain variant
"""

import os as _os, sys as _sys
# Make the package importable when the script is run directly from a
# checkout (scripts/<block>/x.py -> repo root is three dirnames up).
# Without this the script only works under `PYTHONPATH=.` or an
# installed package, which silently looks like the layout is broken.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))

import argparse, os, json
import pandas as pd
from pwkbench.utils import set_seed
from pwkbench import metrics as M
from pwkbench.domains.variant import make_synthetic_cohort, VARIANT_DOMAIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="offline synthetic cohort")
    ap.add_argument("--cohort", help="parquet cohort (real run)")
    ap.add_argument("--domain", choices=["variant"], default="variant")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    set_seed()
    if args.cohort:
        from pwkbench.domains.base import load_real_cohort
        # same empirical-neutral loader as run_benchmark, so the same parquet
        # yields identical channel values in both scripts.
        cohort = load_real_cohort(pd.read_parquet(args.cohort), VARIANT_DOMAIN)
    else:
        cohort = make_synthetic_cohort()

    probe = M.memorization_probe(cohort)
    print("Closed-book two-mask memorization probe (RQ3):")
    for k, v in probe.items():
        print(f"  {k:18s} {v:.3f}")
    outdir = args.outdir or f"results/{args.domain}"
    os.makedirs(outdir, exist_ok=True)
    pd.DataFrame([probe]).to_csv(f"{outdir}/memorization_probe.csv", index=False)
    json.dump(probe, open(f"{outdir}/memorization_probe.json", "w"), indent=2)
    print(f"wrote -> {outdir}/memorization_probe.csv")


if __name__ == "__main__":
    main()
