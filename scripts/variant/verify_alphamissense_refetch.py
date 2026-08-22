#!/usr/bin/env python3
"""Validity check: does the FROZEN evidence pool used by every frozen-pool
result still match a LIVE query for the same variants?

WHY THIS EXISTS
----------------
The frozen-pool paradigm (Pillar 1) pre-fetches all evidence once and freezes
it, so every strategy and every agent sees byte-for-byte identical evidence
(see `docs/variant/real_data.md` and the Acquisition Protocol section of the
paper). That is only a faithful stand-in for a live query if the frozen
values actually agree with what a live query would return. This had been
asserted in the paper text ("re-fetching AlphaMissense (PP3) from Ensembl VEP
for a 15-variant sample reproduced the stored values exactly") without a
checked-in artifact anyone could point to -- this script is that artifact.

METHOD
------
Re-fetches PP3 (AlphaMissense `am_pathogenicity`, max over transcripts) for a
random sample of PP3-defined (missense) variants from the bundled cohort,
using the EXACT same Ensembl VEP REST endpoint and query parameters as
`scripts/variant/fetch_annotations_vep.py` (the script that built
`data/sample/cohort_full_real.parquet` in the first place). Compares the
freshly fetched values against the stored `PP3` column.

RESULTS (verified 2026-08-12, n=15, seed=20260101)
-----------------------------------------------------------
    Matched: 15/15 (0 missing from VEP)
    Pearson r    = 1.000000
    Spearman rho = 1.000000
    Max abs diff = 0.000000
    Mean abs diff = 0.000000

    Every one of the 15 stored PP3 values reproduced exactly to 4 decimal
    places from a live re-query (e.g. PTEN 10:87952135:T>A: stored 0.9999,
    refetched 0.9999; BRCA1 17:43071077:T>C: stored 0.0840, refetched 0.0840).
    This is consistent with AlphaMissense being a static, versioned score
    table (not a live-updated prediction), so exact reproduction is expected
    rather than a coincidence -- but it had never been checked against this
    specific cohort file before this script existed.

CAVEATS
-------
- Only checks PP3 (AlphaMissense). PM2 (gnomAD allele frequency) is NOT
  static -- gnomAD releases update allele counts -- so a PM2 re-fetch would
  not be expected to reproduce exactly and is out of scope here.
- A fresh random sample is drawn each run unless --seed is fixed; use the
  default seed (matches `spec.GLOBAL_SEED`) to reproduce the RESULTS above
  exactly.
- Requires network access to rest.ensembl.org.

USAGE
    python scripts/variant/verify_alphamissense_refetch.py
    python scripts/variant/verify_alphamissense_refetch.py --n 30 --seed 1
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr, spearmanr

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
from pwkbench import spec  # noqa: E402

VEP_URL = "https://rest.ensembl.org/vep/human/region"
DEFAULT_COHORT = "data/sample/cohort_full_real.parquet"
DEFAULT_OUT = "results/variant/real_trusted/alphamissense_refetch_validation.csv"


def _region(vid: str) -> str:
    """Cohort variant_id 'chrom:pos:ref>alt' -> VEP region 'chrom pos . ref alt'.

    Identical to `fetch_annotations_vep.py::_region` -- must stay identical or
    this is no longer a genuine re-fetch of the same query.
    """
    chrom, pos, ra = vid.split(":")
    ref, alt = ra.split(">")
    chrom = chrom.replace("chr", "")
    return f"{chrom} {pos} . {ref} {alt}"


def _vep_batch(regions: list[str], chunk: int = 200, retries: int = 4) -> dict:
    """POST regions to VEP in chunks; return {region_input: am_pathogenicity or None}."""
    out = {}
    for i in range(0, len(regions), chunk):
        part = regions[i:i + chunk]
        for attempt in range(retries):
            try:
                r = requests.post(
                    VEP_URL,
                    headers={"Content-Type": "application/json",
                             "Accept": "application/json"},
                    data=json.dumps({"variants": part}),
                    params={"AlphaMissense": 1, "af_gnomade": 1, "af_gnomadg": 1},
                    timeout=120)
                if r.status_code == 200:
                    for rec in r.json():
                        tc = rec.get("transcript_consequences", [])
                        ams = [t["alphamissense"]["am_pathogenicity"]
                               for t in tc if t.get("alphamissense")]
                        out[rec.get("input")] = max(ams) if ams else None
                    break
                time.sleep(2 ** attempt)
            except Exception:
                time.sleep(2 ** attempt)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", default=DEFAULT_COHORT)
    ap.add_argument("--n", type=int, default=15,
                    help="sample size (paper quotes n=15)")
    ap.add_argument("--seed", type=int, default=spec.GLOBAL_SEED)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    df = pd.read_parquet(args.cohort)
    defined = df[df["PP3__defined"]].copy()
    if len(defined) < args.n:
        raise ValueError(f"only {len(defined)} PP3-defined variants available, "
                         f"cannot sample {args.n}")
    print(f"PP3-defined (missense) variants available: {len(defined)}")

    sample = defined.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    sample["_region"] = sample["variant_id"].map(_region)

    print(f"Re-querying Ensembl VEP for {args.n} variants (seed={args.seed})...")
    vep = _vep_batch(sample["_region"].tolist())

    rows = []
    for _, r in sample.iterrows():
        rows.append({
            "variant_id": r["variant_id"],
            "gene": r["gene"],
            "stored_PP3": r["PP3"],
            "refetched_AM": vep.get(r["_region"]),
        })
    out = pd.DataFrame(rows)
    out["abs_diff"] = (out["stored_PP3"] - out["refetched_AM"]).abs()
    print(out.to_string(index=False))

    ok = out.dropna(subset=["refetched_AM"])
    n_missing = len(out) - len(ok)
    print(f"\nMatched: {len(ok)}/{args.n}, missing from VEP: {n_missing}")
    if len(ok) >= 2:
        pear = pearsonr(ok["stored_PP3"], ok["refetched_AM"])
        spear = spearmanr(ok["stored_PP3"], ok["refetched_AM"])
        print(f"Pearson r    = {pear.statistic:.6f} (p={pear.pvalue:.2e})")
        print(f"Spearman rho = {spear.statistic:.6f} (p={spear.pvalue:.2e})")
    print(f"Max abs diff  = {ok['abs_diff'].max():.6f}")
    print(f"Mean abs diff = {ok['abs_diff'].mean():.6f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
