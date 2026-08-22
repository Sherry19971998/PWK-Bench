#!/usr/bin/env python3
"""Does ClinVar-label circularity concentrate in the same domains as the
headline domain-specific findings?

WHY THIS EXISTS
----------------
Two of this repo's newest findings are domain-specific: arrhythmia's
over-acquisition multiple (`domain_overacquisition.png`, 12.7x, verified
3-run-stable) and lipid_disorder's null-corrected planning-gap excess
(`domain_gap_excess.png`, z=3.94, Holm p=0.0060). Both are argued from a
SMALL number of genes per domain (2 each), so a natural adversarial question
is: are these domains standing out because of unusual ClinVar SUBMISSION
PRACTICES (e.g. one lab's algorithmic pipeline over-represented in that gene
set, citing PM2/PP3-type evidence unusually often or unusually rarely) rather
than a genuine acquisition-behaviour or planning-gap effect? This checks the
label-circularity rate by domain, across all four ACMG-style categories.

RESULT (verified 2026-08-07, n=491, all four channels)
-----------------------------------------------------------
    domain              n    pm2    pp3    pm1    pvs1   any-of-4
    arrhythmia          84   0.607  0.548  0.012  0.333  0.881
    hereditary_cancer  254   0.449  0.476  0.051  0.429  0.870
    cardiomyopathy      84   0.536  0.655  0.036  0.238  0.857
    lipid_disorder      69   0.406  0.507  0.043  0.116  0.638

**lipid_disorder has the LOWEST overall citation rate (63.8%) and by far the
lowest PVS1 rate (11.6% vs 24-43% elsewhere)** -- if anything this argues
AGAINST its planning-gap finding being a circularity artifact: the domain
carrying the strongest verified effect is the one whose labels show the
LEAST evidence of criterion-language overlap with the acquirable channels,
not the most.

arrhythmia (the over-acquisition finding) sits at the HIGH end (88.1%,
highest PM2 rate at 60.7%) but that finding is measured in the LIVE block on
a completely different quantity (spend/need ratio from the agent's own tool
calls), which does not depend on the frozen block's ClinVar-label-vs-channel
scoring at all -- so this domain's high circularity rate has no direct
bearing on that specific result. It is reported here for completeness, not
because the two findings are mechanistically linked.

CAVEAT. n=69-254 per domain (already flagged as thin for statistical claims
elsewhere -- see domain_gap_excess.png's own caveats); this is a descriptive
cross-tabulation, not a test of whether domains differ in circularity rate.

USAGE
    python scripts/variant/make_circularity_by_domain.py
    (seconds -- reads the cached probe output, no network)
"""
import os, sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

PROBE = "results/variant/real_trusted/clinvar_circularity_probe.csv"
OUT = "results/variant/real_trusted/circularity_by_domain.csv"
CHANNELS = ["pm2_cited", "pp3_cited", "pm1_cited", "pvs1_cited"]


def main():
    os.chdir(_ROOT)
    coh = pd.read_parquet("data/sample/cohort_full_real.parquet")
    probe = pd.read_csv(PROBE)
    df = coh[["variant_id", "domain"]].merge(probe, on="variant_id", how="left")
    assert df["pm2_cited"].notna().all(), "probe coverage gap -- rerun probe_clinvar_circularity.py"

    g = df.groupby("domain").agg(n=("variant_id", "size"),
                                 **{c: (c, "mean") for c in CHANNELS})
    g["any_of_4"] = df.groupby("domain").apply(
        lambda x: x[CHANNELS].any(axis=1).mean(), include_groups=False)
    g = g.sort_values("any_of_4", ascending=False)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    g.to_csv(OUT)
    print(g.round(3).to_string())
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
