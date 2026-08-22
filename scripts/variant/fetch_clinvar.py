#!/usr/bin/env python
"""
Fetch REAL ClinVar labels + PVS1 (LoF) for the 12 spec genes, >=2-star,
P/LP vs B/LB, excluding VUS/conflicting. Produces the same schema as
pwkbench.domains.variant.make_synthetic_cohort but with real values for the
label and the PVS1 channel. PM2/PM1/PP3 are left undefined for the other
fetch scripts to fill.

Data source: ClinVar variant_summary.txt.gz (public NCBI FTP).
    https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
(~440 MB compressed.) This exact logic produced the bundled
data/sample/cohort_clinvar_real.parquet: 491 variants (an earlier ClinVar
snapshot targeted 506; PCSK9 is label-limited in ClinVar, only ~5-27
>=2-star P/LP depending on release, so the realized total is
snapshot-dependent -- see spec.py's header note and
docs/variant/real_data.md, "Note on cohort size").

Usage:
    python scripts/variant/fetch_clinvar.py --summary variant_summary.txt.gz \
        --out data/real/cohort_clinvar_real.parquet
"""

import os as _os, sys as _sys
# Make the package importable when the script is run directly from a
# checkout (scripts/<block>/x.py -> repo root is three dirnames up).
# Without this the script only works under `PYTHONPATH=.` or an
# installed package, which silently looks like the layout is broken.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import pandas as pd
from pwkbench import spec

TWO_STAR = {s.lower() for s in spec.REVIEW_STATUS_2STAR_PLUS}


def norm_sig(s):
    low = str(s).strip().lower()
    if "conflict" in low or "uncertain" in low:
        return None
    if low in {x.lower() for x in spec.POSITIVE_LABELS}:
        return 1
    if low in {x.lower() for x in spec.NEGATIVE_LABELS}:
        return 0
    return None


import re

# Loss-of-function is read from the HGVS consequence, NOT loose substrings.
# Substring matching mislabels deep-intronic variants (c.100+15A>G contains
# "+1"; c.7008-21T>C contains "-2") as LoF. We match structured HGVS tokens:
#   - nonsense: p.(Xaa)NNN(Ter|*)            e.g. p.Arg213Ter, p.Arg213*
#   - frameshift: p....fs                    e.g. p.Lys3Argfs*7
#   - canonical splice: c....(+1|+2|-1|-2)   ONLY at an intron boundary, i.e.
#     the digits immediately after +/- are 1 or 2 and not followed by another
#     digit (so +15 / -21 do NOT match).
_NONSENSE = re.compile(r"p\.[A-Za-z]{3}\d+(Ter|\*)", re.I)
_FRAMESHIFT = re.compile(r"p\.[A-Za-z]{3}\d+[A-Za-z]{0,3}fs", re.I)
_SPLICE = re.compile(r"c\.\d+[+-][12](?!\d)", re.I)


def is_lof(name):
    nm = str(name)
    return int(bool(_NONSENSE.search(nm) or _FRAMESHIFT.search(nm)
                    or _SPLICE.search(nm)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="path to variant_summary.txt.gz")
    ap.add_argument("--out", default="data/real/cohort_clinvar_real.parquet")
    ap.add_argument("--seed", type=int, default=spec.GLOBAL_SEED)
    args = ap.parse_args()

    genes = set(spec.GENES)
    # LastEvaluated / NumberSubmitters / VariationID are carried through so a
    # MEMORIZATION-CONTROL slice can be built later: an agent shown the real
    # variant name can recall a classification it saw in training, and the only
    # way to bound that is to score it separately on variants whose
    # classification post-dates the model's training cutoff.
    #
    # CAVEAT, read before using the date as a cutoff filter: LastEvaluated is
    # when the classification was last REVIEWED, not when the variant first
    # appeared in ClinVar. A variant submitted years ago and re-reviewed
    # recently gets a recent date while having been memorizable all along, so
    # this field OVER-counts "new" variants and the slice it builds is a
    # permissive bound, not a clean holdout. A strict first-appearance date
    # requires submission_summary.txt (a separate NCBI file keyed on
    # VariationID), which is why VariationID is retained here.
    keep = ["GeneSymbol", "ClinicalSignificance", "ReviewStatus", "Assembly",
            "Chromosome", "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF",
            "Type", "Name", "LastEvaluated", "NumberSubmitters", "VariationID"]
    parts = []
    for ch in pd.read_csv(args.summary, sep="\t", dtype=str, chunksize=200_000,
                          usecols=lambda c: c in keep, low_memory=False):
        ch = ch[(ch["GeneSymbol"].isin(genes)) & (ch["Assembly"] == "GRCh38")].copy()
        if not len(ch):
            continue
        ch["label"] = ch["ClinicalSignificance"].map(norm_sig)
        ch = ch[ch["label"].notna() &
                ch["ReviewStatus"].str.strip().str.lower().isin(TWO_STAR)]
        if len(ch):
            parts.append(ch)
    if not parts:
        raise SystemExit(
            "No variants passed the >=2-star P/LP/B/LB filter for any of the "
            "12 genes. Check the --summary file is the ClinVar variant_summary "
            "and covers GRCh38.")
    filt = pd.concat(parts, ignore_index=True)
    filt["label"] = filt["label"].astype(int)
    filt["PVS1_real"] = filt["Name"].map(is_lof)
    # Missense = single-residue substitution p.AaaNNNBbb where Bbb is a real
    # amino acid, explicitly NOT Ter (nonsense) and not a frameshift. The old
    # pattern [A-Z][a-z]{2} matched "Ter", so p.Arg213Ter was wrongly counted
    # both LoF and missense; exclude any variant already flagged LoF and any
    # ...Ter / ...fs tail.
    _AA3 = ("Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|"
            "Ser|Thr|Trp|Tyr|Val")
    mis_re = rf"p\.(?:{_AA3})\d+(?:{_AA3})(?![a-z])"
    filt["is_missense"] = (filt["Name"].str.contains(mis_re, regex=True, na=False)
                           & (filt["PVS1_real"] == 0))

    from pwkbench.domains.variant import _per_gene_counts
    counts = _per_gene_counts(spec.N_VARIANTS_TOTAL, spec.N_POSITIVE, spec.GENES)
    rows = []
    for gene in spec.GENES:
        size, npos = counts[gene]
        sub = filt[filt.GeneSymbol == gene]
        # sample with the user-supplied seed (was a hardcoded random_state=1)
        pos = sub[sub.label == 1].sample(min(npos, (sub.label == 1).sum()), random_state=args.seed)
        neg = sub[sub.label == 0].sample(min(size - npos, (sub.label == 0).sum()), random_state=args.seed)
        for _, r in pd.concat([pos, neg]).iterrows():
            mis = bool(r["is_missense"])
            rows.append(dict(
                variant_id=f"{r['Chromosome']}:{r['Start']}:{r['ReferenceAlleleVCF']}>{r['AlternateAlleleVCF']}",
                gene=gene, domain=spec.GENE_TO_DOMAIN[gene], label=int(r["label"]),
                PVS1=float(r["PVS1_real"]), PVS1__defined=True,
                # PM2/PM1/PP3 are placeholders here; fetch_annotations.py fills
                # them. Mark them UNDEFINED (not just missense-gated) -- a
                # placeholder NEUTRAL_VALUE stamped as "defined" would be a
                # constant, zero-information channel masquerading as real
                # evidence (this is the docstring's promise: "left undefined for
                # the other fetch scripts to fill").
                PM2=spec.NEUTRAL_VALUE, PM2__defined=False,
                PM1=spec.NEUTRAL_VALUE, PM1__defined=False,
                PP3=spec.NEUTRAL_VALUE, PP3__defined=False,
                is_missense=mis, source="CLINVAR_REAL", clinvar_name=r["Name"],
                # Memorization-control fields (see `keep` above for what
                # LastEvaluated does and does not mean). Parsed to a real date
                # so a cutoff comparison cannot silently succeed on strings;
                # unparseable/absent dates become NaT rather than a sentinel
                # that would sort as "old" or "new" by accident.
                clinvar_last_evaluated=pd.to_datetime(
                    r.get("LastEvaluated"), errors="coerce"),
                clinvar_n_submitters=pd.to_numeric(
                    r.get("NumberSubmitters"), errors="coerce"),
                clinvar_variation_id=r.get("VariationID")))
    out = pd.DataFrame(rows)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_parquet(args.out)
    print(f"wrote {len(out)} variants -> {args.out} "
          f"({out.label.sum()} P/LP, {(out.label==0).sum()} B/LB)")
    d = out["clinvar_last_evaluated"]
    n_dated = int(d.notna().sum())
    if n_dated:
        print(f"  clinvar_last_evaluated: {n_dated}/{len(out)} dated, "
              f"range {d.min():%Y-%m-%d} .. {d.max():%Y-%m-%d}")
        print("  (LastEvaluated = last REVIEW date, not first appearance -- a "
              "post-cutoff slice built\n   from it is a permissive bound on "
              "memorization, not a clean holdout.)")
    else:
        print("  WARNING: no LastEvaluated parsed -- a post-cutoff "
              "memorization slice cannot be built from this file.")


if __name__ == "__main__":
    main()
