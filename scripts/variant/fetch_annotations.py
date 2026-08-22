#!/usr/bin/env python
"""
Fill the annotation channels PM2 / PM1 / PP3 onto a cohort produced by
fetch_clinvar.py. Each channel has a documented public source:

  PM2  (rarity)      : gnomAD v4 allele frequency via the gnomAD GraphQL API
                       (https://gnomad.broadinstitute.org/api). PM2 = -log10(AF);
                       variants absent from gnomAD are treated as maximally rare.
  PM1  (constraint)  : gene-level missense constraint mis_z from the gnomAD
                       gene constraint table (one value per gene; applied to
                       missense variants only).
  PP3  (in silico)   : AlphaMissense pathogenicity via Ensembl VEP REST
                       (https://rest.ensembl.org, AlphaMissense plugin, GRCh38,
                       max over transcripts) OR the AlphaMissense_hg38.tsv.gz
                       precomputed table (Zenodo). Missense only; non-missense
                       receives the neutral value 0.5 (undefined).

This script contains the request logic for each source. Network access to
these APIs may be restricted in some sandboxes; run it locally where the
domains are reachable. Provide --contact-email for the NCBI/Ensembl user-agent
convention if you have one.

Usage:
    python scripts/variant/fetch_annotations.py \
        --cohort data/real/cohort_clinvar_real.parquet \
        --constraint gnomad_gene_constraint.tsv \
        --alphamissense AlphaMissense_hg38.tsv.gz \
        --out data/real/cohort_full.parquet
"""

import os as _os, sys as _sys
# Make the package importable when the script is run directly from a
# checkout (scripts/<block>/x.py -> repo root is three dirnames up).
# Without this the script only works under `PYTHONPATH=.` or an
# installed package, which silently looks like the layout is broken.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import numpy as np
import pandas as pd
from pwkbench import spec


def gnomad_af(chrom, pos, ref, alt, dataset="gnomad_r4"):
    """Query one variant's global AF from gnomAD GraphQL. Returns AF or None."""
    import requests
    q = """
    query($vid: String!, $ds: DatasetId!) {
      variant(variantId: $vid, dataset: $ds) { genome { af } exome { af } }
    }"""
    vid = f"{str(chrom).replace('chr','')}-{pos}-{ref}-{alt}"
    r = requests.post("https://gnomad.broadinstitute.org/api",
                      json={"query": q, "variables": {"vid": vid, "ds": dataset}},
                      timeout=30)
    r.raise_for_status()
    v = (r.json().get("data") or {}).get("variant")
    if not v:
        return None
    afs = [x["af"] for x in (v.get("genome"), v.get("exome")) if x and x.get("af")]
    return max(afs) if afs else None


# Canonical reviewed UniProt accessions for the 12 benchmark genes.
UNIPROT_ACC = {
    "BRCA1": "P38398", "BRCA2": "P51587", "TP53": "P04637", "PTEN": "P60484",
    "MLH1": "P40692", "MSH2": "P43246", "MYH7": "P12883", "MYBPC3": "Q14896",
    "KCNQ1": "P51787", "SCN5A": "Q14524", "LDLR": "P01130", "PCSK9": "Q8NBP7",
}


def fetch_uniprot_domains(out="uniprot_domains.json"):
    """Fetch UniProt functional features (Domain/Region/Zinc finger/Motif/
    DNA binding/Binding site/Active site) with amino-acid spans for the 12
    benchmark genes, writing {gene: {"acc","seq_len","spans":[[s,e,type],...]}}
    for PM1 (variant-level domain membership). Needs rest.uniprot.org."""
    import requests, json, time
    keep = {"Domain", "Zinc finger", "Motif", "DNA binding",
            "Region", "Binding site", "Active site"}
    out_map = {}
    for g, acc in UNIPROT_ACC.items():
        r = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.json",
                         params={"fields": "ft_domain,ft_region,ft_zn_fing,ft_motif,"
                                 "ft_dna_bind,ft_binding,ft_act_site,sequence"}, timeout=30)
        r.raise_for_status()
        j = r.json()
        spans = []
        for f in j.get("features", []):
            if f.get("type") in keep:
                loc = f.get("location", {})
                s = loc.get("start", {}).get("value")
                e = loc.get("end", {}).get("value")
                if s and e:
                    spans.append([int(s), int(e), f["type"]])
        out_map[g] = {"acc": acc, "seq_len": j.get("sequence", {}).get("length"),
                      "spans": spans}
        time.sleep(0.3)
    json.dump(out_map, open(out, "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--uniprot-domains",
                    help="JSON {gene: {'spans': [[start,end,type],...]}} of UniProt "
                         "functional features; PM1 = variant-level domain membership "
                         "(preferred, ACMG-faithful). Build with fetch_uniprot_domains().")
    ap.add_argument("--constraint", help="gene-level gnomAD mis_z tsv (FALLBACK only; "
                    "gene-level constant -> dead channel under per-gene AUC)")
    ap.add_argument("--alphamissense", help="AlphaMissense_hg38.tsv.gz precomputed table")
    ap.add_argument("--out", default="data/real/cohort_full.parquet")
    ap.add_argument("--query-gnomad", action="store_true",
                    help="hit the gnomAD GraphQL API per variant (slow; needs network)")
    args = ap.parse_args()

    df = pd.read_parquet(args.cohort).copy()

    # PM1 = VARIANT-LEVEL functional-domain membership (ACMG PM1: "located in a
    # mutational hot spot and/or critical, well-established functional domain").
    # A variant's amino-acid position is tested against UniProt functional
    # features for its gene; strength depends on feature specificity:
    #   1.0  precise site  (Active site / Binding site / Zinc finger /
    #                       DNA binding / Motif)
    #   0.6  broad domain  (Domain / Region)
    #   0.2  outside any annotated functional region
    # This replaces the earlier GENE-LEVEL mis_z, which was constant within a
    # gene and therefore contributed ~0 to the per-gene-stratified AUC (a "dead
    # channel"). Domain membership is a genuine per-variant signal, though on a
    # cohort of well-studied genes whose missense set is already
    # pathogenic-enriched its DISCRIMINATION is weak (see docs/variant/real_data.md);
    # it is retained because it is the ACMG-faithful, per-variant definition,
    # not because it is a strong predictor. Requires --uniprot-domains (a JSON
    # {gene: {"spans": [[start,end,type],...]}}) and amino-acid positions parsed
    # from the ClinVar protein change (p.XxxNNNYyy).
    if args.uniprot_domains:
        import json as _json, re as _re
        doms = _json.load(open(args.uniprot_domains))
        _PRECISE = {"Active site", "Binding site", "Zinc finger",
                    "DNA binding", "Motif"}
        _BROAD = {"Domain", "Region"}

        # AA position comes from the ClinVar protein change. fetch_clinvar.py
        # writes this as `clinvar_name` (= ClinVar `Name`, e.g.
        # "NM_007294.4(BRCA1):c.5297T>A (p.Ile1766Asn)"). Fail loudly if no such
        # column exists, so a schema mismatch cannot silently mark every PM1
        # undefined and masquerade as a dead channel.
        _name_col = next((c for c in ("clinvar_name", "Name", "hgvs_p")
                          if c in df.columns), None)
        if _name_col is None:
            raise KeyError("PM1 (variant-level domain) needs a ClinVar protein "
                           "change column (clinvar_name/Name/hgvs_p) to parse the "
                           "amino-acid position; none found in the cohort.")
        _n_missense = int(df["is_missense"].sum())

        def _aa_pos(name):
            m = _re.search(r'p\.[A-Za-z]{3}(\d+)', str(name))
            return int(m.group(1)) if m else None

        def _pm1_strength(row):
            if not row["is_missense"]:
                return None
            pos = _aa_pos(row[_name_col])
            if pos is None:
                return None
            spans = doms.get(row["gene"], {}).get("spans", [])
            if any(s <= pos <= e for s, e, t in spans if t in _PRECISE):
                return 1.0
            if any(s <= pos <= e for s, e, t in spans if t in _BROAD):
                return 0.6
            return 0.2
        strength = df.apply(_pm1_strength, axis=1)
        ok = df["is_missense"] & strength.notna()
        df.loc[ok, "PM1"] = strength[ok]
        df.loc[ok, "PM1__defined"] = True
        df.loc[df["is_missense"] & ~strength.notna(), "PM1__defined"] = False
        _n_pm1 = int(ok.sum())
        if _n_missense and _n_pm1 == 0:
            raise RuntimeError(
                f"PM1 domain membership resolved to 0/{_n_missense} missense "
                f"variants -- AA positions did not parse from '{_name_col}'. "
                "Check the protein-change format; refusing to ship a silently "
                "dead PM1 channel.")
        import sys as _sys
        print(f"[PM1] variant-level domain membership: defined on {_n_pm1}/"
              f"{_n_missense} missense variants", file=_sys.stderr)
    elif args.constraint:
        # FALLBACK (documented): gene-level mis_z. Kept for reproducibility of
        # the earlier pipeline, but note it is a gene-level constant and behaves
        # as a dead channel under per-gene AUC -- prefer --uniprot-domains.
        con = pd.read_csv(args.constraint, sep="\t")
        col = "mis_z" if "mis_z" in con.columns else con.columns[-1]
        gene_z = dict(zip(con["gene"] if "gene" in con.columns else con.iloc[:, 0], con[col]))
        z = df["gene"].map(gene_z)
        has_z = df["is_missense"] & z.notna()
        df.loc[has_z, "PM1"] = z[has_z]
        df.loc[has_z, "PM1__defined"] = True
        df.loc[df["is_missense"] & ~z.notna(), "PM1__defined"] = False

    # PP3 AlphaMissense from the precomputed table (join by chrom:pos:ref:alt)
    if args.alphamissense:
        am = pd.read_csv(args.alphamissense, sep="\t", comment="#",
                         names=["chrom", "pos", "ref", "alt", "genome",
                                "uniprot", "transcript", "prot_var", "am_path", "am_class"])
        key = am.assign(vid=am["chrom"].str.replace("chr", "") + ":" + am["pos"].astype(str)
                        + ":" + am["ref"] + ">" + am["alt"])
        am_map = key.groupby("vid")["am_path"].max().to_dict()
        df["_vid"] = df["variant_id"].str.replace("chr", "")
        hit = df["_vid"].map(am_map)
        # PP3 (AlphaMissense) is a MISSENSE-only channel: only set it defined
        # where the variant is missense AND has an AlphaMissense score, so a
        # non-missense variant that happens to map is not marked defined
        # (contradicting the "missense only" documentation and the 95-missense
        # split).
        pp3_ok = hit.notna() & df["is_missense"].astype(bool)
        df.loc[pp3_ok, "PP3"] = hit[pp3_ok]
        df.loc[pp3_ok, "PP3__defined"] = True
        df.drop(columns=["_vid"], inplace=True)

    # PM2 rarity from gnomAD (optional; network)
    if args.query_gnomad:
        vals, defined = [], []
        for _, r in df.iterrows():
            ok, af = True, None
            try:
                c, p, rr = r["variant_id"].split(":")[0], r["variant_id"].split(":")[1], r["variant_id"].split(":")[2]
                ref, alt = rr.split(">")
                af = gnomad_af(c, p, ref, alt)      # returns None for a genuine 'absent from gnomAD'
            except Exception:
                ok = False                          # a NETWORK/parse failure, NOT 'absent'
            if not ok:
                # Do not fabricate rarity from a failed query: a timeout is not
                # evidence of absence, and 8.0 is the pathogenic extreme. Leave
                # the entry UNDEFINED so it carries the neutral value, not a
                # spurious pathogenic push.
                vals.append(spec.NEUTRAL_VALUE); defined.append(False)
            elif af is None:
                # genuinely absent from gnomAD -> maximally rare (real PM2 signal)
                vals.append(8.0); defined.append(True)
            else:
                vals.append(-np.log10(af)); defined.append(True)
        df["PM2"] = vals
        df["PM2__defined"] = defined

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out)
    print(f"wrote annotated cohort -> {args.out}")
    for c in spec.CHANNELS:
        print(f"  {c}: {int(df[f'{c}__defined'].sum())}/{len(df)} defined")


if __name__ == "__main__":
    main()
