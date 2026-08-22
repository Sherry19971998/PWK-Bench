#!/usr/bin/env python
"""
Fill PM2 / PP3 / PM1 onto a fetch_clinvar.py cohort using a SINGLE reliable
endpoint (Ensembl VEP REST), plus UniProt for PM1. This is the path that built
the bundled data/sample/cohort_full_real.parquet.

Why not fetch_annotations.py's gnomAD GraphQL: that API returns 502s often and
is queried one variant at a time. Ensembl VEP REST returns BOTH the gnomAD exome
allele frequency (af_gnomade=1) AND AlphaMissense (AlphaMissense=1) in ONE POST
to /vep/human/region, batched ~200 variants per call.

Channels:
  PM2 (rarity)   : -log10(gnomAD exome AF); variant absent from gnomAD -> 8.0
                   (maximally rare). defined on every variant VEP can map.
  PP3 (in silico): max AlphaMissense am_pathogenicity over transcripts.
                   Missense only; non-missense -> neutral (undefined).
  PM1 (domain)   : variant-level UniProt functional-domain membership
                   (fetch_uniprot_domains from fetch_annotations.py). Missense
                   only. strength 1.0 precise site / 0.6 broad domain / 0.2 out.

Usage:
    python scripts/variant/fetch_annotations_vep.py \
        --cohort data/sample/cohort_clinvar_real.parquet \
        --uniprot-domains data/sample/uniprot_domains.json \
        --out data/real/cohort_full.parquet
    # --uniprot-domains omitted -> fetched fresh from rest.uniprot.org
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
import re
import time
import numpy as np
import pandas as pd
from pwkbench import spec

VEP_URL = "https://rest.ensembl.org/vep/human/region"


def _region(vid):
    """Cohort variant_id 'chrom:pos:ref>alt' -> VEP region 'chrom pos . ref alt'."""
    try:
        chrom, pos, ra = vid.split(":")
        ref, alt = ra.split(">")
        chrom = chrom.replace("chr", "")
        if ref in ("", "na") or alt in ("", "na"):
            return None
        return f"{chrom} {pos} . {ref} {alt}"
    except Exception:
        return None


def _vep_batch(regions, chunk=200, retries=4):
    """POST regions to VEP in chunks; return {region_input: {af, am}}."""
    import requests
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
                        af = None
                        for c in rec.get("colocated_variants", []):
                            fr = c.get("frequencies")
                            if fr:
                                for _al, d in fr.items():
                                    af = d.get("gnomade") or d.get("gnomadg")
                                    if af is not None:
                                        break
                            if af is not None:
                                break
                        tc = rec.get("transcript_consequences", [])
                        ams = [t["alphamissense"]["am_pathogenicity"]
                               for t in tc if t.get("alphamissense")]
                        out[rec.get("input")] = {"af": af,
                                                 "am": max(ams) if ams else None}
                    break
                time.sleep(2 ** attempt)
            except Exception:
                time.sleep(2 ** attempt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--uniprot-domains",
                    help="JSON from fetch_uniprot_domains(); fetched fresh if omitted")
    ap.add_argument("--out", default="data/real/cohort_full.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.cohort).copy()
    df["_region"] = df["variant_id"].map(_region)
    regions = [x for x in df["_region"].tolist() if x]
    vep = _vep_batch(regions)
    n_af = sum(1 for v in vep.values() if v["af"] is not None)
    n_am = sum(1 for v in vep.values() if v["am"] is not None)
    print(f"[VEP] annotated {len(vep)}/{len(regions)} variants "
          f"(gnomAD AF on {n_af}, AlphaMissense on {n_am})")

    # PM2: -log10(AF); absent from gnomAD -> maximally rare (8.0)
    pm2, pm2_def = [], []
    for _, r in df.iterrows():
        reg = r["_region"]
        if not reg or reg not in vep:
            pm2.append(spec.NEUTRAL_VALUE); pm2_def.append(False); continue
        af = vep[reg]["af"]
        if af is None or af <= 0:
            pm2.append(8.0); pm2_def.append(True)          # absent -> max rare
        else:
            pm2.append(float(-np.log10(af))); pm2_def.append(True)
    df["PM2"] = pm2; df["PM2__defined"] = pm2_def

    # PP3: AlphaMissense (missense only)
    pp3, pp3_def = [], []
    for _, r in df.iterrows():
        reg = r["_region"]; am = vep.get(reg, {}).get("am") if reg else None
        if bool(r["is_missense"]) and am is not None:
            pp3.append(float(am)); pp3_def.append(True)
        else:
            pp3.append(spec.NEUTRAL_VALUE); pp3_def.append(False)
    df["PP3"] = pp3; df["PP3__defined"] = pp3_def

    # PM1: UniProt functional-domain membership (variant-level, missense only)
    if args.uniprot_domains:
        doms = json.load(open(args.uniprot_domains))
    else:
        from scripts.fetch_annotations import fetch_uniprot_domains
        fetch_uniprot_domains("/tmp/_uniprot_domains.json")
        doms = json.load(open("/tmp/_uniprot_domains.json"))
    _PRECISE = {"Active site", "Binding site", "Zinc finger", "DNA binding", "Motif"}
    _BROAD = {"Domain", "Region"}
    name_col = next((c for c in ("clinvar_name", "Name", "hgvs_p")
                     if c in df.columns), None)
    if name_col is None:
        raise KeyError("PM1 needs a protein-change column (clinvar_name/Name/hgvs_p)")

    def _pos(name):
        m = re.search(r"p\.[A-Za-z]{3}(\d+)", str(name))
        return int(m.group(1)) if m else None

    def _pm1(row):
        if not row["is_missense"]:
            return None
        pos = _pos(row[name_col])
        if pos is None:
            return None
        spans = doms.get(row["gene"], {}).get("spans", [])
        if any(s <= pos <= e for s, e, t in spans if t in _PRECISE):
            return 1.0
        if any(s <= pos <= e for s, e, t in spans if t in _BROAD):
            return 0.6
        return 0.2
    strength = df.apply(_pm1, axis=1)
    ok = df["is_missense"] & strength.notna()
    df.loc[ok, "PM1"] = strength[ok]
    df["PM1__defined"] = False; df.loc[ok, "PM1__defined"] = True
    n_mis = int(df["is_missense"].sum())
    if n_mis and int(ok.sum()) == 0:
        raise RuntimeError(f"PM1 resolved 0/{n_mis} missense -- AA positions did "
                           f"not parse from '{name_col}'")
    print(f"[PM1] domain membership defined on {int(ok.sum())}/{n_mis} missense")

    df = df.drop(columns=["_region"])
    df["source"] = "REAL"
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_parquet(args.out)
    for ch in ("PVS1", "PM2", "PP3", "PM1"):
        print(f"  {ch}: defined {int(df[f'{ch}__defined'].sum())}/{len(df)}")
    print(f"wrote {len(df)} variants -> {args.out}")


if __name__ == "__main__":
    main()
