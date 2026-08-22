#!/usr/bin/env python3
"""Empirical check: do the ClinVar labels this cohort scores against overlap
with the same evidence (PM2 rarity, PP3 in-silico) this benchmark treats as
'acquirable'?

WHY THIS EXISTS
----------------
`A(k)` and every downstream planning-gap number scores a strategy's evidence
choices against the real ClinVar P/LP vs B/LB label. That label was itself
assigned by a submitting lab, and many labs apply the ACMG/AMP criteria --
PM2 (population rarity) and PP3 (in-silico prediction) among them -- to reach
it. If a variant's ClinVar classification was informed by the same evidence
category this benchmark exposes as a channel, then a strategy that acquires
that channel is not being tested on independent reasoning; it can partly
reconstruct a rule the label was already derived from. This is exactly the
failure mode this project labels CIRCULAR and excludes wherever caught (see
the 'somatic actionability' cohort) -- but it had never been checked for
THIS benchmark's main 491-variant germline cohort.

METHOD
------
ClinVar's bulk `variant_summary.txt.gz` (what `fetch_clinvar.py` reads) does
NOT expose per-submission ACMG criteria codes -- that granularity only exists
in the free-text `<Comment>` fields of each submission's VCV record (NCBI
eutils, db=clinvar). This script fetches those comments per variant and
keyword-matches them for two signals:

  1. CATEGORY-level (`pm2_cited` / `pp3_cited`): does any submitter's
     rationale reference population-frequency reasoning (gnomAD, "population
     database", "allele frequency", ...) or in-silico-prediction reasoning
     (AlphaMissense, REVEL, PolyPhen, SIFT, CADD, "in silico", ...) at all?
  2. LITERAL source-name level (`gnomad_named` / `alphamissense_named` /
     `revel_named` / `other_insilico_named`): does the comment name the EXACT
     database/tool this repo's own `fetch_annotations.py` queries (gnomAD v4
     for PM2, AlphaMissense via Ensembl-VEP for PP3)?

The two levels give qualitatively different answers (see RESULTS) -- category
overlap is not the same claim as literal same-source reuse, and both matter
for different reasons.

CAVEATS (do not drop these when citing this file)
----------------------------------------------------
- Free-text keyword matching is a PROXY, not a structured criteria extraction.
  False negatives (a lab used PM2 reasoning but phrased it unusually, or left
  no comment at all -- 5.3% of this cohort) and false positives (a keyword
  appearing without being load-bearing for the classification) are both
  possible. This measures "was this category of evidence WRITTEN ABOUT",
  not "how much did it weigh in the final call".
- Tests only whether SOME submitter's comment shows the signal; ClinVar
  classifications are typically an aggregate over multiple submissions.

RESULTS (verified 2026-08-07, n=491, 3 lookup failures)
-----------------------------------------------------------
    Category-level (any wording referencing the evidence type):
        full cohort:        pm2 48.5%   pp3 52.3%   either 78.6%
        missense subset(80): pm2 92.5%   pp3 90.0%   either 96.25%

    Literal source-name level:
        full cohort:        gnomAD named 39.1%   AlphaMissense named 0.0%
        missense subset(80): gnomAD named 83.75%  AlphaMissense named 0.0%
                             any in-silico TOOL (REVEL/PolyPhen/SIFT/CADD) 40%

    ASYMMETRIC RISK. PM2/gnomAD is near-literal circularity: the exact
    database this repo's pipeline queries is independently, frequently named
    by ClinVar submitters. PP3/AlphaMissense is NOT literally circular --
    AlphaMissense (2023) postdates most of these historical submissions, so
    no submitter could have used it; the 40% in-silico-tool citation rate
    among missense variants reflects OLDER, DIFFERENT tools (category-level
    correlation, not same-tool double-counting).

    ROBUSTNESS CHECK (`scripts/variant/make_circularity_robustness_check.py`):
    re-running the RelMax-vs-Oracle gap on the subset of variants where NO
    submitter cited PM2/PP3-type evidence at all (n=105) leaves the gap at
    0.0000 (vs 0.0115 on the full cohort) -- the "no planning gap" finding is
    NOT an artifact of circular labeling; if anything it tightens on the
    cleaner subset. Absolute AUC rises on that subset too (PE 0.93 vs 0.85),
    but this subset is NOT a random control -- variants nobody needed to
    argue about with PM2/PP3 language plausibly skew toward easier,
    already-obvious cases (e.g. clear truncating LoF), so that shift should
    NOT be read as "how much circularity inflated the full cohort's AUC".

USAGE
    python scripts/variant/probe_clinvar_circularity.py
    (~15 min: 491 variants x ~2 NCBI eutils calls, rate-limited to the public
    3 req/s cap; no API key configured. Writes and resumes from
    results/variant/real_trusted/clinvar_circularity_probe.csv, so a killed
    run can be restarted without re-fetching completed variants.)
"""
import argparse, json, os, re, sys, time
import urllib.parse
import urllib.request

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
OUT = "results/variant/real_trusted/clinvar_circularity_probe.csv"
SLEEP = 0.34   # public NCBI eutils rate cap is 3 req/s with no API key

# Category-level: does the comment reference this KIND of evidence at all.
PM2_KW = re.compile(
    r"gnomad|population database|allele frequency|not (?:observed|present|found) in|"
    r"absent from|minor allele frequency|\bmaf\b|exac|topmed|rare(?:ly)? (?:observed|seen)",
    re.I)
PP3_KW = re.compile(
    r"alphamissense|revel|cadd score|polyphen|\bsift\b|in[- ]silico|"
    r"computational (?:predict|tool|analys)|in silico|meta[- ]?predict|"
    r"missense (?:tolerance|prediction)|splice[a-z]* predict",
    re.I)
# Literal level: names the EXACT source this repo's fetch_annotations.py queries.
GNOMAD_RE = re.compile(r"gnomad", re.I)
ALPHAMISSENSE_RE = re.compile(r"alpha[- ]?missense", re.I)
REVEL_RE = re.compile(r"revel", re.I)
OTHER_TOOL_RE = re.compile(r"polyphen|\bsift\b|cadd|meta[- ]?predict|dann|primateai", re.I)
# The other two ACMG-style categories, added 2026-08-07 -- the first pass only
# covered PM2/PP3 and left PVS1/PM1 unchecked, which was an incomplete test of
# a 4-channel benchmark. PVS1 turns out to be the second-most-cited category
# in this cohort (33.6%); PM1 is rare (4.1%) -- see module docstring RESULTS.
PVS1_KW = re.compile(
    r"truncat|frameshift|frame shift|nonsense|premature (?:stop|termination)|"
    r"null variant|loss.of.function|\blof\b|splice (?:donor|acceptor|site)|"
    r"nonsense.mediated decay|\bnmd\b|stop.gain|start.loss|stop codon",
    re.I)
PM1_KW = re.compile(
    r"functional domain|hot.?spot|mutational hot ?spot|critical (?:domain|region)|"
    r"well.established (?:functional )?domain|structurally important|active site|"
    r"binding domain|catalytic domain|important (?:functional )?region",
    re.I)


def _fetch(url, params, retries=5):
    q = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(q, timeout=20) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1.5 * (attempt + 1))
    return None


def _vcv_id(clinvar_name):
    j = _fetch(ESEARCH, dict(db="clinvar", term=clinvar_name, retmode="json"))
    if not j:
        return None
    try:
        ids = json.loads(j).get("esearchresult", {}).get("idlist", [])
        return ids[0] if ids else None
    except Exception:
        return None


def _comments(vcv_id):
    # is_variationid + from_esearch: an esearch UID is NOT directly usable as
    # a VCV accession without these flags (confirmed empirically -- omitting
    # them returns an empty <set/>, silently, not an error).
    xml = _fetch(EFETCH, dict(db="clinvar", id=vcv_id, rettype="vcv",
                              is_variationid="true", from_esearch="true"))
    if not xml:
        return None
    return re.findall(r"<Comment[^>]*>(.*?)</Comment>", xml, re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--n-sample", type=int, default=None,
                    help="probe only a random subsample (for a quick pilot)")
    args = ap.parse_args()
    os.chdir(_ROOT)

    cohort = pd.read_parquet("data/sample/cohort_full_real.parquet")
    if args.n_sample:
        cohort = cohort.sample(n=args.n_sample, random_state=0)

    done = {}
    if os.path.exists(args.out):
        done = {r.variant_id: r._asdict() for r in pd.read_csv(args.out).itertuples()}
        print(f"resuming: {len(done)} already probed")

    rows = list(done.values())
    t0 = time.time()
    for i, row in cohort.iterrows():
        vid, name = row["variant_id"], row["clinvar_name"]
        if vid in done:
            continue
        uid = _vcv_id(name)
        time.sleep(SLEEP)
        comments = _comments(uid) if uid else None
        time.sleep(SLEEP)
        text = " ".join(comments) if comments else ""
        low = text.lower()
        rows.append(dict(
            variant_id=vid, uid=uid, n_comments=len(comments) if comments else 0,
            pm2_cited=bool(PM2_KW.search(text)), pp3_cited=bool(PP3_KW.search(text)),
            pvs1_cited=bool(PVS1_KW.search(text)), pm1_cited=bool(PM1_KW.search(text)),
            gnomad_named=bool(GNOMAD_RE.search(low)),
            alphamissense_named=bool(ALPHAMISSENSE_RE.search(low)),
            revel_named=bool(REVEL_RE.search(low)),
            other_insilico_named=bool(OTHER_TOOL_RE.search(low)),
        ))
        if len(rows) % 25 == 0:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
            el = time.time() - t0
            done_now = len(rows) - len(done)
            remaining = (len(cohort) - len(rows)) * (el / max(done_now, 1))
            print(f"  {len(rows)}/{len(cohort)}  ({el:.0f}s elapsed, "
                  f"~{remaining:.0f}s remaining)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"written: {args.out}  ({len(rows)} variants)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
