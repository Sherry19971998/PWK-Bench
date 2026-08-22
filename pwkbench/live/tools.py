"""Live evidence-acquisition tools for the tool-using agent study.

Each function is one real evidence source the agent may choose to call. Every
call is cached on disk per (variant, tool) and logged, so a rerun is cheap and a
trajectory is reproducible.

WHAT IS DELIBERATELY ABSENT: a ClinVar-assertion tool
----------------------------------------------------
The task label IS the ClinVar classification, so a tool returning prior ClinVar
submissions hands the agent the answer at RUN TIME. That is not a memorization
problem and none of the study's memorization controls (no-tool baseline,
post-cutoff slice) would catch it -- those bound what the model learned in
training, whereas this leaks during inference. The four tools here are all
evidence; none is the adjudicated label. Do not add one back.

ONE UPSTREAM CALL, TWO SEPARATE TOOLS
-------------------------------------
Ensembl VEP returns population frequency AND AlphaMissense in a single response,
so the backend fetches once and caches it. The two tools nevertheless project out
strictly their own field: the whole measurement is which evidence the agent
CHOOSES to acquire, so calling `get_population_freq` must never reveal the
in-silico score as a side effect. `_vep_record` is the shared fetch;
`get_population_freq` and `get_insilico_pathogenicity` are the only readers, and
each returns one field.
"""
from __future__ import annotations
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

# E4: consequence-token masking. The closed-book
# baseline (0.866 AUC, results/variant/real_trusted/memorization_probe.csv)
# is high partly because loss-of-function status is FREE-READABLE from the
# HGVS name the agent is shown (`p.Arg213Ter`, `...fs`) -- PVS1 costs no tool
# call. Masking the parenthetical p.-level consequence (`(p.Arg213Ter)` ->
# `(p.?)`) removes that free read: LoF status becomes something the agent must
# infer or acquire evidence for, not parse off the name. Variants with no
# p.-level notation (splice/UTR changes) are left unchanged -- there was
# nothing to mask, not a failure to mask.
_P_DOT = re.compile(r"\(p\.[^)]*\)")


def mask_consequence(clinvar_name: str) -> str:
    """Redact the p.-level protein consequence from an HGVS name string."""
    return _P_DOT.sub("(p.?)", clinvar_name)


VEP_URL = "https://rest.ensembl.org/vep/human/region"
UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{acc}.json"
MAVEDB_TARGETS = "https://api.mavedb.org/api/v1/target-genes/"

# Reused verbatim from scripts/variant/fetch_annotations.py so the live tools and the
# frozen-pool cohort resolve the same accession per gene.
UNIPROT_ACC = {
    "BRCA1": "P38398", "BRCA2": "P51587", "TP53": "P04637", "PTEN": "P60484",
    "MLH1": "P40692", "MSH2": "P43246", "MYH7": "P12883", "MYBPC3": "Q14896",
    "KCNQ1": "P51787", "SCN5A": "Q14524", "LDLR": "P01130", "PCSK9": "Q8NBP7",
}

# Ordinal acquisition burden, NOT a price or a ratio (same discipline as the
# SPHERE domain): a database lookup is cheaper than a functional assay, and that
# ordering is all that is claimed. Ranks are never summed.
TOOL_COST_RANK = {
    "get_population_freq": 1,
    "get_insilico_pathogenicity": 1,
    "get_domain_context": 2,
    "get_functional_assay": 3,
}
TOOL_COST_BASIS = {
    "get_population_freq": "database lookup (gnomAD via Ensembl VEP)",
    "get_insilico_pathogenicity": "database lookup (AlphaMissense via Ensembl VEP)",
    "get_domain_context": "database lookup + span arithmetic (UniProt)",
    "get_functional_assay": "curated wet-lab assay (MaveDB DMS/MAVE)",
}

_lock = threading.Lock()


class ToolCache:
    """Disk-backed (variant, tool) -> result cache plus a call log.

    The log is the experiment's raw data: which tool the agent called, in what
    order, for which variant. It is appended to on every call INCLUDING cache
    hits, because the measurement is the agent's decision to acquire, not
    whether the bytes happened to be local already.
    """

    def __init__(self, path: str | None = None):
        self.path = path
        self._data = {}
        if path and os.path.exists(path):
            try:
                with open(path) as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._data = {}
        self.calls = []          # [{variant_id, tool, cached, ok}]

    def get(self, key):
        return self._data.get(key)

    def put(self, key, value):
        # Guarded: the runner fans cases out across threads, so without the lock
        # two workers can serialize `_data` concurrently and race on the SAME
        # temp path -- one os.replace then clobbers the other's file, silently
        # dropping cached tool results that were paid for.
        with _lock:
            self._data[key] = value
            if not self.path:
                return
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = f"{self.path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "w") as f:
                json.dump(self._data, f)
            os.replace(tmp, self.path)  # atomic: a kill mid-write must not
                                        # corrupt an expensive cache

    def log(self, variant_id, tool, cached, ok):
        with _lock:
            self.calls.append({"variant_id": variant_id, "tool": tool,
                               "cached": bool(cached), "ok": bool(ok)})


def _http_json(url, data=None, params=None, timeout=60, retries=4):
    """GET/POST JSON with backoff. Returns None on persistent failure rather than
    raising: a dead upstream must show up as a tool that returned no evidence
    (which the agent then has to handle), not as a crashed sweep."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    body = json.dumps(data).encode() if data is not None else None
    headers = {"Accept": "application/json", "User-Agent": "pwkbench-live"}
    if body:
        headers["Content-Type"] = "application/json"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(min(2 ** attempt, 8))
    return None


import urllib.parse  # noqa: E402  (used by _http_json)


def _region(variant_id: str):
    """Cohort 'chrom:pos:ref>alt' -> VEP region string, or None if malformed."""
    try:
        chrom, pos, ra = variant_id.split(":")
        ref, alt = ra.split(">")
        chrom = chrom.replace("chr", "")
        if ref in ("", "na") or alt in ("", "na"):
            return None
        return f"{chrom} {pos} . {ref} {alt}"
    except Exception:
        return None


def _vep_record(variant_id: str, cache: ToolCache):
    """Shared VEP fetch. Cached under a key of its own so the two VEP-backed
    tools share one upstream call without either revealing the other's field."""
    key = f"_vep::{variant_id}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    region = _region(variant_id)
    if region is None:
        # Unparseable coordinate (e.g. 'na>na' in the cohort). Mark it so the
        # tools report "could not query" instead of "absent from gnomAD" -- the
        # latter is affirmative evidence of rarity and would be fabricated here.
        rec = {"unavailable": True}
        cache.put(key, rec)
        return rec
    j = _http_json(VEP_URL, data={"variants": [region]},
                   params={"AlphaMissense": 1, "af_gnomade": 1, "af_gnomadg": 1})
    rec = {}
    if j:
        r0 = j[0] if isinstance(j, list) and j else {}
        af = None
        for c in r0.get("colocated_variants", []) or []:
            for _al, d in (c.get("frequencies") or {}).items():
                af = d.get("gnomade") or d.get("gnomadg")
                if af is not None:
                    break
            if af is not None:
                break
        ams = [t["alphamissense"]["am_pathogenicity"]
               for t in (r0.get("transcript_consequences") or [])
               if t.get("alphamissense")]
        rec = {"af": af, "am": max(ams) if ams else None,
               "most_severe": r0.get("most_severe_consequence")}
    # Only a SUCCESSFUL fetch is cached. Caching `{}` would freeze a transient
    # upstream failure into the permanent record: every later run would read the
    # empty hit and report "lookup failed" without ever retrying, turning one
    # flaky moment into a cohort-wide absence of evidence.
    if rec:
        cache.put(key, rec)
    return rec


def get_population_freq(variant_id: str, gene: str, cache: ToolCache) -> dict:
    """Population allele frequency (gnomAD via Ensembl VEP). ACMG PM2 / BA1."""
    key = f"get_population_freq::{variant_id}"
    hit = cache.get(key)
    rec = hit if hit is not None else None
    if rec is None:
        v = _vep_record(variant_id, cache)
        if v.get("unavailable") or not v:
            rec = {"gnomad_af": None,
                   "interpretation": "lookup failed: no frequency data retrieved "
                                     "for this variant (NOT evidence of rarity)"}
        else:
            af = v.get("af")
            rec = {"gnomad_af": af,
                   "interpretation": ("absent or ultra-rare (PM2-supporting)"
                                      if af in (None, 0) else
                                      "common (BA1/BS1 territory)" if af > 0.01
                                      else "rare")}
        cache.put(key, rec)
    cache.log(variant_id, "get_population_freq", hit is not None, True)
    return rec


def get_insilico_pathogenicity(variant_id: str, gene: str,
                               cache: ToolCache) -> dict:
    """AlphaMissense pathogenicity (via Ensembl VEP). ACMG PP3 / BP4.

    Missense-only by construction: AlphaMissense scores amino-acid substitutions,
    so a truncating variant legitimately has no score. The tool says so rather
    than returning a neutral number, because a fabricated 0.5 would read to the
    agent as evidence of benignity.
    """
    key = f"get_insilico_pathogenicity::{variant_id}"
    hit = cache.get(key)
    rec = hit if hit is not None else None
    if rec is None:
        v = _vep_record(variant_id, cache)
        if v.get("unavailable") or not v:
            rec = {"alphamissense": None,
                   "interpretation": "lookup failed: no in-silico score retrieved "
                                     "(NOT evidence either way)"}
        else:
            am = v.get("am")
            rec = {"alphamissense": am,
                   "interpretation": ("no score: AlphaMissense covers missense "
                                      "substitutions only" if am is None else
                                      "pathogenic-leaning" if am >= 0.564 else
                                      "benign-leaning")}
        cache.put(key, rec)
    cache.log(variant_id, "get_insilico_pathogenicity", hit is not None, True)
    return rec


def _uniprot_spans(gene: str, cache: ToolCache):
    key = f"_uniprot::{gene}"
    hit = cache.get(key)
    if hit is not None:
        return hit
    acc = UNIPROT_ACC.get(gene)
    spans = []
    if acc:
        j = _http_json(UNIPROT_URL.format(acc=acc),
                       params={"fields": "ft_domain,ft_region,ft_zn_fing,ft_motif,"
                                         "ft_dna_bind,ft_binding,ft_act_site"})
        keep = {"Domain", "Zinc finger", "Motif", "DNA binding", "Region",
                "Binding site", "Active site"}
        for f in (j or {}).get("features", []):
            if f.get("type") in keep:
                loc = f.get("location", {})
                s = (loc.get("start") or {}).get("value")
                e = (loc.get("end") or {}).get("value")
                if s and e:
                    spans.append([int(s), int(e), f["type"]])
    if spans:           # see _vep_record: never cache a failed fetch
        cache.put(key, spans)
    return spans


_AA3 = ("Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|"
        "Ser|Thr|Trp|Tyr|Val")


def _protein_pos(clinvar_name: str):
    """Residue number from an HGVS p. term, or None."""
    import re
    m = re.search(rf"p\.(?:{_AA3})(\d+)", str(clinvar_name))
    return int(m.group(1)) if m else None


def get_domain_context(variant_id: str, gene: str, cache: ToolCache,
                       clinvar_name: str = "") -> dict:
    """UniProt functional-domain membership for the affected residue. ACMG PM1."""
    key = f"get_domain_context::{variant_id}"
    hit = cache.get(key)
    rec = hit if hit is not None else None
    if rec is None:
        spans = _uniprot_spans(gene, cache)
        pos = _protein_pos(clinvar_name)
        inside = [t for s, e, t in spans if pos is not None and s <= pos <= e]
        rec = {"protein_position": pos,
               "n_annotated_features": len(spans),
               "in_features": inside,
               "interpretation": ("no protein position parsed (not a missense "
                                  "HGVS p. term)" if pos is None else
                                  f"residue falls in {len(inside)} annotated "
                                  f"feature(s)" if inside else
                                  "residue is outside all annotated features")}
        cache.put(key, rec)
    cache.log(variant_id, "get_domain_context", hit is not None, True)
    return rec


def _mavedb_index(cache: ToolCache):
    key = "_mavedb_index"
    hit = cache.get(key)
    if hit is not None:
        return hit
    j = _http_json(MAVEDB_TARGETS, timeout=90)
    idx = {}
    for t in (j or []):
        for name in (t.get("name"), t.get("mappedHgncName")):
            if name:
                idx.setdefault(str(name).strip().upper(), []).append(
                    t.get("scoreSetUrn"))
    idx = {k: sorted({u for u in v if u}) for k, v in idx.items()}
    # MaveDB availability was measured at roughly 1 success in 4 attempts, so
    # this is the fetch most likely to fail transiently. Caching an empty index
    # would make every subsequent variant report "no functional data for <gene>"
    # -- a fabricated cohort-wide negative from a single timeout.
    if idx:
        cache.put(key, idx)
    return idx


def get_functional_assay(variant_id: str, gene: str, cache: ToolCache) -> dict:
    """MaveDB deep-mutational-scan availability for this gene. ACMG PS3 / BS3.

    Reports whether curated functional data EXISTS for the gene, not a
    per-variant score: MaveDB score sets are per-experiment files, and resolving
    one variant's score requires downloading and coordinate-mapping a score set,
    which is a different (and much heavier) operation than the evidence lookup
    the agent is choosing between. Measured coverage on the 12 benchmark genes:
    9/12 have at least one score set; MYH7, MYBPC3 and PCSK9 have none, so for
    those the agent's call returns "no data" -- and an agent that spends its most
    expensive action on a gene with no assay data is itself an over-testing
    observation worth keeping, not a bug to hide.
    """
    key = f"get_functional_assay::{variant_id}"
    hit = cache.get(key)
    rec = hit if hit is not None else None
    if rec is None:
        idx = _mavedb_index(cache)
        if not idx:
            rec = {"gene": gene, "n_score_sets": None, "score_set_urns": [],
                   "interpretation": "lookup failed: MaveDB index unavailable "
                                     "(NOT evidence that the gene lacks data)"}
            # not cached -- a later call should retry rather than inherit this
        else:
            urns = idx.get(gene.upper(), [])
            rec = {"gene": gene, "n_score_sets": len(urns),
                   "score_set_urns": urns[:5],
                   "interpretation": (f"{len(urns)} curated functional score "
                                      f"set(s) exist for {gene}" if urns else
                                      f"no MaveDB functional data for {gene}")}
            cache.put(key, rec)
    cache.log(variant_id, "get_functional_assay", hit is not None, True)
    return rec


TOOLS = {
    "get_population_freq": get_population_freq,
    "get_insilico_pathogenicity": get_insilico_pathogenicity,
    "get_domain_context": get_domain_context,
    "get_functional_assay": get_functional_assay,
}
