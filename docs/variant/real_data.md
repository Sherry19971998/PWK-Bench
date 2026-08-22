# Reproducing PWK-Bench on real data

The bundled cohorts are for offline demo only. To reproduce the paper's numbers,
build the real cohort from public sources below, then run real models. Every
source is public; the 12 genes, ≥2-star filter, and 4 channels are locked in
`pwkbench/spec.py` and must not change.

## 1. Labels + PVS1 — ClinVar (≥2-star)

Source: ClinVar `variant_summary.txt.gz` (NCBI FTP, ~440 MB):
`https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz`

```bash
curl -O https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
python scripts/variant/fetch_clinvar.py --summary variant_summary.txt.gz \
    --out data/real/cohort_clinvar_real.parquet
```

Filter (matches the paper): GRCh38; review status ≥2 stars (`criteria provided,
multiple submitters, no conflicts` / `reviewed by expert panel` / `practice
guideline`); P/LP = positive, B/LB = negative; **VUS and conflicting excluded**.
PVS1 (LoF) is derived from the ClinVar variant name/consequence.

> **Note on cohort size.** The paper and `pwkbench/spec.py` target 491
> variants (229 P/LP, 262 B/LB), matching the bundled cohort. PCSK9 has only
> ~5 ≥2-star P/LP variants in ClinVar, so its per-gene quota (aimed at ~41,
> `_per_gene_counts(spec.N_VARIANTS_TOTAL, ...)`) can't be filled and it caps
> at 27. This is a real data limit, not a bug: an earlier ClinVar snapshot
> gave 506 total before this shortfall was discovered, and your own pull's
> count will depend on the release date -- if you get a different total,
> update `spec.py` and the paper together (see its header note). Full
> per-gene breakdown: `results/variant/real_trusted/per_gene_cohort_counts.csv`.

> **Verified end-to-end.** This exact command was run against the live
> ~440 MB `variant_summary.txt.gz` (July 2026 release): it produced 491
> variants (229 P/LP, 262 B/LB), the paper's gene quotas (43/43/42.../27 PCSK9),
> PVS1 defined on all 491 with a clean LoF signal (single-slot per-gene
> AUC ≈ 0.81; 66% of P/LP are LoF, 0% of B/LB), and PM2/PM1/PP3 correctly left
> `__defined=False`. The bundled `data/sample/cohort_clinvar_real.parquet` is
> that fresh output. Because ClinVar changes between releases, the specific
> variant set (and counts like the 179 LoF / 129 missense used in the annotated
> example further down) will differ on your pull — the pipeline is
> reproducible, the exact rows are snapshot-dependent.

## 2. PM2 / PM1 / PP3 — annotation channels

> **Reliable one-endpoint path (used to build the bundled `cohort_full_real.parquet`).**
> The gnomAD GraphQL API is frequently down (502s). Ensembl VEP REST returns
> BOTH the gnomAD exome allele frequency (`af_gnomade=1`) AND AlphaMissense
> (`AlphaMissense=1`) in a single POST to `/vep/human/region`, batched ~200
> variants per call. This is how the bundled full cohort was annotated (all 490
> mappable variants in ~24 s, 0 errors): PM2 = −log10(gnomAD AF) with variants
> absent from gnomAD → 8.0 (maximally rare); PP3 = max AlphaMissense over
> transcripts (missense only); PM1 = UniProt functional-domain membership. The
> per-channel sources below remain valid; VEP just consolidates PM2+PP3.

- **PM2 (rarity):** gnomAD v4 allele frequency, `PM2 = -log10(AF)`.
  gnomAD GraphQL API (`https://gnomad.broadinstitute.org/api`); variants absent
  from gnomAD are treated as maximally rare.
- **PM1 (functional domain):** variant-level UniProt functional-domain
  membership (the ACMG-faithful definition; see the PM1 subsection below).
  The gene-level `mis_z` from the gnomAD gene constraint table is retained only
  as a FALLBACK when the UniProt-domain map is absent.
- **PP3/BP4 (in silico):** AlphaMissense via Ensembl VEP REST
  (`https://rest.ensembl.org`, AlphaMissense plugin, GRCh38, max over
  transcripts) **or** the precomputed `AlphaMissense_hg38.tsv.gz` (Zenodo).
  Missense only (paper: 95/506); non-missense → neutral 0.5.

```bash
# constraint table + AlphaMissense table downloaded separately (see their sites)
python scripts/variant/fetch_annotations.py \
    --cohort data/real/cohort_clinvar_real.parquet \
    --constraint gnomad_gene_constraint.tsv \
    --alphamissense AlphaMissense_hg38.tsv.gz \
    --query-gnomad \
    --out data/real/cohort_full.parquet
```

> **Frozen-vs-live validity check (verified 2026-08-12).** The frozen-pool
> paradigm freezes evidence once and reuses it for every strategy and agent
> (see the Acquisition Protocol section of the paper), which is only a
> faithful stand-in for a live query if the frozen values still match one.
> `scripts/variant/verify_alphamissense_refetch.py` re-fetches PP3
> (AlphaMissense) for a random sample of the bundled cohort's PP3-defined
> variants from live Ensembl VEP, using the same endpoint/params as
> `fetch_annotations_vep.py`. On a 15-variant sample (seed=20260101, the
> paper's quoted n): **15/15 matched, Pearson r = Spearman ρ = 1.000000, max
> abs diff = 0.000000** — artifact:
> `results/variant/real_trusted/alphamissense_refetch_validation.csv`. This is
> expected (AlphaMissense is a static, versioned score table, not a
> live-updated prediction) but had no checked-in reproduction before this
> script. It does **not** extend to PM2 (gnomAD allele counts do change
> between releases, so a PM2 re-fetch is out of scope here — see the "Note on
> cohort size" above, which already documents PCSK9 shifting between
> snapshots).

## 3. Run with real models

```bash
pip install -e ".[anthropic,openai,hf]"
export ANTHROPIC_API_KEY=...        # and/or OPENAI_API_KEY
# set exact model versions in configs/models.yaml -> real:
python scripts/variant/run_benchmark.py --cohort data/real/cohort_full.parquet --models real
```

## Definedness diagnosis (order-optimality on real data)

A concern about axis B (order-optimality): how much of the `Oracle − RelMax`
gap is genuine per-instance planning, versus just knowing *which channels are
even defined* for a variant (schema knowledge an agent reads free from the HGVS
name — missense variants have PP3/PM1, non-missense do not)?

We answered it on the real 491-variant cohort with real annotations
(gnomAD v4 allele frequency → PM2, Ensembl-VEP AlphaMissense → PP3, variant-level
UniProt functional-domain membership → PM1). Two schema-only references (no labels, no per-instance
value reasoning) bracket the gap: `DefinednessBaseline` (defined channels
first) and `DefinednessStratifiedBest` (best fixed order *within each
definedness group*).

> **These numbers are from one specific ClinVar snapshot + a full annotation
> run — they are NOT reproducible from the bundled base parquet alone.** The
> bundled `data/sample/cohort_clinvar_real.parquet` is the *base* cohort
> (`fetch_clinvar.py` only): it has real labels + PVS1, but PM2/PM1/PP3 are
> `__defined=False` placeholders (so `load_real_cohort` warns they are
> zero-information). The table below requires additionally running
> `fetch_annotations.py --query-gnomad --alphamissense ... --uniprot-domains ...`
> to fill PM2/PP3/PM1. ClinVar is also a moving target: regenerating the base
> cohort from a later release changes the variant set and the per-channel
> `defined_n` counts (e.g. a July-2026 pull yields PVS1=1 on 151 and 80 missense,
> vs the 179 / 129 the annotated run below was built on). Treat the specific
> figures as a worked example of the *diagnosis*, not fixed constants.

Real-cohort result — regenerated end-to-end (July 2026 ClinVar + live
Ensembl-VEP annotation; n=491; PVS1 defined 491, PM2 490, PP3 80, PM1 80):

| reference | PE |
|---|---|
| RelMax (label-free) | 0.850 |
| DefinednessBaseline | 0.850 |
| DefinednessStratifiedBest (schema-optimal) | 0.833 |
| Oracle (per-instance) | 0.919 |

- `Oracle − RelMax` total gap = **0.070**
- explained by definedness/schema (`DefStratBest − RelMax`) = **−0.016** (schema
  ordering does not help on this cohort)
- residual genuine per-instance planning (`Oracle − DefStratBest`) = **0.086**

On this fully-annotated real cohort the gap is essentially **all genuine
per-instance planning** — the definedness/schema component is ~0 (slightly
negative), because with the current annotation the defined-channel pattern no
longer front-loads the useful evidence. This is the opposite mix from an earlier
sparser-annotation snapshot (which was ~46% schema), and it is exactly why the
diagnosis must be **recomputed per cohort** rather than quoted as a constant —
`definedness_diagnosis.csv` reports it for whatever annotation you build.

Real single-slot per-gene AUCs (which channels carry signal):

| channel | single-slot AUC | defined_n |
|---|---|---|
| PVS1 | 0.807 | 491 |
| PM2  | 0.819 | 490 |
| PP3  | 0.535 | 80 |
| PM1  | 0.456 | 80 |

PVS1 (LoF) and PM2 (gnomAD rarity) carry strong signal; PP3 (AlphaMissense) and
PM1 (UniProt-domain membership) are weak on these 12 densely-annotated genes
whose ≥2-star missense set is pathogenic-enriched (a cohort-composition limit,
not an annotation-source defect — see the PM1 subsection). The `0.481` PM1 in
earlier drafts was the superseded *gene-level* `mis_z`; the current variant-level
UniProt-domain PM1 is `0.456` here.

### PM1 is now variant-level (UniProt functional-domain membership)

PM1 was originally a *gene-level* `mis_z` (one value per gene), so within a gene
it was constant and contributed ≈0 to the per-gene-stratified AUC — a dead
channel. It is now the **ACMG-faithful, variant-level** definition: the
variant's amino-acid position is tested against UniProt functional features for
its gene (`fetch_uniprot_domains()` → `data/sample/uniprot_domains.json`), with
strength 1.0 for a precise site (Active/Binding/Zinc-finger/DNA-binding/Motif),
0.6 for a broad Domain/Region, 0.2 outside any annotated functional region.

Reproduce this PM1 exactly (single-slot AUC **0.432** on the 491-variant real
cohort): build the cohort with `fetch_clinvar.py` (which writes the
`clinvar_name` column the AA-position parser needs), then

```bash
python -c "from scripts.fetch_annotations import fetch_uniprot_domains; fetch_uniprot_domains('data/sample/uniprot_domains.json')"
python scripts/variant/fetch_annotations.py --cohort data/real/cohort_clinvar_real.parquet \
    --uniprot-domains data/sample/uniprot_domains.json --query-gnomad \
    --alphamissense AlphaMissense_hg38.tsv.gz --out data/real/cohort_full.parquet
```

The `--uniprot-domains` path parses each variant's amino-acid position from
`clinvar_name` and prints `[PM1] variant-level domain membership: defined on
N/M missense variants`; it raises rather than silently shipping an empty PM1 if
the protein-change column is missing.

Honest caveat: on this cohort the discrimination is still weak. Real single-slot
AUC ≈ **0.43**, because these 12 genes are densely annotated (most missense fall
in *some* domain) and the ≥2-star missense set is ~89% pathogenic, so
domain membership barely separates classes (P(in-domain | path) ≈ 0.58 vs
P(in-domain | benign) ≈ 0.54). The change fixes the *structural* defect (PM1 is
no longer a gene-level constant and no longer contributes exactly zero to the
stratified AUC) but does not make PM1 a strong predictor on this cohort — a
limitation of the cohort composition, not the annotation source. The bundled
synthetic cohort mirrors this: PM1 is drawn as a weakly label-tied
domain-membership strength (0.2/0.6/1.0), not the old strong gene-level signal,
so demo channel importances match real behaviour. The definedness diagnosis is
essentially unchanged under this PM1 (schema ≈47% / genuine planning ≈53%).

Reproduce: build the real annotated cohort (steps 1–3 above), then
`run_benchmark.py --cohort data/real/cohort_full.parquet --models real --robustness`
writes `definedness_diagnosis.csv` alongside the other robustness CSVs.

## Honesty checklist

- Cohorts are labeled by provenance: `source ∈ {SYNTHETIC, CLINVAR_REAL}`.
- Demo numbers are illustrative and must never be reported as the paper's.
- `spec.py` is the single source of truth for genes/cohort/channels — the whole
  pipeline reads from it, so a real run cannot silently drift from the paper.
