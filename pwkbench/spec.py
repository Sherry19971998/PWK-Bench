"""
Locked benchmark specification — transcribed VERBATIM from the paper
(methods_experiments.tex). Nothing here may change without changing the paper.
Synthetic sample data is generated to match this.

Every constant below is sourced from the Benchmark Construction and
Acquisition Protocol sections of the PWK-Bench manuscript.

NOTE (2026-08-12): these constants track the CURRENT bundled real cohort
(data/sample/cohort_full_real.parquet, 491 variants). An earlier ClinVar
snapshot produced 506/244; the realized cohort size is snapshot-dependent
because PCSK9 has few >=2-star P/LP variants and per-gene quotas below are
always capped at what ClinVar actually has for that gene (see
`_per_gene_counts` in domains/variant.py and docs/variant/real_data.md,
"Note on cohort size"). If you re-fetch from a newer ClinVar release and get
a different total, update these constants and the paper together.
"""

# ---- Cohort (paper: 491 variants, 229 P/LP, 262 B/LB, per-gene balanced) ----
N_VARIANTS_TOTAL = 491
N_POSITIVE = 229   # P/LP
N_NEGATIVE = 262   # B/LB

# 12 genes across 4 domains, in paper order, with framework citation.
GENES_BY_DOMAIN = {
    "hereditary_cancer": ["BRCA1", "BRCA2", "TP53", "PTEN", "MLH1", "MSH2"],  # Strande 2017
    "cardiomyopathy":    ["MYH7", "MYBPC3"],                                   # Ommen 2020
    "arrhythmia":        ["KCNQ1", "SCN5A"],                                   # Priori 2013
    "lipid_disorder":    ["LDLR", "PCSK9"],                                    # Gidding 2015
}
GENES = [g for gs in GENES_BY_DOMAIN.values() for g in gs]
if len(GENES) != 12:  # raise (not assert) so it survives `python -O`
    raise ValueError(f"expected 12 genes, got {len(GENES)}: {GENES}")
GENE_TO_DOMAIN = {g: d for d, gs in GENES_BY_DOMAIN.items() for g in gs}

# ---- Gold label (paper: ClinVar germline_classification, >=2-star) ----
# Positives: Pathogenic / Likely pathogenic ; Negatives: Benign / Likely benign
# EXCLUDED: VUS and conflicting interpretations.
POSITIVE_LABELS = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
NEGATIVE_LABELS = {"Benign", "Likely benign", "Benign/Likely benign"}
# >=2-star ClinVar review statuses:
REVIEW_STATUS_2STAR_PLUS = {
    "criteria provided, multiple submitters, no conflicts",
    "reviewed by expert panel",
    "practice guideline",
}

# ---- Evidence channels (paper: 4 ACMG categories) ----
# CHANNELS order = the fixed global RELEVANCE order used by RelMax (lower ref).
CHANNELS = ["PVS1", "PM2", "PP3", "PM1"]
CHANNEL_META = {
    "PM2":  {"name": "rarity",     "source": "-log10 gnomAD v4 allele frequency",
             "defined_on": "all",      "label_derived": False},
    "PVS1": {"name": "LoF",        "source": "binary LoF from ClinVar molecular_consequence",
             "defined_on": "all",      "label_derived": False},
    "PM1":  {"name": "domain", "source": "variant-level UniProt functional-domain "
             "membership (ACMG PM1: mutational hot spot / critical functional "
             "domain); strength 1.0 precise site / 0.6 broad domain / 0.2 outside",
             "defined_on": "missense", "label_derived": False},
    "PP3":  {"name": "in_silico",  "source": "AlphaMissense via Ensembl VEP (GRCh38, max over transcripts)",
             "defined_on": "missense", "label_derived": False},
}
NEUTRAL_VALUE = 0.5          # non-missense PP3 receives this (paper)
N_MISSENSE_WITH_PP3 = 80     # paper: PP3 defined on 80/491

# ---- Protocol / strategies (paper) ----
STRATEGIES = ["RelMax", "Oracle", "Heuristic", "Random", "Agent"]
LOWER_REFERENCE = "RelMax"   # fixed global relevance ordering
UPPER_REFERENCE = "Oracle"   # global best-subset reference pi*(k) (metrics.py
                              # ::_best_subset_auc); NOT a per-variant policy --
                              # see metrics.py::oracle_order for the separate
                              # per-variant diagnostic ordering (Metric B).

# ---- Metrics ----
BUDGETS = [1, 2, 3, 4]       # k = 1..K, K = number of channels
TAU = 0.85                   # budget-to-target threshold for k*
BOOTSTRAP_B = 2000           # gene-clustered bootstrap
GLOBAL_SEED = 20260101
