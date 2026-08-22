"""
Variant-interpretation domain — the paper's instantiation.

Provides:
  - VARIANT_DOMAIN : the Domain object (4 ACMG channels).
  - make_synthetic_cohort(): a STRUCTURE-IDENTICAL synthetic cohort matching
    spec (12 genes, 491 variants, 229 P/LP + 262 B/LB, per-gene balance,
    PP3 defined only on 80 missense). The generative model is documented below.

    >>> SYNTHETIC DATA WARNING <<<
    The synthetic cohort exists ONLY so the pipeline runs offline with no
    downloads/API keys. Numbers computed on it are ILLUSTRATIVE and are NOT the
    paper's results. To reproduce the paper, build a real cohort with
    scripts/fetch_*.py (see docs/variant/real_data.md) — same schema, real values.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..spec import (GENES, GENE_TO_DOMAIN, CHANNELS, CHANNEL_META, NEUTRAL_VALUE,
                    N_VARIANTS_TOTAL, N_POSITIVE, N_MISSENSE_WITH_PP3, GLOBAL_SEED)
from .base import Domain, Cohort

VARIANT_DOMAIN = Domain(
    name="variant_interpretation",
    channels=list(CHANNELS),
    channel_defined_on={c: CHANNEL_META[c]["defined_on"] for c in CHANNELS},
    neutral_value=NEUTRAL_VALUE,
)


def _per_gene_counts(n_total: int, n_pos: int, genes: list[str]) -> dict:
    """Split n_total across 12 genes, per-gene class-balanced (paper)."""
    g = len(genes)
    base = n_total // g
    rem = n_total - base * g
    sizes = {gene: base + (1 if i < rem else 0) for i, gene in enumerate(genes)}
    # positives per gene ~ proportional, kept close to balanced
    pos = {}
    running = 0
    for i, gene in enumerate(genes):
        p = round(sizes[gene] * n_pos / n_total)
        p = min(p, sizes[gene])
        pos[gene] = p
        running += p
    # fix rounding so total positives == n_pos exactly
    diff = n_pos - running
    for gene in genes:
        if diff == 0:
            break
        step = 1 if diff > 0 else -1
        newp = pos[gene] + step
        if 0 <= newp <= sizes[gene]:
            pos[gene] = newp
            diff -= step
    return {gene: (sizes[gene], pos[gene]) for gene in genes}


def make_synthetic_cohort(seed: int = GLOBAL_SEED) -> Cohort:
    """
    Generative model (SYNTHETIC — not real biology):
      label y ~ per-gene balanced Bernoulli.
      PVS1 (LoF, all):        P(LoF=1) higher for positives.
      PM2 (rarity, all):      -log10 AF; positives rarer (higher).
      PM1 (domain, missense): variant-level functional-domain membership
          strength (0.2/0.6/1.0), only weakly label-tied (matches real data).
      PP3 (in silico, missense only, 80/491): AlphaMissense-like score,
          higher for pathogenic missense; non-missense -> neutral 0.5, undefined.
    Channel signal strengths are chosen so a single channel is insufficient and
    the global best-subset oracle beats a fixed relevance order — matching the
    paper's qualitative structure, NOT its exact numbers.
    """
    rng = np.random.default_rng(seed)
    counts = _per_gene_counts(N_VARIANTS_TOTAL, N_POSITIVE, GENES)

    # Choose EXACTLY N_MISSENSE_WITH_PP3 missense variants globally (paper: 80/491).
    missense_global = np.zeros(N_VARIANTS_TOTAL, dtype=bool)
    missense_global[:N_MISSENSE_WITH_PP3] = True
    rng.shuffle(missense_global)

    rows = []
    vid = 0
    for gene in GENES:
        size, npos = counts[gene]
        labels = np.array([1] * npos + [0] * (size - npos))
        rng.shuffle(labels)
        is_missense = missense_global[vid:vid + size]
        for j in range(size):
            y = int(labels[j])
            mis = bool(is_missense[j])
            # This synthetic cohort is near-separable by construction, so the
            # per-instance oracle reaches A(1) ~ 1.0 -- see README ("oracle=1.0
            # is the true attainable upper bound on this near-separable
            # synthetic data; the paper's ~0.90 needs real ClinVar/gnomAD/
            # AlphaMissense"). It is a ruler, not a claim about real data.
            # PVS1 LoF (binary): pathogenic more likely LoF, and LoF mostly non-missense
            p_lof = (0.55 if y else 0.08) * (0.2 if mis else 1.0)
            pvs1 = float(rng.random() < p_lof)
            # PM2 rarity: -log10 AF ; positives rarer
            pm2 = rng.normal(5.5 if y else 3.0, 1.2)
            pm2 = float(np.clip(pm2, 0, 8))
            # PM1 (missense only): VARIANT-LEVEL functional-domain membership
            # strength (0.2 outside / 0.6 broad domain / 1.0 precise site), to
            # match the real fetch_annotations semantics. On real data this
            # channel discriminates only weakly (well-studied genes are densely
            # annotated and their missense set is pathogenic-enriched), so the
            # synthetic draw is only mildly label-tied -- do NOT give it the
            # strong signal the old gene-level mis_z had, or the demo would
            # overstate PM1's real value.
            if mis:
                pm1 = float(rng.choice([0.2, 0.6, 1.0],
                                       p=([0.40, 0.42, 0.18] if y else [0.46, 0.42, 0.12])))
            else:
                pm1 = NEUTRAL_VALUE
            # PP3 AlphaMissense (missense only)
            pp3 = float(np.clip(rng.normal(0.82 if y else 0.28, 0.15), 0, 1)) if mis else NEUTRAL_VALUE
            rows.append(dict(
                variant_id=f"SYN_{gene}_{vid:04d}", gene=gene,
                domain=GENE_TO_DOMAIN[gene], label=y,
                PVS1=pvs1, PVS1__defined=True,
                PM2=pm2,   PM2__defined=True,
                PM1=pm1,   PM1__defined=mis,
                PP3=pp3,   PP3__defined=mis,
                is_missense=mis, source="SYNTHETIC",
            ))
            vid += 1
    df = pd.DataFrame(rows)
    # raise (not assert) so the invariants survive `python -O`
    if len(df) != N_VARIANTS_TOTAL:
        raise ValueError(f"expected {N_VARIANTS_TOTAL} variants, got {len(df)}")
    if df["label"].sum() != N_POSITIVE:
        raise ValueError(f"expected {N_POSITIVE} positives, got {df['label'].sum()}")
    # Use the SAME empirical per-channel neutral as real cohorts (not the scalar
    # 0.5, which is a strong offset on wide-range channels like PM2 in [0,8]).
    from .base import attach_empirical_neutrals
    return Cohort(df, attach_empirical_neutrals(df, VARIANT_DOMAIN))
