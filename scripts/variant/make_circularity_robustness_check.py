#!/usr/bin/env python3
"""Does the RelMax-vs-Oracle planning gap survive on the non-circular subset?

WHY THIS EXISTS
----------------
`probe_clinvar_circularity.py` found that a large share of this cohort's
ClinVar labels overlap with the same evidence (PM2/gnomAD, PP3-type in-silico
tools) this benchmark treats as acquirable -- see that script's docstring for
the numbers. The natural next question: is the headline "RelMax already ties
the per-instance Oracle, no real planning gap" finding (`results.csv`,
`domain_gap_excess.csv`) an artifact of that overlap?

This re-runs A(k) / the oracle-vs-RelMax gap restricted to variants where NO
ClinVar submitter's comment showed the circularity signal, under three
increasingly strict definitions of "clean", and compares against the full
cohort.

RESULT (verified 2026-08-07)
-------------------------------
    subset                                    n    PE(RelMax)  PE(Oracle)  gap@k=1
    full cohort (baseline)                    491  0.8496      0.8926      0.0115
    no PM2/PP3-type language cited at all      105  0.9322      0.9645      0.0000
    no literal "gnomAD" named                  299  0.9008      0.9431      0.0102
    no literal gnomAD/REVEL/other in-silico    295  0.9169      0.9481      0.0000

The gap does NOT widen on the clean subsets -- if anything it tightens to
exactly 0 in two of the three. This argues AGAINST circular labeling being
the explanation for "no planning gap": removing the variants most likely to
carry circular signal does not resurrect a gap that circularity could have
been masking.

GENE-CLUSTERED BOOTSTRAP CI (added 2026-08-07, B=2000, same method as
`domain_stratified_gap_excess`): every subset's gap@k=1 is statistically
indistinguishable from zero.

    subset                                    95% CI
    full cohort (baseline)                    [-0.069, +0.097]
    no PM2/PP3-type language cited at all      [ 0.000,  0.000]
    no literal gnomAD named                    [-0.121, +0.152]
    no literal gnomAD/REVEL/other in-silico    [ 0.000,  0.000]

The two exactly-[0,0] intervals are STRUCTURAL, not a sampling artifact: on
those subsets the oracle's best single channel at k=1 IS PVS1 -- the same
channel RelMax already leads with -- so the oracle and RelMax scores are
IDENTICAL row-for-row and every bootstrap resample reproduces gap=0 exactly.
Consistent with the same zero-variance pattern documented elsewhere in this
repo for the no-context protocol.

WHAT THIS DOES NOT SHOW. The clean subsets' much higher absolute PE (0.92-0.97
vs 0.85) must NOT be read as "how much circularity inflated the full cohort's
AUC" -- these subsets are not a random control. Variants whose ClinVar
submitters felt no need to argue from population frequency or in-silico
evidence plausibly skew toward the EASIER, already-obvious cases (e.g. clear
truncating LoF calls that need no supporting evidence at all), which would
inflate AUC for reasons unrelated to label circularity. Case difficulty and
circularity-signal presence are confounded in this split; this script cannot
separate them, and no claim here should imply that it does.

USAGE
    python scripts/variant/make_circularity_robustness_check.py
    (seconds -- pure numpy/sklearn over the cached probe output, no network)
"""
from __future__ import annotations
import os, sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench.domains.base import load_real_cohort, Cohort   # noqa: E402
from pwkbench.domains.variant import VARIANT_DOMAIN          # noqa: E402
from pwkbench.strategies import relmax_order                 # noqa: E402
import pwkbench.metrics as M                                 # noqa: E402

PROBE = "results/variant/real_trusted/clinvar_circularity_probe.csv"
OUT = "results/variant/real_trusted/circularity_robustness.csv"
MIN_N = 20
BOOT_B = 2000


def _gap_bootstrap_ci(view: Cohort, B: int = BOOT_B, seed: int = 0):
    """Gene-clustered bootstrap CI for the k=1 oracle-relmax gap. Same
    resampling method as `pwkbench.metrics.domain_stratified_gap_excess`, but
    the label -- clean vs. circular subset -- enters through row FILTERING
    upstream, not through a shuffled-label null, so this is a plain CI on the
    gap, not a permutation test against one."""
    rng = np.random.default_rng(seed)
    genes = np.unique(view.genes)
    rel = relmax_order(view)
    sc_rel_full = M._score_from_evidence(view, M._acquired_mask(rel, 1))
    _, subset = M._best_subset_auc(view, 1)   # oracle's k=1 winning channel, chosen once
    mask = np.zeros((len(view), view.domain.K), dtype=bool)
    mask[:, list(subset)] = True
    sc_ora_full = M._score_from_evidence(view, mask)
    gene_rows = [np.where(view.genes == g)[0] for g in genes]
    vals = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, len(genes), size=len(genes))
        rows = np.concatenate([gene_rows[g] for g in pick])
        cids = np.concatenate([np.full(len(gene_rows[g]), rep * len(genes) + g)
                               for rep, g in enumerate(pick)])
        a_rel = M._macro_auc_by_cluster(view.y[rows], sc_rel_full[rows], cids)
        a_ora = M._macro_auc_by_cluster(view.y[rows], sc_ora_full[rows], cids)
        vals[b] = a_ora - a_rel
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _run(view: Cohort, label: str) -> dict | None:
    n = len(view)
    if n < MIN_N or view.y.sum() == 0 or view.y.sum() == n:
        print(f"{label}: n={n} -- too small / single-class, skipped")
        return None
    rel = relmax_order(view)
    A_rel = M.curve_A(view, rel)
    A_ora = M.curve_A(view, rel, leq_k=True)
    pe_rel, pe_ora = M.planning_efficiency(A_rel), M.planning_efficiency(A_ora)
    ci_lo, ci_hi = _gap_bootstrap_ci(view)
    row = dict(subset=label, n=n, n_genes=len(np.unique(view.genes)),
              pe_relmax=pe_rel, pe_oracle=pe_ora,
              gap_k1=A_ora[1] - A_rel[1], gap_pe=pe_ora - pe_rel,
              gap_k1_ci_lo=ci_lo, gap_k1_ci_hi=ci_hi,
              gap_k1_includes_zero=bool(ci_lo <= 0 <= ci_hi),
              **{f"A_relmax_k{k}": A_rel[k] for k in A_rel},
              **{f"A_oracle_k{k}": A_ora[k] for k in A_ora})
    print(f"{label}: n={n:>4}  PE_relmax={pe_rel:.4f}  PE_oracle={pe_ora:.4f}  "
         f"gap@k1={row['gap_k1']:.4f}  95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    return row


def main():
    os.chdir(_ROOT)
    coh = load_real_cohort(
        pd.read_parquet("data/sample/cohort_full_real.parquet"), VARIANT_DOMAIN)
    probe = pd.read_csv(PROBE)
    df = coh.df.merge(probe, on="variant_id", how="left")
    assert df["pm2_cited"].notna().all(), "probe coverage gap -- rerun probe_clinvar_circularity.py"

    rows = []
    rows.append(_run(Cohort(df.copy(), coh.domain), "full cohort (baseline)"))

    clean_cat = df[~(df.pm2_cited | df.pp3_cited)].reset_index(drop=True)
    rows.append(_run(Cohort(clean_cat, coh.domain),
                     "no PM2/PP3-type language cited at all"))

    clean_gnomad = df[~df.gnomad_named.fillna(False)].reset_index(drop=True)
    rows.append(_run(Cohort(clean_gnomad, coh.domain), "no literal gnomAD named"))

    clean_lit = df[~df.gnomad_named.fillna(False) & ~df.revel_named.fillna(False)
                   & ~df.other_insilico_named.fillna(False)].reset_index(drop=True)
    rows.append(_run(Cohort(clean_lit, coh.domain),
                     "no literal gnomAD/REVEL/other in-silico"))

    out = pd.DataFrame([r for r in rows if r is not None])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwritten: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
