#!/usr/bin/env python3
"""Null-corrected planning gap by clinical domain.

WHY THIS EXISTS, NOT A PLOT OF results/variant/real_trusted/domain_gap.csv
----------------------------------------------------------------------------
That CSV (`M.domain_stratified_gap`) is RAW gap = oracle_A(k=1) − relmax_A(k=1)
per domain, and its own docstring warns the raw magnitude is confounded with
domain size: "a domain with fewer genes shows a larger raw gap for that reason
alone" (max-over-subsets is a selection statistic that inflates under the null
as n shrinks). On this cohort that is not hypothetical -- lipid_disorder
(n=69, smallest) has the largest raw gap and hereditary_cancer (n=254,
largest) has the smallest, exactly the pattern the artifact predicts. Plotting
those four numbers side by side would present a selection-bias fingerprint as
a scientific finding.

This figure plots `M.domain_stratified_gap_excess` instead: gap minus its own
within-gene-permutation null mean, i.e. the same null-correction
`consequence_stratified_gap` already applies to the missense/lof/other split,
now applied to the four ACMG domains. gap_excess and z ARE cross-domain
comparable; raw gap is not.

WHAT SURVIVES CORRECTION (n_perm=2000, seed=1, ~21 min -- rerun to confirm the
faster n_perm=200 default wasn't sitting near 0.05 on permutation-count noise;
it tightened, not reversed: p 0.010->0.0015, Holm 0.040->0.0060)
-----------------------------------------------------------------------
    domain              raw gap   gap_excess    z       p        holm p
    lipid_disorder       0.209      0.141      3.94    0.0015    0.0060  <- survives
    cardiomyopathy        0.086      0.035      1.18    0.116     0.349
    arrhythmia            0.089      0.034      0.91    0.173     0.349
    hereditary_cancer     0.003     -0.024     -1.19    0.956     0.956

Only lipid_disorder's excess is distinguishable from its own null, and it
survives Holm correction for testing all four domains. The other three
domains' raw gaps were, to varying degrees, exactly the size-driven artifact
the docstring warned about -- arrhythmia and cardiomyopathy's raw gaps (0.089,
0.086) looked similar to each other and to lipid_disorder's mainly because all
three domains are much smaller than hereditary_cancer, not because they share
a real effect.

WHAT THIS DOES NOT CLAIM
--------------------------
No mechanism. lipid_disorder has 2 genes (LDLR, PCSK9); this could be a
domain effect, a gene effect, or a property of which 69 real ClinVar variants
happen to fall there -- the same caveat as every other domain-stratified
number in this repo (the paper's own methodology states that the per-domain
gap breakdown is a descriptive, not inferential, quantity by construction --
gap_excess/z is the one exception, being explicitly the null-corrected,
comparable quantity).

COST. n_perm=200 permutations x 4 domains, each re-enumerating the
best-subset oracle at every budget k=1..4 -- no LLM calls, pure numpy/sklearn
over the cohort, but ~2-3 minutes wall clock. Not run inside run_benchmark.py
for that reason; this script computes and caches its own CSV.

USAGE
    python scripts/variant/make_domain_gap_figure.py
"""
import argparse, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench import figstyle as FS                          # noqa: E402
from pwkbench.domains.base import load_real_cohort           # noqa: E402
from pwkbench.domains.variant import VARIANT_DOMAIN          # noqa: E402
from pwkbench.spec import GENES_BY_DOMAIN                    # noqa: E402
import pwkbench.metrics as M                                 # noqa: E402

CSV_OUT = "results/variant/real_trusted/domain_gap_excess.csv"
SIG_ALPHA = 0.05
# Default n_perm=200 (~2-3 min) is enough to rank domains during iteration; the
# CSV/figure actually checked into the paper were generated with --n-perm 2000
# --seed 1 (~21 min), which TIGHTENED lipid_disorder's result rather than
# reversing it (p 0.010->0.0015, Holm 0.040->0.0060) -- confirming it wasn't
# permutation-count noise sitting near the 0.05 line.


def _compute(n_perm=200, seed=0):
    coh = load_real_cohort(
        pd.read_parquet("data/sample/cohort_full_real.parquet"), VARIANT_DOMAIN)
    out = M.domain_stratified_gap_excess(coh, n_perm=n_perm, seed=seed)
    rows = [dict(domain=d, n_perm=n_perm,
                **{k: v for k, v in row.items() if k != "per_k_gap"})
            for d, row in out.items()]
    df = pd.DataFrame(rows).sort_values("gap_excess", ascending=False)
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/variant/domain_gap_excess.png")
    ap.add_argument("--use-cache", action="store_true",
                    help=f"read {CSV_OUT} instead of recomputing (~2-3 min)")
    ap.add_argument("--n-perm", type=int, default=200,
                    help="permutations per domain (paper's checked-in CSV used 2000, ~21 min)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.chdir(_ROOT)
    FS.use()

    if args.use_cache and os.path.exists(CSV_OUT):
        df = pd.read_csv(CSV_OUT).sort_values("gap_excess", ascending=False)
    else:
        df = _compute(n_perm=args.n_perm, seed=args.seed)

    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    xs = range(len(df))
    sig = df["p_value"] < SIG_ALPHA
    colors = [FS.ACCENT if s else FS.STEEL for s in sig]

    ax.bar(xs, df["gap_excess"], width=.56, color=colors, edgecolor="white",
          linewidth=1.0, zorder=3)
    # null_mean shown as a tick per bar: the raw gap sits at null_mean +
    # gap_excess, so this is what "corrected away" looks like, drawn rather
    # than only stated.
    for xi, (_, row) in zip(xs, df.iterrows()):
        ax.plot([xi - .28, xi + .28], [row["gap"], row["gap"]],
               color=FS.INK, lw=1.3, zorder=4)
        if xi == 0:
            ax.text(xi + .32, row["gap"], "raw gap", fontsize=8.6,
                   color=FS.INK, va="center")
    ax.axhline(0, color=FS.MUTE, lw=1.0, zorder=2)

    top = df["gap"].max() + .05
    ax.set_ylim(df["gap_excess"].min() - .05, top)
    for xi, (_, row) in zip(xs, df.iterrows()):
        star = "*" if row["p_value"] < SIG_ALPHA else ""
        ax.text(xi, row["gap_excess"] + (.008 if row["gap_excess"] >= 0 else -.014),
               f"{row['gap_excess']:+.3f}{star}\nz={row['z']:.2f}",
               ha="center", va="bottom" if row["gap_excess"] >= 0 else "top",
               fontsize=10.5, fontweight="bold",
               color=FS.ACCENT if row["p_value"] < SIG_ALPHA else FS.NAVY)

    gene_line = "   ·   ".join(f"{d}: {', '.join(GENES_BY_DOMAIN[d])}"
                               for d in df["domain"])
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{d}\nn={n}" for d, n in zip(df["domain"], df["n"])],
                       fontsize=11)
    ax.text(0.5, -.145, gene_line, transform=ax.transAxes, ha="center",
           va="top", fontsize=8.4, color=FS.MUTE)

    ax.set_ylabel("planning-gap EXCESS over each domain's own null\n"
                 "(oracle − RelMax, minus what domain size alone predicts)")
    ax.set_title("Only lipid_disorder's planning gap survives null-correction",
                pad=14)
    n_perm_used = int(df["n_perm"].iloc[0]) if "n_perm" in df.columns else args.n_perm
    ax.text(0.02, .97,
           f"* p < 0.05 (within-gene permutation, {n_perm_used} draws)",
           transform=ax.transAxes, fontsize=9, color=FS.MUTE, va="top")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"written: {args.out}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
