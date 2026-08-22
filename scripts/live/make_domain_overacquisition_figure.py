#!/usr/bin/env python3
"""Live-agent over-acquisition, broken down by clinical domain.

WHY THIS FIGURE
---------------
`figures/live/more_is_not_better.png` headlines a single cohort-wide multiple
(spend / need, pooled over all 491 cases: 3.19x). That number is an average
over four clinical domains that are NOT interchangeable -- pooling erases the
one that is actually driving the claim. Split by the four ACMG domains this
cohort is drawn from (arrhythmia: KCNQ1/SCN5A; cardiomyopathy: MYH7/MYBPC3;
lipid_disorder: LDLR/PCSK9; hereditary_cancer: BRCA1/BRCA2/TP53/PTEN/MLH1/MSH2
-- see pwkbench/spec.py:GENES_BY_DOMAIN), arrhythmia sits at ~12-13x while the
other three cluster at 2.6-2.9x. That gap is the more striking -- and more
specific -- claim: this is not a uniform 3x tax, it concentrates in one
domain.

VERIFIED, NOT A ONE-RUN ARTIFACT. This repo's own history is full of numbers
that did not survive a second run, so before this figure
existed, the per-domain multiple was checked against each of the 3 independent
live runs SEPARATELY, not just the pooled table:
    run 0: 12.00x   run 1: 12.67x   run 2: 13.33x   (arrhythmia; other three
    domains stay in the 2.5-3.1x band in every run). The per-run dots on top
    of each bar ARE that check, kept visible rather than collapsed into the
    bar so a reader can see the claim is not resting on pooling alone.

WHAT THIS FIGURE DOES NOT CLAIM
--------------------------------
1. No mechanism. Nothing here says WHY arrhythmia is different -- only two
   genes carry that domain (KCNQ1, SCN5A), so this could be a gene effect, a
   domain effect, or an artifact of which 78 real ClinVar variants happen to
   fall in it. Do not caption this "arrhythmia variants are harder"; caption
   it as what it is, a measured concentration.
2. No cross-domain significance test. The bootstrap CI (case-level,
   resampling WITHIN a domain's pooled 3-run rows) says how uncertain each
   domain's OWN point estimate is; it is not a paired contrast between
   domains, and gene-clustering the resample is not meaningful here --
   arrhythmia and cardiomyopathy have exactly 2 genes each, so a
   gene-clustered bootstrap would draw from a pool of 2 and produce a
   near-discrete, uninformative interval. Case-level resampling is reported
   instead, with this limitation stated rather than hidden.
3. The functional-assay caveat still applies (get_functional_assay cannot
   lower the reference sufficiency point -- see metrics_live.py docstring),
   so `mult_mapped` (the headline number) excludes it, matching every other
   figure's convention. It was checked that this is not what drives the
   arrhythmia gap: including assay calls (`mult_all`) makes arrhythmia's
   number HIGHER (15.7x), not lower, and the 218 real functional-assay calls
   concentrate in MYH7/MYBPC3/PCSK9 (cardiomyopathy/lipid_disorder), not the
   arrhythmia genes.

Every number is read from the artifacts at run time.

USAGE
    python scripts/live/make_domain_overacquisition_figure.py
"""
import argparse, ast, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench import figstyle as FS                          # noqa: E402
from pwkbench.domains.base import load_real_cohort           # noqa: E402
from pwkbench.domains.variant import VARIANT_DOMAIN          # noqa: E402
from pwkbench.spec import GENES_BY_DOMAIN                    # noqa: E402
from pwkbench.live.metrics_live import (                     # noqa: E402
    sufficiency_points, TOOL_TO_CHANNEL)
from pwkbench.live.tools import TOOL_COST_RANK               # noqa: E402

LIVE_RUNS = ["results/live/trajectories_full.csv",
             "results/live/trajectories_full_r1.csv",
             "results/live/trajectories_full_r2.csv"]
CH2TOOL = {v: k for k, v in TOOL_TO_CHANNEL.items() if v}
DOMAIN_ORDER = ["arrhythmia", "cardiomyopathy", "lipid_disorder",
                "hereditary_cancer"]
N_BOOT = 3000


def _cost_of_subset(sub):
    if not isinstance(sub, str) or not sub.strip():
        return 0.0
    parts = [x.strip().strip("[]'\" ") for x in sub.replace("|", ",").split(",")]
    return sum(TOOL_COST_RANK[CH2TOOL[c]] for c in parts if c in CH2TOOL)


def _per_case_table():
    """One row per (variant, run) with need/spend, restricted to cases with a
    defined sufficiency point -- the same subset every other live headline
    number in this repo is computed over."""
    coh = load_real_cohort(
        pd.read_parquet("data/sample/cohort_full_real.parquet"), VARIANT_DOMAIN)
    suff = sufficiency_points(coh)
    si = suff.set_index("variant_id")
    meta = coh.df.set_index("variant_id")[["gene", "domain"]]

    rows = []
    for run_i, p in enumerate(LIVE_RUNS):
        d = pd.read_csv(p).set_index("variant_id")
        ids = [i for i in d.index
               if i in si.index and pd.notna(si.loc[i, "k_star"])]
        for i in ids:
            need = _cost_of_subset(si.loc[i, "k_star_example_subset"])
            L = d.loc[i, "tools_called"]
            L = ast.literal_eval(L) if isinstance(L, str) else (L or [])
            spend = sum(TOOL_COST_RANK[t] for t in L if TOOL_TO_CHANNEL.get(t))
            rows.append(dict(variant_id=i, run=run_i, need=need, spend=spend,
                             gene=meta.loc[i, "gene"], domain=meta.loc[i, "domain"]))
    return pd.DataFrame(rows)


def _bootstrap_ci(sub, B=N_BOOT, seed=0):
    """Case-level percentile CI on spend.mean()/need.mean(), within one
    domain's pooled rows. See module docstring §2 for why this is case-level
    rather than gene-clustered."""
    rng = np.random.default_rng(seed)
    need, spend = sub["need"].to_numpy(), sub["spend"].to_numpy()
    n = len(sub)
    idx = rng.integers(0, n, size=(B, n))
    m = spend[idx].mean(axis=1) / need[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/live/domain_overacquisition.png")
    args = ap.parse_args()
    os.chdir(_ROOT)
    FS.use()

    df = _per_case_table()
    overall_mult = df["spend"].mean() / df["need"].mean()

    stats = {}
    for dom in DOMAIN_ORDER:
        sub = df[df.domain == dom]
        mult = sub["spend"].mean() / sub["need"].mean()
        lo, hi = _bootstrap_ci(sub)
        per_run = [sub[sub.run == r]["spend"].mean() / sub[sub.run == r]["need"].mean()
                  for r in range(3)]
        stats[dom] = dict(n=len(sub) // 3, mult=mult, lo=lo, hi=hi, per_run=per_run)

    order = sorted(DOMAIN_ORDER, key=lambda d: -stats[d]["mult"])

    fig, ax = plt.subplots(figsize=(8.6, 6.3))
    xs = np.arange(len(order))
    colors = [FS.ACCENT if d == "arrhythmia" else FS.STEEL for d in order]
    mults = [stats[d]["mult"] for d in order]
    lo_err = [stats[d]["mult"] - stats[d]["lo"] for d in order]
    hi_err = [stats[d]["hi"] - stats[d]["mult"] for d in order]
    top = max(stats[d]["hi"] for d in order) + 3.4
    ax.set_ylim(0, top)
    ax.set_xlim(-.62, len(order) - .38)

    ax.bar(xs, mults, width=.56, color=colors, edgecolor="white", linewidth=1.0,
          zorder=3)
    ax.errorbar(xs, mults, yerr=[lo_err, hi_err], fmt="none", ecolor=FS.INK,
               elinewidth=1.4, capsize=5, capthick=1.4, zorder=4)

    # Per-run point estimates, overlaid rather than collapsed into the bar --
    # the reproducibility check IS part of the claim (see module docstring).
    for xi, d in zip(xs, order):
        jitter = np.linspace(-.13, .13, 3)
        h_run = ax.scatter(xi + jitter, stats[d]["per_run"], s=26, color=FS.NAVY,
                           zorder=5, edgecolor="white", linewidth=.6)

    h_flat = ax.axhline(1.0, color=FS.MUTE, ls="--", lw=1.2, zorder=2)
    h_avg = ax.axhline(overall_mult, color=FS.GREY, ls=":", lw=1.6, zorder=2)
    ax.legend([h_run, h_flat, h_avg],
             ["individual run",
              "1.0x — spent exactly what was needed",
              f"{overall_mult:.2f}x — cohort-wide average (all 4 domains pooled)"],
             loc="upper right", framealpha=.95, fontsize=9.3)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d}\nn={stats[d]['n']}" for d in order], fontsize=11.5)
    ax.set_ylabel("evidence acquired  /  evidence the case needed")
    ax.set_title("Over-acquisition concentrates in one clinical domain",
                pad=14)
    for xi, d in zip(xs, order):
        ax.text(xi, stats[d]["hi"] + .55, f"{stats[d]['mult']:.1f}x",
               ha="center", fontsize=14, fontweight="bold",
               color=FS.ACCENT if d == "arrhythmia" else FS.NAVY)

    # Gene composition, one caption line under the axis rather than crammed
    # into the tick labels (6 genes for hereditary_cancer collided with its
    # neighbours' labels at any font size that still fit the column).
    gene_line = "   ·   ".join(f"{d}: {', '.join(GENES_BY_DOMAIN[d])}"
                               for d in order)
    ax.text(0.5, -.145, gene_line, transform=ax.transAxes, ha="center",
           va="top", fontsize=8.6, color=FS.MUTE)

    # KCNQ1 callout: its 36 cases have a sufficiency point of ZERO in every
    # one (the reference call needs no acquired evidence at all), yet the
    # agent still spent evidence on them -- an infinite ratio that the bar
    # chart cannot show (arrhythmia's bar is finite only because SCN5A, the
    # domain's other gene, has need > 0). Stated as spend, not as a multiple.
    kcnq1 = df[df.gene == "KCNQ1"]
    ax.annotate(
        f"KCNQ1 alone: {len(kcnq1)//3} cases need ZERO acquired evidence,\n"
        f"agent spends {kcnq1['spend'].mean():.2f} tool-units anyway",
        xy=(0.18, stats["arrhythmia"]["mult"] / top),
        xycoords="axes fraction",
        xytext=(0.99, 0.66), textcoords="axes fraction",
        fontsize=9.3, color=FS.INK, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=.45", facecolor=FS.PALE,
                 edgecolor=FS.ACCENT, linewidth=1.1),
        arrowprops=dict(arrowstyle="-|>", color=FS.ACCENT, lw=1.3,
                       connectionstyle="arc3,rad=.25"))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"written: {args.out}")
    for d in order:
        s = stats[d]
        print(f"  {d:<18} n={s['n']:>3}  mult={s['mult']:.2f}x  "
             f"95% CI [{s['lo']:.2f}, {s['hi']:.2f}]  "
             f"per-run={[f'{v:.2f}' for v in s['per_run']]}")
    print(f"  cohort-wide  {overall_mult:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
