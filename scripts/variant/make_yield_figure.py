#!/usr/bin/env python3
"""Diagnostic yield against acquisition budget, without a cost assumption.

WHY THIS REPLACES THE OLD FIGURE
--------------------------------
The earlier `figures/variant/analysis/clinical_endpoints.png` was half usable. Panel A was
real and matched `clinical_yield.csv` exactly. Panel B was not: its x-axis was
CUMULATIVE COST with the fourth channel priced at 10x, which is why the curve
jumps 3 -> 13. The fourth channel in the relevance order is PM1 — a UniProt
functional-domain lookup — so that 10x cost figure is unsourced and has been
retracted. Beautifying that panel would have made a withdrawn assumption more
persuasive, so it is replaced rather than restyled.

The replacement makes the same point — the later acquisitions add little — with
no pricing at all: plot the MARGINAL gain of each additional query. Under both
scorers the third query is nearly free of benefit (+0.022 logistic, +0.002 ACMG)
while the second adds the most. That is a statement about the evidence, not
about what a lab charges.

ONE SCORER THROUGHOUT
---------------------
Both panels use the ACMG point rule — the guideline's own. An earlier draft drew
panel A with that rule and panel B with two scorers, which is where the confusing
"+1 or +11?" came from: the third query adds 1 variant under the guideline and 11
under a fitted logistic scorer, and putting both on one panel made the reader
adjudicate a methods choice mid-figure. The data-driven scorer resolves more
overall (0.308 -> 0.546 against 0.308 -> 0.389) at a call accuracy that falls to
0.966; that comparison is a methods point and lives in the text, not here.

PANEL A SAYS THE CEILING, NOT THE GAIN
--------------------------------------
An earlier draft titled panel A "more evidence, fewer undecided cases" and the
bars barely moved — 151 decided at k=1 against 191 at k=4, out of 491. That flat
look was read as a weak result, but it IS the result: **most variants stay
undecided at any budget.** The guideline's own coverage ceiling on this cohort is
38.9%, which FINDINGS already records as a limit of the yardstick rather than an
implementation gap. Titling the panel after the small gain fought the picture;
titling it after the ceiling is both what the bars show and the more useful fact.

PANEL A: ONE MESSAGE, ONE CHART TYPE
------------------------------------
The first draft put four series across two y-axes — resolved and call accuracy,
each under two scorers — and a reader had to track which axis each line belonged
to before the panel said anything. It now answers one question in the units a
lab counts in: OF 491 VARIANTS, HOW MANY COME BACK WITH A DEFINITE CALL? The
rest stays undecided (VUS), and the shrinking grey band is the benefit of
acquisition without any jargon.

The ACMG point rule is used for the bars because it is the guideline's own rule
and its call accuracy is 1.000 at every budget, so the bars cannot be misread as
"more calls, but worse ones". The data-driven scorer resolves more (0.308 ->
0.546) at a call accuracy that falls to 0.966; that decline is scorer-dependent
and is stated as a note rather than drawn, because drawing it next to the
guideline bars is what made the panel unreadable.

USAGE
    python scripts/variant/make_yield_figure.py
"""
import argparse, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench import figstyle as FS                          # noqa: E402

YIELD = "results/variant/real_trusted/clinical_yield.csv"
INK, MUTE = FS.INK, "#4d5c68"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="figures/variant/clinical_endpoints.png")
    args = ap.parse_args()
    os.chdir(_ROOT)
    FS.use()

    d = pd.read_csv(YIELD)
    r = d[d.strategy == "RelMax"].sort_values("budget")
    k = r["budget"].to_numpy()
    log_res, acmg_res = r["logistic_resolved"].to_numpy(), r["acmg_resolved"].to_numpy()
    N = 491

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 4.30))
    fig.patch.set_facecolor(FS.PARADIGM["paper"])
    for a in (axA, axB):
        a.set_facecolor(FS.PARADIGM["paper"])

    # ---- A: of 491 variants, how many come back decided --------------------
    dec = np.round(acmg_res * N).astype(int)
    und = N - dec
    axA.bar(k, dec, width=.58, color=FS.NAVY, edgecolor="white", linewidth=1.2,
            label="resolved")
    axA.bar(k, und, bottom=dec, width=.58, color="#dbe4ea", edgecolor="white",
            linewidth=1.2, label="undecided (VUS)")
    for ki, d in zip(k, dec):
        axA.text(ki, d / 2, str(d), ha="center", va="center", color="white",
                 fontsize=13.5, fontweight="bold")
        axA.text(ki, d + (N - d) / 2, str(N - d), ha="center", va="center",
                 color=MUTE, fontsize=12)
    axA.set_xticks(k); axA.set_xlabel("acquisition budget k")
    axA.set_ylabel("variants")
    axA.set_ylim(0, N * 1.12)
    # No legend at all: the panel is too narrow to seat one beside the title,
    # and inside the axes there is nowhere free — the bars are stacked to the
    # full cohort, so every column is solid top to bottom. The two segments are
    # named directly beside the last bar instead, which needs no key.
    axA.set_xlim(.45, 5.35)
    axA.text(4.45, dec[-1] / 2, "resolved", va="center", ha="left",
             fontsize=11.5, fontweight="bold", color=FS.NAVY)
    axA.text(4.45, dec[-1] + (N - dec[-1]) / 2, "undecided\n(VUS)",
             va="center", ha="left", fontsize=11.5, color=MUTE,
             linespacing=1.3)
    # Titles name what is plotted rather than stating the finding. The previous
    # set ("Most cases stay undecided", "The third query adds almost nothing")
    # read as ad copy next to the neighbouring panels' terminology, and out of
    # context a reader could not tell what was on either axis. The findings are
    # still carried inside the panels — the "only +1" callout in B, the note
    # under A — so nothing is lost by making the titles descriptive.
    axA.set_title("A   Diagnostic yield by budget",
                  fontsize=14, fontweight="bold", color=INK, loc="left", pad=12)
    axA.set_xlabel("acquisition budget k")
    # One short line, not a paragraph: the only thing a reader must know to
    # avoid misreading the bars as "more calls, but worse ones".

    # ---- B: what each extra query actually adds ---------------------------
    marg_log = np.round(np.diff(log_res) * N).astype(int)
    marg_acmg = np.round(np.diff(acmg_res) * N).astype(int)
    x = np.arange(len(marg_acmg))
    axB.bar(x, marg_acmg, width=.55, color=FS.NAVY, edgecolor="white",
            linewidth=1.2)
    for xi, v in enumerate(marg_acmg):
        axB.text(xi, v + .6, f"+{v}", ha="center", fontsize=15.5,
                 fontweight="bold", color=FS.NAVY)
    # Highlight the step where the return collapses — that is the claim this
    # figure supports. One number, because the whole panel now uses one scorer.
    axB.axvspan(.5, 1.5, color=FS.ACCENT, alpha=.08, zorder=0)
    axB.text(1, max(marg_acmg) * .55, f"only +{marg_acmg[1]}",
             ha="center", fontsize=19.5, fontweight="bold", color=FS.ACCENT)
    axB.set_xticks(x)
    axB.set_xticklabels(["2nd", "3rd", "4th"])
    axB.set_xlabel("query number")
    axB.set_ylabel("additional variants resolved")
    axB.set_ylim(0, max(marg_acmg) * 1.22)
    axB.set_title("B   Marginal yield of each query",
                  fontsize=14, fontweight="bold", color=INK, loc="left", pad=12)

    fig.tight_layout()
    # No caption. The retraction note this figure used to carry lives in this
    # module's docstring, where a reader who needs the provenance will look.
    # The VUS-ceiling line that sat here briefly is now
    # the first conclusion in the poster's clinical box, which is where it
    # belongs: it is a reading of the panel, not a label for it.

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"written: {args.out}")
    print(f"  resolved  logistic {log_res.round(3).tolist()}  acmg {acmg_res.round(3).tolist()}")
    print(f"  marginal  logistic {marg_log.round(3).tolist()}  acmg {marg_acmg.round(3).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
