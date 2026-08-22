#!/usr/bin/env python3
"""Hero figure: More Evidence · Same Answer · Wasted Cost.

WHY THIS SHAPE
--------------
The paper's thesis has three beats — MORE EVIDENCE, SAME ANSWER, WASTED COST —
so those are the figure's three columns, and each row is read straight across.
Earlier drafts showed only the first two beats and then headlined a percentage,
which is why the cost claim never landed: it was never drawn.

The two information conditions are two row-groups rather than two side-by-side
panels. That change matters: as separate panels the eye compares the two big
numbers, and those numbers are NOT comparable (a ratio against a flat budget on
one side, a share of cases on the other). As row-groups under shared column
headers, the eye compares each row against the row above it — which is the
comparison the experiment actually made.

COST IS SHOWN AS A MULTIPLE, not in raw units, because the two conditions are
priced in different currencies: the frozen block counts queries (every channel
costs the same there), the live block sums TOOL_COST_RANK (gnomAD 1, in-silico
1, UniProt 2, MaveDB 3). A shared bar scale across those would be meaningless;
"x what the case needed" is the same thing on both sides.

The information condition is drawn on the subject rather than described — the
blinded agent is shown no variant identity at all (four channel names, nothing
else); the unblinded one is handed gene + full HGVS name.

TERMINOLOGY. The subject is written as an "LLM agent", never an "AI clinician".
The system is a language model choosing among four database lookups; it does not
see patients, prescribe, or sit in a clinical workflow, and naming it after a
licensed role would claim a deployment context this work does not have. (The
system prompt does say "You are a clinical variant scientist" — that is an
instruction given to the model, not a description of the system.) The clinician
GLYPH stays: an icon may be a metaphor, a title may not.

WHAT THIS FIGURE IS CAREFUL NOT TO CLAIM
----------------------------------------
Three overclaims an earlier draft would have made:

1. **The two panels are NOT the same measurement**, so they are not drawn as one
   "tests ordered" comparison. In the frozen block the budget is imposed by the
   harness (k=1..4 swept) and the agent chooses an ORDER, not an amount; its
   over-acquisition surfaces as ranking AUC that peaks at k=2 and falls by 0.110
   by k=4. Only the live count is the agent's own decision. Merging the two into
   one bar chart would invent a quantity neither experiment measured.
2. **The blindfolded clinician is not "choosing to over-order"** in the main
   frozen arm — the protocol runs everything. The STOP arm is where it does
   choose, so that is drawn as its own comparison rather than folded in.
3. **The sighted clinician does not know the evidence is sufficient.** The
   sufficiency point is retrospective, computed with gold labels. What the repo
   measured is closed-book AUC 0.865 from the name alone — it very often already
   has the answer. "Already knew it was enough" would overstate; "very likely
   already has the answer, and orders anyway" is what holds.

The "tests" are database lookups: the cost is time, money and API calls, not
patient risk, and nothing here should imply clinical harm.

Every number is read from the artifacts at run time.

USAGE
    python scripts/live/make_hero_figure.py
"""
import argparse, ast, glob, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench import figglyphs as G                          # noqa: E402
from pwkbench import figstyle as FS                          # noqa: E402
from pwkbench.domains.base import load_real_cohort           # noqa: E402
from pwkbench.domains.variant import VARIANT_DOMAIN          # noqa: E402
from pwkbench.live.metrics_live import (                     # noqa: E402
    sufficiency_points, over_acquisition, TOOL_TO_CHANNEL)
from pwkbench.live.tools import TOOL_COST_RANK               # noqa: E402

LIVE_RUNS = ["results/live/trajectories_full.csv",
             "results/live/trajectories_full_r1.csv",
             "results/live/trajectories_full_r2.csv"]
STOP_GLOB = "results/variant/stopping/stopping_frontier_B_run*.csv"

BLIND, SIGHT = FS.STEEL, FS.NAVY
INK = FS.INK
# figstyle's MUTE (#7d8b96) is tuned for on-screen reading and disappears at
# poster distance. Secondary text here uses a darker slate; only the single
# lowest-priority line keeps the lighter tone.
MUTE = "#4d5c68"
CH2TOOL = {v: k for k, v in TOOL_TO_CHANNEL.items() if v}

# Five columns, shared by both conditions, so the channel difference is visible
# in place rather than implied by two different-length rows. Each condition
# marks every column with one of three states:
#   "acq"  acquirable — costs a query, and is FILLED when this row acquired it
#   "free" available at no cost (PVS1 in the live block: loss-of-function is
#          legible in the HGVS name the agent is already shown)
#   "n/a"  not in this condition's pool at all
#
# ORDER IS THE OBSERVED ACQUISITION ORDER, not the ACMG listing order, because
# left-to-right filling otherwise asserts a sequence the agent never used.
# Measured: frozen picks PVS1 first in 491/491 cases and yields one distinct
# order; live picks population frequency first in 1309/1309, then reaches for
# the assay (43.9% of cases), domain (42.3%) and in-silico (18.3%).
TEST = {
    "PVS1":  (G.truncation, "PVS1", "loss of\nfunction"),
    "PM2":   (G.bars,       "PM2",  "population\nfrequency"),
    "PM1":   (G.domain,     "PM1",  "functional\ndomain"),
    "PP3":   (G.curve,      "PP3",  "in-silico\nprediction"),
    "assay": (G.vial,       "PS3",  "functional\nassay"),
}
# Each condition shows ONLY the tests it actually offers. An earlier draft drew
# all five in both rows and greyed out the absent one, which spent a column on
# something that is not there.
#
# PVS1 is drawn inside the UNBLINDED condition panel rather than as a column,
# because it is not a choice there: disclosing the variant name IS what makes
# loss-of-function legible for free. Listing it beside the four purchasable
# tools would imply the agent decided to acquire it.
#
# Column order is the observed acquisition order, by MEAN CALL POSITION.
# An earlier draft ordered by overall call frequency, which put the assay second
# because it is called in 43.9% of cases — but positionally it is last:
#     population frequency 1.00 · in-silico 2.19 · domain 2.51 · assay 3.15
# That distinction matters, and not only for the drawing: the assay is the most
# expensive tool (TOOL_COST_RANK 3 against 1/1/2), and the agent defers it. The
# agent is cost-aware in ORDERING; what it is not cost-aware about is STOPPING.
FROZEN_ORDER = ["PVS1", "PM2", "PM1", "PP3"]
LIVE_ORDER = ["PM2", "PP3", "PM1", "assay"]


def _stop_arm():
    s = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(STOP_GLOB))])
    g = s.groupby("rule")
    ch, fx = g.get_group("agent_stop:frontier_B"), g.get_group("agent_fixed_k2")
    return (ch["mean_queries"].mean(), int(ch["n_called"].mean()),
            fx["mean_queries"].mean(), int(fx["n_called"].mean()))


def _cost_of_subset(sub):
    """Cost of the sufficient channel set, priced with the live tool ranks."""
    if not isinstance(sub, str) or not sub.strip():
        return 0.0
    parts = [x.strip().strip("[]'\" ") for x in sub.replace("|", ",").split(",")]
    return sum(TOOL_COST_RANK[CH2TOOL[c]] for c in parts if c in CH2TOOL)


def _live():
    coh = load_real_cohort(
        pd.read_parquet("data/sample/cohort_full_real.parquet"), VARIANT_DOMAIN)
    suff = sufficiency_points(coh)
    si = suff.set_index("variant_id")
    dom = coh.df.set_index("variant_id")["domain"]
    need_c, spend_c, extra_ch, exceed, ran = [], [], [], [], []
    # Domain callout for the panel annotation (see
    # figures/live/domain_overacquisition.png): arrhythmia's
    # need/spend ratio is tracked alongside the cohort-wide one, same
    # per-run/per-case values, just filtered to domain=="arrhythmia" before
    # averaging -- not a separate computation that could drift from it.
    need_arr, spend_arr = [], []
    for p in LIVE_RUNS:
        d = pd.read_csv(p)
        oa = over_acquisition(d, suff)
        col = next(c for c in oa.columns if "over" in c.lower())
        v = oa.loc[oa[col].notna(), col]
        extra_ch.append(v.mean()); exceed.append((v > 0).mean())
        d = d.set_index("variant_id")
        ids = [i for i in d.index if i in si.index and pd.notna(si.loc[i, "k_star"])]
        need_c.append(np.mean([_cost_of_subset(si.loc[i, "k_star_example_subset"])
                               for i in ids]))
        # MAPPED tools only, so both sides of the ratio are priced over the same
        # channel set; the assay has no channel here and would inflate only the
        # numerator (see over_acq_mapped vs over_acq_all).
        sp = []
        for i in ids:
            L = d.loc[i, "tools_called"]
            L = ast.literal_eval(L) if isinstance(L, str) else (L or [])
            sp.append(sum(TOOL_COST_RANK[t] for t in L if TOOL_TO_CHANNEL.get(t)))
        spend_c.append(np.mean(sp))
        ran.append(d["n_tools_called"].median())
        arr_ids = [i for i in ids if dom.get(i) == "arrhythmia"]
        need_arr.append(np.mean([_cost_of_subset(si.loc[i, "k_star_example_subset"])
                                 for i in arr_ids]))
        spend_arr.append(np.mean(
            [sp[ids.index(i)] for i in arr_ids]))
    arr_mult = float(np.mean(spend_arr) / np.mean(need_arr))
    return (suff, float(np.mean(need_c)), float(np.mean(spend_c)),
            float(np.mean(extra_ch)), float(np.mean(exceed)), int(np.median(ran)),
            arr_mult)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/live/more_is_not_better.png")
    args = ap.parse_args()
    os.chdir(_ROOT)
    FS.use()

    q_ch, n_ch, q_fx, n_fx = _stop_arm()
    suff, need_c, spend_c, extra_ch, exceed, ran, arr_mult = _live()
    blind_extra, blind_mult = q_ch - q_fx, q_ch / q_fx
    live_mult = spend_c / need_c

    # TALLER WITHOUT DISTORTION. The axes has aspect "auto" and a 0-100 data
    # space, so one y-unit is currently 8.4/100 in while one x-unit is 15.6/100
    # in. Raising the figure height alone would therefore stretch every glyph
    # vertically — the doctor's head would go oval. Raising the Y RANGE by the
    # same factor keeps the physical size of a y-unit fixed, so nothing is
    # distorted and the extra units become breathing room between the two
    # blocks, which is what made the panel read as flat.
    #
    # savefig uses bbox_inches="tight", so outer margin is trimmed anyway; only
    # the gaps BETWEEN elements change the final aspect (1.86 -> ~1.65).
    # WHY THE CANVAS IS SMALL RATHER THAN THE FONTS BIG. The poster draws this
    # at its column width (11.1 in) whatever size it is authored at, so a label
    # lands on the board at fontsize x (11.1 / figure width). Authored at 15.6 in
    # that is 0.71x — a 9.2 pt row label printed at 6.5 pt, unreadable across a
    # poster hall. Authoring at 11.1 in makes the ratio 1.0 and every label
    # prints at its nominal size, which is one edit instead of rescaling ~25
    # fontsize arguments and re-tuning the collisions that would follow.
    #
    # Both dimensions scale by the same 0.712, so unit_x / unit_y is unchanged
    # and no glyph is distorted. Glyphs are drawn in DATA units and so shrink
    # with the canvas, while text is in points and does not — which is the
    # intended effect: text grows relative to the icons.
    # Bottom of ylim used to be 0, but the lowest artist (the UNBLINDED box)
    # sits at y=8.5 — group(10, ...) draws its box at y0-1.5 — so the 8.5
    # units below it were dead axis space that bbox_inches="tight" does not
    # trim (the invisible axes patch still spans the full data range). On the
    # poster that 8.5-unit strip became a 0.68 in blank band UNDER the figure,
    # stacking with the panel's own image-to-header gap to read as a missing
    # block between "Main result" and "Theoretical Framework". Cropping the
    # ylim to 7.0 (a 1.5-unit margin, matching the box's own pad) removes it.
    # Figure height shrinks in the same proportion as the ylim range so
    # unit_y (in/data-unit) is unchanged and nothing already on the canvas is
    # stretched.
    fig = plt.figure(figsize=(8.7, 7.386))
    fig.patch.set_facecolor(FS.PARADIGM["paper"])
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 100); ax.set_ylim(7.0, 149.0)

    ax.text(50, 145.0, "More Is Not Better", ha="center", fontsize=23,
            fontweight="bold", color=INK)
    # Not "AI clinician": the system is a language model choosing among database
    # lookups, and naming it after a licensed role would claim a deployment
    # context this work does not have. "Clinical" describes the DOMAIN, which is
    # accurate, and "agent's choice" is what the figure actually measures.
    ax.text(50, 140.8, "evidence acquisition by an LLM agent  ·  "
            "clinical variant interpretation",
            ha="center", fontsize=14.0, color=MUTE)

    # ---- the three beats, as column headers ------------------------------
    # "SAME ANSWER" was too strong. Under the ACMG point rule call accuracy is
    # 1.000 at every budget, but under a logistic scorer it falls 1.000 -> 0.966
    # across k=1..4, and in the live block cases with more tools score lower
    # (0.96 at 0-1 tools vs 0.83 at 3-4). That live gap is CONFOUNDED — the agent
    # runs more tools on the cases it finds hard — so it cannot be read as
    # evidence that extra tests cause worse answers. What survives both is the
    # weaker, defensible claim: the extra evidence does not make the answer
    # better.
    # Spread out and one point down: at the larger text size the first two
    # ran into each other ("MORE EVIDENCENO BETTER ANSWER").
    COLS = [(30, "MORE EVIDENCE"), (60, "NO BETTER ANSWER"), (87, "HIGHER COST")]
    for cx, name in COLS:
        ax.text(cx, 135.0, name, ha="center", fontsize=11.8, fontweight="bold",
                color=FS.BLUE)
    ax.plot([6, 94], [132.5, 132.5], color="#c3ced6", lw=1.2)

    def group(y0, c, tag, told, order, rows, mult, delta, free_pvs1=False,
              footnote=None):
        ax.add_patch(FancyBboxPatch((5, y0 - 1.5), 90, 56,
                                    boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor="white", edgecolor=c, linewidth=1.6,
                                    zorder=1))
        ax.add_patch(FancyBboxPatch((5.5, y0 + 50.0), 15, 3.6,
                                    boxstyle="round,pad=0,rounding_size=1.8",
                                    facecolor=c, edgecolor="none", zorder=4))
        ax.text(13, y0 + 51.8, tag, ha="center", va="center", color="white",
                fontsize=11.8, fontweight="bold", zorder=5)
        G.doctor(ax, 12.5, y0 + 34.0, 9.0, c, blindfold=(tag == "BLINDED"), z=4)
        ax.text(12.5, y0 + 22.0, told, ha="center", va="center", fontsize=10.2,
                color=MUTE, zorder=5, linespacing=1.5)

        # PVS1 belongs to the CONDITION when the name is disclosed: it is free
        # precisely because of the disclosure, not because the agent chose it.
        if free_pvs1:
            # Kept inside the condition column (x < 22) so it cannot reach the
            # row labels, which are right-aligned at x=28.5.
            G.truncation(ax, 6.0, y0 + 13.0, 4.4, FS.ACCENT, filled=True)
            ax.text(11.2, y0 + 14.6, "PVS1 free", fontsize=9.4,
                    fontweight="bold", color=FS.ACCENT, va="center")
            ax.text(11.2, y0 + 11.4, "legible in the\ndisclosed name",
                    fontsize=7.9, color=MUTE, va="center", linespacing=1.3)

        n_cols = len(order)
        x0 = 30.0
        for i, key in enumerate(order):
            cx = x0 + i * 6.2 + 2.0
            ax.text(cx, y0 + 49.4, TEST[key][1], ha="center", fontsize=8.6,
                    fontweight="bold", color=FS.BLUE)
            ax.text(cx, y0 + 45.2, TEST[key][2], ha="center", fontsize=6.8,
                    color=MUTE, linespacing=1.3)

        for j2, (lbl, n_on, ink, bar) in enumerate(rows):
            y = y0 + 30.5 - j2 * 25.0
            ax.text(28.5, y, lbl, fontsize=10.8, color=INK, va="center",
                    ha="right", fontweight="bold", linespacing=1.4)
            for i in range(n_cols):
                gx = x0 + i * 6.2
                TEST[order[i]][0](ax, gx, y - 2.0, 4.0,
                                  ink if i < n_on else FS.OFF, filled=i < n_on)
            G.check(ax, 59, y, 2.3, ink)
            w = 18 * bar / mult
            ax.add_patch(FancyBboxPatch((66, y - 1.5), max(w, .6), 3.0,
                                        boxstyle="round,pad=0,rounding_size=.5",
                                        facecolor=ink if j2 == 0 else FS.ACCENT,
                                        edgecolor="none", zorder=4))
            ax.text(66 + max(w, .6) + 1.2, y, f"{bar:.2f}×", va="center",
                    fontsize=11.0, fontweight="bold",
                    color=ink if j2 == 0 else FS.ACCENT)
        # The delta, centred between the two rows it separates.
        ax.text(x0 + (n_cols * 6.2) / 2 - 0.8, y0 + 18.0, delta, ha="center",
                va="center", fontsize=13.5, fontweight="bold", color=FS.ACCENT)
        ax.text(90.5, y0 + 18.0, f"{mult:.2f}×", ha="right", va="center",
                fontsize=30, fontweight="bold", color=FS.ACCENT)
        # Cohort-wide average dilutes a concentration that is real in one
        # clinical domain (see figures/live/domain_overacquisition.png,
        # verified 3-run-stable) -- one line, not a
        # second panel, so the headline number stays the cohort-wide one and
        # this reads as a footnote rather than a competing claim.
        if footnote:
            ax.text(90.5, y0 + 12.3, footnote, ha="right", va="center",
                    fontsize=8.4, fontweight="bold", color=MUTE)

    group(72, BLIND, "BLINDED", "variant identity\nWITHHELD",
          FROZEN_ORDER,
          [("minimum", 2, BLIND, 1.00),
           ("agent's choice", 3, BLIND, blind_mult)],
          blind_mult, f"+{blind_extra:.2f} tests")

    group(10, SIGHT, "UNBLINDED", "variant identity\nDISCLOSED",
          LIVE_ORDER,
          [("minimum", 0, SIGHT, 1.00),
           ("agent's choice", max(ran, 1), SIGHT, live_mult)],
          live_mult, f"+{extra_ch:.2f} tests", free_pvs1=True,
          footnote=f"up to {arr_mult:.1f}× in arrhythmia cases")

    # No closing line here. It restated, almost word for word, the Takeaway
    # panel three columns to its right, and on a poster the same sentence twice
    # reads as a layout mistake rather than emphasis.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"written: {args.out}")
    print(f"  blinded  {q_fx:.2f} -> {q_ch:.2f} queries = {blind_mult:.2f}x, "
          f"+{blind_extra:.2f}/case, {n_ch} vs {n_fx} solved")
    print(f"  unblinded cost {need_c:.2f} -> {spend_c:.2f} = {live_mult:.2f}x, "
          f"+{extra_ch:.2f} channels/case, exceed {exceed:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
