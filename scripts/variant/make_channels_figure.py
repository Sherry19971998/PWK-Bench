#!/usr/bin/env python3
"""What the five pieces of evidence are, in plain terms.

WHY THIS EXISTS
---------------
The poster asks a reader to care about how much evidence an agent acquires
before it classifies a variant. That question is meaningless to anyone who does
not already know what one "piece of evidence" is. This figure answers only that:
five ACMG/AMP criteria, the question each one asks, what it is worth, and how
they combine into a call.

IT IS TYPESET AT POSTER SCALE, NOT SCREEN SCALE
-----------------------------------------------
`FIG_W` is the poster's column width. That is the whole reason the type on this
figure is legible: matplotlib font sizes are literal points, so a figure drawn
15 in wide and then placed in an 11.1 in column has every one of its labels
shrunk by 0.74 before it reaches the paper. Earlier versions were authored at
15 in with 7.9 pt body text, which printed at 5.9 pt — about a fifth the size of
the poster's own body text, and unreadable from any distance.

Drawing at exactly the column width makes the sizes below mean what they say:
the 14 pt question line is 14 pt on the printed board. If the poster's column
count or margins change, change `FIG_W` with them.

The layout follows from that. One card per row, full width, is the only
arrangement that gives a sentence enough inches to be set large: at two cards
per row each text column is 3.1 in, which forces ~7 pt type. The rows are short
because a row of N text lines caps the type at (row height / N), so every line
this layout does not draw is size the remaining ones get to keep.

WHAT THIS FIGURE DELIBERATELY DOES NOT DO
-----------------------------------------
It does not carry the experimental design: which criteria each arm offers, why
PVS1 is free once the variant name is disclosed, or why an assay call cannot
lower the sufficiency point. That material turned a primer into a design
argument, and a reader who does not yet know what PM1 is cannot follow either
one. It lives in the poster text and in `metrics_live.py`, its primary source.

One consequence must not be lost in the simplification: this cohort carries no
PS3 column, so the reference cannot score a functional assay and every assay
call counts as excess BY CONSTRUCTION. That is a limitation of the yardstick,
not a finding about the agent, and it is load-bearing wherever `over_acq_all`
is quoted.

Nor does it carry the data sources or the coverage counts any more — PP3 and PM1
are defined on 80 of 491 variants, PM2 on 490. Those are still computed and
printed to stdout on every build, and they belong in the poster text, because
"supporting evidence" and "supporting evidence that exists for a sixth of the
cohort" are not the same claim.

WHY THE PICTURES ARE DRAWN, NOT DOWNLOADED
------------------------------------------
Every illustration is matplotlib vector art from `pwkbench.figglyphs`. A stock
image would be prettier per unit of effort and would also (a) carry a licence
this figure would inherit, (b) break the property that every artifact in this
tree regenerates from the repository alone, and (c) not be able to show real
numbers — the PP3 picture is a real step function over the real ClinGen bins
that `metrics.py` scores with, not a picture of a graph.

EVERYTHING NUMERIC IS READ FROM THE CODE AND THE COHORT
-------------------------------------------------------
Point values, bin edges, classification thresholds and the worked example are pulled
from `pwkbench.metrics` and the real cohort at run time and asserted against what
the figure draws. A figure that hard-codes "+4" drifts the first time a threshold
is revised; this one fails loudly instead.

USAGE
    python scripts/variant/make_channels_figure.py
"""
import argparse, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench import figglyphs as G                          # noqa: E402
from pwkbench import figstyle as FS                          # noqa: E402
from pwkbench.domains.base import Cohort                     # noqa: E402
from pwkbench.domains.variant import VARIANT_DOMAIN          # noqa: E402
from pwkbench.metrics import (_acmg_channel_points, _ACMG_PATH_MIN,
                              _ACMG_BEN_MAX, _ACMG_PP3_SUP, _ACMG_PP3_MOD,
                              _ACMG_PP3_STRONG)              # noqa: E402

INK, MUTE = FS.INK, FS.MUTE
CARD_FILL, CARD_EDGE = "#f5f9fc", "#d3e1ec"

# The poster's column width (scripts/make_poster.py: COL_W). Drawing at exactly
# this width is what makes the point sizes below literal on the printed board.
# Authored NARROWER than the poster column (11.1 in) deliberately: the poster
# scales this to column width whatever it is, so a label prints at
# fontsize x (11.1 / FIG_W). At 9.7 that is 1.14x — every point size in this
# file lands 14% larger on the board without editing a single one of them.
# Aspect is set by YTOP, so this does not change how tall the panel sits.
FIG_W = 9.7
# Height in the same 0..100 units as x, so one unit is the same number of inches
# in both directions: circles come out round and a square picture is square,
# without a correction factor at every call site.
YTOP = 119.2          # = MARGIN*2 + 5*ROW_H + 5*GAP + SUM_H (asserted below)
FIG_H = FIG_W * YTOP / 100

# Colour encodes WEIGHT, not identity: the heavier the criterion, the darker it
# is drawn. Same hue family throughout, so the ordering survives greyscale and
# the reader picks up the ladder without reading a single number. PS3 and PP3
# share a colour because they share a weight — that is the encoding working,
# not a collision.
WEIGHT_COLOR = {8: FS.NAVY, 4: FS.BLUE, 2: FS.STEEL, 1: "#6fa9cf"}

# The worked example. Chosen, not invented: a real cohort variant whose two
# heaviest criteria already reach the threshold, so the same picture that
# explains the points scale also shows that the last piece changed nothing.
EXAMPLE_ID = "17:7674230:C>A"

# Type scale, in PRINTED points (see the header note). Every size on the figure
# is here, so the whole thing can be resized in one place if the column changes.
# Type scale for the taller rows. Capped at 1.05 by the figure's own
# assert_text_fits guard, not by taste: the text column is fixed at ~82 units
# and the longest explanation line overflows the card above that. Row HEIGHT
# grew 1.68x, so the extra vertical room goes to leading and to the icon
# (8.6 -> 12.0 units), which are not width-constrained.
_S = 1.05
FS_NAME, FS_BADGE, FS_PTS = 16.5 * _S, 14.5 * _S, 12.5 * _S
FS_Q, FS_BODY, FS_SMALL = 16.5 * _S, 14.0 * _S, 11.0 * _S


# --------------------------------------------------------------- helpers
def square_panel(fig, cx, cy, s):
    """A square axes of side `s` units centred at (cx, cy) of the main axes."""
    axg = fig.add_axes([(cx - s / 2) / 100, (cy - s / 2) / YTOP,
                        s / 100, s / YTOP])
    axg.set_xlim(0, 1)
    axg.set_ylim(0, 1)
    axg.set_axis_off()
    axg.patch.set_alpha(0)
    return axg


def card(ax, x, y, w, h, fill=CARD_FILL, edge=CARD_EDGE, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=0.9",
                                facecolor=fill, edgecolor=edge, linewidth=lw,
                                zorder=1))


def badge(ax, cx, cy, w, h, text, color, fs=FS_BADGE):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={h * .45}",
                                facecolor=color, edgecolor="none", zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", color="white",
            fontsize=fs, fontweight="bold", zorder=4)


# ----------------------------------------------------------------- content
# Ordered by weight, heaviest first, so reading down the figure walks down the
# strength ladder. One explanation line each, measured against the text column:
# matplotlib wraps in display space, so an auto-wrapped line re-breaks itself
# whenever the figure is resized and goes stale without anyone noticing.
CRITERIA = [
    dict(
        code="PVS1", pts=8, name="loss of function",
        question="Is the protein destroyed?",
        body="A stop codon, frameshift or splice error means no protein is made.",
        draw=lambda a: G.stop_bar(a, 0, 0, 1, FS.NAVY, pale="#a9c2d4"),
    ),
    dict(
        # +4 is the guideline's Strong tier, the same value `metrics.py` gives
        # PP3_Strong. It is not read from the cohort because this cohort has no
        # PS3 column to read it from.
        code="PS3", pts=4, name="functional assay",
        question="Was the variant tested in a laboratory?",
        body="An experiment measures whether the variant protein still works.",
        draw=lambda a: G.well_plate(a, 0, 0, 1, FS.BLUE),
    ),
    dict(
        code="PP3", pts=4, name="computational prediction",
        question="Do computer models predict harm?",
        body="A model predicts how damaging the amino-acid change is.",
        draw=lambda a: G.calibration_steps(a, 0, 0, 1, FS.BLUE, ink=INK,
                                           mute=MUTE, at=.997, fs=8.0),
    ),
    dict(
        code="PM1", pts=2, name="functional domain",
        question="Is it inside an important region?",
        body="The variant sits in a region where many known disease variants are "
             "found.",
        draw=lambda a: G.lollipop(a, 0, 0, 1, FS.STEEL, pale="#c2d9e8"),
    ),
    dict(
        code="PM2", pts=1, name="population rarity",
        question="Is it rare in the general population?",
        body="The variant is absent or very rare in population databases.",
        draw=lambda a: G.population(a, 0, 0, 1, "#4f8fbe", pale="#c8dceb"),
    ),
]

# Row geometry, in units.
X0, W = 1.5, 97.0
# Taller rows and summary panel: the poster column this sits in was ending
# short of its neighbours, and stretching the CONTENT is the only way to fill
# that without distorting the image. YTOP above is derived from these — the
# assert further down is what keeps the two in step.
ROW_H, GAP, SUM_H, MARGIN = 16.0, 1.4, 29.0, 1.6
# The icon is capped by the horizontal room before the text column, not by
# the row height, so it does not take the full row scale factor.
X_PIC, PIC_S = 6.3, 10.4              # picture centre-x and side
X_TEXT, X_END = 13.8, X0 + W - 2.0    # text column
Y_HEAD, Y_RULE = 12.2, 9.5            # header baseline, hairline
Y_Q, Y_BODY = 6.3, 2.5                # question, explanation


def draw_row(fig, ax, y0, c):
    card(ax, X0, y0, W, ROW_H)
    col = WEIGHT_COLOR[c["pts"]]

    c["draw"](square_panel(fig, X0 + X_PIC, y0 + ROW_H / 2, PIC_S))

    badge(ax, X_TEXT + 4.6, y0 + Y_HEAD, 9.2, 3.4, c["code"], col)
    ax.text(X_TEXT + 10.6, y0 + Y_HEAD, c["name"], fontsize=FS_NAME, color=INK,
            fontweight="bold", va="center")
    # The bar carries the strength visually; the tier WORD lives in the foot
    # line, where there is room for it. Setting both here took 30 characters of
    # header and pushed the bar into the criterion name.
    G.weight_bar(ax, X_END - 27.5, y0 + Y_HEAD - 1.1, 15.0, 2.2,
                 c["pts"], 8, col, mark=int(_ACMG_PATH_MIN))
    ax.text(X_END, y0 + Y_HEAD, f"+{c['pts']} point{'s' if c['pts'] > 1 else ''}",
            ha="right", va="center", fontsize=FS_PTS, color=col,
            fontweight="bold")
    ax.plot([X_TEXT, X_END], [y0 + Y_RULE] * 2, color=CARD_EDGE, lw=0.9,
            zorder=2)

    ax.text(X_TEXT, y0 + Y_Q, c["question"], fontsize=FS_Q, color=INK,
            fontweight="bold", va="center")
    ax.text(X_TEXT, y0 + Y_BODY, c["body"], fontsize=FS_BODY, color=MUTE,
            va="center")


def draw_summary(ax, y0, ex):
    """The sixth row: the rule that turns the five criteria into one answer.

    The stack is drawn heaviest-first, which for this variant is also the order
    that reaches the threshold soonest. That is the point of showing it at all:
    the same three pieces in an arbitrary order give the same total and would
    imply the opposite reading.
    """
    card(ax, X0, y0, W, SUM_H, fill="white", edge="#c8d5df")
    ax.text(X_TEXT, y0 + 25.4, "From points to a classification",
            fontsize=FS_NAME, color=INK, fontweight="bold", va="center")
    ax.plot([X_TEXT, X_END], [y0 + 22.6] * 2, color="#dbe4eb", lw=0.9, zorder=2)
    ax.text(X_TEXT, y0 + 20.0,
            "Points can be positive or negative; no single criterion decides "
            "\u2014 the total does.",
            fontsize=FS_BODY, color=MUTE, va="center")
    # Whose scale this is. Asked and worth answering on the figure itself: the
    # criteria are the ACMG/AMP standard (Richards 2015) and the point values
    # and cutoffs are ClinGen's Bayesian calibration of it (Tavtigian 2020),
    # quoted in `metrics.acmg_points_curve`. Neither was fitted to this data.
    ax.text(X_TEXT, y0 + 17.2,
            "Official ACMG/AMP standard  ·  ClinGen point scale "
            "(Tavtigian 2020)",
            fontsize=FS_SMALL, color=MUTE, style="italic", va="center")

    lo, hi = -4.0, 10.0
    bx0, bx1 = X_TEXT, X_END

    def X(p):
        return bx0 + (bx1 - bx0) * (p - lo) / (hi - lo)

    ybar, hbar = y0 + 10.6, 3.4
    for a, b, fc, lab in (
        (lo, _ACMG_BEN_MAX, "#dce7ef", f"benign (LB/B)  ≤ {_ACMG_BEN_MAX:.0f}"),
        (_ACMG_BEN_MAX, _ACMG_PATH_MIN, "#f2f5f7", "uncertain (VUS)"),
        (_ACMG_PATH_MIN, hi, FS.MIST, f"pathogenic (LP/P)  ≥ +{_ACMG_PATH_MIN:.0f}"),
    ):
        ax.add_patch(FancyBboxPatch((X(a), ybar), X(b) - X(a), hbar,
                                    boxstyle="round,pad=0,rounding_size=0",
                                    facecolor=fc, edgecolor="white",
                                    linewidth=1.0, zorder=2))
        ax.text((X(a) + X(b)) / 2, ybar + hbar / 2, lab, ha="center",
                va="center", fontsize=FS_SMALL + 0.5, color=INK, zorder=3)

    ystk, hstk = y0 + 2.4, 3.6
    ax.text(bx0, y0 + 7.6,
            f"{ex['label']}   ·   total {sum(ex['pts'].values()):+.0f}"
            "   →   likely pathogenic",
            fontsize=FS_BODY, color=INK, fontweight="bold", va="center")
    run = 0.0
    for ch in ex["order"]:
        p = ex["pts"][ch]
        ax.add_patch(FancyBboxPatch((X(run), ystk), X(run + p) - X(run), hstk,
                                    boxstyle="round,pad=0,rounding_size=0",
                                    facecolor=WEIGHT_COLOR[int(p)],
                                    edgecolor="white", linewidth=1.0, zorder=3))
        # A one-point block is 0.65 in wide: "PM2 +1" does not fit inside it,
        # and setting it outside collides with the threshold line that lands on
        # that block's edge. The code alone fits, and the scale states the value.
        wide = X(run + p) - X(run) > 8.0
        ax.text((X(run) + X(run + p)) / 2, ystk + hstk / 2,
                f"{ch} +{p:.0f}" if wide else ch, ha="center", va="center",
                fontsize=FS_SMALL + 0.5 if wide else FS_SMALL - 1.5,
                color="white", fontweight="bold", zorder=4)
        run += p
    # Two segments, not one: the threshold line's job is to line the band's +6
    # edge up with the stack's, and drawn as a single rule it strikes through
    # the label sitting between them.
    for a, b in ((ybar - 1.0, ybar + hbar + 0.4), (ystk - 0.9, ystk + hstk + 1.0)):
        ax.plot([X(_ACMG_PATH_MIN)] * 2, [a, b], color=INK, lw=1.1,
                ls=(0, (3, 2)), zorder=5)


def assert_text_fits(fig, ax):
    """No label may run past the card edge.

    Hand-measured line lengths are only as good as the last time someone
    measured them, and this figure is set at poster scale where an overrun is
    not a small ugliness: `bbox_inches="tight"` grows the saved image to contain
    it, which silently changes the figure's aspect and so its height in the
    poster column. That is how a 91-character summary line made this figure
    0.25 in taller than the layout asked for. Measure with the real renderer
    rather than trusting the copy.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    right = ax.transData.transform((X0 + W - 0.6, 0))[0]
    left = ax.transData.transform((X0 + 0.6, 0))[0]
    bad = [(t.get_text()[:60], round(t.get_window_extent(r).x1 - right, 1))
           for t in ax.texts
           if t.get_window_extent(r).x1 > right
           or t.get_window_extent(r).x0 < left]
    assert not bad, f"text outside the card (px past edge): {bad}"


# -------------------------------------------------------------------- main
def load_example(cohort_path):
    """Read the worked example's per-criterion points off the real cohort."""
    if not os.path.exists(cohort_path):
        raise SystemExit(
            f"missing cohort: {cohort_path}\n"
            "This figure quotes a real worked example, so it cannot be drawn "
            "from the synthetic cohort without printing numbers that are not "
            "the paper's. Build the real cohort first "
            "(docs/variant/real_data.md).")
    df = pd.read_parquet(cohort_path)
    coh = Cohort(df, VARIANT_DOMAIN)
    i = int(df.index[df["variant_id"] == EXAMPLE_ID][0])
    pts = {ch: float(v[i]) for ch, v in _acmg_channel_points(coh).items()}
    scoring = {c: p for c, p in pts.items() if p > 0}
    name = str(df.loc[i, "clinvar_name"])
    short = name.split("(")[-1].rstrip(")") if "p." in name else name
    return dict(
        # Only the criteria that actually score appear in the stack. A zero
        # would draw a zero-width block, which reads as a piece of evidence that
        # was somehow weightless rather than one that did not apply here.
        pts=scoring,
        order=sorted(scoring, key=lambda c: -scoring[c]),
        label=f"{df.loc[i, 'gene']} {short}",
        n=len(df),
        defined={ch: int(df[f"{ch}__defined"].sum())
                 for ch in VARIANT_DOMAIN.channels},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/variant/channels.png")
    ap.add_argument("--cohort", default="data/sample/cohort_full_real.parquet")
    args = ap.parse_args()
    os.chdir(_ROOT)
    FS.use()

    # ---- tie the figure to the code, so it cannot drift from it ----------
    # The figure prints point values, bin edges and thresholds: exactly the
    # things a guideline revision would silently invalidate, so each is checked
    # against its source.
    assert set(VARIANT_DOMAIN.channels) == {"PVS1", "PM2", "PM1", "PP3"}, \
        VARIANT_DOMAIN.channels
    assert [c["code"] for c in CRITERIA] == ["PVS1", "PS3", "PP3", "PM1", "PM2"]
    assert (_ACMG_PP3_SUP, _ACMG_PP3_MOD, _ACMG_PP3_STRONG) == (.906, .972, .99)
    assert (_ACMG_PATH_MIN, _ACMG_BEN_MAX) == (6.0, -1.0)
    # The rows must exactly fill the frame, or the figure prints with a blank
    # band that bbox_inches="tight" will not crop away.
    assert abs(MARGIN * 2 + 5 * ROW_H + 5 * GAP + SUM_H - YTOP) < 1e-6, YTOP

    ex = load_example(args.cohort)
    assert ex["n"] == 491, ex["n"]
    assert ex["pts"] == {"PP3": 4.0, "PM1": 2.0, "PM2": 1.0}, ex["pts"]
    assert ex["order"] == ["PP3", "PM1", "PM2"], ex["order"]
    # The stack must cross the threshold before its last block, or the summary
    # row is describing a different picture from the one drawn.
    assert sum(ex["pts"][c] for c in ex["order"][:-1]) >= _ACMG_PATH_MIN, ex

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor(FS.PARADIGM["paper"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, YTOP)
    ax.set_autoscale_on(False)

    y = YTOP - MARGIN - ROW_H
    for c in CRITERIA:
        draw_row(fig, ax, y, c)
        y -= ROW_H + GAP
    draw_summary(ax, y + ROW_H - SUM_H, ex)

    assert_text_fits(fig, ax)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"written: {args.out}   ({FIG_W:.2f} x {FIG_H:.2f} in, poster scale)")
    print(f"  n={ex['n']}  defined={ex['defined']}")
    print(f"  example={ex['label']}  pts={ex['pts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
