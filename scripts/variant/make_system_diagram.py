#!/usr/bin/env python3
"""The hero/system diagram: same case, agent behavior differs, outcome
differs -- once for the frozen pool, once for live tools.

WHY THIS VERSION
-----------------
The first draft (a 6-box flowchart + a cohort-fanout panel) was correct but
too dense to read at a glance. Bo's request, and the reference image
supplied alongside it, ask for a simple, iconic, two-row "same input ->
agent -> behavior branches -> outcome" template, exactly the shape of the
classic "same situation, different internal state, different outcome"
picture (their example: calm vs. anxious human, confident vs. uncertain
LLM). This is that template, applied to what this paper's two paradigms
actually show:

  (a) FROZEN: the same evidence pool, budget spent well vs. budget spent
      past the optimum -> ranking AUC 0.917 (k=2) vs. 0.807 (k=4).
  (b) LIVE: the same real case, agent stops once sufficient vs. keeps
      querying past sufficiency -> no accuracy gain for the extra 62% of
      cases that over-acquire.

ICONS, NOT AN ICON PACK
-------------------------
The reference image uses downloaded flat-illustration clipart (emoji faces,
a stock robot icon). This repo's figures never do that -- figglyphs.py's own
docstring explains why (licence + reproducibility: every glyph must be
regenerable from this repo alone). So the agent here is a simple line-art
robot bust in the same hand-drawn-vector style as figglyphs.doctor().

RED/GREEN, DELIBERATELY SOFTENED
-----------------------------------
The reference uses saturated red/green for bad/good. figstyle.py's palette
comment explains why this repo avoids that pair (least colour-blind-
accessible common combination). The verdict here is carried by the
check/slash glyphs already used across this figure set, with only a light
tint behind each box -- shape carries the claim, not hue, so it survives
greyscale print like every other figure in this tree.

LAYOUT NOTE (read before editing coordinates)
------------------------------------------------
Every position below is a concrete, hand-checked number, not a formula
applied to an arbitrary row_center -- an earlier formulaic version let the
outcome boxes drift past the canvas edge and into the other row before
anyone looked at the render. If you change a row's vertical position, re-
render and LOOK before trusting the arithmetic.

USAGE
    python scripts/variant/make_system_diagram.py
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench import figglyphs as G  # noqa: E402
from pwkbench import figstyle as FS  # noqa: E402

GOOD_FILL = "#eaf1f6"   # pale blue tint -- "good" outcome
BAD_FILL = "#fbeceb"    # pale accent tint -- "bad" outcome


def _box(ax, x0, y0, x1, y1, edge, fill, lw=1.8, r=0.10, z=2):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z))


def _agent(ax, cx, cy, s, color, live=False, z=5):
    """A simple line-art agent bust, same visual family as
    figglyphs.doctor(). `live=True` adds two short antenna ticks."""
    lw = max(1.9, s * .08)
    bw, bh = s * .78, s * .60
    ax.add_patch(FancyBboxPatch((cx - bw / 2, cy - s * .42), bw, bh,
                                boxstyle=f"round,pad=0,rounding_size={bw * .22}",
                                facecolor="white", edgecolor=color,
                                linewidth=lw, zorder=z))
    hr = s * .30
    hy = cy + s * .18
    ax.add_patch(FancyBboxPatch((cx - hr, hy - hr * .85), hr * 2, hr * 1.7,
                                boxstyle=f"round,pad=0,rounding_size={hr * .5}",
                                facecolor="white", edgecolor=color,
                                linewidth=lw, zorder=z + 1))
    for dx in (-hr * .42, hr * .42):
        ax.add_patch(Circle((cx + dx, hy), hr * .13, facecolor=color,
                            edgecolor="none", zorder=z + 2))
    ax.plot([cx - hr * .30, cx + hr * .30], [hy - hr * .40, hy - hr * .40],
            color=color, lw=lw * .8, solid_capstyle="round", zorder=z + 2)
    if live:
        for dx in (-hr * .55, hr * .55):
            ax.plot([cx + dx, cx + dx * 1.35], [hy + hr * .85, hy + hr * 1.35],
                    color=color, lw=lw * .8, solid_capstyle="round", zorder=z)
            ax.add_patch(Circle((cx + dx * 1.35, hy + hr * 1.35), s * .035,
                                facecolor=color, edgecolor="none", zorder=z))


def _outcome(ax, x, y, w, h, good, title, sub, good_color, z=4):
    fill = GOOD_FILL if good else BAD_FILL
    edge = good_color if good else FS.ACCENT
    _box(ax, x, y, x + w, y + h, edge, fill, lw=1.8, z=z)
    r = h * .30
    cx, cy = x + r * 1.25, y + h / 2
    if good:
        G.check(ax, cx, cy, r, good_color, z=z + 2)
    else:
        G.slash(ax, cx, cy, r, FS.ACCENT, z=z + 2)
    tx = x + r * 2.7
    ax.text(tx, y + h * .64, title, ha="left", va="center", fontsize=13.5,
            fontweight="bold", color=FS.INK, zorder=z + 2)
    ax.text(tx, y + h * .27, sub, ha="left", va="center", fontsize=10.8,
            color=FS.MUTE, zorder=z + 2)


def _evidence_pool_glyph(ax, cx, cy, s, color):
    """Frozen row's input: the four pre-fetched ACMG evidence chips,
    reusing the same per-criterion glyphs as channels.png."""
    from pwkbench.figglyphs import bars, curve, domain, truncation
    chips = [truncation, bars, curve, domain]
    chip_s = s * .46
    for i, fn in enumerate(chips):
        x = cx - chip_s * 1.05 + (i % 2) * chip_s * 1.05
        y = cy + chip_s * .55 - (i // 2) * chip_s * 1.05
        _box(ax, x, y, x + chip_s, y + chip_s, color, "white", lw=1.3,
            r=0.05, z=4)
        fn(ax, x + chip_s * .08, y + chip_s * .08, chip_s * .84, color,
          filled=True, z=5)


def _live_query_glyph(ax, cx, cy, s, color):
    """Live row's input: a magnifying glass over a record -- querying a
    real external source, not a pre-fetched pool."""
    G.notes(ax, cx - s * .38, cy - s * .42, s * .76, color, filled=False, z=4)
    gx, gy, gr = cx + s * .16, cy - s * .02, s * .26
    ax.add_patch(Circle((gx, gy), gr, facecolor="white", edgecolor=color,
                        linewidth=max(1.8, s * .09), zorder=6))
    ax.plot([gx + gr * .70, gx + gr * 1.30], [gy - gr * .70, gy - gr * 1.30],
            color=color, lw=max(1.8, s * .09), solid_capstyle="round", zorder=6)


def _branch(ax, bx, by, tx, ty, label, above, color):
    """One straight branch arrow + its action label, offset off the line
    so the text never sits on top of the arrowhead or the box edge."""
    ax.annotate("", xy=(tx, ty), xytext=(bx, by),
                arrowprops=dict(arrowstyle="-|>", color=FS.MUTE, lw=1.5,
                                shrinkA=2, shrinkB=4))
    mx, my = (bx + tx) * .5, (by + ty) * .5
    ax.text(mx, my + (0.20 if above else -0.20), label, ha="center",
            va="bottom" if above else "top", fontsize=10.6,
            color=color, fontweight="bold", zorder=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/variant/system_diagram.png")
    args = ap.parse_args()
    os.chdir(_ROOT)
    FS.use()

    frozen_c = FS.PARADIGM["frozen"]
    live_c = FS.PARADIGM["live"]

    fig, ax = plt.subplots(figsize=(11.6, 8.8))
    fig.patch.set_facecolor(FS.PARADIGM["paper"])
    ax.set_facecolor(FS.PARADIGM["paper"])
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 8.8)
    ax.set_axis_off()

    ax.text(5.8, 8.60, "Same case, different acquisition behavior, "
            "different outcome",
            ha="center", va="top", fontsize=16,
            fontweight="bold", color=FS.INK)

    # Column x-positions shared by both rows.
    ROW_LABEL_X = 0.18
    INPUT_X = 1.55
    AGENT_X = 3.15
    BRANCH_X = 3.95
    BOX_X = 6.35
    BOX_W = 4.85
    BOX_H = 1.05

    def draw_row(top_by, bot_by, label, color, input_label, input_glyph,
                good_action, bad_action, good_title, good_sub,
                bad_title, bad_sub, live):
        # Every y below is anchored to the two box positions the caller
        # supplies, NOT derived from a single centre -- see the module
        # docstring's layout note.
        y_center = (top_by + BOX_H + bot_by) / 2

        ax.text(ROW_LABEL_X, y_center, label, ha="left", va="center",
                fontsize=14.5, fontweight="bold", color=color, rotation=90)

        input_glyph(ax, INPUT_X, y_center + 0.10, 1.30, color)
        ax.text(INPUT_X, y_center - 0.98, input_label, ha="center",
                va="top", fontsize=10.6, color=FS.MUTE, style="italic")

        G.arrow_chevron(ax, INPUT_X + 0.62, y_center - 0.17, 0.55, 0.34,
                        FS.MUTE, z=2)

        _agent(ax, AGENT_X, y_center, 1.45, color, live=live, z=5)
        ax.text(AGENT_X, y_center - 0.98, "agent", ha="center", va="top",
                fontsize=10.6, color=FS.MUTE, style="italic")

        _branch(ax, BRANCH_X, y_center, BOX_X, top_by + BOX_H / 2,
                good_action, True, FS.NAVY)
        _branch(ax, BRANCH_X, y_center, BOX_X, bot_by + BOX_H / 2,
                bad_action, False, FS.ACCENT)

        _outcome(ax, BOX_X, top_by, BOX_W, BOX_H, True, good_title, good_sub,
                color)
        _outcome(ax, BOX_X, bot_by, BOX_W, BOX_H, False, bad_title, bad_sub,
                color)

    # Row A box band: 4.65 -> 7.90 (title sits at 8.60, so a 0.70 clearance
    # above the top box; row A's own bottom box floor is 4.65).
    draw_row(top_by=6.85, bot_by=4.65, label="(a) Frozen", color=frozen_c,
            input_label="same evidence pool", input_glyph=_evidence_pool_glyph,
            good_action="stop at k=2",
            bad_action="keep going to k=4",
            good_title="AUC 0.917", good_sub="ranking near its peak",
            bad_title="AUC 0.807", bad_sub="ranking degrades (RQ1)",
            live=False)

    # Separator sits in the 0.35-wide gap between row A's box floor (4.65)
    # and row B's box ceiling (3.70).
    ax.plot([0.4, 11.2], [4.28, 4.28], color=FS.GRID, lw=1.2, ls=(0, (4, 3)))

    # Row B box band: 0.45 -> 3.70, mirroring row A's shape, shifted down.
    draw_row(top_by=2.65, bot_by=0.45, label="(b) Live", color=live_c,
            input_label="same real case", input_glyph=_live_query_glyph,
            good_action="stop once sufficient",
            bad_action="query past sufficiency",
            good_title="Efficient", good_sub="decision quality unchanged",
            bad_title="No accuracy gain", bad_sub="extra queries, extra cost",
            live=True)

    fig.savefig(args.out, bbox_inches="tight", facecolor=fig.get_facecolor(),
                dpi=200)
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
