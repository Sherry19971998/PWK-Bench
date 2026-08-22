#!/usr/bin/env python3
"""The problem, as one picture: several sources feed one decision.

WHAT IT HAS TO SAY, AND WHAT IT MUST NOT
----------------------------------------
A clinical decision draws on several sources at once, and each one costs time and
money. That is the setup. The FINDING — that the agent keeps acquiring past the
point where the evidence already settled the case — belongs to the hero figure,
not here. So this panel shows no verdict, no cost figures, no counts: it
establishes why acquisition is a decision at all, and stops.

IT IS DELIBERATELY NOT ABOUT VARIANTS
-------------------------------------
This is the opening panel, so it has to motivate the problem for clinical
decision-making generally; the variant cohort is the instrument, introduced one
section later. An earlier version drew exactly the four ACMG channels this study
happens to use, which announced the specific experiment before the reader knew
what question it answered. The sources here are ordinary clinical ones and the
count is six, not four, so no one reads them as our evidence set.

The centre is the AGENT — the thing that decides — not the call it produces.
Labelling the hub with the outcome made the arrows look like they were being
combined into an answer, when what the panel is about is who is choosing to
acquire them. The doctor glyph is `figglyphs.doctor`, the same one the hero
figure uses, so the two panels read as the same character.

USAGE
    python scripts/variant/make_problem_figure.py
"""
import argparse, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench import figglyphs as G                          # noqa: E402
from pwkbench import figstyle as FS                          # noqa: E402

INK = FS.INK

# Icons only, no captions: the panel is a schematic of "several sources feed one
# decision", and the bullet list beside it already names what they are. Labelling
# each box repeated that list inside the figure and made a simple diagram look
# like a taxonomy.
SOURCES = ["notes", "vial", "scan", "helix", "bars", "curve"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/variant/problem.png")
    args = ap.parse_args()
    os.chdir(_ROOT)
    FS.use()

    # Wide and short on purpose. Six sources stacked three-a-side wants a tall
    # canvas, but this panel sits at the head of a poster column, where every
    # inch of height it takes is an inch the whole board's type size pays for —
    # going from a 11x5.0 to an 11x6.3 canvas cost the solved type scale 1.65
    # -> 1.28. Widening instead keeps the same six boxes at 4.8 in of column.
    fig, ax = plt.subplots(figsize=(9.2, 2.89))
    fig.patch.set_facecolor(FS.PARADIGM["paper"])
    ax.set_facecolor(FS.PARADIGM["paper"])
    ax.set_xlim(0, 11.0); ax.set_ylim(0, 3.45); ax.set_axis_off()

    cx, cy = 5.5, 1.75
    R = 0.80

    # --- the agent, centred ----------------------------------------------
    ax.add_patch(Circle((cx, cy), R, facecolor=FS.PALE,
                        edgecolor=FS.NAVY, lw=2.4, zorder=3))
    G.doctor(ax, cx, cy - 0.05, 0.64, FS.NAVY, z=5)
    ax.text(cx, cy - R - 0.10, "AGENT", ha="center", va="top",
            fontsize=13, fontweight="bold", color=FS.NAVY)

    # --- the sources, three per side, feeding in --------------------------
    glyph = {"bars": G.bars, "curve": G.curve, "vial": G.vial,
             "notes": G.notes, "scan": G.scan, "helix": None}
    half = 0.50
    for i, kind in enumerate(SOURCES):
        left = i < 3
        x = 1.22 if left else 9.78
        y = (2.85, 1.75, 0.65)[i % 3]
        colour = FS.BLUE if i % 2 == 0 else FS.STEEL
        ax.add_patch(FancyBboxPatch((x - half, y - half), half * 2, half * 2,
                                    boxstyle="round,pad=0,rounding_size=0.16",
                                    facecolor="white", edgecolor=FS.SKY,
                                    lw=1.8, zorder=3))
        if kind == "helix":
            # helix is drawn along a span, not from a corner like the others
            G.helix(ax, x - 0.32, x + 0.32, y, 0.14, colour, FS.SKY,
                    turns=1.5, lw=2.0, z=5)
        else:
            glyph[kind](ax, x - 0.26, y - 0.26, 0.52, colour, z=5)

        # arrow from the box edge toward the circle, stopping short of both
        p0 = np.array([x + (half if left else -half), y])
        d = np.array([cx, cy]) - p0
        d = d / np.linalg.norm(d)
        ax.annotate("", xy=np.array([cx, cy]) - d * (R + 0.10),
                    xytext=p0 + d * 0.10,
                    arrowprops=dict(arrowstyle="-|>", color=FS.OFF, lw=2.2,
                                    shrinkA=0, shrinkB=0), zorder=2)

    # No title inside the figure: the poster's own "The problem" header sits
    # directly above it, and a second heading two inches away read as a caption
    # for a caption.

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor=fig.get_facecolor(),
                dpi=200)
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
