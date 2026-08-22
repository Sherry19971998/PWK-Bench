#!/usr/bin/env python3
"""Figure 1: the benchmark overview -- compact horizontal row, drawn
NATIVELY at print size.

WHY NATIVE SIZE, NOT A RESCALED LARGE CANVAS
-------------------------------------------------
Two earlier attempts tried to reuse a large (15.6in), richly-labelled
canvas and get it print-legible after \\includegraphics[width=\\textwidth]
scaling -- first by shrinking the physical canvas alone (broke: box widths
stayed sized for the old smaller relative text and overflowed), then by
shrinking the canvas AND bumping every fontsize by a fixed multiplier
(still broke: the multiplier wasn't tied to what any specific container
actually had room for). Both failed for the same reason: text width in
points and box width in data-units don't co-scale automatically, and
retrofitting a fix onto a dense, richly-worded layout just moves the
overflow around.

This version instead draws at the ACTUAL target size (~7.2in wide, IEEE
\\figure* \\textwidth) with font sizes chosen directly as their real
printed point size -- no hidden multiplier anywhere. To fit that width
without overflowing, on-figure text is cut to the minimum that still reads
as a diagram (short labels, codes, one-line annotations); the mechanism
detail that a denser version tried to cram in (PVS1's free-vs-costed
status, what "sufficient" means, the evaluation axis list) moves to the
caption, which is exactly where prose explanation belongs relative to a
diagram.

WHY LIVE, NOT FROZEN
-------------------------
system_diagram.png answers "what did we FIND" for both paradigms; this
figure answers "what does PWK-Bench DO", and only live's mechanics fit one
merged row (PVS1 is free and never drawn as a tool call here; frozen has
no such free channel and no native stop -- see pwkbench/live/metrics_live.py
FREE=("PVS1",) vs. ACQUIRABLE=("PM2","PP3","PM1")). Frozen's own mechanics
are system_diagram.png's job.

THE TRAJECTORY IS ILLUSTRATIVE
-----------------------------------
PM2-first is real (first tool queried in 442/491 live cases). The
sufficiency flag and PS3's mark are the qualitative shape of the 62%
over-acquisition finding, not one logged case -- the caption says so.

USAGE
    python scripts/variant/make_overview_figure.py
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.patheffects import SimplePatchShadow, Normal

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench import figglyphs as G  # noqa: E402
from pwkbench import figstyle as FS  # noqa: E402

SHADOW = [SimplePatchShadow(offset=(0.018, -0.022), shadow_rgbFace="#1c2b38",
                            alpha=0.18), Normal()]
CARD = "#eef2f5"
PANEL = "#fbfcfd"


def _panel(ax, x0, y0, x1, y1, fill=PANEL, edge=None, lw=0.9, r=0.055, z=2,
          shadow=True):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                       boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fill, edgecolor=edge if edge else "#dfe5ea",
                       linewidth=lw, zorder=z)
    if shadow:
        p.set_path_effects(SHADOW)
    ax.add_patch(p)


def _card_round(ax, cx, cy, r, fill=CARD, z=3, shadow=True):
    p = Circle((cx, cy), r, facecolor=fill, edgecolor="none", zorder=z)
    if shadow:
        p.set_path_effects(SHADOW)
    ax.add_patch(p)


def _agent(ax, cx, cy, s, color, z=5):
    """The AI-doctor hybrid: figglyphs.doctor()'s clinician bust (white
    coat, stethoscope, simple dot eyes -- the same silhouette pillars.png
    already uses for "the clinical decision-maker") plus two antenna ticks
    on top, the one addition that reads as "AI" rather than "human
    clinician". Coat + stethoscope answers "why does this look like a
    doctor"; the antenna answers "why is it acquiring evidence
    autonomously" -- neither alone was the right picture."""
    _card_round(ax, cx, cy, s * .62, z=z - 1)
    lw = max(0.9, s * .085)
    G.doctor(ax, cx, cy, s, color, blindfold=False, z=z, lw=lw)
    hr = s * .26
    hy = cy + s * .30
    top = hy + hr
    for dx in (-hr * .55, hr * .55):
        ax.plot([cx + dx * .55, cx + dx], [top, top + hr * .60],
                color=color, lw=lw * .8, solid_capstyle="round",
                zorder=z + 4)
        ax.add_patch(Circle((cx + dx, top + hr * .60), s * .032,
                            facecolor=color, edgecolor="none", zorder=z + 4))
        ax.add_patch(Circle((cx + dx * 1.3, hy + hr * 1.3), s * .035,
                            facecolor=color, edgecolor="none", zorder=z))


def _arrow(ax, x0, x1, y, z=3):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color="#9aa7b0", lw=1.1),
                zorder=z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/variant/overview.png")
    args = ap.parse_args()
    os.chdir(_ROOT)
    FS.use()

    live_c = FS.PARADIGM["live"]

    # Native print size: IEEE \figure* at \textwidth is ~7.2in. 1 data
    # unit == 1 real inch here, so every fontsize below IS its final
    # printed point size -- no multiplier, no rescale step to get wrong.
    W, H = 7.20, 2.85
    fig, ax = plt.subplots(figsize=(W, H))
    fig.patch.set_facecolor(FS.PARADIGM["paper"])
    ax.set_facecolor(FS.PARADIGM["paper"])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    YM = 1.56

    # ---- Slot 1: Clinical variant ------------------------------------------
    s1_x0, s1_x1 = 0.10, 1.42
    s1_y0, s1_y1 = YM - 0.44, YM + 0.44
    _panel(ax, s1_x0, s1_y0, s1_x1, s1_y1, fill=FS.PARADIGM["live_fill"], z=3)
    ax.text((s1_x0 + s1_x1) / 2, YM + 0.15, "Clinical\nvariant",
            ha="center", va="center", fontsize=9.8, fontweight="bold",
            color=FS.INK, zorder=4)
    ax.text((s1_x0 + s1_x1) / 2, YM - 0.28, "HGVS + gene",
            ha="center", va="center", fontsize=6.4, color=FS.MUTE,
            style="italic", zorder=4)

    _arrow(ax, s1_x1 + 0.03, s1_x1 + 0.20, YM)

    # ---- Slot 2: Agent ---------------------------------------------------
    s2_cx = s1_x1 + 0.20 + 0.28
    _agent(ax, s2_cx, YM + 0.10, 0.50, live_c, z=4)
    ax.text(s2_cx, YM - 0.34, "agent", ha="center", va="top", fontsize=7.4,
            fontweight="bold", color=FS.INK, zorder=4)

    a2_x0 = s2_cx + 0.28 + 0.03
    a2_x1 = a2_x0 + 0.62
    _arrow(ax, a2_x0, a2_x1, YM)
    ax.text((a2_x0 + a2_x1) / 2, YM + 0.06, "acquire \\&\nstop?",
            ha="center", va="bottom", fontsize=5.6, color=FS.MUTE,
            style="italic", zorder=3)

    # ---- Slot 3: Evidence chips == trajectory taken -------------------------
    s3_x0, s3_x1 = a2_x1 + 0.06, 5.55
    chip_r = 0.185
    n = 4
    cxs = [s3_x0 + (s3_x1 - s3_x0) * (i + 0.5) / n for i in range(n)]
    chip_y = YM

    chips = [
        (G.bars, live_c, "PM2", False),
        (G.curve, live_c, "PP3", False),
        (G.domain, live_c, "PM1", False),
        (G.vial, live_c, "PS3", True),
    ]
    for i, (cx, (glyph, color, label, dashed)) in enumerate(zip(cxs, chips)):
        _card_round(ax, cx, chip_y, chip_r, fill=CARD, z=3)
        if dashed:
            ax.add_patch(Circle((cx, chip_y), chip_r, facecolor="none",
                                edgecolor="#c3cbd1", linewidth=0.8,
                                linestyle=(0, (1.6, 1.3)), zorder=4))
        glyph(ax, cx - chip_r * .78, chip_y - chip_r * .78, chip_r * 1.56,
             "#97a3ab" if dashed else color, filled=not dashed, z=5)
        ax.text(cx, chip_y - chip_r - 0.09, label, ha="center", va="top",
                fontsize=7.6, fontweight="bold",
                color=FS.MUTE if dashed else FS.INK, zorder=4)
        if i < n - 1:
            ax.annotate("", xy=(cxs[i + 1] - chip_r - 0.02, chip_y),
                        xytext=(cx + chip_r + 0.02, chip_y),
                        arrowprops=dict(arrowstyle="-|>", color="#9aa7b0",
                                        lw=0.9), zorder=3)

    flag_x = (cxs[2] + cxs[3]) / 2
    flag_top = chip_y + chip_r + 0.14
    ax.plot([flag_x, flag_x], [chip_y + chip_r + 0.02, flag_top],
            color=FS.NAVY, lw=1.0, zorder=4)
    ax.text(flag_x, flag_top + 0.03, "sufficient", ha="center", va="bottom",
            fontsize=6.2, color=FS.NAVY, fontweight="bold", zorder=4)

    ax.text(cxs[3], chip_y - chip_r - 0.30, "62% unnecessary",
            ha="center", va="top", fontsize=6.0, color=FS.ACCENT,
            fontweight="bold", zorder=4)

    _arrow(ax, s3_x1 + 0.03, s3_x1 + 0.20, YM)

    # ---- Slot 4: Decision --------------------------------------------------
    s4_x0, s4_x1 = s3_x1 + 0.20, 7.10
    s4_y0, s4_y1 = YM - 0.44, YM + 0.44
    _panel(ax, s4_x0, s4_y0, s4_x1, s4_y1, fill=FS.PARADIGM["live_fill"], z=3)
    ax.text((s4_x0 + s4_x1) / 2, YM, "Pathogenic\n/ Benign",
            ha="center", va="center", fontsize=8.6, fontweight="bold",
            color=FS.INK, zorder=4)

    # ---- Bottom strip: PWK-Bench finding -------------------------------------
    ax.plot([0.10, 7.10], [YM - 0.72, YM - 0.72], color="#e2e7ea", lw=0.7)
    ax.text(3.60, YM - 0.98,
            "More evidence is not always better -- PWK-Bench measures "
            "evidence choice, ordering, sufficiency \\& over-acquisition, "
            "not just accuracy",
            ha="center", va="center", fontsize=6.8, color=FS.ACCENT,
            fontweight="bold", zorder=4)

    fig.savefig(args.out, bbox_inches="tight", facecolor=fig.get_facecolor(),
                dpi=400)
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
