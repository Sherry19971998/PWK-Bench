"""Hand-drawn vector glyphs for the figures.

WHY NOT AN ICON PACK
--------------------
Downloaded icon sets carry licence obligations that a paper figure should not
inherit, and an external asset breaks the property every other artifact in this
tree has: a figure must be regenerable from this repository alone. Everything
here is matplotlib paths, so the glyphs are resolution-independent, vector in
the PDF, and version-controlled alongside the numbers they illustrate.

Each glyph is drawn in a local 0..1 box and placed by (x, y, size), so a caller
composes a scene without doing trigonometry. `color` is the ink; `filled`
distinguishes acquired from not-yet-acquired evidence, which is the one visual
distinction that carries meaning rather than style.
"""
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Wedge


def _box(ax, x, y, s, fc, ec, lw=1.6, r=0.16, z=4):
    ax.add_patch(FancyBboxPatch((x, y), s, s,
                                boxstyle=f"round,pad=0,rounding_size={r * s}",
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))


def helix(ax, x0, x1, cy, amp, color_a="#4a7fa5", color_b="#9dbdd6",
          bases=None, turns=3.2, lw=2.6, z=3):
    """DNA duplex: two phase-shifted strands with nucleotide-coloured rungs."""
    bases = bases or {"A": "#3f8f4f", "C": "#2f6fb0", "G": "#e0a020", "T": "#c0392b"}
    t = np.linspace(0, turns * 2 * np.pi, 400)
    x = np.linspace(x0, x1, 400)
    for ph, c in ((0, color_a), (np.pi, color_b)):
        ax.plot(x, cy + amp * np.sin(t + ph), color=c, lw=lw, zorder=z,
                solid_capstyle="round")
    seq = list("ACGT" * 6)
    for i, ti in enumerate(np.linspace(0, turns * 2 * np.pi, 18)):
        xi = x0 + (x1 - x0) * ti / (turns * 2 * np.pi)
        ax.plot([xi, xi], [cy + amp * np.sin(ti), cy + amp * np.sin(ti + np.pi)],
                color=bases[seq[i]], lw=1.4, alpha=.85, zorder=z - 1)


# One stroke weight for every glyph, scaled to the icon box, so a row of icons
# has an even visual weight the way a professional icon set does.
def _lw(s, filled=False):
    base = max(1.8, s * .26)
    return base if filled else base * 1.25


def vial(ax, x, y, s, color, filled=False, z=5):
    """Assay tube — PS3 functional evidence."""
    w, lw = s * .34, _lw(s, filled)
    ax.add_patch(FancyBboxPatch((x + s * .33, y + s * .12), w, s * .60,
                                boxstyle=f"round,pad=0,rounding_size={w * .48}",
                                facecolor=color if filled else "none",
                                edgecolor=color, linewidth=lw, zorder=z))
    # Meniscus: present in both states so the glyph reads as a vial, not a pill.
    ax.plot([x + s * .36, x + s * .64], [y + s * .40, y + s * .40],
            color="white" if filled else color, lw=lw * .8,
            solid_capstyle="round", zorder=z + 1)
    ax.plot([x + s * .26, x + s * .74], [y + s * .78, y + s * .78],
            color=color, lw=lw, solid_capstyle="round", zorder=z + 1)


def bars(ax, x, y, s, color, filled=False, z=5):
    """Allele-frequency histogram — PM2 population evidence."""
    lw = _lw(s, filled)
    ax.plot([x + s * .14, x + s * .86], [y + s * .14, y + s * .14],
            color=color, lw=lw, solid_capstyle="round", zorder=z)
    for i, h in enumerate([.26, .58, .40, .20]):
        bx = x + s * (.20 + i * .17)
        ax.add_patch(FancyBboxPatch((bx, y + s * .14), s * .11, s * h,
                                    boxstyle="round,pad=0,rounding_size=0.006",
                                    facecolor=color if filled else "none",
                                    edgecolor=color, linewidth=lw * .8, zorder=z))


def curve(ax, x, y, s, color, filled=False, z=5):
    """Sigmoid on an axis — PP3 in-silico score."""
    lw = _lw(s, filled)
    ax.plot([x + s * .14, x + s * .14, x + s * .88],
            [y + s * .82, y + s * .14, y + s * .14],
            color=color, lw=lw * .8, solid_capstyle="round",
            solid_joinstyle="round", zorder=z)
    t = np.linspace(-6, 6, 90)
    cx = x + s * (.20 + .62 * (t + 6) / 12)
    cy = y + s * (.20 + .54 / (1 + np.exp(-t)))
    ax.plot(cx, cy, color=color, lw=lw, zorder=z + 1, solid_capstyle="round")
    if filled:
        ax.fill_between(cx, y + s * .16, cy, color=color, alpha=.22, zorder=z)


def domain(ax, x, y, s, color, filled=False, z=5):
    """Protein track with one annotated domain — PM1 positional evidence."""
    lw = _lw(s, filled)
    ax.plot([x + s * .10, x + s * .90], [y + s * .50, y + s * .50],
            color=color, lw=lw * .7, zorder=z, solid_capstyle="round")
    ax.add_patch(FancyBboxPatch((x + s * .32, y + s * .33), s * .34, s * .34,
                                boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor=color if filled else "none",
                                edgecolor=color, linewidth=lw, zorder=z + 1))
    # Residue tick: the thing PM1 is actually about.
    ax.plot([x + s * .49, x + s * .49], [y + s * .58, y + s * .72],
            color=color, lw=lw * .7, solid_capstyle="round", zorder=z + 2)


def check(ax, cx, cy, r, color, z=6):
    """Outcome mark — a correct guideline call."""
    ax.add_patch(Circle((cx, cy), r, facecolor=color, edgecolor="none", zorder=z))
    ax.plot([cx - r * .42, cx - r * .10, cx + r * .46],
            [cy + r * .04, cy - r * .34, cy + r * .38],
            color="white", lw=r * 3.4, solid_capstyle="round",
            solid_joinstyle="round", zorder=z + 1)


def coins(ax, x, y, s, n, color, z=5):
    """A stack of n cost units — the price of the acquisition, drawn not stated."""
    for i in range(n):
        ax.add_patch(Circle((x, y + i * s * .46), s * .5, facecolor=color,
                            edgecolor="white", linewidth=1.6, zorder=z + i))


def clock(ax, cx, cy, r, color, frac=.75, z=5):
    """Elapsed-time dial — the other axis a clinician pays on."""
    ax.add_patch(Circle((cx, cy), r, facecolor="white", edgecolor=color,
                        linewidth=2.0, zorder=z))
    ax.add_patch(Wedge((cx, cy), r * .92, 90 - 360 * frac, 90,
                       facecolor=color, alpha=.35, zorder=z + 1))


def slash(ax, cx, cy, r, color, z=8):
    """Prohibition ring — marks the acquisition that returned nothing."""
    ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=color,
                        linewidth=3.0, zorder=z))
    d = r * .70
    ax.plot([cx - d, cx + d], [cy + d, cy - d], color=color, lw=3.0,
            solid_capstyle="round", zorder=z + 1)


def arrow_chevron(ax, x, y, w, h, color, z=3, alpha=1.0):
    """Flat chevron — a flow step that does not need a line to be read as one."""
    ax.add_patch(Polygon([[x, y], [x + w * .72, y], [x + w, y + h / 2],
                          [x + w * .72, y + h], [x, y + h], [x + w * .28, y + h / 2]],
                         closed=True, facecolor=color, edgecolor="none",
                         zorder=z, alpha=alpha))


def doctor(ax, cx, cy, s, color, blindfold=False, z=5, lw=None):
    """Line-art clinician bust: head, coat, stethoscope.

    `blindfold=True` draws a band over the eyes. That state is not decoration —
    it is how the frozen block's information condition is depicted: the agent is
    shown no variant identity at all, so it picks an acquisition ORDER without
    knowing which case it is looking at.

    Proportions follow the usual bust icon: head radius ~0.26s, shoulders
    starting AT the chin (a gap there is what made the first draft read as two
    unrelated shapes rather than one figure).
    """
    lw = lw or max(1.9, s * .075)
    hr = s * .26
    hy = cy + s * .30
    chin = hy - hr

    # Coat first, so the head overlaps it and the two read as one silhouette.
    # 0.72 x 0.52 is roughly 1.4:1. The first draft was 0.86 x 0.40 — a 2.15:1
    # slab that read as a squashed bust rather than shoulders.
    bw, bh = s * .66, s * .56
    ax.add_patch(FancyBboxPatch((cx - bw / 2, chin - bh + s * .10), bw, bh,
                                boxstyle=f"round,pad=0,rounding_size={bw * .22}",
                                facecolor="white", edgecolor=color,
                                linewidth=lw, zorder=z))
    # V neckline, drawn from the chin so the collar sits where a collar sits.
    ax.plot([cx - bw * .19, cx, cx + bw * .19],
            [chin + s * .04, chin - s * .12, chin + s * .04],
            color=color, lw=lw * .9, solid_capstyle="round",
            solid_joinstyle="round", zorder=z + 4)
    # Stethoscope: one tube down the left, ending in the bell.
    t = np.linspace(0, 1, 60)
    ax.plot(cx - bw * .24 - s * .05 * np.sin(t * 2.6),
            chin - s * .02 - t * s * .22,
            color=color, lw=lw * .8, zorder=z + 4, solid_capstyle="round")
    ax.add_patch(Circle((cx - bw * .29, chin - s * .25), s * .052,
                        facecolor="white", edgecolor=color, linewidth=lw * .8,
                        zorder=z + 5))

    ax.add_patch(Circle((cx, hy), hr, facecolor="white", edgecolor=color,
                        linewidth=lw, zorder=z + 2))
    ax.add_patch(Wedge((cx, hy), hr, 22, 158, width=hr * .32,
                       facecolor=color, edgecolor="none", zorder=z + 3))
    if blindfold:
        # Width clipped to the head: a band running past the outline reads as a
        # stripe across the picture rather than as something worn.
        bwid = hr * 1.72
        ax.add_patch(FancyBboxPatch((cx - bwid / 2, hy - hr * .16),
                                    bwid, hr * .36,
                                    boxstyle="round,pad=0,rounding_size=0.006",
                                    facecolor=color, edgecolor="none", zorder=z + 4))
    else:
        for dx in (-hr * .36, hr * .36):
            ax.add_patch(Circle((cx + dx, hy + hr * .02), hr * .11,
                                facecolor=color, edgecolor="none", zorder=z + 4))


# ---------------------------------------------------------------------------
# EXPLANATORY ILLUSTRATIONS
#
# The glyphs above are ICONS: 5-unit marks that label a row. The ones below are
# small PICTURES, meant to be read at ~1.2 in square, and each one carries the
# meaning of an ACMG criterion rather than merely tagging it — a reader who does
# not know what PM1 is should be able to see "the residue sits inside a domain
# where variants cluster" from the drawing alone.
#
# They are drawn in a local 0..1 box like everything else here, but they assume
# the caller has given them a SQUARE axes (equal inches per unit in x and y).
# The icons above tolerate a stretched axes because a stretched histogram is
# still a histogram; a stretched population grid is a smear, and a stretched
# well plate reads as an unrelated object.
# ---------------------------------------------------------------------------

def stop_bar(ax, x, y, s, color, pale="#c9e0ef", z=5):
    """A gene, and the protein it fails to finish making — PVS1.

    Two registers, which is what makes the criterion legible rather than merely
    labelled: the coding exons on top are all present, and the product beneath
    them stops at the premature stop and is drawn hollow from there on. A single
    row of blocks (the first draft) showed a sequence with a sign in it and left
    the reader to supply the consequence themselves.

    The hollow half must never be filled: what PVS1 asserts is precisely that
    there is nothing there.
    """
    def X(u):
        return x + s * u

    def Y(u):
        return y + s * u

    lw = max(1.2, s * .022)
    stop_u = .495

    # register 1 — the gene: every exon is present, whatever the variant does.
    ax.plot([X(.05), X(.95)], [Y(.64), Y(.64)], color=pale, lw=lw * 1.0,
            zorder=z)
    for i in range(6):
        ax.add_patch(FancyBboxPatch((X(.05 + i * .155), Y(.56)), s * .12,
                                    s * .16,
                                    boxstyle=f"round,pad=0,rounding_size={s * .016}",
                                    facecolor=pale, edgecolor=color,
                                    linewidth=lw * .8, zorder=z + 1))

    # the stop, dropped through both registers so the cause meets its effect
    ax.plot([X(stop_u), X(stop_u)], [Y(.18), Y(.80)], color=color,
            lw=lw * .8, ls=(0, (2, 2)), zorder=z + 2)
    r = s * .085
    oct_ = [(X(stop_u) + r * np.cos(a), Y(.84) + r * np.sin(a))
            for a in np.linspace(np.pi / 8, 2 * np.pi + np.pi / 8, 9)[:-1]]
    ax.add_patch(Polygon(oct_, closed=True, facecolor=color, edgecolor="none",
                         zorder=z + 3))
    ax.plot([X(stop_u) - r * .46, X(stop_u) + r * .46], [Y(.84), Y(.84)],
            color="white", lw=lw * 1.6, solid_capstyle="round", zorder=z + 4)

    # register 2 — the product: made up to the stop, absent after it
    ax.add_patch(FancyBboxPatch((X(.05), Y(.20)), s * (stop_u - .085), s * .18,
                                boxstyle=f"round,pad=0,rounding_size={s * .018}",
                                facecolor=color, edgecolor="none", zorder=z + 1))
    ax.add_patch(FancyBboxPatch((X(stop_u + .035), Y(.20)),
                                s * (.95 - stop_u - .035), s * .18,
                                boxstyle=f"round,pad=0,rounding_size={s * .018}",
                                facecolor="none", edgecolor=pale,
                                linewidth=lw, linestyle=(0, (2.2, 1.8)),
                                zorder=z + 1))


def population(ax, x, y, s, color, pale="#c9e0ef", rows=6, cols=11, hit=(3, 4),
               z=5):
    """A population reference with one carrier in it — PM2 rarity.

    Deliberately literal. "Absent or vanishingly rare in gnomAD" is a statement
    about a crowd, and one dark dot in a field of pale ones says it faster than
    any allele-frequency axis at this size.
    """
    x0, x1, y0, y1 = .07, .93, .13, .80
    for r in range(rows):
        for c in range(cols):
            cx = x + s * (x0 + (x1 - x0) * c / (cols - 1))
            cy = y + s * (y0 + (y1 - y0) * r / (rows - 1))
            on = (r, c) == hit
            ax.add_patch(Circle((cx, cy), s * (.037 if on else .028),
                                facecolor=color if on else pale,
                                edgecolor="none", zorder=z + (1 if on else 0)))
    # A ring around the carrier, so the single dark dot is unmistakably marked
    # rather than merely darker — the distinction survives greyscale print.
    cx = x + s * (x0 + (x1 - x0) * hit[1] / (cols - 1))
    cy = y + s * (y0 + (y1 - y0) * hit[0] / (rows - 1))
    ax.add_patch(Circle((cx, cy), s * .085, facecolor="none", edgecolor=color,
                        linewidth=max(1.1, s * .018), zorder=z + 2))


def calibration_steps(ax, x, y, s, color, ink="#1c2b38", mute="#7d8b96",
                      lo=.85, hi=1.0, at=None, fs=6.0, z=5):
    """Score → evidence strength as a step function — PP3.

    The x-axis starts at `lo`, not at 0. On a full 0..1 axis the three ClinGen
    bins (0.906 / 0.972 / 0.99) occupy 9% of the width and the last two are
    invisible; the point of the picture is that the bins are unevenly spaced and
    steep at the top, which only a zoomed axis shows. The axis end labels are
    drawn so the zoom cannot be mistaken for the full range.
    """
    bins = [(lo, .906, 0), (.906, .972, 1), (.972, .99, 2), (.99, hi, 4)]
    ax0, ax1, ay0, ay1 = .22, .96, .22, .82
    lw = max(1.1, s * .018)

    def X(v):
        return x + s * (ax0 + (ax1 - ax0) * (v - lo) / (hi - lo))

    def Y(p):
        return y + s * (ay0 + (ay1 - ay0) * p / 4.0)

    ax.plot([X(lo), X(lo)], [Y(0), Y(4.35)], color=mute, lw=lw * .8, zorder=z)
    ax.plot([X(lo), X(hi)], [Y(0), Y(0)], color=mute, lw=lw * .8, zorder=z)
    for a, b, p in bins:
        if p:
            ax.add_patch(FancyBboxPatch((X(a), Y(0)), X(b) - X(a), Y(p) - Y(0),
                                        boxstyle="round,pad=0,rounding_size=0",
                                        facecolor=color, alpha=.30,
                                        edgecolor="none", zorder=z + 1))
        ax.plot([X(a), X(b)], [Y(p), Y(p)], color=color, lw=lw * 1.7,
                solid_capstyle="butt", zorder=z + 2)
    for (a, _, p), (_, _, prev) in zip(bins[1:], bins[:-1]):
        ax.plot([X(a), X(a)], [Y(prev), Y(p)], color=color, lw=lw * 1.7,
                solid_capstyle="butt", zorder=z + 2)
    for p in (1, 2, 4):
        ax.plot([X(lo) - s * .015, X(lo)], [Y(p), Y(p)], color=mute,
                lw=lw * .8, zorder=z)
        ax.text(X(lo) - s * .035, Y(p), f"+{p}", ha="right", va="center",
                fontsize=fs, color=mute)
    for v in (lo, hi):
        ax.text(X(v), Y(0) - s * .075, f"{v:.2f}", ha="center", va="top",
                fontsize=fs, color=mute)
    if at is not None:
        p = next(p for a, b, p in bins if a <= at <= b)
        ax.add_patch(Circle((X(at), Y(p)), s * .036, facecolor=ink,
                            edgecolor="white", linewidth=lw * .7, zorder=z + 4))


def lollipop(ax, x, y, s, color, pale="#c9e0ef", z=5):
    """Variants stacked on a protein track, clustered in one domain — PM1.

    The mutation-lollipop plot is the field's own idiom for "this is a hot
    spot", so it needs no legend: the eye reads the pile inside the shaded box
    and the strays outside it without being told what either means.
    """
    def X(u):
        return x + s * u

    def Y(u):
        return y + s * u

    lw = max(1.1, s * .018)
    ax.add_patch(FancyBboxPatch((X(.05), Y(.20)), s * .90, s * .085,
                                boxstyle=f"round,pad=0,rounding_size={s * .042}",
                                facecolor=pale, edgecolor="none", zorder=z))
    ax.add_patch(FancyBboxPatch((X(.36), Y(.150)), s * .34, s * .185,
                                boxstyle=f"round,pad=0,rounding_size={s * .02}",
                                facecolor=color, alpha=.34, edgecolor=color,
                                linewidth=lw * 1.1, zorder=z + 1))
    stems = [(.13, .20, False), (.41, .48, True), (.47, .66, True),
             (.53, .40, True), (.62, .56, True), (.87, .17, False)]
    for u, h, inside in stems:
        top = Y(.30 + h)
        ax.plot([X(u), X(u)], [Y(.30), top],
                color=color if inside else pale, lw=lw * 1.1, zorder=z + 2,
                solid_capstyle="round")
        ax.add_patch(Circle((X(u), top), s * (.040 if inside else .032),
                            facecolor=color if inside else "white",
                            edgecolor=color, linewidth=lw * .9,
                            zorder=z + 3))


def well_plate(ax, x, y, s, color, z=5):
    """A microtitre plate read across a dose gradient — PS3 functional evidence.

    Wells darken left to right. That is what a functional assay produces: not a
    yes/no annotation but a measured response, which is why PS3 is Strong
    evidence rather than Supporting.
    """
    lw = max(1.1, s * .018)
    ax.add_patch(FancyBboxPatch((x + s * .05, y + s * .18), s * .90, s * .64,
                                boxstyle=f"round,pad=0,rounding_size={s * .05}",
                                facecolor="white", edgecolor=color,
                                linewidth=lw * 1.3, zorder=z))
    cols, rows = 7, 5
    for c in range(cols):
        for r in range(rows):
            cx = x + s * (.14 + .72 * c / (cols - 1))
            cy = y + s * (.26 + .48 * r / (rows - 1))
            a = .12 + .88 * (c / (cols - 1)) ** 1.4
            ax.add_patch(Circle((cx, cy), s * .043, facecolor=color, alpha=a,
                                edgecolor=color, linewidth=lw * .5, zorder=z + 1))


def weight_bar(ax, x, y, w, h, n, total, color, pale="#dde8f0", mark=None,
               z=5, ec="white"):
    """`n` of `total` ACMG points, drawn as filled cells, with the decision
    threshold marked at cell `mark`.

    Stating "+8 points" tells a reader nothing until they know what the points
    are counted towards. Showing eight cells with a notch at six makes the
    criterion's weight and the threshold the same picture: PVS1 fills past the
    notch on its own, PM2 fills one cell of the six needed.

    Unlike the illustrations above this is drawn in the CALLER's units and does
    not need a square axes — it is a row of rectangles either way.
    """
    cw = w / total
    for i in range(total):
        ax.add_patch(FancyBboxPatch((x + i * cw + w * .006, y),
                                    cw - w * .012, h,
                                    boxstyle="round,pad=0,rounding_size=0",
                                    facecolor=color if i < n else pale,
                                    edgecolor=ec, linewidth=.8, zorder=z))
    if mark is not None:
        mx = x + mark * cw
        ax.plot([mx, mx], [y - h * .55, y + h * 1.55], color="#1c2b38",
                lw=1.3, solid_capstyle="butt", zorder=z + 2)


def truncation(ax, x, y, s, color, filled=False, z=5):
    """A protein cut short — PVS1, the null-variant / loss-of-function evidence.

    Drawn as a coding bar that stops partway, with a break mark where it stops.
    The remaining stub is hollow in both states: what PVS1 asserts is that the
    product is ABSENT past that point, so filling it in would say the opposite
    of the evidence it stands for.
    """
    lw = _lw(s, filled)
    y0 = y + s * .38
    # The part that survives.
    ax.add_patch(FancyBboxPatch((x + s * .10, y0), s * .38, s * .24,
                                boxstyle="round,pad=0,rounding_size=0.008",
                                facecolor=color if filled else "none",
                                edgecolor=color, linewidth=lw, zorder=z))
    # The break.
    for dx in (.53, .60):
        ax.plot([x + s * dx, x + s * (dx - .05)],
                [y0 - s * .07, y0 + s * .31],
                color=color, lw=lw, solid_capstyle="round", zorder=z + 1)
    # The part that is lost — dashed, never filled.
    ax.plot([x + s * .68, x + s * .90], [y0 + s * .12, y0 + s * .12],
            color=color, lw=lw * .8, ls=(0, (2, 2)), zorder=z,
            solid_capstyle="butt")


def notes(ax, x, y, s, color, filled=False, z=5):
    """A record page — clinical history, or a literature/guideline source.

    Domain-neutral on purpose: the framing panel has to read as clinical
    decision-making in general, not as variant interpretation, so it needs
    glyphs that are not tied to one assay type.
    """
    lw = _lw(s, filled)
    ax.add_patch(FancyBboxPatch((x + s * .20, y + s * .10), s * .60, s * .78,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor=color if filled else "none",
                                edgecolor=color, linewidth=lw, zorder=z))
    for i, frac in enumerate((.68, .52, .36)):
        ax.plot([x + s * .32, x + s * (.68 - i * .08)],
                [y + s * frac, y + s * frac],
                color="white" if filled else color, lw=lw * .7,
                solid_capstyle="round", zorder=z + 1)


def scan(ax, x, y, s, color, filled=False, z=5):
    """A cross-sectional image — imaging as an evidence source."""
    lw = _lw(s, filled)
    ax.add_patch(FancyBboxPatch((x + s * .12, y + s * .16), s * .76, s * .68,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor=color if filled else "none",
                                edgecolor=color, linewidth=lw, zorder=z))
    ax.add_patch(Circle((x + s * .50, y + s * .50), s * .20,
                        facecolor="none",
                        edgecolor="white" if filled else color,
                        linewidth=lw * .8, zorder=z + 1))
    ax.plot([x + s * .50, x + s * .50], [y + s * .30, y + s * .70],
            color="white" if filled else color, lw=lw * .55,
            solid_capstyle="round", zorder=z + 1)
