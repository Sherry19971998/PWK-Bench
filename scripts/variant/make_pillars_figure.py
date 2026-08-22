#!/usr/bin/env python3
"""How the two pillars relate: complementary, not redundant.

WHY THIS FIGURE EXISTS
-----------------------
The single most-repeated reviewer comment on this section ("IMPORTANT! I
don't understand the relations among different pillars", flagged twice) was
never answered with a picture, only with a paragraph the reader has to hold
in their head across two subsections. This panel says in one glance what the
prose says in five sentences: each pillar is missing exactly what the other
one has, and the over-acquisition claim needs both.

THE TWO AXES, DRAWN, NOT JUST NAMED
-------------------------------------
  * ENUMERABLE ORACLE. Frozen has one (the evidence pool is fixed and small
    enough to enumerate exactly, S*(k) in Sec. formulation); live does not
    (no computable ceiling exists once the agent can call real APIs).
  * NATIVE STOPPING. Frozen does not have it (the harness imposes the budget
    grid k=1..4; the agent never decides when to stop); live does (the agent
    calls `final_answer` whenever it chooses).
  * INFORMATION STRENGTH (the `doctor(blindfold=...)` glyph, reused from
    `make_problem_figure.py` so the same character reads consistently across
    the paper's figures). Frozen's default arm is BLINDED: no variant
    identity at all. Live is UNBLINDED: the agent is shown the full HGVS
    name. This third axis is not decorative -- it is why the live-pillar
    over-acquisition finding is the STRONGER claim (the agent has enough to
    answer and still keeps acquiring), while the frozen-pillar one is the
    weaker form (the agent cannot tell it could stop). Burying this
    asymmetry in prose is exactly what produced the reviewer's confusion.

Neither pillar subsumes the other, which is why the bottom box has two
arrows feeding it, not one: the frozen pool cannot observe an agent choosing
to over-acquire (its budget is imposed), and the live agent has no
computable optimum to be measured against. The over-acquisition claim
requires both.

USAGE
    python scripts/variant/make_pillars_figure.py
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench import figglyphs as G  # noqa: E402
from pwkbench import figstyle as FS  # noqa: E402


def _box(ax, x0, y0, x1, y1, edge, fill, lw=2.2, r=0.14, z=1):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z))


def _feature_row(ax, x, y, ok, label, color, fontsize=11.5):
    """One check/slash + label row. `ok=True` -> check, `ok=False` -> slash."""
    if ok:
        G.check(ax, x, y, 0.155, color, z=6)
    else:
        G.slash(ax, x, y, 0.155, FS.MUTE, z=6)
    ax.text(x + 0.34, y, label, ha="left", va="center",
            fontsize=fontsize, color=FS.INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/variant/pillars.png")
    args = ap.parse_args()
    os.chdir(_ROOT)
    FS.use()

    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    fig.patch.set_facecolor(FS.PARADIGM["paper"])
    ax.set_facecolor(FS.PARADIGM["paper"])
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 6.5)
    ax.set_axis_off()

    frozen_c, frozen_fill = FS.PARADIGM["frozen"], FS.PARADIGM["frozen_fill"]
    live_c, live_fill = FS.PARADIGM["live"], FS.PARADIGM["live_fill"]
    shared_c, shared_fill = FS.PARADIGM["shared"], FS.PARADIGM["shared_fill"]

    # --- Pillar 1: Frozen Pool ------------------------------------------
    fx0, fx1 = 0.35, 5.55
    fy0, fy1 = 2.15, 6.15
    _box(ax, fx0, fy0, fx1, fy1, frozen_c, frozen_fill, z=1)
    ax.text((fx0 + fx1) / 2, fy1 - 0.42, "Pillar 1 — Frozen Pool",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=frozen_c)

    G.doctor(ax, fx0 + 1.05, fy1 - 1.55, 0.95, frozen_c, blindfold=True, z=5)
    ax.text(fx0 + 1.05, fy1 - 2.42, "blinded\n(no variant identity,\ndefault arm)",
            ha="center", va="top", fontsize=10.5, color=FS.MUTE, style="italic")

    fr_x = fx0 + 2.55
    _feature_row(ax, fr_x, fy1 - 1.30, True,
                 "enumerable cohort-level\noracle (exact ceiling)", frozen_c)
    _feature_row(ax, fr_x, fy1 - 2.15, False,
                 "no native stopping\n(harness imposes k=1..4)", frozen_c)

    ax.text((fx0 + fx1) / 2, fy0 + 0.35,
            "prices the structural cost of\nexhausting a fixed budget",
            ha="center", va="center", fontsize=11, color=FS.INK,
            style="italic")

    # --- Pillar 2: Live Agent --------------------------------------------
    lx0, lx1 = 6.05, 11.25
    ly0, ly1 = 2.15, 6.15
    _box(ax, lx0, ly0, lx1, ly1, live_c, live_fill, z=1)
    ax.text((lx0 + lx1) / 2, ly1 - 0.42, "Pillar 2 — Live Agent",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=live_c)

    G.doctor(ax, lx0 + 1.05, ly1 - 1.55, 0.95, live_c, blindfold=False, z=5)
    ax.text(lx0 + 1.05, ly1 - 2.42, "unblinded\n(full HGVS name\ngiven)",
            ha="center", va="top", fontsize=10.5, color=FS.MUTE, style="italic")

    lr_x = lx0 + 2.55
    _feature_row(ax, lr_x, ly1 - 1.30, False,
                 "no enumerable oracle\n(real APIs, no fixed pool)", live_c)
    _feature_row(ax, lr_x, ly1 - 2.15, True,
                 "native stopping\n(agent calls final_answer)", live_c)

    ax.text((lx0 + lx1) / 2, ly0 + 0.35,
            "measures the agent's own\nchoice to keep acquiring",
            ha="center", va="center", fontsize=11, color=FS.INK,
            style="italic")

    # --- convergence: both feed the over-acquisition claim ----------------
    bx0, bx1 = 2.55, 9.05
    by0, by1 = 0.20, 1.55
    _box(ax, bx0, by0, bx1, by1, shared_c, shared_fill, lw=2.4, z=1)
    ax.text((bx0 + bx1) / 2, by1 - 0.40, "Over-acquisition claim",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color=shared_c)
    ax.text((bx0 + bx1) / 2, by0 + 0.42,
            "supported only by both together — frozen cannot see an agent\n"
            "choose to over-acquire; live has no computable ceiling to price it against",
            ha="center", va="center", fontsize=10.3, color=FS.INK)

    for (x0, x1, color) in [(fx0 + 1.3, (bx0 + bx1) / 2 - 0.55, frozen_c),
                            (lx1 - 1.3, (bx0 + bx1) / 2 + 0.55, live_c)]:
        ax.annotate("", xy=(x1, by1 + 0.28),
                    xytext=(x0, fy0 - 0.12),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2,
                                    shrinkA=0, shrinkB=0,
                                    connectionstyle="arc3,rad=0.0"),
                    zorder=2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor=fig.get_facecolor(),
                dpi=200)
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
