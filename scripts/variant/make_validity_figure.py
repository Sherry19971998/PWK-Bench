#!/usr/bin/env python3
"""Memorization control: is the answer recalled, or actually acquired?

WHY THIS FIGURE EXISTS
----------------------
Axis C of the framework was carried on the poster as prose plus one formula, and
prose is the wrong medium for it: the claim is a comparison between FOUR closed-
book conditions, and the whole argument is which of them sit at chance and which
do not. Drawn as bars against a chance line, it is a one-glance read.

THE TWO-MASK ARGUMENT
---------------------
This is the part that a single "masked AUC -> 0.5" line cannot express, and
getting it wrong would put two contradictory numbers on the same poster:

  * COORDINATE MASK (gene identity only) = 0.43. At chance. The label is NOT
    recoverable from parametric memory of the locus — that is the memorization
    control passing. This bar is a validity GATE, not the G_acq baseline (see
    below): it answers "can the control detect memorization at all", not "how
    much did acquisition add".
  * FULL-HGVS MASK (the variant name) = 0.865. Far above chance, and this is
    NOT a failure of the control: an HGVS name exposes the molecular consequence
    (p.Arg213Ter is loss-of-function on its face), and reading it is in-context
    ACMG reasoning worth the PVS1 category, not recall. `metrics.py` says so
    directly, and the PVS1-alone bar at 0.830 is the evidence: the name gets you
    almost exactly what the LoF channel gets you, and no more.
  * ALL EVIDENCE = 0.942, so acquisition adds G_acq = +0.077.

G_acq IS MEASURED FROM THE FULL-HGVS BAR, NOT FROM THE IDENTITY BAR. The
identity-only bar sits near chance and is a validity gate (does the control
detect memorization at all), not a deployment baseline. A model shown the
real variant name already has full_hgvs's worth of performance for free --
that is what a deployed system actually starts from -- so the acquisition-
attributable quantity the paper reports is what evidence adds ON TOP of that:
G_acq = full_acquisition - full_hgvs = 0.942 - 0.865 = 0.077. Anchoring the
arrow at the identity bar instead would credit acquisition for beating a
near-chance baseline the agent was never actually starting from, inflating
the number to 0.44 -- `metrics.py::memorization_probe`'s `memory_floor`
field still reports that quantity separately, for the validity-gate check,
but it is not G_acq.

That distinction is also why the blinded block withholds the name at all.

USAGE
    python scripts/variant/make_validity_figure.py
"""
import argparse, glob, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench import figstyle as FS                          # noqa: E402

PROBE = "results/variant/yield_repeat/run*/memorization_probe.csv"
INK, MUTE = FS.INK, "#4d5c68"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/variant/validity.png")
    args = ap.parse_args()
    os.chdir(_ROOT)
    FS.use()

    files = sorted(glob.glob(PROBE))
    if not files:
        raise SystemExit(f"no memorization probe under {PROBE}")
    m = pd.concat([pd.read_csv(f) for f in files]).mean(numeric_only=True)

    rows = [("gene name only", m.coord_masked, FS.MUTE),
            ("loss-of-function only", m.pvs1_alone, FS.SKY),
            ("variant name only", m.full_hgvs, FS.STEEL),
            ("all evidence", m.full_acquisition, FS.NAVY)]

    # 9.3 in wide against 11.6 for every other figure on the board meant this
    # one's type was scaled up by a quarter relative to its neighbours, and its
    # 0.49 aspect made it the tallest block in the column. Matching the common
    # width both shrinks it and puts its labels on the same visual scale.
    fig, ax = plt.subplots(figsize=(11.6, 3.5))
    fig.patch.set_facecolor(FS.PARADIGM["paper"])
    ax.set_facecolor(FS.PARADIGM["paper"])

    ys = range(len(rows))
    ax.barh(list(ys), [r[1] for r in rows], height=.58,
            color=[r[2] for r in rows], edgecolor="white", linewidth=1.4)
    # A value under the chance line has no room to its right: "0.43" set at
    # x=0.438 runs straight through the dashed line at 0.50. Those labels go
    # inside the bar instead.
    for y, (lab, v, _) in zip(ys, rows):
        inside = v < .5
        ax.text(v - .01 if inside else v + .008, y, f"{v:.2f}", va="center",
                ha="right" if inside else "left", fontsize=15,
                fontweight="bold", color="white" if inside else INK)

    # Chance is the reference the control is read against, so it is a labelled
    # line rather than an axis tick a reader has to find.
    ax.axvline(.5, color=FS.ACCENT, lw=2.2, ls="--", zorder=5)
    # Above the first bar, not beside the last: at the foot it collided with
    # the G_acq arrow label.
    ax.text(.5, -.52, "  chance (0.50)", ha="left", va="center",
            fontsize=13, fontweight="bold", color=FS.ACCENT)

    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=14)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.06)
    ax.set_xticks([0, .25, .5, .75, 1.0])
    ax.set_xlabel("closed-book AUC", fontsize=13)
    ax.grid(axis="x", alpha=.28)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    # G_acq is measured from the full-HGVS bar (the deployment-realistic
    # closed-book baseline), NOT from the near-chance identity-only floor --
    # see the module docstring's "G_acq IS MEASURED FROM..." section.
    gain = m.full_acquisition - m.full_hgvs
    assert abs(gain - m.G_acq) < 1e-6, "G_acq drawn differently from how it is stored"
    ax.annotate("", xy=(m.full_acquisition, 3.52), xytext=(m.full_hgvs, 3.52),
                arrowprops=dict(arrowstyle="<->", color=FS.ACCENT, lw=2.0))
    ax.text((m.full_hgvs + m.full_acquisition) / 2, 3.72,
            f"evidence adds +{gain:.3f}",
            ha="center", va="top", fontsize=14, fontweight="bold",
            color=FS.ACCENT)
    # 3.95 put the bottom spine through the "evidence adds" label, which sits
    # at y=3.72 and is set in bold at 13 pt. Extra room below the arrow rather
    # than moving the label up, where it would run into the "all evidence" bar.
    ax.set_ylim(4.15, -.75)

    # "Identity alone is no better than chance" is true of the gene bar (0.43)
    # and false of the variant-name bar (0.865) sitting two rows below it, so
    # the title contradicted its own panel. Naming the gene keeps the claim to
    # the bar that carries it; the two-mask argument is in this module's
    # docstring and in the poster's box beside the figure.
    ax.set_title("Gene identity alone is at chance",
                 fontsize=15, fontweight="bold", color=INK, loc="left", pad=12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"written: {args.out}")
    for lab, v, _ in rows:
        print(f"  {lab:28s} {v:.3f}")
    print(f"  G_acq {gain:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
