#!/usr/bin/env python3
"""Consequence-and-coordinate masking (E4): is the closed-book floor read off
the name, or acquired?

WHY THIS FIGURE EXISTS
-----------------------
The live pillar's headline validity number --- closed-book accuracy 0.866 vs.
full-tool accuracy 0.902, p=0.154 --- was, until this figure, only stated as
prose in the paper. A reviewer skimming past a paragraph of numbers can miss
the point entirely: that 0.866 is not a fixed floor, it is READ OFF THE NAME,
and it falls hard once the name stops giving the answer away. Drawn as a
staircase against a fixed 0.866 reference, it is a one-glance read, matching
`make_validity_figure.py`'s two-mask argument for the frozen pillar.

THE THREE-STEP ARGUMENT
------------------------
  * FULL NAME (no mask) = 0.866. The agent is shown the real HGVS name and
    calls no tools. This is the number Table~tab:live reports as the
    closed-book floor.
  * CONSEQUENCE MASKED = 0.794. Redacting the p.-level protein consequence
    (`p.Arg213Ter` -> `p.?`, `pwkbench/live/tools.py::mask_consequence`)
    removes the free-readable loss-of-function signal. Accuracy drops by
    0.072 -- real, but modest.
  * STRONGEST MASK (consequence + coordinate identity) = 0.621. Masking gene
    identity ON TOP of consequence removes far more: a further 0.173 drop,
    more than double the consequence-only effect. MOST of the 0.866->0.621
    gap is the coordinate mask, not the consequence mask -- do not caption
    this "masking the consequence token drops accuracy to 0.621"; that
    conflates the two.
  * CLAUDE-SONNET-5, same strongest mask = 0.616 (485/491 valid; 6 cases
    ended in `error:nonzero_exit` and are excluded, NOT counted as
    incorrect). Drawn as a second, hatched bar in the same shade as the
    gpt-5.5 strongest-mask bar because it is the same condition, a second
    model -- not a new hue, per figstyle's one-hue-family rule.

Every number is read from the trajectory CSVs at run time, not hardcoded.

USAGE
    python scripts/live/make_consequence_mask_figure.py
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench import figstyle as FS  # noqa: E402

FULL = "results/live/trajectories_notools.csv"
CONSEQ_MASKED = "results/live/masked/trajectories_masked_notools.csv"
STRONGEST_GPT = "results/live/masked/trajectories_masked_coords_notools.csv"
STRONGEST_CLAUDE = "results/live/claude_code/trajectories_masked_coords_notools.csv"


def _acc(path, drop_unparsed=False):
    df = pd.read_csv(path)
    n_total = len(df)
    if drop_unparsed:
        n_unparsed = int(df["predicted"].isna().sum())
        df = df.dropna(subset=["predicted"])
    else:
        n_unparsed = 0
    return float(df["correct"].mean()), n_total, n_unparsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/live/consequence_mask.png")
    args = ap.parse_args()
    os.chdir(_ROOT)
    FS.use()

    full_acc, full_n, _ = _acc(FULL)
    conseq_acc, conseq_n, _ = _acc(CONSEQ_MASKED)
    strong_gpt_acc, strong_gpt_n, _ = _acc(STRONGEST_GPT)
    strong_claude_acc, claude_n, claude_unparsed = _acc(STRONGEST_CLAUDE, drop_unparsed=True)
    claude_valid = claude_n - claude_unparsed

    rows = [
        ("full name\n(no mask)", full_acc, FS.NAVY, False,
         f"n={full_n}"),
        ("consequence\nmasked", conseq_acc, FS.STEEL, False,
         f"n={conseq_n}"),
        ("strongest mask\n(gpt-5.5)", strong_gpt_acc, FS.SKY, False,
         f"n={strong_gpt_n}"),
        ("strongest mask\n(claude-sonnet-5)", strong_claude_acc, FS.SKY, True,
         f"n={claude_valid}/{claude_n} valid"),
    ]

    fig, ax = plt.subplots(figsize=(11.6, 3.9))
    fig.patch.set_facecolor(FS.PARADIGM["paper"])
    ax.set_facecolor(FS.PARADIGM["paper"])

    ys = range(len(rows))
    for y, (lab, v, color, hatched, note) in zip(ys, rows):
        ax.barh(y, v, height=.58, color=color, edgecolor="white",
                linewidth=1.4, hatch="////" if hatched else None)
        ax.text(v + .012, y, f"{v:.3f}", va="center", ha="left",
                fontsize=15, fontweight="bold", color=FS.INK)
        ax.text(1.045, y, note, va="center", ha="left",
                fontsize=9.5, color=FS.MUTE, style="italic")

    # Reference line at the unmasked floor, so every later bar is read as a
    # drop FROM this point, mirroring the chance line in the validity figure.
    ax.axvline(full_acc, color=FS.ACCENT, lw=2.0, ls="--", zorder=5)
    ax.text(full_acc, -.62, f"  full-name floor ({full_acc:.3f})", ha="left",
            va="center", fontsize=12, fontweight="bold", color=FS.ACCENT)

    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=13)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.28)
    ax.set_xticks([0, .25, .5, .75, 1.0])
    ax.set_xlabel("live closed-book accuracy", fontsize=13)
    ax.grid(axis="x", alpha=.28)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    # Bracket the TOTAL drop (full name -> gpt-5.5 strongest mask), and
    # separately annotate how much of it is the consequence step alone, so
    # the figure cannot be read as "consequence masking alone explains the
    # drop" -- exactly the conflation the docstring above warns against.
    total_drop = full_acc - strong_gpt_acc
    conseq_drop = full_acc - conseq_acc
    coord_drop = conseq_acc - strong_gpt_acc
    ax.annotate("", xy=(strong_gpt_acc, 3.55), xytext=(full_acc, 3.55),
                arrowprops=dict(arrowstyle="<->", color=FS.ACCENT, lw=2.0))
    ax.text((strong_gpt_acc + full_acc) / 2, 3.78,
            f"total drop −{total_drop:.3f}  "
            f"(consequence −{conseq_drop:.3f}, coordinate −{coord_drop:.3f})",
            ha="center", va="top", fontsize=12.5, fontweight="bold",
            color=FS.ACCENT)
    ax.set_ylim(4.35, -.95)

    ax.set_title(
        "Coordinate masking, not consequence masking alone, drives the drop",
        fontsize=15, fontweight="bold", color=FS.INK, loc="left", pad=12)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"written: {args.out}")
    for lab, v, _, _, note in rows:
        print(f"  {lab.replace(chr(10), ' '):32s} {v:.3f}  {note}")
    print(f"  total drop (full -> gpt-5.5 strongest mask): {total_drop:.3f}")
    print(f"  of which consequence-only: {conseq_drop:.3f}, coordinate: {coord_drop:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
