"""Shared figure styling.

WHY THIS EXISTS
---------------
The figures in this tree were written at different times and drifted: different
vendor colours, different fonts, bare matplotlib defaults next to hand-tuned
panels. On a poster that reads as several unrelated projects rather than one.
Everything visual that is shared between figures lives here so it can only be
changed in one place.

Two rules encoded below that are about correctness, not taste:

* `VENDOR` maps a model to a colour. It is keyed by the model string that
  actually appears in the artifacts, so a figure cannot silently paint two
  different models the same colour.
* `ROLE` reserves the single warm ACCENT for one claim per figure — the gateway
  tool. Nothing else in the palette is warm, so a red mark always means the same
  thing wherever it appears, and spending it twice in one figure would dilute it.
"""
from __future__ import annotations

import matplotlib as mpl

# ---------------------------------------------------------------- palette
# One BLUE FAMILY plus a single warm accent, following the reference posters:
# light-blue bars with one dark-navy highlight and a lone red accent line. An
# earlier draft used red-vs-green, which is both the least accessible pair in
# common use (red-green deficiency is ~8% of men) and, at poster scale, simply
# discordant. Everything below is one hue family, so figures read as one project
# and value alone carries the distinction in greyscale print.
NAVY = "#1f4e79"      # darkest — emphasis, primary series
BLUE = "#2e77ab"      # primary
STEEL = "#4f97c4"     # secondary
SKY = "#8fbfdc"       # tertiary / second series
MIST = "#c9e0ef"      # fills
PALE = "#e8f1f8"      # panel tints
ACCENT = "#c0504d"    # the ONE warm colour; spend it on a single claim per figure
INK = "#1c2b38"
MUTE = "#7d8b96"
GREY = "#b9c4cc"
# The "available but not acquired" state. Must be clearly DRAWN, not a ghost:
# an earlier draft used a near-white grey, and because the median case needs
# zero channels most glyphs in the hero figure are in this state — the figure
# read as "a pile of grey icons". Same hue family as everything else, light
# enough to be unmistakably secondary, dark enough to survive projection.
OFF = "#a9c6dc"

# Vendor identity: same family, different VALUE, so the pair survives greyscale
# and is not a hue contrast a colour-deficient reader has to resolve.
VENDOR = {"gpt-5.5": NAVY, "gemini-2.5-pro": SKY}
VENDOR_LABEL = {"gpt-5.5": "gpt-5.5\n(OpenAI API)",
                "gemini-2.5-pro": "gemini-2.5-pro\n(Vertex)"}

# Ablation roles. `fungible` is deliberately the SAME colour as the baseline —
# the claim is that removing that tool changes nothing, so its bar should not
# look like a different kind of thing. `gateway` gets the accent because it is
# the one result in that panel worth spending it on.
ROLE = {"gateway": ACCENT, "fungible": NAVY, "neutral": GREY}

# Sequential map for heatmaps, same family as everything else.
HEAT = "Blues"

# Schematic tints.
STAGE = {"frozen": PALE, "live": MIST, "shared": "#eef4f8"}
PARADIGM = {
    "frozen": NAVY, "frozen_fill": PALE,
    "live": "#2f7f92", "live_fill": "#e3f0f2",   # teal: cool, harmonises with navy
    "shared": BLUE, "shared_fill": "#eef4f8",
    "paper": "#fbfcfd",
}
# Nucleotide colours stay conventional (genome-browser standard) so the helix
# reads as DNA; this is the one place a non-blue hue is load-bearing.
BASES = {"A": "#3f8f4f", "C": "#2f6fb0", "G": "#e0a020", "T": "#c0392b"}

GRID = "#dde5ea"


def use():
    """Apply the shared rcParams. Call once at the top of a figure script."""
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.9,
        # Top/right spines removed everywhere: the reference posters do this and
        # it materially increases the ink-to-data ratio at poster viewing
        # distance.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,          # grid behind bars, never through them
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.55,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
    })


def panel_tag(ax, letter, dx=-0.14, dy=1.06):
    """Bold panel letter in the corner, poster convention."""
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=15, fontweight="bold", va="top", ha="left")
