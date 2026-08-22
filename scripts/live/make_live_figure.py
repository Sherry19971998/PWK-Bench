#!/usr/bin/env python3
"""The live block's figure: four panels, four different chart types.

WHY THIS SHAPE
--------------
`figures/live/` was empty, so Pillar 2 existed only as prose and tables. The
first version of this figure was three bar charts — one encoding answering four
different questions, so a reader cannot tell from the shape of a panel what kind
of claim it makes. Each panel now uses the encoding its claim actually needs:

  A  DISTRIBUTION — over-acquisition is a spread over cases, not one number. A
     bar of the mean hides that a quarter of cases over-acquire by 2+ channels.
     The exceed-fraction is annotated above so the headline stays readable at
     poster distance.
  B  HEATMAP — the gateway finding is POSITIONAL: `get_population_freq` is
     442/442 first calls and appears at no other position. That is structure in
     (tool x call-index) space, which no bar chart can show.
  C  ORDERED BAR — the ablation is a comparison against one baseline, so bars
     plus a baseline rule is right. Colour is reserved and meaningful: red =
     removing it collapses the sequence, green = removing it changes nothing.
  D  GROUPED BAR — per-case reproducibility is two distributions over the same
     discrete axis; grouping beats overlaying.

Every number is read from the artifacts, never hard-coded, so the figure cannot
drift from the tables it illustrates.

USAGE
    python scripts/live/make_live_figure.py
"""
import argparse, ast, os, sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench import figstyle as FS                           # noqa: E402
from pwkbench.domains.base import load_real_cohort            # noqa: E402
from pwkbench.domains.variant import VARIANT_DOMAIN           # noqa: E402
from pwkbench.live.metrics_live import (                      # noqa: E402
    sufficiency_points, over_acquisition)

OPENAI_RUNS = ["results/live/trajectories_full.csv",
               "results/live/trajectories_full_r1.csv",
               "results/live/trajectories_full_r2.csv"]
GEMINI_RUNS = [f"results/live/gemini_vertex_pro/trajectories_vertex_pro_r{i}.csv"
               for i in range(3)]
# Ablation arms exist for gpt-5.5 only — the second vendor's hand-off covers the
# full arm. Panel C says so rather than implying both.
ABLATIONS = [("full", "all\nfour", "base"),
             ("ablate_get_population_freq", "−pop_freq\n(gateway)", "gateway"),
             ("ablate_get_functional_assay", "−assay", "neutral"),
             ("ablate_get_domain_context", "−domain", "neutral"),
             ("ablate_get_insilico_pathogenicity", "−in-silico\n(fungible)",
              "fungible")]
TOOLS = ["get_population_freq", "get_insilico_pathogenicity",
         "get_domain_context", "get_functional_assay"]
TOOL_SHORT = ["population\nfreq", "in-silico", "domain\ncontext", "functional\nassay"]

OA, GM = FS.VENDOR["gpt-5.5"], FS.VENDOR["gemini-2.5-pro"]


def _over(paths, suff):
    """Pooled per-case over-acquisition values, and the per-run exceed rate."""
    pooled, ex = [], []
    for p in paths:
        oa = over_acquisition(pd.read_csv(p), suff)
        col = next(c for c in oa.columns if "over" in c.lower())
        v = oa.loc[oa[col].notna(), col]
        pooled.append(v.to_numpy())
        ex.append((v > 0).mean())
    return np.concatenate(pooled), np.array(ex)


def _position_matrix(paths):
    """counts[tool, call-index] pooled over runs, normalised BY POSITION.

    Column-normalised, not row-normalised: the question is "given that this is
    the agent's n-th call, which tool is it?" — that is what makes the gateway
    structure legible. Row-normalising would answer a different question.
    """
    c = Counter()
    for p in paths:
        for s in pd.read_csv(p)["tools_called"]:
            seq = ast.literal_eval(s) if isinstance(s, str) else (s or [])
            for i, t in enumerate(seq):
                c[(t, i)] += 1
    m = np.zeros((len(TOOLS), 4))
    for (t, i), v in c.items():
        if t in TOOLS and i < 4:
            m[TOOLS.index(t), i] = v
    colsum = m.sum(axis=0, keepdims=True)
    return np.divide(m, colsum, out=np.zeros_like(m), where=colsum > 0), m


def _spread(paths):
    fr = [pd.read_csv(p).set_index("variant_id") for p in paths]
    ids = sorted(set.intersection(*[set(d.index) for d in fr]))
    n = pd.DataFrame({i: d.loc[ids, "n_tools_called"] for i, d in enumerate(fr)})
    a = pd.DataFrame({i: d.loc[ids, "answer"] for i, d in enumerate(fr)})
    ok = a.notna().all(axis=1)
    return (n.max(axis=1) - n.min(axis=1)), float((a[ok].nunique(axis=1) == 1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/live/live_agent.png")
    args = ap.parse_args()
    os.chdir(_ROOT)
    FS.use()

    cohort = load_real_cohort(
        pd.read_parquet("data/sample/cohort_full_real.parquet"), VARIANT_DOMAIN)
    suff = sufficiency_points(cohort)

    po, ex_o = _over(OPENAI_RUNS, suff)
    pg, ex_g = _over(GEMINI_RUNS, suff)

    # Two panels, one row. The ablation and repeat-spread panels were cut:
    # the ablation invites a causal reading the design cannot support (the
    # gateway tool is also the first tool in the schema list), and the
    # repeat-spread panel duplicates 'same evidence' in the reliability
    # figure two columns to its right.
    # Taller than wide-and-flat: on the poster every figure is scaled to the
    # column width, so a figure's printed SIZE is set by its aspect ratio, not
    # by its figsize. 4.3 in of height at 11.6 in wide rendered this panel at
    # about 3.5 in on a 11 in column, smaller than the two clinical figures
    # below it despite carrying the headline result.
    fig = plt.figure(figsize=(11.6, 6.9))
    gs = fig.add_gridspec(1, 2, wspace=.24,
                          left=.07, right=.96, top=.80, bottom=.22)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ---- A  distribution -------------------------------------------------
    bins = np.arange(-1.5, 4.0, 1.0)
    for v, c, lab, off in ((po, OA, "gpt-5.5", -.18), (pg, GM, "gemini-2.5-pro", .18)):
        h, _ = np.histogram(v, bins=bins)
        axA.bar(bins[:-1] + .5 + off, h / h.sum(), width=.34, color=c,
                edgecolor="white", linewidth=.8, label=lab)
    axA.axvline(0.5, color=FS.ROLE["gateway"], ls="--", lw=1.6, zorder=5)
    # Axes coords, parked in the empty upper-middle: data coords put this on
    # top of the k=1 bar at every plausible y.
    axA.annotate("over-acquiring", xy=(.60, .62), xytext=(.72, .62),
                 xycoords="axes fraction", textcoords="axes fraction",
                 color=FS.ROLE["gateway"], fontsize=12.5, fontweight="bold",
                 va="center", ha="left",
                 arrowprops=dict(arrowstyle="<-", color=FS.ROLE["gateway"], lw=1.4))
    axA.set_xticks([-1, 0, 1, 2, 3])
    axA.set_xlabel("queries beyond the minimum the case needed")
    axA.set_ylabel("fraction of cases")
    axA.set_title("Both models acquire more evidence than needed", pad=30)
    axA.legend(loc="upper right", framealpha=.95)
    axA.text(0, 1.06,
             f"gpt-5.5: {ex_o.mean():.0%} of cases     "
             f"gemini-2.5-pro: {ex_g.mean():.0%} of cases",
             transform=axA.transAxes, ha="left", fontsize=11.5, color=FS.MUTE)
    FS.panel_tag(axA, "A")

    # ---- B  heatmap ------------------------------------------------------
    mo, raw_o = _position_matrix(OPENAI_RUNS)
    mg, _ = _position_matrix(GEMINI_RUNS)
    both = np.concatenate([mo, mg], axis=1)
    im = axB.imshow(both, cmap=FS.HEAT, vmin=0, vmax=1, aspect="auto")
    axB.set_xticks(range(8)); axB.set_xticklabels(["1", "2", "3", "4"] * 2)
    axB.set_yticks(range(4)); axB.set_yticklabels(TOOL_SHORT, fontsize=11)
    axB.axvline(3.5, color="white", lw=3.5)
    axB.set_xlabel("order of the query")
    for r in range(4):
        for c in range(8):
            v = both[r, c]
            if v > .02:
                axB.text(c, r, f"{v:.0%}", ha="center", va="center", fontsize=10.5,
                         color="white" if v > .55 else FS.INK)
    axB.text(1.5, -.80, "gpt-5.5", ha="center", fontsize=12.5, color=OA,
             fontweight="bold")
    axB.text(5.5, -.80, "gemini-2.5-pro", ha="center", fontsize=12.5, color=GM,
             fontweight="bold")
    axB.set_title("Population frequency is always the first query", pad=34)
    axB.grid(False)
    fig.colorbar(im, ax=axB, fraction=.03, pad=.02).set_label(
        "share of queries at this position", fontsize=11)
    FS.panel_tag(axB, "B")


    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"written: {args.out}")
    print(f"  A exceed  {ex_o.mean():.4f}±{ex_o.std(ddof=1):.4f} / "
          f"{ex_g.mean():.4f}±{ex_g.std(ddof=1):.4f}")
    print(f"  B pos-1 share  gpt-5.5={mo[0, 0]:.3f}  gemini={mg[0, 0]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
