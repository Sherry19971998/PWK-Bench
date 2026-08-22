#!/usr/bin/env python3
"""Redraw the RQ1 four-slot gate figure (paper fig:gate) in house style.

WHY THIS EXISTS
----------------
This is the direct replacement for a stale copy of this figure
(`variant_gate_4slot-2.png`) that was traced back to a superseded prototype
checkout (`pwk_variant`/`pwk_variant 2`, pre-`pwkbench`, no `figstyle.py`) --
its Oracle A(k) values (0.9029 / 0.9798 / 0.9833 / 0.9357) and its single-slot
AUCs (PM2 0.69, PM1 0.59, PP3 0.65) do not match the current 491-variant
authoritative results (`results/variant/real_trusted/results.csv`: Oracle
0.8189/0.9172/0.9172/0.9172; `docs/variant/real_data.md`: PM2 0.819, PM1
0.456, PP3 0.535) -- it is not a colour problem, it is the wrong run.

Panel A is drawn purely from `results/<...>/results.csv` (every strategy's
A(k1..k4) is already cached there -- no cohort load, no agent re-run needed).
Panel B is drawn from the real cohort parquet directly (`per_gene_auc` on
each single unmasked channel is a closed-form computation over the cohort,
not an agent output, so this also needs no live run).

USAGE
    python scripts/variant/make_gate_figure.py \
        --results results/variant/real_trusted/results.csv \
        --cohort data/sample/cohort_full_real.parquet \
        --agent agent:frontier_B --agent-label "agent: gpt-5.5" \
        --out figures/variant/real_trusted/gate_4slot.png
"""
import argparse, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench import figstyle as FS                          # noqa: E402
from pwkbench import metrics as M                             # noqa: E402
from pwkbench.domains.base import load_real_cohort            # noqa: E402
from pwkbench.domains.variant import VARIANT_DOMAIN           # noqa: E402

BUDGETS = [1, 2, 3, 4]
LINES = [
    ("Oracle", "oracle", FS.NAVY, "o-"),
    ("RelMax", "relmax", FS.STEEL, "o-"),
    ("Heuristic", "heuristic", FS.MUTE, "o--"),
]


def draw_panel_a(ax, results_csv, agent, agent_label):
    d = pd.read_csv(results_csv).set_index("strategy")
    kcols = [f"A_k{k}" for k in BUDGETS]
    for name, label, color, style in LINES:
        if name not in d.index:
            continue
        y = d.loc[name, kcols].to_numpy(float)
        ax.plot(BUDGETS, y, style, color=color, lw=2.2, ms=6, label=label)
    if agent in d.index:
        y = d.loc[agent, kcols].to_numpy(float)
        ax.plot(BUDGETS, y, "s-", color=FS.ACCENT, lw=2.6, ms=8, label=agent_label, zorder=5)
    else:
        print(f"WARNING: agent strategy '{agent}' not in {results_csv}; "
              f"available: {list(d.index)}", file=sys.stderr)
    ax.axhline(0.5, ls=":", color=FS.GREY, lw=1)
    ax.set_xlabel("acquisition budget $k$")
    ax.set_ylabel("ranking AUC (per-gene)")
    ax.set_title("A   Planning vs. relevance, and where the agent lands",
                  fontsize=11.5, color=FS.INK, loc="left")
    ax.set_xticks(BUDGETS)
    ax.legend(frameon=False, fontsize=8.5)
    ax.margins(0.06)
    return d


def draw_panel_b(ax, cohort_path):
    cohort = load_real_cohort(pd.read_parquet(cohort_path), VARIANT_DOMAIN)
    y, genes = cohort.y, cohort.genes
    names, vals = [], []
    for ch in cohort.domain.channels:
        mask = np.zeros((len(cohort), cohort.domain.K), bool)
        j = cohort.domain.channels.index(ch)
        mask[:, j] = True
        vals.append(M.per_gene_auc(y, M._score_from_evidence(cohort, mask), genes))
        names.append(ch)
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    ax.bar(names, vals, color=FS.NAVY)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, color=FS.INK)
    ax.axhline(0.5, ls=":", lw=0.8, color=FS.GREY)
    ax.set_ylim(min(0.4, min(vals) - 0.05), max(vals) + 0.08)
    ax.set_ylabel("single-slot AUC")
    ax.set_title("B   No single ACMG category suffices",
                  fontsize=11.5, color=FS.INK, loc="left")
    return dict(zip(names, vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/variant/real_trusted/results.csv")
    ap.add_argument("--cohort", default="data/sample/cohort_full_real.parquet")
    ap.add_argument("--agent", default="agent:frontier_B",
                     help="strategy name in results.csv for the headline agent line")
    ap.add_argument("--agent-label", default="agent: gpt-5.5")
    ap.add_argument("--out", default="figures/variant/real_trusted/gate_4slot.png")
    args = ap.parse_args()
    os.chdir(_ROOT)
    FS.use()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    fig.patch.set_facecolor(FS.PARADIGM["paper"])
    for a in (axA, axB):
        a.set_facecolor(FS.PARADIGM["paper"])

    d = draw_panel_a(axA, args.results, args.agent, args.agent_label)
    slots = draw_panel_b(axB, args.cohort)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=fig.get_facecolor())
    print(f"written: {args.out}")
    kcols = [f"A_k{k}" for k in BUDGETS]
    for name, *_ in LINES + [(args.agent,)]:
        if name in d.index:
            print(f"  A: {name:20s} " + " ".join(f"{v:.4f}" for v in d.loc[name, kcols]))
    for name, v in slots.items():
        print(f"  B: {name:6s} {v:.4f}")


if __name__ == "__main__":
    main()
