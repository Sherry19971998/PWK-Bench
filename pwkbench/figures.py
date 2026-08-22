"""
Figure generation — four figures total: three that mirror the paper's figure
types (budget curve, single-slot bars, order-optimality histograms) plus the
new Plan-A scaling plot. All are drawn from the numbers the pipeline produces;
on the bundled synthetic/partial-real data the values are illustrative, not the
paper's.

Figures:
  fig_budget_curve   : A(k) for oracle / relmax / heuristic / agent (paper Fig A)
  fig_single_slot    : single-channel AUC bars (paper Fig B)
  fig_order_hist     : order-optimality rho histograms vs relevance & oracle
  fig_scaling        : NEW (Plan A) capability proxy -> oracle gap
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from .domains.base import Cohort
from . import strategies as S, metrics as M
from . import figstyle as FS

# House palette (see figstyle.py): one navy family + the single warm ACCENT
# reserved for the agent line, which is the subject of every figure this
# module draws. Was {"Oracle": "#2ca02c", "RelMax": "#d62728", "Heuristic":
# "#8c8c8c", "Agent": "#1f77b4"} -- unstyled matplotlib defaults, and the
# red/green pair is both the least accessible common combination (~8% of men
# have red-green deficiency) and inconsistent with every other figure in the
# tree, which uses figstyle.
_COL = {"Oracle": FS.NAVY, "RelMax": FS.STEEL, "Heuristic": FS.MUTE, "Agent": FS.ACCENT}


def fig_budget_curve(cohort: Cohort, agent_order=None, budgets=None, outpath="figures/budget_curve.png"):
    FS.use()
    budgets = budgets or [k for k in M.BUDGETS if k <= cohort.domain.K]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for name in ["Oracle", "RelMax", "Heuristic"]:
        A = M.curve_A(cohort, S.STRATEGY_FNS[name](cohort), budgets=budgets,
                      leq_k=(name == "Oracle"))
        ax.plot(list(A), list(A.values()), "o-", color=_COL[name], label=name.lower())
    if agent_order is not None:
        A = M.curve_A(cohort, agent_order, budgets=budgets)
        ax.plot(list(A), list(A.values()), "s-", color=_COL["Agent"], lw=2.5, label="agent")
    ax.axhline(0.5, ls=":", color="grey", lw=1)
    ax.set_xlabel("acquisition budget $k$"); ax.set_ylabel("ranking AUC (per-gene)")
    ax.set_title("Planning vs. relevance, and where the agent lands")
    ax.set_xticks(budgets); ax.legend(frameon=False, fontsize=8); ax.margins(0.04)
    fig.tight_layout(); fig.savefig(outpath, dpi=200); plt.close(fig)
    return outpath


def fig_single_slot(cohort: Cohort, outpath="figures/single_slot.png"):
    FS.use()
    y, genes = cohort.y, cohort.genes
    fig, ax = plt.subplots(figsize=(5, 3))
    names, vals = [], []
    for ch in cohort.domain.channels:
        mask = np.ones((len(cohort), cohort.domain.K), bool) * False
        j = cohort.domain.channels.index(ch)
        m = mask.copy(); m[:, j] = True
        vals.append(M.per_gene_auc(y, M._score_from_evidence(cohort, m), genes))
        names.append(ch)
    # Was a bare "#4c72b0" (seaborn-default blue, not the house family).
    ax.bar(names, vals, color=FS.NAVY)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)
    # floor at the lower of 0.5 and the smallest bar so a below-chance channel
    # is not silently clipped off the axis.
    ax.set_ylim(min(0.5, min(vals) - 0.03), max(vals) + 0.08)
    ax.set_ylabel("single-slot AUC")
    ax.axhline(0.5, ls=":", lw=0.8, color="0.6")
    ax.set_title("No single evidence category suffices")
    fig.tight_layout(); fig.savefig(outpath, dpi=200); plt.close(fig)
    return outpath


def fig_order_hist(cohort: Cohort, order, outpath="figures/order_optimality.png"):
    FS.use()
    oo = M.order_optimality(cohort, order)
    fig, axes = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
    # House palette: the top panel is the headline claim (the agent converges
    # to the relevance order), so it gets the single warm ACCENT; the bottom
    # panel is the reference comparison and stays in the navy family. Was
    # red/green (#d62728 / #2ca02c) -- unstyled matplotlib defaults, not
    # figstyle.
    axes[0].hist(oo["po_rel_per_variant"], bins=20, range=(0, 1), color=FS.ACCENT, alpha=0.75)
    axes[0].axvline(oo["po_vs_relevance"], color=FS.ACCENT)
    axes[0].set_title(f"vs. global relevance order  mean prefix-overlap={oo['po_vs_relevance']:.2f}", fontsize=9)
    axes[0].set_ylabel("count")
    axes[1].hist(oo["po_ora_per_variant"], bins=20, range=(0, 1), color=FS.NAVY, alpha=0.75)
    axes[1].axvline(oo["po_vs_oracle"], color=FS.NAVY)
    axes[1].set_title(f"vs. per-variant oracle order  mean prefix-overlap={oo['po_vs_oracle']:.2f}", fontsize=9)
    axes[1].set_ylabel("count")
    # Shortened from "...(higher = more aligned; drives $A(k)$)": that clause
    # was getting hard-clipped at the right edge before bbox_inches="tight"
    # was added below. The dropped context belongs in the caption, not an
    # axis label that has to fit in the figure width.
    axes[1].set_xlabel("per-variant prefix-set overlap")
    fig.suptitle("Agent acquires in relevance-greedy, not variant-adaptive, order", fontsize=10)
    fig.tight_layout()
    # bbox_inches="tight" so a long title/label cannot be clipped by a fixed
    # canvas size again; dpi raised from 200 to match the other figures in
    # the tree (1900-2340px wide -- this one was shipping at 1000x800).
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath


def fig_scaling(results_df, outpath="figures/scaling_gap.png", watermark=False):
    """Plan A: capability proxy -> oracle gap, one point per agent.

    `watermark=True` stamps 'SYNTHETIC — illustrative only' across the panel;
    the runner sets it whenever the model matrix is the mock/demo one, because
    the mock's capability axis is a stand-in that produces an inverted trend
    (see note below). Pass watermark=False for real-model runs.
    """
    if "capability_proxy" not in results_df.columns:
        return None
    d = results_df.dropna(subset=["capability_proxy"]).sort_values("capability_proxy")
    if len(d) < 2:                    # nothing to scale (e.g. the context ablation)
        return None
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(d["capability_proxy"], d["delta_oracle"], "o-", color="#1f77b4")
    for _, r in d.iterrows():
        ax.annotate(str(r["strategy"]).replace("agent:", ""),
                    (r["capability_proxy"], r["delta_oracle"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("capability proxy"); ax.set_ylabel("oracle gap $\\Delta_{\\mathrm{Oracle}}$")
    ax.set_title("Does capability close the planning gap?")
    ax.margins(0.08)
    if watermark:
        # The mock agent ties `adherence` to both capability AND P(follow
        # relevance), so lower-capability mocks mix in MORE oracle-aligned
        # orders and land CLOSER to the oracle — an inverted, constructive
        # artifact of the stand-in, not a finding. Real models overwrite this.
        ax.text(0.5, 0.5, "SYNTHETIC\nillustrative only", transform=ax.transAxes,
                ha="center", va="center", fontsize=20, color="0.80",
                rotation=25, zorder=0, alpha=0.6, fontweight="bold")
    fig.tight_layout(); fig.savefig(outpath, dpi=200); plt.close(fig)
    return outpath
