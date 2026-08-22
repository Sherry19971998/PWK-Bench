#!/usr/bin/env python3
"""Clinical endpoints: what the acquisition policy costs and yields, in lab KPIs.

WHY THIS EXISTS
---------------
An earlier `figures/variant/analysis/clinical_endpoints.png` was half usable — its
panel B rests on a 10x cost assumption that was retracted, so putting it on a
poster would carry a withdrawn assumption along with it. This is its
replacement, computed from the artifacts.

WHAT IT HAS TO GET RIGHT
------------------------
The behaviour figure says the agent acquires more than it needs. Read alone,
that invites the conclusion that acquisition does not help — which the data
contradicts: under the ACMG rule a flat budget resolves 151 -> 177 -> 178 -> 191
cases at k=1..4. Both facts are true, and the resolution is that the claim is
about ALLOCATION, not about evidence being useless. So the panels are ordered to
make that impossible to misread:

  A  the agent's self-chosen stopping sits off the yield frontier: it spends
     2.66 queries for what a flat 2.00 already resolves.

     STATE THIS AS "no gain", NEVER "worse". The counts are 176 against 177 —
     a ONE-variant difference, which is noise, and FINDINGS §1.2 says so
     outright. The defensible claim is +33% spend for no yield benefit; a
     figure that lets a reader read 176 < 177 as a deficit is overclaiming on
     a single case.
  B  where the agent's ordering does win — at k=3 it resolves 27 cases RelMax
     misses against 4 the other way (paired McNemar, p = 3.4e-05)

  A yield-vs-budget panel was dropped from this figure on 2026-08-02: it said
  the same thing as `figures/variant/clinical_endpoints.png`, and two
  figures making one point is what made this one hard to read. The "acquisition
  does help" premise that panel A supplied now lives in that figure, and panel A
  here presupposes it — the frontier line IS the yield curve.
  D  and the endpoint a lab would care about most: run the SAME case three times
     and 12.4% of variants come back with a different pathogenic/benign call.

     Three checks were run before this was allowed onto a figure, because a
     number this quotable has to survive the same scrutiny that collapsed five
     other effects in this work:
       * WHAT KIND of disagreement — decisive. `final_answer` offers only
         {Pathogenic, Benign}; there is no VUS option, so all 61 disagreements
         are direct pathogenic<->benign FLIPS, not a softening to uncertain.
       * HOW STABLE — pairwise disagreement across the three run pairs is
         9.6% / 7.1% / 8.1% (mean 8.3%, SD 1.2%), so it is not one run's
         artifact; gene-clustered bootstrap (B=2000) puts the three-run figure
         at 12.4%, 95% CI [8.9%, 16.2%].
       * DIRECTION OF THE BOUND — with only three repeats this is a LOWER
         bound: more runs give more opportunities to observe a flip.

     Frame this as RELIABILITY, not safety. "Safety" carries a regulatory
     meaning that requires evidence of patient harm, and none was collected
     here. Reliability is a precondition for deployment, which is a claim the
     data does support: a system whose pathogenic/benign call changes on a
     rerun cannot be used for a clinical decision. The clinical analogue is
     intra-observer agreement, a standard QA measure.

     The frozen block cannot see this at all — its no-context arm is
     bit-identical across repeats.

TWO SCORERS, AND THEY DISAGREE ABOUT ACCURACY
---------------------------------------------
`acmg_*` is the guideline point rule; `logistic_*` is a data-driven scorer. Yield
rises under both. Call accuracy is flat at 1.000 under the ACMG rule and falls
1.000 -> 0.966 under the logistic scorer. Reporting only the second would present
a scorer-dependent effect as a property of the guideline, so panel A shows both.

USAGE
    python scripts/variant/make_clinical_figure.py
"""
import argparse, glob, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench import figstyle as FS                          # noqa: E402

SIG = "results/variant/real_trusted/yield_significance.csv"
STOP_GLOB = "results/variant/stopping/stopping_frontier_B_run*.csv"
STAB = "results/live/variance_openai_full_stability.json"
INK, MUTE = FS.INK, "#4d5c68"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/variant/clinical_reliability.png")
    args = ap.parse_args()
    os.chdir(_ROOT)
    FS.use()

    st = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(STOP_GLOB))])
    g = st.groupby("rule")[["mean_queries", "n_called"]].mean()
    ag_q = g.loc["agent_stop:frontier_B", "mean_queries"]
    ag_n = g.loc["agent_stop:frontier_B", "n_called"]
    fixed = [(g.loc[f"agent_fixed_k{i}", "mean_queries"],
              g.loc[f"agent_fixed_k{i}", "n_called"]) for i in range(1, 5)]

    sig = pd.read_csv(SIG)
    row = sig[(sig.strategy == "frontier_B") & (sig.reference == "RelMax")
              & (sig.budget == 3)].iloc[0]

    stab = json.load(open(STAB))

    fig, (axB, axC, axD) = plt.subplots(1, 3, figsize=(11.6, 4.15))
    fig.patch.set_facecolor(FS.PARADIGM["paper"])
    for a in (axB, axC, axD):
        a.set_facecolor(FS.PARADIGM["paper"])

    # ---- B: but the agent's stopping is off the frontier ------------------
    fx, fy = zip(*fixed)
    axB.plot(fx, fy, "o-", color=FS.NAVY, lw=2.4, ms=7,
             label="flat budget")
    for xx, yy, lab in zip(fx, fy, ["k=1", "k=2", "k=3", "k=4"]):
        axB.annotate(lab, (xx, yy), textcoords="offset points", xytext=(7, 7),
                     fontsize=10, color=MUTE)
    axB.plot([ag_q], [ag_n], "*", color=FS.ACCENT, ms=22, zorder=5,
             label="agent's own stopping")
    axB.annotate(f"agent: {ag_q:.2f} queries\n+33% for no gain",
                 xy=(ag_q, ag_n), xytext=(ag_q - 1.55, ag_n - 15),
                 fontsize=11, fontweight="bold", color=FS.ACCENT, linespacing=1.4,
                 arrowprops=dict(arrowstyle="->", color=FS.ACCENT, lw=1.6))
    # The 176/177 gap is one variant. Saying so on the figure is cheaper than
    # letting a reader infer a deficit from it.
    axB.set_xlabel("queries per variant"); axB.set_ylabel("variants resolved")
    axB.set_xlim(.6, 4.4)
    axB.legend(fontsize=10.5, loc="upper left", frameon=False,
               markerscale=.45, handletextpad=.5, borderpad=0)
    # Descriptive titles, matching the yield figure. "More queries, no more
    # resolved" and "Better order, though" stated the findings, but a reader
    # meeting them cold could not tell what was plotted, and "though" is a
    # transition out of a sentence that is not on the poster. Each finding is
    # still on its own panel in accent colour: "+33% for no gain", the p value,
    # and the 12.4%.
    axB.set_title("A   Agent stopping vs. flat budget",
                  fontsize=13, fontweight="bold", color=INK, loc="left", pad=10)

    # ---- C: where its ordering does win -----------------------------------
    a_only, b_only = int(row.b_resolved_only_by_a), int(row.c_resolved_only_by_b)
    axC.bar([0, 1], [a_only, b_only], width=.55,
            color=[FS.NAVY, FS.SKY], edgecolor="white", linewidth=1.2)
    for x, v in ((0, a_only), (1, b_only)):
        axC.text(x, v + .8, str(v), ha="center", fontsize=13, fontweight="bold",
                 color=INK)
    axC.set_xticks([0, 1])
    axC.set_xticklabels(["only the agent\nresolved these",
                         "only the fixed\norder resolved these"], fontsize=10)
    axC.set_ylabel("variants (budget k=3)")
    axC.set_ylim(0, a_only * 1.75)
    axC.set_title("B   Discordant pairs at k = 3", fontsize=13,
                  fontweight="bold", color=INK, loc="left", pad=10)
    axC.text(.5, a_only * 1.52, f"p = {row.p_value:.1e}", ha="center",
             fontsize=13, fontweight="bold", color=FS.ACCENT)

    # ---- D: the same case, three times -----------------------------------
    same = stab["answer_identical_frac"]
    diff, n_diff = 1 - same, stab["n_answer_differing"]
    axD.barh([1], [same], color=FS.NAVY, edgecolor="white", height=.42)
    axD.barh([1], [diff], left=[same], color=FS.ACCENT, edgecolor="white",
             height=.42)
    axD.text(same / 2, 1, f"{same:.1%}", ha="center", va="center", color="white",
             fontsize=13, fontweight="bold")
    axD.text(same + diff / 2, 1.42, f"{diff:.1%}", ha="center", va="bottom",
             color=FS.ACCENT, fontsize=15, fontweight="bold")
    axD.text(same + diff / 2, 0.55, f"{n_diff} variants", ha="center", va="top",
             color=FS.ACCENT, fontsize=9)

    ge2 = stab["spread_ge2_frac"]
    axD.barh([0], [1 - ge2], color=FS.SKY, edgecolor="white", height=.42)
    axD.barh([0], [ge2], left=[1 - ge2], color="#d99a8f", edgecolor="white",
             height=.42)
    axD.text((1 - ge2) / 2, 0, f"{1 - ge2:.1%}", ha="center", va="center",
             color="white", fontsize=13, fontweight="bold")
    axD.text(1 - ge2 / 2, -.42, f"{ge2:.1%}", ha="center", va="top",
             color="#b0554a", fontsize=13, fontweight="bold")

    axD.set_yticks([1, 0])
    axD.set_yticklabels(["same answer", "same evidence"], fontsize=11)
    # Lifted clear of the 12.4% figure, which sits at y=1.42 on the right.
    axD.text(0.0, 1.80, "every disagreement is a reversal", ha="left",
             fontsize=10.5, color=FS.ACCENT)
    axD.set_xlim(0, 1.02); axD.set_xticks([0, .5, 1])
    axD.set_xticklabels(["0", "50%", "100%"])
    axD.set_ylim(-.9, 1.9)
    axD.set_xlabel("share of 491 variants")
    axD.grid(axis="x", alpha=.3)
    # "The answer is not reproducible" was both lay phrasing and an overstatement
    # of what this panel plots: 87.6% of calls ARE identical across the three
    # repeats. What is plotted is an agreement rate, and the QA measure it is the
    # analogue of has a name -- this file's own docstring already calls it
    # intra-observer agreement. Use the term; let the accent numbers carry the
    # claim.
    axD.set_title("C   Intra-observer agreement", fontsize=13,
                  fontweight="bold", color=INK, loc="left", pad=8)

    fig.tight_layout()
    # The "177 vs 176 is one variant — noise" line that used to sit here is
    # gone by request. The misreading it guarded against — 176 < 177 read as a
    # deficit — is now closed in the poster's clinical box instead, which says
    # "one variant, i.e. no yield benefit" in words. If this figure is ever
    # used outside that poster, the caveat has to travel with it.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"written: {args.out}")
    print(f"  A  agent {ag_q:.2f}q -> {int(ag_n)};  flat k=2 {fixed[1][0]:.2f}q "
          f"-> {int(fixed[1][1])}")
    print(f"  B  {a_only} vs {b_only} discordant, p={row.p_value:.2e}")
    print(f"  C  call changes {1 - stab['answer_identical_frac']:.1%} "
          f"({stab['n_answer_differing']} variants)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
