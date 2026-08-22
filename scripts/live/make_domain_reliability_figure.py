#!/usr/bin/env python3
"""Repeat-run answer reliability, broken down by clinical domain -- an HONEST NULL.

WHY THIS FIGURE, AND WHY IT LOOKS DIFFERENT FROM THE OTHER TWO
------------------------------------------------------------------
`figures/live/domain_overacquisition.png` and
`figures/variant/domain_gap_excess.png` both found a real, statistically
distinguishable concentration in one clinical domain (arrhythmia's spend/need
multiple; lipid_disorder's planning-gap excess). The natural next question is
whether the cohort's THIRD live-block headline -- 12.4% of cases flip their
pathogenic/benign call across 3 identical repeats
(`variance_openai_full_stability.json`) -- shows the same kind of domain
concentration. It does NOT, and this figure exists to show that
checked-and-absent result rather than let the question go unanswered.

    domain              n    flip rate   95% Wilson CI      p vs rest (Fisher)
    arrhythmia          84    17.9%      [11.1%, 27.4%]         0.104
    lipid_disorder      69    13.0%      [ 7.0%, 22.9%]         0.845
    hereditary_cancer  254    11.8%      [ 8.4%, 16.4%]         0.684
    cardiomyopathy      84     8.3%      [ 4.1%, 16.2%]         0.276

Omnibus chi-square (domain x flip): chi2=3.68, p=0.298. arrhythmia's point
estimate is the highest, but its CI overlaps every other domain's, the
omnibus test is not significant, and none of the four per-domain-vs-rest
Fisher tests reaches p<0.05 even BEFORE a Holm correction for testing four
domains (which would only push these p-values higher). Contrast with
`domain_gap_excess.png`, where lipid_disorder's z=3.44 survives Holm
correction across the same four domains.

THE POINT OF PUBLISHING A NULL. Reliability failing to concentrate the same
way over-acquisition and the planning gap do is itself informative: it means
arrhythmia is not uniformly "the hard domain" across every metric this repo
measures, only on evidence acquisition. Selectively showing only the two
figures that found something would imply every axis was checked and
concentrated; this figure is what "checked, and did not concentrate" looks
like, and belongs alongside the other two for that reason.

CAVEAT. n=84 (or 69) per domain gives Wilson intervals ~10-15 points wide at
this base rate -- absence of significance here is compatible with a real but
smaller domain effect than the ~4x concentration the other two figures found;
it is not evidence that no domain difference in reliability could exist at
larger n.

Every number is read from the artifacts at run time. No LLM calls (uses the
already-collected `answer` column across the 3 existing live runs).

USAGE
    python scripts/live/make_domain_reliability_figure.py
"""
import argparse, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench import figstyle as FS                          # noqa: E402
from pwkbench.domains.base import load_real_cohort           # noqa: E402
from pwkbench.domains.variant import VARIANT_DOMAIN          # noqa: E402
from pwkbench.spec import GENES_BY_DOMAIN                    # noqa: E402

LIVE_RUNS = ["results/live/trajectories_full.csv",
             "results/live/trajectories_full_r1.csv",
             "results/live/trajectories_full_r2.csv"]
Z95 = 1.959963984540054


def _wilson_ci(k, n, z=Z95):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return center - half, center + half


def _compute():
    coh = load_real_cohort(
        pd.read_parquet("data/sample/cohort_full_real.parquet"), VARIANT_DOMAIN)
    meta = coh.df.set_index("variant_id")[["domain"]]

    frames = [pd.read_csv(p).set_index("variant_id") for p in LIVE_RUNS]
    ids = sorted(set.intersection(*[set(d.index) for d in frames]))
    a = pd.DataFrame({i: d.loc[ids, "answer"] for i, d in enumerate(frames)})
    complete = a.notna().all(axis=1)
    agree = a.nunique(axis=1) == 1

    df = pd.DataFrame({"domain": meta.loc[ids, "domain"].to_numpy(),
                       "complete": complete.to_numpy(),
                       "agree": agree.to_numpy()}, index=ids)
    scored = df[df["complete"]].copy()
    scored["flip"] = ~scored["agree"]

    tab = pd.crosstab(scored["domain"], scored["flip"])
    chi2, p_omni, _, _ = chi2_contingency(tab)

    rows = []
    for d in sorted(scored["domain"].unique()):
        m = scored["domain"] == d
        n_d, k_d = int(m.sum()), int(scored.loc[m, "flip"].sum())
        n_rest, k_rest = int((~m).sum()), int(scored.loc[~m, "flip"].sum())
        _, p = fisher_exact([[k_d, n_d - k_d], [k_rest, n_rest - k_rest]])
        lo, hi = _wilson_ci(k_d, n_d)
        rows.append(dict(domain=d, n=n_d, n_flip=k_d, flip_rate=k_d / n_d,
                         ci_lo=lo, ci_hi=hi, p_vs_rest=p))
    res = pd.DataFrame(rows).sort_values("flip_rate", ascending=False)
    overall = scored["flip"].mean()
    return res, overall, chi2, p_omni, len(scored)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures/live/domain_reliability.png")
    args = ap.parse_args()
    os.chdir(_ROOT)
    FS.use()

    res, overall, chi2, p_omni, n_scored = _compute()

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    xs = np.arange(len(res))
    lo_err = res["flip_rate"] - res["ci_lo"]
    hi_err = res["ci_hi"] - res["flip_rate"]

    ax.bar(xs, res["flip_rate"], width=.56, color=FS.STEEL, edgecolor="white",
          linewidth=1.0, zorder=3)
    ax.errorbar(xs, res["flip_rate"], yerr=[lo_err, hi_err], fmt="none",
               ecolor=FS.INK, elinewidth=1.4, capsize=5, capthick=1.4, zorder=4)

    h_avg = ax.axhline(overall, color=FS.GREY, ls=":", lw=1.6, zorder=2)
    ax.legend([h_avg], [f"{overall:.1%} cohort-wide average"],
             loc="upper right", framealpha=.95, fontsize=9.5)

    for xi, (_, row) in zip(xs, res.iterrows()):
        ax.text(xi, row["ci_hi"] + .015, f"{row['flip_rate']:.1%}",
               ha="center", fontsize=13, fontweight="bold", color=FS.NAVY)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{d}\nn={n}" for d, n in zip(res["domain"], res["n"])],
                       fontsize=11)
    ax.set_ylim(0, res["ci_hi"].max() + .13)
    ax.set_ylabel("fraction of cases: different call across 3 identical repeats")
    ax.set_title("Reliability does NOT concentrate by domain (honest null)",
                pad=14)
    ax.text(0.02, .90,
           f"omnibus χ²={chi2:.2f}, p={p_omni:.2f} (n={n_scored}) — no domain\n"
           f"reaches p<0.05 vs. the rest, even before correcting for testing 4",
           transform=ax.transAxes, fontsize=9, color=FS.MUTE, va="top")

    gene_line = "   ·   ".join(f"{d}: {', '.join(GENES_BY_DOMAIN[d])}"
                               for d in res["domain"])
    ax.text(0.5, -.16, gene_line, transform=ax.transAxes, ha="center",
           va="top", fontsize=8.4, color=FS.MUTE)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"written: {args.out}")
    print(res.to_string(index=False))
    print(f"overall flip rate {overall:.4f}  omnibus chi2={chi2:.3f} p={p_omni:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
