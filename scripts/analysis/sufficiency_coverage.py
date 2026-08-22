#!/usr/bin/env python3
"""E3 -- sufficiency selection-bias analysis.

Reviewer concern: "characterize how the excluded cases differ by
gene/variant type/class; does exclusion bias the over-acquisition rate?" The
retrospective sufficiency point k* (pwkbench/live/metrics_live.py) is defined
only for cases the reference ACMG-points classifier can resolve from SOME
subset of the three acquirable channels (PM2/PP3/PM1) plus the free PVS1 --
491 - 193 = 298 cases have no such subset (k*=NaN) and are excluded from
every over-acquisition number in the paper.

What this answers:
1. Do the 193 included / 298 excluded cases differ by gene, consequence type
   (LoF / missense / other), or gold label? (proportion tables + chi-square)
2. Is the headline "62% of cases exceed sufficiency" rate stable across the
   resolvable population's own strata, or concentrated in a few genes/types?

Uses only already-recorded live trajectories (results/live/trajectories_full*
.csv, the same 3-run gpt-5.5 pool scripts/live/make_live_figure.py uses) and
the frozen real cohort -- no new API calls.

USAGE
    python scripts/analysis/sufficiency_coverage.py \
        --cohort data/sample/cohort_full_real.parquet \
        --runs results/live/trajectories_full.csv results/live/trajectories_full_r1.csv results/live/trajectories_full_r2.csv \
        --out results/live/sufficiency_coverage.csv
"""
import argparse, os, sys

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench.domains.base import load_real_cohort              # noqa: E402
from pwkbench.domains.variant import VARIANT_DOMAIN             # noqa: E402
from pwkbench.live.metrics_live import sufficiency_points, over_acquisition  # noqa: E402

DEFAULT_RUNS = ["results/live/trajectories_full.csv",
                "results/live/trajectories_full_r1.csv",
                "results/live/trajectories_full_r2.csv"]


def consequence_type(row):
    if row["PVS1"] == 1:
        return "LoF"
    if bool(row["is_missense"]):
        return "missense"
    return "other"


def proportion_table(profile, group_col):
    """Included-vs-excluded counts/proportions within each level of group_col,
    plus a chi-square test of independence (does inclusion depend on this
    stratum?)."""
    ct = pd.crosstab(profile[group_col], profile["included"])
    chi2, p, dof, _ = chi2_contingency(ct)
    prop = ct.div(ct.sum(axis=1), axis=0)
    return ct, prop, chi2, p, dof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/sample/cohort_full_real.parquet")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--out", default="results/live/sufficiency_coverage.csv")
    args = ap.parse_args()
    os.chdir(_ROOT)

    cohort = load_real_cohort(pd.read_parquet(args.cohort), VARIANT_DOMAIN)
    suff = sufficiency_points(cohort)

    profile = cohort.df[["variant_id", "gene", "PVS1", "is_missense", "label"]].merge(
        suff[["variant_id", "k_star"]], on="variant_id")
    profile["included"] = profile["k_star"].notna()
    profile["consequence"] = profile.apply(consequence_type, axis=1)
    profile["gold"] = profile["label"].map({1: "P/LP", 0: "B/LB"})

    n_incl, n_excl = int(profile["included"].sum()), int((~profile["included"]).sum())
    print(f"Resolvable (k* defined): {n_incl} / {len(profile)}   "
          f"Excluded (k*=NaN): {n_excl} / {len(profile)}\n")

    print("=" * 78)
    print("PART 1 — does exclusion depend on gene / consequence / gold label?")
    print("=" * 78)
    strat_rows = []
    for col in ("gene", "consequence", "gold"):
        ct, prop, chi2, p, dof = proportion_table(profile, col)
        print(f"\n-- by {col} --")
        print(ct)
        print(f"chi2={chi2:.3f}  dof={dof}  p={p:.4f}"
              + ("  (exclusion is NOT independent of this stratum)" if p < 0.05
                 else "  (no evidence exclusion depends on this stratum)"))
        for level in ct.index:
            strat_rows.append(dict(
                part="coverage", stratifier=col, level=level,
                n_included=int(ct.loc[level, True]) if True in ct.columns else 0,
                n_excluded=int(ct.loc[level, False]) if False in ct.columns else 0,
                chi2=chi2, p=p))

    print("\n" + "=" * 78)
    print("PART 2 — is the 62% over-acquisition rate stable within the")
    print("         resolvable (included) population's own strata?")
    print("=" * 78)
    pooled = []
    for p in args.runs:
        oa = over_acquisition(pd.read_csv(p), suff)
        pooled.append(oa)
    oa = pd.concat(pooled, ignore_index=True)
    # `oa` (from trajectories_*.csv) already carries its own "gene" column;
    # merge only the profile columns that are NOT already there to avoid a
    # gene_x/gene_y suffix collision.
    oa = oa[oa["over_acq_mapped"].notna()].merge(
        profile[["variant_id", "consequence", "gold"]], on="variant_id")
    oa["exceeds"] = oa["over_acq_mapped"] > 0

    overall_rate = oa["exceeds"].mean()
    print(f"\nOverall (pooled {len(args.runs)} runs, resolvable cases only, "
          f"n={len(oa)}): {overall_rate:.1%} exceed sufficiency point")

    for col in ("gene", "consequence", "gold"):
        print(f"\n-- by {col} --")
        g = oa.groupby(col)["exceeds"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        g.columns = ["exceed_rate", "n_case_rows"]
        print(g.round(3))
        for level, r in g.iterrows():
            strat_rows.append(dict(
                part="over_acq_rate", stratifier=col, level=level,
                exceed_rate=r["exceed_rate"], n_case_rows=int(r["n_case_rows"])))

    spread = oa.groupby("gene")["exceeds"].mean()
    print(f"\nRange of per-gene exceed rate: [{spread.min():.1%}, {spread.max():.1%}] "
          f"vs overall {overall_rate:.1%}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(strat_rows).to_csv(args.out, index=False)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
