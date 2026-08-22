#!/usr/bin/env python3
"""Cross-vendor comparison table for both blocks, generated from the artifacts.

WHY A SCRIPT AND NOT A HAND-TYPED TABLE
---------------------------------------
The vendor comparison is the one place where a transcription slip is invisible:
two columns of plausible numbers look correct whichever run they came from. This
regenerates the whole table from the trajectory / variance CSVs, so the document
that quotes it can always be re-derived rather than trusted.

It also computes the one metric neither vendor's hand-off package contained:
over-acquisition against the per-case sufficiency point. Both packages reported
`mean_tools`, which is NOT comparable as an over-acquisition claim -- a model
that acquires less could simply be under-acquiring. The sufficiency point is the
smallest acquirable channel set that already yields the correct guideline call,
so "exceeds it" is the actual quantity the paper argues about.

USAGE
    python scripts/live/build_vendor_comparison.py \\
        --openai results/live/trajectories_full.csv,results/live/trajectories_full_r1.csv,results/live/trajectories_full_r2.csv \\
        --gemini <dir>/trajectories_vertex_pro_r0.csv,...r1.csv,...r2.csv \\
        --out results/live/vendor_comparison.csv
"""
import argparse, json, os, sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench.domains.base import load_real_cohort           # noqa: E402
from pwkbench.domains.variant import VARIANT_DOMAIN          # noqa: E402
from pwkbench.live.metrics_live import (                     # noqa: E402
    sufficiency_points, over_acquisition)


def _arm(paths, suff):
    """Per-run metrics for one vendor, then mean/SD across runs."""
    rows = []
    for p in paths:
        t = pd.read_csv(p)
        oa = over_acquisition(t, suff)
        col = next(c for c in oa.columns if "over" in c.lower())
        ok = oa[col].notna()
        err = t["stop_reason"].astype(str).str.startswith("error:")
        rows.append({
            "run": os.path.basename(p),
            # served_model, not the requested id: a run answered by a different
            # model is identical in every other column.
            "served_model": (t["served_model"].dropna().unique()[0]
                             if "served_model" in t.columns and t["served_model"].notna().any()
                             else t.get("model_id", pd.Series(["?"])).iloc[0]),
            "backend": (t["backend"].dropna().unique()[0]
                        if "backend" in t.columns and t["backend"].notna().any() else "api"),
            "n": len(t),
            "mean_tools": t["n_tools_called"].mean(),
            "zero_tool_frac": (t["n_tools_called"] == 0).mean(),
            "cost_rank_sum": t["cost_rank_sum"].mean(),
            "error_frac": err.mean(),
            # Accuracy is reported but is NOT the headline: the agent sees the
            # real HGVS name and this repo measured closed-book AUC 0.865 from
            # the name alone against 0.942 with all evidence, so the tool layer
            # has ~0.077 of room. Read the acquisition columns, not this one.
            "accuracy": (t["correct"].mean() if "correct" in t.columns
                         else np.nan),
            "n_suff_defined": int(ok.sum()),
            "over_acq_mean": oa.loc[ok, col].mean(),
            "exceed_suff_frac": (oa.loc[ok, col] > 0).mean(),
        })
    df = pd.DataFrame(rows)
    num = df.select_dtypes(include=[np.number]).columns
    mean = df[num].mean()
    sd = df[num].std(ddof=1)
    return df, mean, sd


def _stability(paths):
    """Per-case reproducibility across the runs of ONE vendor."""
    fr = {os.path.basename(p): pd.read_csv(p).set_index("variant_id") for p in paths}
    ids = sorted(set.intersection(*[set(d.index) for d in fr.values()]))
    n = pd.DataFrame({k: d.loc[ids, "n_tools_called"] for k, d in fr.items()})
    spread = n.max(axis=1) - n.min(axis=1)
    a = pd.DataFrame({k: d.loc[ids, "answer"] for k, d in fr.items()})
    complete = a.notna().all(axis=1)
    return {
        "identical_tools_frac": float((spread == 0).mean()),
        "spread_ge2_frac": float((spread >= 2).mean()),
        "identical_answer_frac": (float((a[complete].nunique(axis=1) == 1).mean())
                                  if complete.any() else np.nan),
        "n_cases": len(ids),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openai", required=True, help="comma-separated trajectory CSVs")
    ap.add_argument("--gemini", required=True, help="comma-separated trajectory CSVs")
    ap.add_argument("--cohort", default="data/sample/cohort_full_real.parquet")
    ap.add_argument("--out", default="results/live/vendor_comparison.csv")
    args = ap.parse_args()

    o_paths = [p.strip() for p in args.openai.split(",")]
    g_paths = [p.strip() for p in args.gemini.split(",")]

    cohort = load_real_cohort(pd.read_parquet(args.cohort), VARIANT_DOMAIN)
    suff = sufficiency_points(cohort)

    # Same cohort on both sides or the comparison is meaningless. Checked rather
    # than assumed: a subsampled run would otherwise produce a smaller but
    # entirely plausible-looking table.
    sets = [set(pd.read_csv(p)["variant_id"]) for p in o_paths + g_paths]
    if len({frozenset(s) for s in sets}) != 1:
        raise SystemExit("cohorts differ across the supplied runs -- refusing to "
                         "build a comparison table from mismatched case sets")

    od, om, osd = _arm(o_paths, suff)
    gd, gm, gsd = _arm(g_paths, suff)
    ost, gst = _stability(o_paths), _stability(g_paths)

    rows = []
    METRICS = [("mean_tools", "tools acquired per case"),
               ("exceed_suff_frac", "cases exceeding sufficiency point"),
               ("over_acq_mean", "over-acquisition (channels)"),
               ("zero_tool_frac", "answered with zero tools"),
               ("cost_rank_sum", "cost-rank sum per case"),
               ("accuracy", "accuracy (see caveat)"),
               ("error_frac", "API error rate")]
    for k, label in METRICS:
        rows.append({"metric": label, "key": k,
                     "openai_mean": om.get(k), "openai_sd": osd.get(k),
                     "gemini_mean": gm.get(k), "gemini_sd": gsd.get(k)})
    for k, label in [("identical_tools_frac", "per-case: identical tool count in all runs"),
                     ("spread_ge2_frac", "per-case: differs by >=2 tools"),
                     ("identical_answer_frac", "per-case: identical diagnosis in all runs")]:
        rows.append({"metric": label, "key": k,
                     "openai_mean": ost[k], "openai_sd": np.nan,
                     "gemini_mean": gst[k], "gemini_sd": np.nan})
    out = pd.DataFrame(rows)

    print("=== per-run detail ===")
    for name, d in (("OpenAI", od), ("Gemini", gd)):
        print(f"\n{name}: served={d['served_model'].unique().tolist()} "
              f"backend={d['backend'].unique().tolist()}")
        print(d.drop(columns=["served_model", "backend"]).to_string(
            index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== cross-vendor comparison (mean +- SD over runs) ===")
    for _, r in out.iterrows():
        o_s = "" if pd.isna(r.openai_sd) else f" ± {r.openai_sd:.4f}"
        g_s = "" if pd.isna(r.gemini_sd) else f" ± {r.gemini_sd:.4f}"
        print(f"  {r.metric:<44} {r.openai_mean:.4f}{o_s:<10}   {r.gemini_mean:.4f}{g_s}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    meta = {"openai_runs": o_paths, "gemini_runs": g_paths,
            "openai_served": sorted(od["served_model"].unique()),
            "gemini_served": sorted(gd["served_model"].unique()),
            "openai_backend": sorted(od["backend"].unique()),
            "gemini_backend": sorted(gd["backend"].unique()),
            "n_cases": int(od["n"].iloc[0]),
            "n_sufficiency_defined": int(od["n_suff_defined"].iloc[0])}
    with open(args.out.replace(".csv", "_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
