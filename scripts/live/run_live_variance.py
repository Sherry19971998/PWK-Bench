#!/usr/bin/env python3
"""Repeated live-agent runs + a run-to-run variance summary.

WHY THIS EXISTS
---------------
The live block had three runs of the `full` arm (`full`, `full_r1`, `full_r2`)
produced by hand with `--tag`, and NO code that summarised them -- so the spread
existed on disk but was never reported, and a reader had no way to tell whether
`mean tools/case = 1.955` was stable or a single draw. The frozen block has
`run_variance.py` for exactly this; this is its live-block counterpart.

It also answers a question the aggregate numbers hide. Measured on the three
existing OpenAI runs: the aggregate acquisition rate is stable (CV ~1.5%) while
only 47% of INDIVIDUAL variants get the same tool count in all three runs. So
aggregate claims replicate and per-case anecdotes do not -- this script reports
both so that distinction cannot be lost.

TWO MODES
---------
1. Run and summarise (default):
       python scripts/live/run_live_variance.py --vendor gemini \\
           --model gemini-3.5-flash-lite --runs 3 --tag gemini_full

   Each repeat gets its own `--tag` (`<tag>_r0`, `_r1`, ...) so run_live.py's
   checkpoint files stay separate and a repeat can resume after a quota wall
   without contaminating another repeat.

2. Summarise runs that already exist (no API calls, no spend):
       python scripts/live/run_live_variance.py --summarize-only \\
           --tags full,full_r1,full_r2 --label openai_full

RATE LIMITS
-----------
Default `--workers 1`. Gemini's free tier is 15 requests/minute and the live
loop issues ~3 requests per case; with 4 workers a 40-case smoke lost 31/37
cases to 429s before retry handling existed (measured 2026-08-02). Raise
`--workers` only against an endpoint you have confirmed can take it.
"""
import argparse, json, os, subprocess, sys

import numpy as np
import pandas as pd

# Scripts live one level deeper than the repo root (scripts/<block>/x.py),
# so the root is three dirnames up, not two.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

# Aggregate quantities summarised across repeats. `n_tools_called` is the
# headline (the over-acquisition measurement); the rest characterise the tails
# that a single mean would hide.
_METRICS = ("mean_tools", "zero_tool_frac", "all_tools_frac", "cost_rank_sum",
            "answered_frac", "error_frac")


def _summarise_one(df: pd.DataFrame) -> dict:
    n = max(len(df), 1)
    ntools = df["n_tools_called"]
    # An API error is recorded as a row with n_tools_called == 0, which is
    # INDISTINGUISHABLE from a genuine "acquired nothing" decision in every
    # aggregate. It is reported separately so a run with a bad error rate
    # cannot be read as a run with low acquisition.
    err = df["stop_reason"].astype(str).str.startswith("error:").sum()
    return {
        "n": len(df),
        "mean_tools": ntools.mean(),
        "zero_tool_frac": (ntools == 0).mean(),
        "all_tools_frac": (ntools == ntools.max()).mean() if len(df) else np.nan,
        "cost_rank_sum": df["cost_rank_sum"].mean(),
        "answered_frac": df["answer"].notna().mean(),
        "error_frac": err / n,
    }


def _per_case_stability(frames: dict) -> dict:
    """Reproducibility at the level of the individual case.

    The aggregate CV can look reassuring while individual trajectories are
    almost unrelated between runs, because cases that acquire more and cases
    that acquire less cancel out in the mean (measured on the three OpenAI runs:
    81 variants went up, 101 went down, net -2.8%). Two consequences, and this
    function exists so both are traceable to an artifact rather than recomputed
    ad hoc:

    1. A specific case CANNOT be quoted as an anecdote unless it is in the
       stable fraction -- someone re-running the code will get a different
       number for it.
    2. The instability is itself a reportable clinical result, stated as a
       population statistic. "N% of patients receive a materially different
       workup on a repeat consultation" is a reproducibility claim about the
       agent, computed over the whole cohort, and it is the live paradigm's
       sharpest finding -- the frozen no-context arm has exactly zero spread
       and cannot see it at all.

    `spread` is max-min tool count across repeats; `>=2` is the threshold for
    "materially different workup" (one extra test is arguably noise, two is a
    different plan). `answer_identical_frac` is the same question asked of the
    DIAGNOSIS rather than the tests -- the clinically weightier of the two.
    """
    ids = set.intersection(*[set(d["variant_id"]) for d in frames.values()])
    if not ids:
        return {"common_cases": 0, "identical_frac": np.nan, "n_differing": np.nan}
    ids = sorted(ids)
    n = pd.DataFrame({t: d.set_index("variant_id").loc[ids, "n_tools_called"]
                      for t, d in frames.items()})
    spread = n.max(axis=1) - n.min(axis=1)
    out = {
        "common_cases": len(ids),
        "identical_frac": float((spread == 0).mean()),
        "n_differing": int((spread > 0).sum()),
        "spread_ge2_frac": float((spread >= 2).mean()),
        "n_spread_ge2": int((spread >= 2).sum()),
        "spread_distribution": {int(k): int(v)
                                for k, v in spread.value_counts().sort_index().items()},
    }
    # Diagnosis agreement. NaN answers (an errored case) are excluded rather
    # than counted as a disagreement, so a bad run degrades coverage instead of
    # masquerading as clinical instability.
    a = pd.DataFrame({t: d.set_index("variant_id").loc[ids, "answer"]
                      for t, d in frames.items()})
    complete = a.notna().all(axis=1)
    if complete.any():
        agree = a[complete].nunique(axis=1) == 1
        out["answer_cases_scored"] = int(complete.sum())
        out["answer_identical_frac"] = float(agree.mean())
        out["n_answer_differing"] = int((~agree).sum())
    else:
        out["answer_cases_scored"] = 0
        out["answer_identical_frac"] = np.nan
        out["n_answer_differing"] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", choices=["openai", "gemini"], default="gemini")
    ap.add_argument("--model", default=None,
                    help="model id; defaults per vendor (gemini-2.5-pro / "
                         "gpt-5.5-2026-04-23), matching run_live.py")
    ap.add_argument("--cohort", default="data/sample/cohort_full_real.parquet")
    ap.add_argument("--outdir", default="results/live")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tag", default=None,
                    help="base tag; repeats become <tag>_r0, _r1, ...")
    ap.add_argument("--max-variants", type=int, default=None,
                    help="gene-stratified subsample for a cheap smoke")
    ap.add_argument("--workers", type=int, default=1,
                    help="keep at 1 on rate-limited endpoints (Gemini free tier "
                         "is 15 req/min and each case costs ~3 requests)")
    ap.add_argument("--summarize-only", action="store_true",
                    help="summarise existing trajectory files; makes NO API calls")
    ap.add_argument("--tags", default=None,
                    help="with --summarize-only: comma-separated existing tags")
    ap.add_argument("--label", default=None,
                    help="name for the output summary file")
    args = ap.parse_args()
    if args.model is None:
        args.model = ("gemini-2.5-pro" if args.vendor == "gemini"
                      else "gpt-5.5-2026-04-23")

    if args.summarize_only:
        if not args.tags:
            raise SystemExit("--summarize-only requires --tags a,b,c")
        tags = [t.strip() for t in args.tags.split(",")]
        label = args.label or "summary"
    else:
        base = args.tag or f"{args.vendor}_full"
        tags = [f"{base}_r{i}" for i in range(args.runs)]
        label = args.label or base
        for i, t in enumerate(tags):
            print(f"\n[variance] run {i+1}/{len(tags)}  tag={t}", flush=True)
            cmd = [sys.executable, os.path.join(_ROOT, "scripts/live/run_live.py"),
                   "--vendor", args.vendor, "--model", args.model,
                   "--cohort", args.cohort, "--outdir", args.outdir,
                   "--tag", t, "--workers", str(args.workers)]
            if args.max_variants:
                cmd += ["--max-variants", str(args.max_variants)]
            # check=True on purpose: a failed repeat must stop the sweep rather
            # than silently produce a 2-run "variance" reported as 3.
            subprocess.run(cmd, check=True)

    frames = {}
    for t in tags:
        p = os.path.join(args.outdir, f"trajectories_{t}.csv")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run it first, or fix --tags")
        frames[t] = pd.read_csv(p)

    rows = [dict(run=t, **_summarise_one(d)) for t, d in frames.items()]
    tab = pd.DataFrame(rows).set_index("run")
    stats = tab[list(_METRICS)]
    tab.loc["MEAN"] = {**{"n": tab["n"].mean()}, **stats.mean().to_dict()}
    tab.loc["SD"] = {**{"n": 0}, **stats.std(ddof=1).to_dict()}

    stab = _per_case_stability(frames)

    print(f"\n=== live variance: {label}  ({len(frames)} runs) ===")
    print(tab.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nper-case reproducibility over {stab['common_cases']} shared cases:")
    print(f"  identical tool count in ALL runs   : {stab['identical_frac']:.1%}")
    print(f"  differing in at least one run      : {stab['n_differing']}")
    print(f"  differing by >=2 tools             : {stab['n_spread_ge2']} "
          f"({stab['spread_ge2_frac']:.1%})   <- materially different workup")
    print(f"  spread distribution (max-min)      : {stab['spread_distribution']}")
    if stab["answer_cases_scored"]:
        print(f"  identical DIAGNOSIS in ALL runs    : "
              f"{stab['answer_identical_frac']:.1%} "
              f"({stab['n_answer_differing']} cases disagree, "
              f"n={stab['answer_cases_scored']})")
    print("  -> Report these as POPULATION statistics ('N% of cases are not "
          "reproducible'), which is a valid and clinically meaningful claim. "
          "Do NOT quote an individual case as an anecdote unless it is in the "
          "identical fraction -- a re-run will contradict it.")

    if (tab.loc["MEAN", "error_frac"] or 0) > 0.01:
        print(f"\n  WARNING: mean error_frac = {tab.loc['MEAN','error_frac']:.1%}. "
              f"Errored cases are recorded with n_tools_called=0 and are "
              f"indistinguishable from a genuine zero-acquisition decision in "
              f"every aggregate above. Fix the errors before reporting.")

    out_csv = os.path.join(args.outdir, f"variance_{label}.csv")
    tab.to_csv(out_csv)
    with open(os.path.join(args.outdir, f"variance_{label}_stability.json"), "w") as fh:
        json.dump(stab, fh, indent=2)
    print(f"\nwritten: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
