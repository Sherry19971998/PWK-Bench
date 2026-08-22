#!/usr/bin/env python3
"""E6 -- calibration + coverage-accuracy-cost Pareto.

Reviewer concern: "too much AUC, add complementary metrics (ECE/Brier,
coverage-accuracy-cost Pareto)". Uses only already-recorded live trajectories
(`confidence` column, self-reported per-case by the agent) -- no new API
calls.

Two deliverables, computed for BOTH the closed-book arm
(`results/live/trajectories_notools.csv`) and the full-acquisition arm
(pooled 3-run `results/live/trajectories_full*.csv`):

1. ECE (equal-width confidence bins) and Brier score, treating `confidence`
   as the model's self-reported P(its own answer is correct) -- the standard
   reliability metrics for a binary correct/incorrect outcome.
2. A coverage-accuracy-cost Pareto: sweep a confidence ABSTENTION threshold
   tau in [0.5, 1.0]; at each tau, "answer" iff confidence >= tau, else
   abstain. Report coverage (fraction answered), accuracy among answered, and
   mean tools/case among answered, at each tau. This is what "the agent could
   trade cost for coverage" looks like as an operating curve, not a single
   number.

USAGE
    python scripts/analysis/calibration_pareto.py \
        --closed-book results/live/trajectories_notools.csv \
        --full-runs results/live/trajectories_full.csv results/live/trajectories_full_r1.csv results/live/trajectories_full_r2.csv \
        --out-prefix results/live/calibration_pareto
"""
import argparse, os, sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)


def ece_brier(df, n_bins=10):
    """Equal-width-bin ECE and Brier score for `confidence` vs `correct`."""
    conf = df["confidence"].to_numpy(float)
    correct = df["correct"].to_numpy(float)
    brier = float(np.mean((conf - correct) ** 2))
    edges = np.linspace(0.5, 1.0, n_bins + 1)  # confidence is reported >= 0.5
    bin_idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    ece, rows = 0.0, []
    n = len(df)
    for b in range(n_bins):
        m = bin_idx == b
        if not m.any():
            continue
        bin_conf, bin_acc, bin_n = conf[m].mean(), correct[m].mean(), int(m.sum())
        ece += (bin_n / n) * abs(bin_conf - bin_acc)
        rows.append(dict(bin_lo=edges[b], bin_hi=edges[b + 1],
                          mean_confidence=bin_conf, empirical_accuracy=bin_acc, n=bin_n))
    return ece, brier, pd.DataFrame(rows)


def pareto_curve(df, taus=None):
    """Coverage / accuracy-if-answered / mean-tools-if-answered at each
    confidence abstention threshold tau."""
    if taus is None:
        taus = np.round(np.arange(0.5, 1.001, 0.05), 2)
    rows = []
    n = len(df)
    for tau in taus:
        ans = df[df["confidence"] >= tau]
        rows.append(dict(
            tau=tau, coverage=len(ans) / n,
            accuracy_if_answered=ans["correct"].mean() if len(ans) else np.nan,
            mean_tools_if_answered=ans["n_tools_called"].mean() if len(ans) else np.nan,
            n_answered=len(ans)))
    return pd.DataFrame(rows)


def report_arm(name, df):
    print(f"\n{'='*70}\n{name}  (n={len(df)}, overall accuracy={df['correct'].mean():.3f}, "
          f"mean confidence={df['confidence'].mean():.3f})\n{'='*70}")
    ece, brier, bins = ece_brier(df)
    print(f"ECE={ece:.4f}   Brier={brier:.4f}")
    print(bins.round(3).to_string(index=False))
    pareto = pareto_curve(df)
    print("\ncoverage-accuracy-cost Pareto:")
    print(pareto.round(3).to_string(index=False))
    return dict(arm=name, ece=ece, brier=brier), bins.assign(arm=name), pareto.assign(arm=name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed-book", default="results/live/trajectories_notools.csv")
    ap.add_argument("--full-runs", nargs="+", default=[
        "results/live/trajectories_full.csv",
        "results/live/trajectories_full_r1.csv",
        "results/live/trajectories_full_r2.csv"])
    ap.add_argument("--out-prefix", default="results/live/calibration_pareto")
    args = ap.parse_args()
    os.chdir(_ROOT)

    cb = pd.read_csv(args.closed_book)
    full = pd.concat([pd.read_csv(p) for p in args.full_runs], ignore_index=True)

    summary_rows, bin_rows, pareto_rows = [], [], []
    for name, df in [("closed-book (no tools)", cb),
                      ("full-acquisition (pooled 3 runs)", full)]:
        s, b, p = report_arm(name, df)
        summary_rows.append(s); bin_rows.append(b); pareto_rows.append(p)

    pd.DataFrame(summary_rows).to_csv(f"{args.out_prefix}_summary.csv", index=False)
    pd.concat(bin_rows, ignore_index=True).to_csv(f"{args.out_prefix}_bins.csv", index=False)
    pd.concat(pareto_rows, ignore_index=True).to_csv(f"{args.out_prefix}_curve.csv", index=False)
    print(f"\nwritten: {args.out_prefix}_summary.csv, _bins.csv, _curve.csv")


if __name__ == "__main__":
    main()
