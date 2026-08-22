#!/usr/bin/env python3
"""SPHERE second-domain runner: acquisition curve + single-modality tables + figure.

>>> SYNTHETIC DATA <<<
SPHERE (Stanford ADRC SPHERE release) is a FULLY SYNTHETIC cohort intended for
methods development, pilot analysis, grant applications and teaching -- NOT for
clinical findings. Participant ids are literally `Synthetic537`, ... Nothing
produced by this script is evidence about Alzheimer's disease. The claim it
supports is FRAMEWORK TRANSFERABILITY only.

USAGE
-----
    python3 scripts/sphere/prep_sphere.py --src /path/to/ADRC_SPHERE_all_data --out data/sphere
    python3 scripts/sphere/run_sphere.py --data data/sphere --task ad_vs_hc \
        --outdir results/sphere --figdir figures/sphere

Writes
  results/sphere/sphere_acquisition_curve.csv     curve (fold-averaged), fixed cohort
  results/sphere/sphere_curve_seed_stability.csv  the same curve per fold seed
  results/sphere/sphere_curve_bootstrap.csv       participant-bootstrap CIs
  results/sphere/sphere_budget_contrasts.csv      PAIRED between-budget contrasts
  results/sphere/sphere_staging_sensitivity.csv   curve with/without the CDR block
  results/sphere/sphere_single_auc.csv            per channel, own covered samples
  results/sphere/sphere_single_auc_fixed.csv      per channel, FIXED cohort
  results/sphere/sphere_modality_bootstrap.csv    per-modality participant-boot CIs
  results/sphere/sphere_differential_gain.csv     differential pairs, cog vs +3mod
  figures/sphere/sphere.png                       2-panel figure
  figures/sphere/sphere_differential.png          differential grouped bars

THE DIFFERENTIAL TABLE IS NOT A UNIFORM RESULT
----------------------------------------------
`sphere_differential_gain` asks whether appending the three costly modalities to
the cheap cognitive battery helps on the harder overlapping-dementia pairs. On
this release it does not, in three of the four resolvable pairs -- but ad_vs_mci
goes the other way, and the table prints `frac_seeds_negative` beside every gain
so a direction that holds in 25/25 fold seeds is not read the same as one that
holds in 6/25. Where the gain is negative, part of the drop is small-n/high-dim
over-fitting rather than an information-theoretic fact; see the caveat in
`sphere_differential_gain`'s docstring, which is the wording any write-up must
carry.

WHICH SPREAD TO QUOTE
---------------------
Three are produced and only one is a confidence interval. The CV-fold SD says
how much the estimate moves when the SAME participants are split differently
(~0.010 here) -- an estimator property, not sampling uncertainty. The
participant bootstrap says how much it would move on a different sample of the
same size (~0.047, i.e. 4.4x wider). The PAIRED contrast is the one that answers
"did adding this modality change discrimination?", and it is much tighter than
the marginal intervals because cohort-level variation is shared by both budgets
and cancels -- reading "no difference" off overlapping marginal intervals is the
overlapping-CI fallacy.

WHY TWO SINGLE-MODALITY TABLES
------------------------------
`sphere_single_auc` scores each channel on its own covered participants, so its
rows differ in n and class balance and are NOT a like-for-like ranking (on this
release tau covers 102 participants against cognitive's 358).
`sphere_single_auc_fixed` holds the participants fixed. Both are written because
they answer different questions, and reporting only the first would let a
coverage artifact read as a modality ranking.
"""
import argparse, os, sys
import numpy as np
import pandas as pd

# Scripts live one level deeper than the repo root (scripts/<block>/x.py),
# so the root is three dirnames up, not two.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from pwkbench.domains.sphere import (
    load_sphere_cohort, sphere_curve, sphere_single_auc, sphere_single_auc_fixed,
    sphere_curve_bootstrap, sphere_budget_contrasts, sphere_differential_gain,
    sphere_modality_bootstrap,
    SPHERE_FILES, SPHERE_CHANNELS_WELLCOVERED)


def make_figure(curve, single_fixed, single_own, path, task, seeds_df=None,
                boot=None, contrasts=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- Panel A: cost-ordered acquisition curve on the fixed cohort ---------
    k = curve.budget_k.to_numpy()
    auc = curve.cum_auc.to_numpy()
    # Uncertainty band: the PARTICIPANT bootstrap when available, never the
    # CV-fold spread alone. The fold spread here is ~0.010 against a sampling SD
    # of ~0.047, so drawing it as the error band would shrink the visual
    # uncertainty ~4x and make every wiggle in this curve look resolved.
    if boot is not None:
        b = boot.set_index("budget_k").loc[k]
        ax1.fill_between(k, b.ci_lo.to_numpy(), b.ci_hi.to_numpy(),
                         alpha=0.16, color="tab:blue", lw=0,
                         label=f"95% CI, {int(b.n_draws.iloc[0])}-draw "
                               f"participant bootstrap")
    if seeds_df is not None:
        g = seeds_df.groupby("budget_k").cum_auc.agg(["mean", "std"])
        ax1.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"],
                         alpha=0.45, color="tab:blue", lw=0,
                         label="+/-1 SD over CV folds (split noise only,\n"
                               "NOT a confidence interval)")
    ax1.plot(k, auc, "o-", color="tab:blue", lw=2, ms=7,
             label="cumulative AUC (mean over CV folds)")
    # Headroom so the annotations sit inside the axes instead of being clipped
    # by the top spine or colliding with the tick labels (the tick labels here
    # are three lines tall, so the usual "put it below the point" placement
    # overlaps them).
    # Scale the axis to the widest band actually drawn, so the uncertainty is
    # not clipped off the panel -- a CI you cannot see reads as no CI at all.
    lo_v, hi_v = auc.min(), auc.max()
    if boot is not None:
        lo_v = min(lo_v, float(boot.ci_lo.min()))
        hi_v = max(hi_v, float(boot.ci_hi.max()))
    span = hi_v - lo_v
    ax1.set_ylim(lo_v - 0.10 * span, hi_v + 0.20 * span)
    pspan = auc.max() - auc.min()
    peak = int(np.argmax(auc))
    # "Peak" only if the paired contrast to the next budget actually separates;
    # on this cohort k=1 vs k=2 is -0.002 [-0.028, +0.020], so calling k=1 the
    # peak would dress a coin flip up as a finding.
    peak_txt = f"highest point: {curve.added_channel.iloc[peak]} alone"
    if contrasts is not None and len(contrasts):
        nxt = contrasts[(contrasts.from_k == k[peak])]
        if len(nxt) and nxt.iloc[0].ci_lo <= 0 <= nxt.iloc[0].ci_hi:
            peak_txt += "\n(tied with the next budget:\nCI on the difference spans 0)"
    ax1.annotate(peak_txt,
                 xy=(k[peak], auc[peak]),
                 xytext=(k[peak] + 0.30, auc[peak] + 0.30 * span),
                 fontsize=8, ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="->", lw=1))
    lo = int(np.argmin(auc))
    # State the paired contrast WITH its interval rather than the bare drop: at
    # this cohort size the interval spans zero, so annotating "-0.059" alone
    # would present a sample description as a demonstrated effect.
    drop_txt = f"adding {curve.added_channel.iloc[lo]}: {auc[lo]-auc[lo-1]:+.3f}"
    if contrasts is not None and len(contrasts):
        row = contrasts[contrasts.to_k == k[lo]]
        if len(row):
            r = row.iloc[0]
            spans0 = r.ci_lo <= 0 <= r.ci_hi
            drop_txt = (f"adding {curve.added_channel.iloc[lo]}: "
                        f"{r.mean_diff:+.3f}\n[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] "
                        f"paired bootstrap\n"
                        + ("CI spans 0 -- not resolved at this n"
                           if spans0 else f"{r.frac_same_sign:.0%} of resamples"))
    ax1.annotate(drop_txt,
                 xy=(k[lo], auc[lo]),
                 xytext=(k[lo] - 0.30, auc[lo] - 0.30 * pspan),
                 fontsize=8, ha="right",
                 arrowprops=dict(arrowstyle="->", lw=1))
    ax1.set_xticks(k)
    # Words, not a raw "$": matplotlib parses $...$ as mathtext and errors out.
    ax1.set_xticklabels([f"{i}\n+{c}\n(rank {int(cc)})" for i, c, cc in
                         zip(k, curve.added_channel, curve.cost_rank)], fontsize=8)
    # No cumulative-cost axis on purpose: burden is ordinal here, and summing
    # ranks would put an invented ratio on the x-axis (see SPHERE_COST_RANK).
    ax1.set_xlabel("modalities acquired, added in increasing burden rank")
    ax1.set_ylabel("cross-validated ROC-AUC")
    ax1.set_title(f"A. Acquisition in increasing burden rank ({task}, fixed n="
                  f"{int(curve.n_fixed.iloc[0])})", fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="lower left")

    # --- Panel B: single-modality AUC, fixed cohort vs own coverage ---------
    s = single_fixed.sort_values("auc_fixed")
    ypos = np.arange(len(s))
    ax2.barh(ypos, s.auc_fixed, color="tab:blue", height=0.55,
             label=f"fixed cohort (n={int(s.n_fixed.iloc[0])})")
    own = single_own.set_index("channel").reindex(s.channel)
    ax2.plot(own.auc.to_numpy(), ypos, "D", color="tab:orange", ms=6,
             label="own covered samples (n varies)")
    for i, (a, n) in enumerate(zip(own.auc, own.n)):
        if not np.isnan(a):            # was `a == a`; reads as a typo, not a NaN test
            ax2.annotate(f"n={int(n)}", (a, i), textcoords="offset points",
                         xytext=(6, -3), fontsize=7, color="tab:orange")
    ax2.axvline(0.5, color="grey", ls="--", lw=1)
    ax2.annotate("chance", (0.5, -0.65), fontsize=7.5, color="grey", ha="center")
    ax2.set_yticks(ypos)
    ax2.set_yticklabels([f"{c}\n(rank {int(v)})" for c, v in
                         zip(s.channel, s.cost_rank)], fontsize=8)
    ax2.set_xlabel("cross-validated ROC-AUC")
    ax2.set_xlim(0.4, 1.0)
    ax2.set_title("B. No single saturating modality", fontsize=10)
    ax2.grid(alpha=0.3, axis="x")
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle("SPHERE (SYNTHETIC cohort) — framework transferability only; "
                 "no clinical claim about AD", fontsize=9, y=1.005)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_differential_figure(gain, path, seeds):
    """Grouped bars: cognitive alone vs cognitive+3 costly modalities, per pair.

    Error bars are the +/-1 SD over CV-fold seeds -- split noise, NOT sampling
    uncertainty -- and are labelled as such, because on these 59-230 participant
    cohorts that spread is large enough that omitting it would make single-seed
    differences look like results.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    x = np.arange(len(gain))
    w = 0.36
    scored = ~gain.skipped.to_numpy()

    ax.bar(x[scored] - w / 2, gain.auc_cognitive[scored], w, color="tab:blue",
           yerr=gain.auc_cognitive_sd[scored], capsize=3, ecolor="0.35",
           label="cognitive alone (burden rank 1)")
    ax.bar(x[scored] + w / 2, gain.auc_3mod[scored], w, color="tab:orange",
           yerr=gain.auc_3mod_sd[scored], capsize=3, ecolor="0.35",
           label="+ biomarkers + wgs + plasma (ranks 2-3)")

    for i, r in gain.iterrows():
        if r.skipped:
            # Drawn as an empty slot rather than dropped: a reader should see
            # that the pair was examined and refused, not that it never existed.
            ax.annotate(f"not scored\n{r.skip_reason}", (i, 0.72), ha="center",
                        va="center", fontsize=7.5, color="0.35", style="italic")
            continue
        top = max(r.auc_cognitive + (r.auc_cognitive_sd or 0),
                  r.auc_3mod + (r.auc_3mod_sd or 0))
        # Direction stability, not just magnitude: a -0.03 that holds in 22/25
        # seeds and one that holds in 13/25 are different claims.
        ax.annotate(f"{r.gain:+.3f}\n{int(round(r.frac_seeds_negative * seeds))}"
                    f"/{seeds} seeds fall",
                    (i, top + 0.012), ha="center", va="bottom", fontsize=7.5,
                    color="0.15")

    ax.axhline(0.5, color="grey", ls="--", lw=1)
    # Left margin widened so the chance label sits in clear space rather than on
    # top of the first group's bar.
    ax.set_xlim(-0.78, len(gain) - 0.32)
    ax.annotate("chance", (-0.74, 0.507), fontsize=7.5, color="grey", ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={int(n)}, {int(p)}/{int(q)})" for t, n, p, q in
                        zip(gain.task, gain.n, gain.n_pos, gain.n_neg)],
                       fontsize=8)
    ax.set_ylabel("cross-validated ROC-AUC")
    # Headroom for the per-group gain labels AND the legend, which sits in the
    # empty band above the bars rather than on top of the shortest group.
    ax.set_ylim(0.4, 1.16)
    ax.set_xlabel("differential pair (positive/negative class sizes on the "
                  "four-channel common cohort)")
    ax.set_title(f"Adding costly modalities to the cheap channel, "
                 f"differential pairs (mean of {seeds} CV-fold seeds; "
                 f"bars +/-1 SD = split noise, not a CI)", fontsize=9.5)
    ax.grid(alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    ax.set_yticks(np.arange(0.4, 1.01, 0.1))
    ax.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.9)

    fig.suptitle("SPHERE (SYNTHETIC cohort) — framework transferability only; "
                 "no clinical claim about AD", fontsize=9, y=1.005)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sphere")
    ap.add_argument("--task", default="ad_vs_hc")
    ap.add_argument("--outdir", default="results/sphere")
    ap.add_argument("--figdir", default="figures/sphere")
    ap.add_argument("--seeds", type=int, default=10,
                    help="CV-fold seeds used for the stability band (default 10)")
    ap.add_argument("--bootstrap", type=int, default=200,
                    help="participant-resampling bootstrap draws for the "
                         "sampling-uncertainty interval; 0 disables (default 200)")
    ap.add_argument("--differential-seeds", type=int, default=25,
                    help="CV-fold seeds averaged for the differential-pair "
                         "table; 0 skips that supplement (default 25)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("SYNTHETIC cohort (Stanford ADRC SPHERE) -- methods development only.\n")

    all_channels = list(SPHERE_FILES)
    cohort = load_sphere_cohort(args.data, task=args.task, channels=all_channels)
    print(f"loaded {len(cohort)} labelled participants "
          f"({int(cohort.y.sum())} positive) for task={args.task}")

    seeds_df = pd.concat(
        [sphere_curve(cohort, order=SPHERE_CHANNELS_WELLCOVERED, seed=s)
         .assign(seed=s) for s in range(args.seeds)], ignore_index=True)
    seeds_df.to_csv(os.path.join(args.outdir, "sphere_curve_seed_stability.csv"),
                    index=False)

    # The headline curve is the MEAN over fold seeds, not one arbitrary split.
    # A single seed is a legitimate estimate but an arbitrary one, and on this
    # release seed 0 sits about +0.009 above the 10-seed mean at k=1 -- plotting
    # it as the curve would put the line along the top edge of its own error
    # band and overstate every budget by the same amount.
    agg = (seeds_df.groupby(["budget_k", "added_channel", "cost_rank",
                             "n_fixed", "n_features"], as_index=False)
           .cum_auc.agg(["mean", "std"]))
    agg = agg.rename(columns={"mean": "cum_auc", "std": "cum_auc_sd"})
    curve = agg.sort_values("budget_k").reset_index(drop=True)
    curve["n_seeds"] = args.seeds
    curve.to_csv(os.path.join(args.outdir, "sphere_acquisition_curve.csv"),
                 index=False)

    # Both modality tables are averaged over the SAME fold seeds as the curve.
    # Panel A's k=1 point and Panel B's `cognitive` bar are the same quantity
    # (that channel alone on the fixed cohort), so computing one at the seed mean
    # and the other at a single seed would print two different numbers for one
    # measurement inside a single figure.
    def _over_seeds(fn, **kw):
        parts = [fn(seed=s, **kw).assign(seed=s) for s in range(args.seeds)]
        d = pd.concat(parts, ignore_index=True)
        val = "auc_fixed" if "auc_fixed" in d.columns else "auc"
        keys = [c for c in d.columns if c not in (val, "seed")]
        out = (d.groupby(keys, as_index=False, dropna=False)[val]
               .agg(["mean", "std"]).rename(columns={"mean": val,
                                                     "std": f"{val}_sd"}))
        out["n_seeds"] = args.seeds
        return out.sort_values(val, ascending=False,
                               na_position="last").reset_index(drop=True)

    single_own = _over_seeds(sphere_single_auc, cohort=cohort,
                             channels=all_channels)
    single_own.to_csv(os.path.join(args.outdir, "sphere_single_auc.csv"),
                      index=False)
    single_fixed = _over_seeds(sphere_single_auc_fixed, cohort=cohort,
                               order=SPHERE_CHANNELS_WELLCOVERED)
    single_fixed.to_csv(os.path.join(args.outdir,
                                     "sphere_single_auc_fixed.csv"), index=False)

    # Circularity sensitivity: the same curve with the CDR/cognitive-status
    # columns put BACK. Published so a reader can see how much of the cognitive
    # channel's strength is the staging instrument predicting its own diagnosis,
    # rather than having to take the guard on trust.
    # Aggregated over the same fold seeds as the stability band, NOT a single
    # run: on this release seed 0 happens to give bit-identical AUC with and
    # without the staging block (dropping those columns does not reorder any
    # fold's predictions at that particular split), so a one-seed sensitivity
    # would make the guard look like a no-op when over 10 seeds it moves the
    # cognitive channel by about -0.010.
    unguarded = load_sphere_cohort(args.data, task=args.task,
                                   channels=all_channels, exclude_staging=False)
    sens = pd.concat(
        [sphere_curve(co, order=SPHERE_CHANNELS_WELLCOVERED, seed=s)
         .assign(cognitive_cols=lab, seed=s)
         for lab, co in (("neuropsych only (staging excluded)", cohort),
                         ("all (staging included)", unguarded))
         for s in range(args.seeds)], ignore_index=True)
    sens["staging_cols_excluded"] = len(
        cohort.dropped_staging.get("cognitive", []))
    sens.to_csv(os.path.join(args.outdir, "sphere_staging_sensitivity.csv"),
                index=False)

    fmt = lambda v: f"{v:.3f}"
    print("\n=== A. cost-ordered acquisition curve (fixed cohort) ===")
    print(curve.to_string(index=False, float_format=fmt))

    g = seeds_df.groupby("budget_k").cum_auc.agg(["mean", "std"])
    print(f"\n    stability over {args.seeds} CV-fold seeds:")
    for kk, r in g.iterrows():
        print(f"      k={kk}  {r['mean']:.4f} +/- {r['std']:.4f}")
    w = seeds_df.pivot(index="seed", columns="budget_k", values="cum_auc")
    pk = int(curve.cum_auc.idxmax()) + 1
    wins = int((w[pk] >= w.drop(columns=[pk]).max(axis=1)).sum())
    print(f"      peak sits at k={pk} in {wins}/{args.seeds} seeds")

    dropped = cohort.dropped_staging.get("cognitive", [])
    print(f"\n    circularity guard: dropped {len(dropped)} CDR/status columns "
          f"from `cognitive` ({', '.join(dropped[:4])}...)")
    for lab, sub in sens[sens.budget_k == 1].groupby("cognitive_cols"):
        print(f"      k=1 cognitive alone, {lab}: "
              f"{sub.cum_auc.mean():.4f} +/- {sub.cum_auc.std(ddof=1):.4f} "
              f"(over {args.seeds} fold seeds)")

    print("\n=== B1. single modality, FIXED cohort (like-for-like) ===")
    print(single_fixed.to_string(index=False, float_format=fmt))
    print("\n=== B2. single modality, own covered samples (n differs -- NOT a "
          "like-for-like ranking) ===")
    print(single_own.to_string(index=False, float_format=fmt))

    boot = contrasts = None
    if args.bootstrap:
        print(f"\n=== C. sampling uncertainty: {args.bootstrap}-draw participant "
              f"bootstrap ===")
        boot, draws = sphere_curve_bootstrap(
            cohort, order=SPHERE_CHANNELS_WELLCOVERED, B=args.bootstrap,
            return_draws=True)
        boot.to_csv(os.path.join(args.outdir, "sphere_curve_bootstrap.csv"),
                    index=False)
        merged = curve.merge(boot[["budget_k", "boot_mean", "boot_sd",
                                   "ci_lo", "ci_hi", "n_draws"]], on="budget_k")
        merged["fold_sd"] = curve.cum_auc_sd
        print(merged[["budget_k", "added_channel", "cum_auc", "fold_sd",
                      "boot_sd", "ci_lo", "ci_hi", "n_draws"]]
              .to_string(index=False, float_format=fmt))
        ratio = (merged.boot_sd / merged.fold_sd).mean()
        print(f"\n    sampling SD is {ratio:.1f}x the CV-fold SD -- the fold "
              f"spread is NOT a confidence interval.")
        print("    Point estimate = cum_auc (fold-averaged, full sample). The "
              "bootstrap mean sits low")
        print("    (~63% distinct participants per resample => less training "
              "data); read its WIDTH, not its position.")

        contrasts = sphere_budget_contrasts(draws)
        contrasts.to_csv(os.path.join(args.outdir,
                                      "sphere_budget_contrasts.csv"), index=False)
        print("\n    PAIRED contrasts (same resampled people at both budgets) --")
        print("    the marginal intervals above overlap, but concluding "
              "'no difference' from")
        print("    overlapping intervals is the overlapping-CI fallacy; these "
              "are the right test:")
        print(contrasts.to_string(index=False, float_format=fmt))

        # Per-modality CIs answer a different question from the curve's: is the
        # cheap channel's lead over the expensive ones bigger than sampling
        # noise? Same resampling and same participant-level guard, one channel
        # at a time instead of cumulative blocks.
        mboot = sphere_modality_bootstrap(
            cohort, channels=SPHERE_CHANNELS_WELLCOVERED, B=args.bootstrap)
        mboot.to_csv(os.path.join(args.outdir, "sphere_modality_bootstrap.csv"),
                     index=False)
        print("\n    per-MODALITY participant bootstrap (point estimate is "
              "single_auc_fixed above;")
        print("    read these as the WIDTH -- boot_mean sits low, ~63% distinct "
              "participants per resample):")
        print(mboot.to_string(index=False, float_format=fmt))
        best = mboot.sort_values("boot_mean", ascending=False).iloc[0]
        sep = mboot[mboot.ci_hi < best.ci_lo]
        if len(sep):
            print(f"    {best.channel}'s CI clears {', '.join(sep.channel)} "
                  f"outright; overlap with the rest means those gaps are NOT "
                  f"resolved at n={int(best.n_fixed)}.")

    if args.differential_seeds:
        print(f"\n=== D. differential pairs: cognitive alone vs +3 costly "
              f"modalities ({args.differential_seeds} fold seeds) ===")
        gain = sphere_differential_gain(args.data, seeds=args.differential_seeds)
        gain.to_csv(os.path.join(args.outdir, "sphere_differential_gain.csv"),
                    index=False)
        print(gain[["task", "n", "n_pos", "n_neg", "auc_cognitive", "auc_3mod",
                    "gain", "gain_sd", "frac_seeds_negative", "skipped"]]
              .to_string(index=False, float_format=fmt))
        for _, r in gain[gain.skipped].iterrows():
            print(f"    skipped {r.task}: {r.skip_reason}")
        s = gain[~gain.skipped]
        print(f"\n    gain range {s.gain.min():+.3f} to {s.gain.max():+.3f} "
              f"over {len(s)} resolvable pairs "
              f"({int((s.gain < 0).sum())} negative, "
              f"{int((s.gain > 0).sum())} positive) -- report the RANGE and the")
        print("    per-pair seed stability, not a single headline number. Where "
              "the gain is negative,")
        print("    part of it is small-n/high-dimensional over-fitting, not an "
              "information-theoretic")
        print("    fact: the claim is scoped to this regime, NOT 'more evidence "
              "always hurts'.")
        make_differential_figure(
            gain, os.path.join(args.figdir, "sphere_differential.png"),
            args.differential_seeds)

    figpath = os.path.join(args.figdir, "sphere.png")
    make_figure(curve, single_fixed, single_own, figpath, args.task, seeds_df,
                boot=boot, contrasts=contrasts if args.bootstrap else None)
    print(f"\nwritten: {args.outdir}/*.csv and {figpath}")


if __name__ == "__main__":
    main()
