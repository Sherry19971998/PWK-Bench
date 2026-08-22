# SYNTHETIC data (Stanford ADRC SPHERE). Methods-development use only; no clinical claim.
"""
SPHERE — second domain, used to show the benchmark's framing TRANSFERS.

>>> SYNTHETIC DATA WARNING — READ BEFORE CITING ANY NUMBER <<<
SPHERE is a FULLY SYNTHETIC cohort (Stanford ADRC SPHERE release) generated from
the real ADRC cohort and intended by its authors for methods development, pilot
analysis, grant applications and teaching -- explicitly NOT for clinical
findings. Subject ids are literally `Synthetic537`, `Synthetic236`, ...
Nothing computed here is evidence about Alzheimer's disease. The only claim this
module supports is FRAMEWORK TRANSFERABILITY: the budgeted-acquisition framing
(cost-ordered channels, an acquisition curve, a peak that is not the full
budget) applies to a heterogeneous-cost, multi-modal problem, not just to the
4-channel ACMG variant task.

WHY THIS DOES NOT REUSE `Cohort` / `metrics.curve_A`
-----------------------------------------------------
The variant domain's `Cohort` stores ONE numeric value per channel per instance
and scores by summing calibrated (s - 0.5) contributions across acquired
channels -- an ACMG-shaped rank score. A SPHERE channel is a MATRIX (cognitive
73 features, WGS 167 SNPs, proteomics 500 after truncation), and its channels
carry no per-feature calibrated direction, so that scorer is undefined here.
The acquisition curve therefore uses a cross-validated logistic model over the
cumulative feature block (see `sphere_curve`), which is the standard way to read
"how much does adding this modality buy?" for continuous multi-modal features.
This divergence is deliberate and is the reason the two domains' AUC columns are
comparable in MEANING (discrimination at a budget) but not produced by identical
code.

COST ORDERING (ORDINAL ONLY)
----------------------------
`SPHERE_COST_RANK` is an ORDINAL rank of acquisition burden, never a price and
never a ratio. The order is defensible from what each modality physically
requires (see the table's comment and `SPHERE_COST_BASIS`); the magnitudes are
not, so none are asserted and ranks are never summed. This is why the curve's
x-axis is the NUMBER of modalities acquired rather than a cumulative cost: a
"cumulative cost" would be an invented ratio presented as a measurement.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

ID = "adrc_id"
LABEL_COL = "diagnosis_consensus"

# Channel -> source CSV. Ordered cheapest-first; that ordering IS the acquisition
# policy evaluated by `sphere_curve`.
SPHERE_FILES = {
    "cognitive":  "cognitive_scores.csv",
    "biomarkers": "biomarkers.csv",
    "wgs":        "wgs.csv",
    "plasma":     "proteomics_plasma.csv",
    "csf":        "proteomics_csf.csv",
    "amyloid":    "imaging_amyloid.csv",
    "tau":        "imaging_tau.csv",
}

SPHERE_CHANNELS = ["cognitive", "biomarkers", "wgs", "plasma", "tau"]

# CIRCULARITY GUARD -- excluded from the `cognitive` channel by default.
#
# The label is `diagnosis_consensus`, a CLINICAL CONSENSUS diagnosis. The columns
# below are the instruments that consensus is built from: `b4_cdrglob` /
# `b4_cdrsum` are the Clinical Dementia Rating global score and sum-of-boxes, the
# `b4_*` domain items are CDR's six boxes, and `c2_cogstat` is an adjudicated
# cognitive-status field. Predicting the diagnosis from the staging instrument
# used to make it is close to definitional, not a finding about acquisition.
#
# Measured on this release (fixed cohort n=221, 10 CV-fold seeds): these 11
# columns ALONE score AUC 0.882, higher than all 73 cognitive columns together
# (0.875); the remaining 62 neuropsychological tests score 0.865. So the channel
# stays comfortably the strongest either way -- the exclusion changes the number,
# not the conclusion -- but reporting 0.875 as "a cheap modality predicts AD"
# would be quoting the diagnosis back to itself.
#
# Set `exclude_staging=False` to reproduce the un-guarded number; the runner
# writes both as a sensitivity table.
SPHERE_STAGING_COLS = [
    "b4_memory", "b4_orient", "b4_judgment", "b4_commun", "b4_homehobb",
    "b4_perscare", "b4_cdrsum", "b4_cdrglob", "b4_comport", "b4_cdrlang",
    "c2_cogstat",
]
SPHERE_STAGING_CHANNEL = "cognitive"

# Acquisition burden as an ORDINAL RANK -- deliberately not a price, a ratio, or
# anything that can be summed.
#
# We can defend the ORDER from what each modality physically requires: an
# in-clinic cognitive battery takes no specimen; a targeted plasma assay needs one
# blood draw; WGS and plasma proteomics need a draw plus a send-out assay; CSF
# proteomics needs a lumbar puncture; PET needs a scanner visit with a
# radiotracer. We have NO sourced figure for how much more any of these costs
# than another, so no magnitude is asserted. An earlier version of this table used
# 1/2/8/8/15 and summed it into a "cumulative cost" plotted on the figure's
# x-axis, which silently published invented ratios ("19x the cheapest"); ranks
# cannot be summed, which is exactly why they are used here.
#
# Ties are meant: WGS and plasma proteomics are both blood-draw send-out assays
# and no order between them is claimed. Same for the two PET tracers.
SPHERE_COST_RANK = {"cognitive": 1, "biomarkers": 2, "wgs": 3, "plasma": 3,
                    "csf": 4, "tau": 5, "amyloid": 5}

# What each rank rests on -- stated so a reader can check the ordering rather
# than take it on faith.
SPHERE_COST_BASIS = {
    "cognitive":  "in-clinic battery, no specimen",
    "biomarkers": "one blood draw, targeted assay (PTAU217/GFAP/NFL)",
    "wgs":        "blood draw + genome sequencing",
    "plasma":     "blood draw + proteomic panel",
    "csf":        "lumbar puncture + proteomic panel",
    "tau":        "PET scan with radiotracer",
    "amyloid":    "PET scan with radiotracer",
}

# The acquisition curve is computed on a FIXED cohort -- the samples present in
# every channel of the evaluated order. Including `tau` collapses that
# intersection from 221 to 37 participants (measured on this release), which
# would make the curve a statement about 37 people rather than about acquisition.
# So the curve runs on the four well-covered channels and `tau` is reported only
# in the single-modality table, on its own covered samples.
SPHERE_CHANNELS_WELLCOVERED = ["cognitive", "biomarkers", "wgs", "plasma"]

# Binary tasks: name -> (positive label, negative label). Rows carrying any other
# consensus diagnosis are DROPPED rather than folded into the negative class --
# MCI, Parkinson's and Lewy body cases are not "healthy", and pooling them would
# make the contrast a mixture of unrelated comparisons.
#
# The `diagnosis_consensus` strings below were read off this release rather than
# transcribed from a plan: `Parkinsons Disease only` and `Parkinsons Disease and
# Dementia` are distinct values, and the file also carries `Possible Alzheimers
# Disease`, `Parkinsons Disease and Mild cognitive impairment` and `Other`, none
# of which are folded into any task here.
SPHERE_TASKS = {
    "ad_vs_hc": ("Probable Alzheimers Disease", "Healthy Control"),
    # Differential pairs: overlapping dementia syndromes rather than
    # disease-vs-healthy. These are the harder, clinically realistic contrasts,
    # and their cohorts are small (see SPHERE_MIN_CLASS_N).
    "ad_vs_lbd": ("Probable Alzheimers Disease", "Lewy Body Disease"),
    "ad_vs_mci": ("Probable Alzheimers Disease", "Mild Cognitive Impairment"),
    "mci_vs_hc": ("Mild Cognitive Impairment", "Healthy Control"),
    "ad_vs_pd":  ("Probable Alzheimers Disease", "Parkinsons Disease only"),
    # Kept ONLY so the size guard is exercised on a real task rather than being
    # untested code: this release has 6 `Parkinsons Disease and Dementia`
    # participants, 3 of them in the four-channel common cohort, so
    # `sphere_differential_gain` refuses to score it.
    "ad_vs_pd_dementia": ("Probable Alzheimers Disease",
                          "Parkinsons Disease and Dementia"),
}

SPHERE_DIFFERENTIAL_TASKS = ["ad_vs_lbd", "ad_vs_mci", "mci_vs_hc", "ad_vs_pd",
                             "ad_vs_pd_dementia"]

# Below this many members in EITHER class, a 5-fold stratified CV AUC is not a
# measurement worth reporting: with <8 in a class some folds carry one or two
# members of it, so the fold AUC is a near-degenerate statistic and its mean
# moves by tenths with the split seed. Tasks under the threshold are returned
# with a stated reason rather than dropped, so a reader sees that they were
# looked at and refused, not quietly omitted.
SPHERE_MIN_CLASS_N = 8


class SphereCohort:
    """Per-channel feature matrices over a shared participant index.

    Attributes
    ----------
    ids       : (n,) participant ids, the row order every matrix follows
    y         : (n,) int 0/1 label
    channels  : evaluated channel names
    task      : task name (used as the `domain` grouping label)
    X         : {channel: (n, d_c) float array}, NaN where a participant is not
                covered by that channel's assay
    defined   : {channel: (n,) bool}, True where that participant appears in the
                channel's CSV
    cost_rank : {channel: ordinal acquisition-burden rank} -- comparable by
                order only; never summed, never read as a ratio
    """

    def __init__(self, ids, y, X, defined, channels, task):
        self.ids = np.asarray(ids)
        self._y = np.asarray(y, int)
        self.X = X
        self.defined = defined
        self.channels = list(channels)
        self.task = task
        self.cost_rank = {c: SPHERE_COST_RANK[c] for c in self.channels}
        self.dropped_staging = {}   # channel -> columns removed by the
                                    # circularity guard (see load_sphere_cohort)

    @property
    def y(self) -> np.ndarray:
        return self._y

    def value(self, channel: str) -> np.ndarray:
        """The channel's feature matrix (n, d). Named `value` to echo the variant
        domain's accessor, but returns a MATRIX, not a scalar per instance."""
        return self.X[channel]

    def covered(self, channel: str) -> np.ndarray:
        return self.defined[channel]

    def common_index(self, channels) -> np.ndarray:
        """Boolean mask of participants covered by EVERY listed channel."""
        m = np.ones(len(self), bool)
        for c in channels:
            m &= self.defined[c]
        return m

    def __len__(self):
        return len(self._y)


def load_sphere_cohort(data_dir: str, task: str = "ad_vs_hc",
                       channels=None, exclude_staging: bool = True) -> SphereCohort:
    """Load the SYNTHETIC SPHERE cohort as per-channel feature matrices.

    Proteomics files are expected to be the variance-truncated ones written by
    `scripts/sphere/prep_sphere.py` (default 500 columns each). Running against the full
    download changes the plasma/CSF feature count and therefore the numbers --
    regenerate with the same `--n-proteins` to reproduce.

    The label column (`diagnosis_consensus`) lives only in
    demographics_diagnosis.csv and is never merged into a channel matrix, so no
    channel can leak the outcome; `tests/test_sphere.py::test_no_label_leakage`
    pins that.

    `exclude_staging` (default True) additionally drops the CDR / cognitive-status
    columns from the `cognitive` channel. That is a CONSTRUCT-level guard rather
    than a leakage fix: those columns are not the label, but they are the staging
    instrument the consensus diagnosis is made with, so keeping them makes the
    channel partly predict itself. See `SPHERE_STAGING_COLS` for the measured
    effect. `cohort.dropped_staging` records exactly what was removed.
    """
    if task not in SPHERE_TASKS:
        raise ValueError(f"task must be one of {sorted(SPHERE_TASKS)}, got {task!r}")
    channels = list(channels or SPHERE_CHANNELS)
    unknown = set(channels) - set(SPHERE_FILES)
    if unknown:
        raise ValueError(f"unknown channel(s): {sorted(unknown)}")

    pos, neg = SPHERE_TASKS[task]
    demo = pd.read_csv(os.path.join(data_dir, "demographics_diagnosis.csv"),
                       usecols=[ID, LABEL_COL])
    demo = demo[demo[LABEL_COL].isin([pos, neg])].drop_duplicates(subset=[ID])
    demo = demo.sort_values(ID).reset_index(drop=True)
    ids = demo[ID].to_numpy()
    y = (demo[LABEL_COL] == pos).to_numpy(int)

    X, defined, dropped = {}, {}, {}
    for c in channels:
        path = os.path.join(data_dir, SPHERE_FILES[c])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- run scripts/sphere/prep_sphere.py first")
        d = pd.read_csv(path)
        if ID not in d.columns:
            raise ValueError(f"{path}: missing '{ID}' column")
        if LABEL_COL in d.columns:      # defensive: a channel must never carry y
            raise ValueError(f"{path} unexpectedly contains the label column "
                             f"'{LABEL_COL}'")
        d = d.drop_duplicates(subset=[ID]).set_index(ID)
        d = d.select_dtypes("number")
        if exclude_staging and c == SPHERE_STAGING_CHANNEL:
            hit = [col for col in SPHERE_STAGING_COLS if col in d.columns]
            d = d.drop(columns=hit)
            dropped[c] = hit
        # Reindex onto the label cohort: uncovered participants become all-NaN
        # rows, which `defined` marks and the curve's imputer fills.
        aligned = d.reindex(ids)
        X[c] = aligned.to_numpy(float)
        defined[c] = np.isin(ids, d.index.to_numpy())
    cohort = SphereCohort(ids, y, X, defined, channels, task)
    cohort.dropped_staging = dropped
    return cohort


def _cv_auc(Xb: np.ndarray, y: np.ndarray, n_splits: int = 5,
            seed: int = 0) -> float:
    """Stratified 5-fold CV ROC-AUC of a regularized logistic model.

    Imputation and standardization are fitted INSIDE each fold (via a Pipeline),
    not once on the full matrix: fitting them on all rows first would let the
    held-out fold's feature distribution influence the training transform, which
    inflates AUC. C=0.1 keeps the model regularized -- necessary because the
    cumulative block reaches ~740 features against 221 participants.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=5000)),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return float(cross_val_score(pipe, Xb, y, cv=cv, scoring="roc_auc").mean())


def sphere_curve(cohort: SphereCohort, order=None, seed: int = 0) -> pd.DataFrame:
    """Cost-ordered acquisition curve on a FIXED cohort.

    The participant set is fixed to those covered by EVERY channel in `order`
    before the first budget is scored, so a rise or fall along k reflects the
    added modality and not a change of population. (Letting n shrink as channels
    are added is the obvious trap here: the k=4 column would then describe a
    different, smaller cohort than k=1.)

    Returns budget_k / added_channel / cost_rank / cum_auc / n_fixed /
    n_features. `cost_rank` is the ADDED channel's ordinal burden rank; there is
    deliberately no cumulative-cost column, because ranks cannot be summed.
    """
    order = list(order or SPHERE_CHANNELS_WELLCOVERED)
    mask = cohort.common_index(order)
    y = cohort.y[mask]
    if len(np.unique(y)) < 2:
        raise ValueError("fixed cohort has a single class -- cannot compute AUC")

    rows, blocks = [], []
    for k, c in enumerate(order, start=1):
        blocks.append(cohort.X[c][mask])
        Xb = np.hstack(blocks)
        rows.append({"budget_k": k, "added_channel": c,
                     "cost_rank": cohort.cost_rank[c],
                     "cum_auc": _cv_auc(Xb, y, seed=seed),
                     "n_fixed": int(mask.sum()), "n_features": Xb.shape[1]})
    return pd.DataFrame(rows)


def sphere_single_auc(cohort: SphereCohort, channels=None,
                      seed: int = 0) -> pd.DataFrame:
    """Per-channel AUC, each on ITS OWN covered participants.

    Unlike `sphere_curve` these rows are NOT mutually comparable as a curve: each
    is measured on a different subset (and a different class balance), so a
    higher AUC here can reflect an easier subset rather than a better modality.
    The column exists to answer one question -- is any single modality already
    saturating? -- and `n` / `n_pos` are reported alongside so the reader can see
    which comparisons are even meaningful.
    """
    out = []
    for c in list(channels or cohort.channels):
        m = cohort.defined[c]
        y = cohort.y[m]
        if len(np.unique(y)) < 2:
            out.append({"channel": c, "cost_rank": cohort.cost_rank[c],
                        "auc": float("nan"),
                        "n": int(m.sum()), "n_pos": int(y.sum()),
                        "n_features": cohort.X[c].shape[1]})
            continue
        out.append({"channel": c, "cost_rank": cohort.cost_rank[c],
                    "auc": _cv_auc(cohort.X[c][m], y, seed=seed),
                    "n": int(m.sum()), "n_pos": int(y.sum()),
                    "n_features": cohort.X[c].shape[1]})
    return pd.DataFrame(out).sort_values("auc", ascending=False,
                                         na_position="last").reset_index(drop=True)


def sphere_single_auc_fixed(cohort: SphereCohort, order=None,
                            seed: int = 0) -> pd.DataFrame:
    """Per-channel AUC on the SAME fixed cohort the acquisition curve uses.

    `sphere_single_auc` measures each channel on its own covered participants,
    which makes its rows a ranking of {modality x subset x class balance} rather
    than of modalities: on this release `tau` covers 102 participants and
    `cognitive` 358, so their AUCs are not on comparable footing and the apparent
    winner can flip with coverage alone. This function holds the participants
    fixed (the intersection over `order`, the same set `sphere_curve` scores), so
    the resulting ranking isolates the modality.

    It necessarily excludes channels with thin coverage -- adding `tau` to the
    intersection drops it from 221 participants to 37 -- so the two tables answer
    different questions and both are reported.
    """
    order = list(order or SPHERE_CHANNELS_WELLCOVERED)
    mask = cohort.common_index(order)
    y = cohort.y[mask]
    rows = []
    for c in order:
        rows.append({"channel": c, "cost_rank": cohort.cost_rank[c],
                     "auc_fixed": _cv_auc(cohort.X[c][mask], y, seed=seed),
                     "n_fixed": int(mask.sum()), "n_pos": int(y.sum()),
                     "n_features": cohort.X[c].shape[1]})
    return pd.DataFrame(rows).sort_values("auc_fixed", ascending=False
                                          ).reset_index(drop=True)


def _grouped_cv(n_splits: int = 5, seed: int = 0):
    """The splitter every bootstrap in this module fits through.

    Factored out so `tests/test_sphere.py::test_no_bootstrap_over_cv` can assert
    the no-shared-participant property against the ACTUAL splits rather than
    against a comment claiming they are clean.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=seed)


def _cv_auc_grouped(Xb: np.ndarray, y: np.ndarray, groups: np.ndarray,
                    n_splits: int = 5, seed: int = 0) -> float:
    """CV AUC where all copies of one participant stay in the same fold.

    Needed by the bootstrap: resampling participants with replacement puts the
    SAME person in the data several times, and ordinary k-fold would then train
    on one copy and test on another. That is memorization, not generalization,
    and it biases every resampled AUC upward -- which would make the bootstrap
    interval both too high and too narrow, the opposite of its purpose.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=5000)),
    ])
    return float(cross_val_score(pipe, Xb, y, groups=groups,
                                 cv=_grouped_cv(n_splits, seed),
                                 scoring="roc_auc").mean())


def sphere_curve_bootstrap(cohort: SphereCohort, order=None, B: int = 200,
                           seed: int = 0, alpha: float = 0.05,
                           return_draws: bool = False):
    """Percentile bootstrap CI for the acquisition curve, resampling PARTICIPANTS.

    WHY THIS IS NOT THE FOLD-SEED SPREAD
    ------------------------------------
    `sphere_curve` run over several CV-fold seeds measures how much the estimate
    moves when the SAME 221 participants are split differently. That is a
    property of the estimator, not of the sample: it says nothing about how much
    the curve would move on a different 221 people, which is the uncertainty a
    reader cares about and the only one an interval in a paper should claim.
    This function resamples participants with replacement (stratified by label,
    so every resample keeps the observed class balance and no resample can lose
    a class) and re-runs the whole pipeline, so the interval reflects sampling
    variability. It is materially wider than the fold-seed spread; reporting the
    latter as a confidence interval would understate the uncertainty.

    Duplicate participants are kept in one fold via `_cv_auc_grouped` -- see its
    docstring for why the naive version is optimistically biased.

    B=200 by default rather than the variant domain's BOOTSTRAP_B=2000: each
    resample here refits a 5-fold logistic pipeline over up to ~730 features, so
    2000 resamples costs hours for an interval whose width is already stable.
    Raise B if a published interval needs finer percentile resolution.

    READ THE WIDTH, NOT THE POSITION. A bootstrap resample contains only ~63%
    distinct participants, so each grouped-CV fit trains on materially less data
    than the full sample and `boot_mean` lands BELOW the full-sample estimate
    (measured here: about -0.027 at k=1). That is an artifact of resampling with
    replacement under cross-validation, not evidence the full-sample number is
    optimistic. Use `sphere_curve`'s fold-averaged value as the point estimate
    and this function's spread (`boot_sd`, or ci_hi - ci_lo) as the sampling
    uncertainty; quoting `ci_lo`/`ci_hi` as the interval around the reported
    point estimate would shift it low by that same bias.
    """
    order = list(order or SPHERE_CHANNELS_WELLCOVERED)
    mask = cohort.common_index(order)
    y_all = cohort.y[mask]
    blocks_all, names = [], []
    for c in order:
        blocks_all.append(cohort.X[c][mask])
        names.append(c)

    pos = np.flatnonzero(y_all == 1)
    neg = np.flatnonzero(y_all == 0)
    rng = np.random.default_rng(seed)

    draws = {k: [] for k in range(1, len(order) + 1)}
    for b in range(B):
        # Stratified resample: keeps the class balance fixed across resamples so
        # the interval reflects the curve's variability, not the extra noise of
        # a randomly drifting positive rate.
        idx = np.concatenate([rng.choice(pos, size=len(pos), replace=True),
                              rng.choice(neg, size=len(neg), replace=True)])
        yb = y_all[idx]
        cum = []
        for k in range(1, len(order) + 1):
            cum.append(blocks_all[k - 1][idx])
            try:
                draws[k].append(_cv_auc_grouped(np.hstack(cum), yb, groups=idx,
                                                seed=seed + b))
            except ValueError:
                # A resample can leave too few distinct participants of one class
                # for a grouped stratified split. Record NaN rather than skipping:
                # skipping would leave the per-budget lists at DIFFERENT lengths,
                # silently destroying the row-wise correspondence that paired
                # budget contrasts depend on (draw i must mean the same resample
                # at every k). NaNs are dropped per-statistic instead.
                draws[k].append(float("nan"))

    rows = []
    for k in range(1, len(order) + 1):
        d = np.asarray(draws[k], float)
        d = d[~np.isnan(d)]
        rows.append({
            "budget_k": k, "added_channel": names[k - 1],
            "cost_rank": cohort.cost_rank[names[k - 1]],
            "boot_mean": float(np.mean(d)) if d.size else float("nan"),
            "boot_sd": float(np.std(d, ddof=1)) if d.size > 1 else float("nan"),
            "ci_lo": float(np.percentile(d, 100 * alpha / 2)) if d.size else float("nan"),
            "ci_hi": float(np.percentile(d, 100 * (1 - alpha / 2))) if d.size else float("nan"),
            "n_draws": int(d.size), "B_requested": B, "n_fixed": int(mask.sum()),
        })
    summary = pd.DataFrame(rows)
    if not return_draws:
        return summary
    D = pd.DataFrame({k: draws[k] for k in sorted(draws)})
    return summary, D


def sphere_modality_bootstrap(cohort: SphereCohort, channels=None, B: int = 400,
                              seed: int = 0, alpha: float = 0.05,
                              return_draws: bool = False):
    """Participant-bootstrap CI for EACH single modality on the fixed cohort.

    SYNTHETIC cohort (Stanford ADRC SPHERE): framework-transfer evidence only,
    no clinical claim about Alzheimer's disease is made or supported here.

    `sphere_curve_bootstrap` intervals the CUMULATIVE blocks along the
    acquisition order; this intervals the modalities one at a time, which is the
    statistic behind "the cheapest channel is the strongest". Same fixed
    participant set, same stratified participant resampling, same leakage guard
    -- it deliberately calls `_cv_auc_grouped` rather than reimplementing a
    split, so there is exactly one place in this module where the
    duplicate-participant rule lives and exactly one place it can regress.

    WHY GROUPED CV AND NOT A SINGLE INTERNAL HOLDOUT
    -----------------------------------------------
    The requirement is that no participant appears in both the training and the
    evaluation half of a resample; a bootstrap draw contains the same person up
    to ~5 times, so an ordinary k-fold (or an index-level holdout) trains on one
    copy and scores another. Passing the ORIGINAL participant positions as
    `groups` to `StratifiedGroupKFold` satisfies that requirement exactly -- all
    copies of a participant land in one fold -- while still scoring every
    participant once instead of only the ~30% a single holdout would keep. A
    one-shot holdout meets the same guarantee but throws away most of the
    evaluation set, so at 46 positives its per-resample AUC is far noisier and
    the resulting interval mixes sampling variability with holdout-size noise.
    `tests/test_sphere.py::test_no_bootstrap_over_cv` asserts the no-overlap
    property against the actual splits rather than trusting this paragraph.

    READ THE WIDTH, NOT THE POSITION -- see `sphere_curve_bootstrap`. A resample
    holds only ~63% distinct participants, so every grouped fit trains on less
    data than the full sample and `boot_mean` sits below the full-sample
    estimate. Use `sphere_single_auc_fixed` for the point estimate and this
    spread for the sampling uncertainty.

    MEASURED ON THIS RELEASE (ad_vs_hc, fixed n=221 / 46 positive, B=400):
    cognitive [0.740, 0.935] and wgs [0.327, 0.636] do not overlap, so the
    cheapest-vs-most-expensive modality gap is resolvable even at this n.
    cognitive vs biomarkers [0.611, 0.823] DOES overlap -- the gap that
    separates is cognitive against the two weak high-dimensional channels, not
    against every other channel. These intervals are ~0.10-0.16 wide, several
    times the CV-fold SD used in the acquisition curve; that fold SD is not a
    confidence interval and must not be reported as one.
    """
    channels = list(channels or SPHERE_CHANNELS_WELLCOVERED)
    mask = cohort.common_index(channels)
    y_all = cohort.y[mask]
    if len(np.unique(y_all)) < 2:
        raise ValueError("fixed cohort has a single class -- cannot compute AUC")
    Xs = {c: cohort.X[c][mask] for c in channels}

    pos = np.flatnonzero(y_all == 1)
    neg = np.flatnonzero(y_all == 0)
    rng = np.random.default_rng(seed)

    draws = {c: [] for c in channels}
    for b in range(B):
        idx = np.concatenate([rng.choice(pos, size=len(pos), replace=True),
                              rng.choice(neg, size=len(neg), replace=True)])
        yb = y_all[idx]
        for c in channels:
            try:
                # `groups=idx` is the guard: idx holds ORIGINAL participant
                # positions, so every duplicate of one person shares a group.
                draws[c].append(_cv_auc_grouped(Xs[c][idx], yb, groups=idx,
                                                seed=seed + b))
            except ValueError:
                # Too few distinct participants of one class for a grouped
                # stratified split. NaN, not skip -- the per-channel lists must
                # stay row-aligned so draw i means the same resample everywhere.
                draws[c].append(float("nan"))

    rows = []
    for c in channels:
        d = np.asarray(draws[c], float)
        d = d[~np.isnan(d)]
        rows.append({
            "channel": c, "cost_rank": cohort.cost_rank[c],
            "boot_mean": float(np.mean(d)) if d.size else float("nan"),
            "boot_sd": float(np.std(d, ddof=1)) if d.size > 1 else float("nan"),
            "ci_lo": float(np.percentile(d, 100 * alpha / 2)) if d.size else float("nan"),
            "ci_hi": float(np.percentile(d, 100 * (1 - alpha / 2))) if d.size else float("nan"),
            "n_draws": int(d.size), "B_requested": B, "n_fixed": int(mask.sum()),
            "n_pos": int(y_all.sum()), "n_features": Xs[c].shape[1],
        })
    summary = pd.DataFrame(rows)
    if not return_draws:
        return summary
    return summary, pd.DataFrame(draws)


def sphere_differential_gain(cohort_or_data_dir, tasks=None, channels=None,
                             seeds: int = 25, min_class_n: int = SPHERE_MIN_CLASS_N,
                             exclude_staging: bool = True) -> pd.DataFrame:
    """Cognitive alone vs cognitive+3 costly modalities, per differential pair.

    SYNTHETIC cohort (Stanford ADRC SPHERE): methods development only. Nothing
    below is evidence about Alzheimer's disease or about how any real patient
    should be worked up; the claim is FRAMEWORK TRANSFERABILITY -- that an
    over-acquisition point exists outside the easy disease-vs-healthy contrast.

    AD-vs-HC is nearly saturated by one cheap channel, so its budget points sit
    close together. The differential pairs (AD vs LBD / MCI / PD, MCI vs HC) are
    the harder contrasts, and this table asks the acquisition question in its
    starkest form: does appending the three costly modalities to the cheap
    cognitive battery help at all? Each pair is scored on ITS OWN common cohort
    (participants covered by all four channels), because the pairs have
    different coverage and a shared cohort would be the intersection of all of
    them.

    Pairs where either class has fewer than `min_class_n` members are returned
    with `skipped=True` and a reason instead of a number.

    AVERAGED OVER FOLD SEEDS, AND WHY THAT IS NOT OPTIONAL HERE
    ----------------------------------------------------------
    These cohorts are small (59-230 participants), so a single CV split is a
    very noisy read: measured on this release, the ad_vs_lbd gain ranges from
    -0.02 to -0.32 across 25 fold seeds. A one-seed table would let the seed
    choose the headline. Every cell is therefore the mean over `seeds` fold
    seeds, and `gain_sd` / `frac_seeds_negative` are reported beside it so a
    reader can see which rows are direction-stable and which are a coin flip.

    THE HONEST CAVEAT -- REQUIRED WORDING FOR ANY WRITE-UP
    -----------------------------------------------------
    Where the gain is negative, part of the drop is STATISTICAL, not
    information-theoretic. The added modalities are high-dimensional (plasma 500
    columns, wgs 167 after prep) and the smaller class here holds 13-55 people,
    so a linear model over them over-fits and dilutes a strong cheap signal. The
    supportable claim is scoped:

        "In this small-sample, high-dimensional regime, appending costly
        high-dimensional modalities to an already-sufficient cheap channel
        degrades discrimination."

    It is NOT "more evidence always hurts". It remains a genuine
    over-acquisition instance -- under a cost constraint, ordering these assays
    on these cohorts is net-harmful -- but the mechanism must be stated, or a
    reviewer is right to read a small-n artifact dressed up as a general law.

    MEASURED DIRECTION ON THIS RELEASE (25 fold seeds, staging excluded)
    -------------------------------------------------------------------
    The drop is NOT universal across the pairs. ad_vs_lbd (-0.162) and ad_vs_pd
    (-0.083) fall in every one of 25 seeds; mci_vs_hc (-0.033) falls in most;
    ad_vs_mci RISES (+0.026, negative in only 6 of 25 seeds). So the honest
    summary is "negative in three of four resolvable pairs, one of them
    marginal, and positive in the fourth" -- not a uniform decline. Anything
    claiming all four fall has not been run against this data.
    """
    channels = list(channels or SPHERE_CHANNELS_WELLCOVERED)
    tasks = list(tasks or SPHERE_DIFFERENTIAL_TASKS)
    if "cognitive" not in channels:
        raise ValueError("`cognitive` must be in channels -- it is the baseline")

    if isinstance(cohort_or_data_dir, SphereCohort):
        # A cohort is built for ONE task's label pair, so it cannot answer for
        # any other pair. Accepted for the single-task call; anything else needs
        # the data dir so each pair gets its own load.
        co = cohort_or_data_dir
        if list(tasks) != [co.task]:
            raise ValueError(
                f"cohort was built for task={co.task!r}; pass the data directory "
                f"to score {tasks} (each pair needs its own label subset)")
        loaded = {co.task: co}
        data_dir = None
    else:
        loaded, data_dir = {}, str(cohort_or_data_dir)

    rows = []
    for t in tasks:
        co = loaded.get(t)
        if co is None:
            co = load_sphere_cohort(data_dir, task=t, channels=channels,
                                    exclude_staging=exclude_staging)
        mask = co.common_index(channels)
        y = co.y[mask]
        n_pos, n_neg = int(y.sum()), int((y == 0).sum())
        row = {"task": t, "positive": SPHERE_TASKS[t][0],
               "negative": SPHERE_TASKS[t][1],
               "n": int(mask.sum()), "n_pos": n_pos, "n_neg": n_neg,
               "n_seeds": seeds, "skipped": False, "skip_reason": ""}
        if min(n_pos, n_neg) < min_class_n:
            row.update({"skipped": True,
                        "skip_reason": f"smallest class has {min(n_pos, n_neg)} "
                                       f"< {min_class_n} members",
                        "auc_cognitive": float("nan"),
                        "auc_cognitive_sd": float("nan"),
                        "auc_3mod": float("nan"), "auc_3mod_sd": float("nan"),
                        "gain": float("nan"), "gain_sd": float("nan"),
                        "frac_seeds_negative": float("nan"),
                        "n_features_3mod": sum(co.X[c].shape[1] for c in channels)})
            rows.append(row)
            continue

        Xc = co.X["cognitive"][mask]
        X3 = np.hstack([co.X[c][mask] for c in channels])
        a1 = np.array([_cv_auc(Xc, y, seed=s) for s in range(seeds)])
        a3 = np.array([_cv_auc(X3, y, seed=s) for s in range(seeds)])
        # Paired by seed: both blocks see the SAME split at seed s, so the
        # per-seed difference isolates the added modalities from split noise.
        g = a3 - a1
        row.update({
            "auc_cognitive": float(a1.mean()),
            "auc_cognitive_sd": float(a1.std(ddof=1)) if seeds > 1 else float("nan"),
            "auc_3mod": float(a3.mean()),
            "auc_3mod_sd": float(a3.std(ddof=1)) if seeds > 1 else float("nan"),
            "gain": float(g.mean()),
            "gain_sd": float(g.std(ddof=1)) if seeds > 1 else float("nan"),
            "frac_seeds_negative": float(np.mean(g < 0)),
            "n_features_3mod": int(X3.shape[1]),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def sphere_budget_contrasts(draws: pd.DataFrame, alpha: float = 0.05
                            ) -> pd.DataFrame:
    """Paired bootstrap contrasts between budgets, from `sphere_curve_bootstrap`
    draws (call it with `return_draws=True`).

    WHY MARGINAL INTERVALS ARE THE WRONG TOOL HERE
    ----------------------------------------------
    The per-budget intervals overlap heavily, but concluding "no difference" from
    overlapping intervals is the overlapping-CI fallacy: every budget is measured
    on the SAME resampled participants, so most of each interval's width is
    cohort-level variation shared by all budgets and cancels in the difference.
    The paired difference is the statistic that answers "does adding this channel
    change discrimination?", and it is far tighter than the marginals suggest.

    Returns, for each ordered pair (i, j) of consecutive budgets plus (1, K):
    the mean paired difference, its percentile interval, and the fraction of
    resamples in which the difference had the reported sign.
    """
    ks = list(draws.columns)
    pairs = [(ks[i], ks[i + 1]) for i in range(len(ks) - 1)]
    if len(ks) > 2:
        pairs.append((ks[0], ks[-1]))
    rows = []
    for a, b in pairs:
        d = (draws[b] - draws[a]).dropna().to_numpy()
        if d.size == 0:
            continue
        m = float(np.mean(d))
        rows.append({
            "from_k": a, "to_k": b, "mean_diff": m,
            "ci_lo": float(np.percentile(d, 100 * alpha / 2)),
            "ci_hi": float(np.percentile(d, 100 * (1 - alpha / 2))),
            "frac_same_sign": float(np.mean(d < 0) if m < 0 else np.mean(d > 0)),
            "n_draws": int(d.size),
        })
    return pd.DataFrame(rows)
