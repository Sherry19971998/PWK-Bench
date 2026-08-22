# SYNTHETIC data (Stanford ADRC SPHERE). Methods-development use only; no clinical claim.
"""Tests for the SPHERE second domain.

These skip (rather than fail) when `data/sphere/` is absent, so the suite still
runs in a checkout that has not fetched the second-domain data -- the SPHERE CSVs
are produced by `scripts/sphere/prep_sphere.py` from a download that not every user has.
"""
import os
import numpy as np
import pandas as pd
import pytest

from pwkbench.domains.sphere import (
    load_sphere_cohort, sphere_curve, sphere_single_auc_fixed,
    sphere_differential_gain, sphere_modality_bootstrap,
    SPHERE_CHANNELS, SPHERE_CHANNELS_WELLCOVERED, SPHERE_COST_RANK,
    SPHERE_COST_BASIS, SPHERE_FILES, SPHERE_STAGING_COLS, SPHERE_TASKS,
    SPHERE_DIFFERENTIAL_TASKS, SPHERE_MIN_CLASS_N,
    LABEL_COL, ID)

DATA = "data/sphere"
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, "demographics_diagnosis.csv")),
    reason="data/sphere not built; run scripts/sphere/prep_sphere.py")


@pytest.fixture(scope="module")
def cohort():
    return load_sphere_cohort(DATA)


def test_loader_builds_binary_task(cohort):
    assert len(cohort) > 0
    assert set(np.unique(cohort.y)) <= {0, 1}
    assert cohort.y.sum() > 0 and (cohort.y == 0).sum() > 0
    assert cohort.channels == SPHERE_CHANNELS
    for c in cohort.channels:
        assert cohort.X[c].shape[0] == len(cohort)
        assert cohort.defined[c].shape == (len(cohort),)


def test_curve_is_on_fixed_cohort(cohort):
    """Every budget must score the SAME participants.

    If n were allowed to shrink as channels are added, the k=4 row would describe
    a smaller, differently-composed cohort than k=1 and the curve would not be an
    acquisition result at all.
    """
    curve = sphere_curve(cohort, order=SPHERE_CHANNELS_WELLCOVERED)
    assert len(curve) == len(SPHERE_CHANNELS_WELLCOVERED)
    assert curve.n_fixed.nunique() == 1
    mask = cohort.common_index(SPHERE_CHANNELS_WELLCOVERED)
    assert int(curve.n_fixed.iloc[0]) == int(mask.sum())
    # features accumulate strictly: each budget adds a whole channel block
    assert curve.n_features.is_monotonic_increasing
    assert list(curve.added_channel) == list(SPHERE_CHANNELS_WELLCOVERED)


def test_no_label_leakage(cohort):
    """The outcome column must not appear in any channel's source CSV."""
    for c, fname in SPHERE_FILES.items():
        path = os.path.join(DATA, fname)
        if not os.path.exists(path):
            continue
        cols = pd.read_csv(path, nrows=0).columns
        assert LABEL_COL not in cols, f"{fname} carries the label column"
        assert ID in cols


def test_cost_monotone():
    """The evaluated order must be non-decreasing in burden rank, or
    'cheapest-first' is not what the acquisition curve actually does."""
    ranks = [SPHERE_COST_RANK[c] for c in SPHERE_CHANNELS_WELLCOVERED]
    assert ranks == sorted(ranks)
    assert all(c in SPHERE_COST_RANK for c in SPHERE_CHANNELS)
    # Every rank must state what it rests on, so the ordering stays checkable
    # rather than becoming folklore.
    assert set(SPHERE_COST_BASIS) == set(SPHERE_COST_RANK)


def test_no_cumulative_cost_column(cohort):
    """Burden is ordinal, so nothing may publish a summed cost: a cumulative
    figure would assert ratios the ranks do not carry."""
    curve = sphere_curve(cohort, order=SPHERE_CHANNELS_WELLCOVERED)
    assert "cum_cost" not in curve.columns
    assert "cost_rank" in curve.columns
    assert list(curve.cost_rank) == [SPHERE_COST_RANK[c]
                                     for c in SPHERE_CHANNELS_WELLCOVERED]


def test_fixed_single_auc_shares_the_curve_cohort(cohort):
    """The like-for-like modality table must use the curve's participant set, so
    the two tables in the figure are talking about the same people."""
    curve = sphere_curve(cohort, order=SPHERE_CHANNELS_WELLCOVERED)
    single = sphere_single_auc_fixed(cohort, order=SPHERE_CHANNELS_WELLCOVERED)
    assert single.n_fixed.nunique() == 1
    assert int(single.n_fixed.iloc[0]) == int(curve.n_fixed.iloc[0])
    assert set(single.channel) == set(SPHERE_CHANNELS_WELLCOVERED)


def test_unknown_task_and_channel_rejected():
    with pytest.raises(ValueError):
        load_sphere_cohort(DATA, task="not_a_task")
    with pytest.raises(ValueError):
        load_sphere_cohort(DATA, channels=["cognitive", "not_a_channel"])


def test_staging_columns_excluded_by_default(cohort):
    """The CDR / cognitive-status block must be out of the `cognitive` channel
    unless explicitly asked for.

    These columns are the staging instrument the consensus diagnosis is built
    from, so leaving them in makes the channel partly predict itself -- a
    construct problem the label-column check cannot catch.
    """
    assert cohort.dropped_staging.get("cognitive")
    dropped = set(cohort.dropped_staging["cognitive"])
    assert dropped <= set(SPHERE_STAGING_COLS)

    raw = pd.read_csv(os.path.join(DATA, "cognitive_scores.csv"), nrows=0).columns
    present = [c for c in SPHERE_STAGING_COLS if c in raw]
    assert dropped == set(present), "guard must drop every staging column present"
    assert cohort.X["cognitive"].shape[1] == len(
        [c for c in raw if c != ID]) - len(present)

    unguarded = load_sphere_cohort(DATA, exclude_staging=False)
    assert unguarded.X["cognitive"].shape[1] > cohort.X["cognitive"].shape[1]
    assert unguarded.dropped_staging.get("cognitive", []) == []


def test_bootstrap_is_wider_than_fold_spread(cohort):
    """Participant-resampling uncertainty must exceed CV-fold-split spread.

    The fold-seed SD only says how much the estimate moves when the SAME people
    are split differently; it is not a confidence interval and is much smaller
    than the real sampling uncertainty. Pinning the inequality stops anyone from
    later quoting the fold spread as a CI.

    B is tiny here to keep the suite fast -- the assertion is about the
    ORDERING of the two spreads, which is already unambiguous at this size.
    """
    from pwkbench.domains.sphere import sphere_curve_bootstrap

    boot = sphere_curve_bootstrap(cohort, order=SPHERE_CHANNELS_WELLCOVERED,
                                  B=8, seed=0)
    assert list(boot.budget_k) == list(range(1, len(SPHERE_CHANNELS_WELLCOVERED) + 1))
    assert (boot.n_draws > 0).all()
    assert (boot.ci_hi >= boot.ci_lo).all()

    folds = pd.concat([sphere_curve(cohort, order=SPHERE_CHANNELS_WELLCOVERED,
                                    seed=s) for s in range(5)])
    fold_sd = folds.groupby("budget_k").cum_auc.std(ddof=1)
    assert (boot.set_index("budget_k").boot_sd > fold_sd).all(), \
        "bootstrap spread must exceed fold-split spread"


# Cohort sizes measured on this release, NOT copied from a plan: the four-channel
# common cohort for each differential pair. They are structural (who is covered
# by which assay), so they are safe to pin -- unlike the AUCs, which move with
# the fold seed and are deliberately not asserted here.
DIFFERENTIAL_N = {
    "ad_vs_lbd":         (59, 46, 13),
    "ad_vs_mci":         (101, 46, 55),
    "mci_vs_hc":         (230, 55, 175),
    "ad_vs_pd":          (105, 46, 59),
    "ad_vs_pd_dementia": (49, 46, 3),
}


def test_differential_tasks_build():
    """Each differential pair must load as a clean binary task on the expected
    participants, and the size guard must refuse exactly the ones too small to
    cross-validate.

    The pairs are drawn from `diagnosis_consensus`; a typo'd label string would
    silently produce an empty or half-sized class rather than an error, so the
    per-class counts are pinned.
    """
    for t in SPHERE_DIFFERENTIAL_TASKS:
        assert t in SPHERE_TASKS
        co = load_sphere_cohort(DATA, task=t,
                                channels=SPHERE_CHANNELS_WELLCOVERED)
        assert set(np.unique(co.y)) <= {0, 1}
        assert co.y.sum() > 0 and (co.y == 0).sum() > 0, \
            f"{t}: a label string matched nothing -- check the spelling"
        # The staging guard applies to every task, not just ad_vs_hc.
        assert co.dropped_staging.get("cognitive")

        mask = co.common_index(SPHERE_CHANNELS_WELLCOVERED)
        n, n_pos, n_neg = DIFFERENTIAL_N[t]
        assert (int(mask.sum()), int(co.y[mask].sum()),
                int((co.y[mask] == 0).sum())) == (n, n_pos, n_neg), t

    gain = sphere_differential_gain(DATA, seeds=2)
    assert list(gain.task) == list(SPHERE_DIFFERENTIAL_TASKS)
    assert list(gain.n) == [DIFFERENTIAL_N[t][0] for t in gain.task]

    too_small = {t for t, (_, p, q) in DIFFERENTIAL_N.items()
                 if min(p, q) < SPHERE_MIN_CLASS_N}
    assert set(gain.loc[gain.skipped, "task"]) == too_small
    assert (gain.loc[gain.skipped, "skip_reason"].str.len() > 0).all(), \
        "a skipped task must say why, not just vanish"
    scored = gain[~gain.skipped]
    assert len(scored) == len(SPHERE_DIFFERENTIAL_TASKS) - len(too_small)
    assert scored[["auc_cognitive", "auc_3mod", "gain"]].notna().all().all()
    # `gain` must be the difference actually reported, not an independent recompute
    assert np.allclose(scored.gain, scored.auc_3mod - scored.auc_cognitive)


def test_no_bootstrap_over_cv():
    """A participant may never appear in both train and test of a resample.

    Bootstrapping over CV folds is the leak this guards against: a resample
    contains the same person several times, so an ordinary k-fold trains on one
    copy and scores another, inflating every draw. The assertion runs against
    the SPLITS the bootstrap actually fits through -- `sphere_modality_bootstrap`
    is called with a recording stand-in for `_cv_auc_grouped`, so a future
    version that stopped passing `groups`, or passed the wrong ones, fails here.
    """
    from sklearn.model_selection import StratifiedKFold
    from pwkbench.domains import sphere as S

    cohort = load_sphere_cohort(DATA, channels=SPHERE_CHANNELS_WELLCOVERED)
    seen = []
    real = S._cv_auc_grouped

    def recorder(Xb, y, groups, n_splits=5, seed=0):
        seen.append((np.asarray(y), np.asarray(groups), n_splits, seed))
        return real(Xb, y, groups, n_splits=n_splits, seed=seed)

    S._cv_auc_grouped = recorder
    try:
        boot = sphere_modality_bootstrap(cohort, channels=["cognitive"], B=3,
                                         seed=0)
    finally:
        S._cv_auc_grouped = real

    assert len(seen) == 3, "the bootstrap must route through the grouped estimator"
    assert (boot.n_draws > 0).all() and (boot.ci_hi >= boot.ci_lo).all()

    leaks_without_guard = 0
    for y, groups, n_splits, seed in seen:
        # Vacuity check: with replacement the resample MUST contain repeats,
        # otherwise "no participant on both sides" is trivially true.
        assert len(np.unique(groups)) < len(groups)

        X = np.zeros((len(y), 1))
        for tr, te in S._grouped_cv(n_splits, seed).split(X, y, groups):
            shared = set(groups[tr]) & set(groups[te])
            assert not shared, f"participant(s) {sorted(shared)[:3]} on both sides"

        # And the guard is load-bearing: the ungrouped split on the same
        # resample does leak, so the assertion above is not passing by luck.
        for tr, te in StratifiedKFold(n_splits=n_splits, shuffle=True,
                                      random_state=seed).split(X, y):
            leaks_without_guard += len(set(groups[tr]) & set(groups[te]))
    assert leaks_without_guard > 0
