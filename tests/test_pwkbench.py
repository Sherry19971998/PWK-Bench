"""Core correctness tests for PWK-Bench."""
import numpy as np
import pytest
from pwkbench import spec, strategies as S, metrics as M
from pwkbench.domains.variant import make_synthetic_cohort, VARIANT_DOMAIN


@pytest.fixture(scope="module")
def cohort():
    return make_synthetic_cohort()


def test_spec_locked():
    assert len(spec.GENES) == 12
    assert spec.N_POSITIVE + spec.N_NEGATIVE == spec.N_VARIANTS_TOTAL == 491
    assert spec.CHANNELS == ["PVS1", "PM2", "PP3", "PM1"]
    # genes match the paper's four domains exactly
    assert spec.GENES_BY_DOMAIN["hereditary_cancer"] == ["BRCA1", "BRCA2", "TP53", "PTEN", "MLH1", "MSH2"]


def test_cohort_matches_spec(cohort):
    assert len(cohort) == 491
    assert int(cohort.y.sum()) == 229
    assert cohort.df["gene"].nunique() == 12
    assert int(cohort.df["PP3__defined"].sum()) == spec.N_MISSENSE_WITH_PP3  # exactly 80


def test_consequence_stratified_gap_uses_applicable_action_space(cohort):
    """Consequence stratification must (a) score each stratum only over the
    ACMG categories applicable to it -- so the missense stratum EXCLUDES PVS1
    (LoF-only, always inapplicable to missense) and cannot win a spurious gap
    from a fixed order wasting its first pick on a dead channel -- and
    (b) report a permutation-null p-value per stratum."""
    out = M.consequence_stratified_gap(cohort, n_perm=30)
    assert "missense" in out
    # missense action space must not contain the inapplicable PVS1
    assert "PVS1" not in out["missense"]["channels"]
    for name, v in out.items():
        assert v["n"] > 0 and v["n_pos"] > 0
        assert 0.0 <= v["p_value"] <= 1.0
        # gap is oracle minus relmax on the applicable pool -> non-negative
        assert v["gap"] >= -1e-9, (name, v["gap"])


def test_channel_complementarity_reports_pairs(cohort):
    """The complementarity diagnostic must return, for defined channel pairs,
    a rank correlation, single-channel AUCs, the paired AUC, and a lift
    (paired minus best-single). Positive lift = genuine complementarity; lift<=0
    is NOT read as redundancy (an equal-weight combiner can dilute a strong
    channel). See the function docstring for the one-directional reading."""
    cc = M.channel_complementarity(cohort)
    assert cc, "expected at least one channel pair"
    for pair, v in cc.items():
        assert v["n"] >= 10
        assert v["rho"] is None or -1.0 <= v["rho"] <= 1.0
        assert 0.0 <= v["auc_a"] <= 1.0 and 0.0 <= v["auc_b"] <= 1.0
        # paired score is never worse than the best single channel by construction margin
        assert v["lift"] == pytest.approx(v["auc_pair"] - max(v["auc_a"], v["auc_b"]), abs=1e-9)


def test_oracle_is_upper_bound(cohort):
    """The per-instance oracle must dominate RelMax at every budget k."""
    Ao = M.curve_A(cohort, S.oracle_order(cohort))
    Ar = M.curve_A(cohort, S.relmax_order(cohort))
    for k in Ao:
        assert Ao[k] >= Ar[k] - 1e-9, f"oracle < relmax at k={k}"


def test_relmax_is_fixed_order(cohort):
    """RelMax must be the SAME permutation for every variant."""
    o = S.relmax_order(cohort)
    assert (o == o[0]).all()


def test_relmax_is_label_free_channel_order(cohort):
    """RelMax must be the domain's declared (label-free) channel order, i.e.
    the spec CHANNELS permutation [0,1,...,K-1], NOT a label-fitted order."""
    o = S.relmax_order(cohort)
    assert list(o[0]) == list(range(cohort.domain.K))


def test_definedness_baselines_bracket_gap(cohort):
    """The schema-only references must be bracketed by RelMax (below, a fixed
    label-free order) and the oracle (above), and the stratified best must be at
    least as strong as the plain defined-first baseline. This makes
    Oracle-DefinednessStratifiedBest a valid 'genuine per-instance planning'
    residual (the rest of the Oracle-RelMax gap is schema/definedness lookup)."""
    K = cohort.domain.K
    def PE(order):
        A = M.curve_A(cohort, order, budgets=list(range(1, K + 1)))
        return float(np.mean(list(A.values())))
    pe_rel = PE(S.relmax_order(cohort))
    pe_db = PE(S.definedness_baseline_order(cohort))
    pe_dsb = PE(S.definedness_stratified_best_order(cohort))
    pe_ora = PE(S.oracle_order(cohort))
    assert pe_dsb >= pe_db - 1e-9            # stratified-best dominates defined-first
    assert pe_ora + 1e-9 >= pe_dsb           # oracle is the ceiling
    assert pe_dsb + 1e-9 >= pe_rel           # schema-optimal is >= label-free RelMax


def test_best_fixed_is_optimal_over_all_fixed_orders(cohort):
    """BestFixed must be a single fixed permutation whose PE is >= every other
    fixed permutation's PE (so no hand-picked or brute-forced fixed order beats
    it), and it must be bracketed by RelMax (below) and the oracle (above)."""
    import itertools
    K = cohort.domain.K
    bf = S.best_fixed_order(cohort)
    assert (bf == bf[0]).all()                       # one fixed order
    def PE(order):
        A = M.curve_A(cohort, order, budgets=list(range(1, K + 1)))
        return float(np.mean(list(A.values())))
    pe_bf = PE(bf)
    n = len(cohort)
    best_brute = max(PE(np.tile(p, (n, 1))) for p in itertools.permutations(range(K)))
    assert pe_bf >= best_brute - 1e-9                # nothing fixed beats BestFixed
    assert PE(S.oracle_order(cohort)) + 1e-9 >= pe_bf >= PE(S.relmax_order(cohort)) - 1e-9


def test_order_optimality_signature(cohort):
    """RelMax aligns with relevance, oracle aligns with oracle -- under BOTH the
    primary prefix-overlap metric (which drives A(k)) and the secondary rho."""
    rel = M.order_optimality(cohort, S.relmax_order(cohort))
    ora = M.order_optimality(cohort, S.oracle_order(cohort))
    # primary: prefix-set overlap must be exactly 1.0 for the matching reference
    assert abs(rel["po_vs_relevance"] - 1.0) < 1e-9
    assert abs(ora["po_vs_oracle"] - 1.0) < 1e-9
    assert rel["po_vs_relevance"] > rel["po_vs_oracle"]
    # secondary: rho retained for continuity
    assert rel["rho_vs_relevance"] > 0.99
    assert ora["rho_vs_oracle"] > 0.99
    assert rel["rho_vs_relevance"] > rel["rho_vs_oracle"]


def test_masked_auc_near_chance(cohort):
    """Memorization control: gene-only scorer must be near 0.5."""
    assert abs(M.masked_auc(cohort) - 0.5) < 0.15


def test_mock_agent_adherence_differentiates():
    """Different adherence -> different order-optimality (config is respected)."""
    from pwkbench.agents import build_agent
    c = make_synthetic_cohort()
    hi = build_agent("mock", "m", adherence=0.95).order(c)
    lo = build_agent("mock", "m", adherence=0.40).order(c)
    rho_hi = M.order_optimality(c, hi)["rho_vs_relevance"]
    rho_lo = M.order_optimality(c, lo)["rho_vs_relevance"]
    assert rho_hi > rho_lo


def test_offline_demo_smoke():
    """The whole harness runs offline with the demo matrix."""
    import yaml
    from pwkbench import harness
    c = make_synthetic_cohort()
    cfg = yaml.safe_load(open("configs/models.yaml"))["demo"]
    df = harness.run(c, model_matrix=cfg)
    assert len(df) == 8 + 4                     # 8 strategies (RelMax/BestFixed/2x Definedness/Oracle/Heuristic/Random/DiscriminativeGreedy) + 4 demo agents
    assert (df["delta_oracle"] >= -1e-9).all()  # oracle gap non-negative


def test_context_ablation_raises_oracle_alignment():
    """P1(b): supplying variant context should let the agent plan more
    instance-adaptively -> higher rho_vs_oracle than the no-context agent."""
    from pwkbench.agents import build_agent
    c = make_synthetic_cohort()
    off = build_agent("mock", "m", adherence=0.85, context=False).order(c)
    on = build_agent("mock", "m", adherence=0.85, context=True).order(c)
    rho_off = M.order_optimality(c, off)["rho_vs_oracle"]
    rho_on = M.order_optimality(c, on)["rho_vs_oracle"]
    assert rho_on > rho_off, (rho_on, rho_off)


def test_all_three_axes_in_results_table():
    """Axis C (masked_auc) must appear in the results table, not just in tests
    — Bo/reviewer concern that only two of three axes were reported."""
    from pwkbench.domains.variant import make_synthetic_cohort
    from pwkbench import harness
    c = make_synthetic_cohort()
    df = harness.run(c, model_matrix=None, with_ci=False)
    assert "masked_auc" in df.columns
    assert "PE" in df.columns and "rho_vs_oracle" in df.columns


def test_memorization_probe_two_mask(cohort):
    """Two-mask probe: coordinate mask near chance, full acquisition high,
    G_acq = full_acquisition - full_hgvs is the paper's positive acquisition
    gain over the deployment-realistic (identity-disclosed) closed-book
    baseline."""
    mp = M.memorization_probe(cohort)
    assert abs(mp["coord_masked"] - 0.5) < 0.15          # near chance
    assert mp["full_acquisition"] > 0.85                 # open-book ceiling
    assert mp["G_acq"] > 0.1                             # acquisition-attributable gain


def test_paired_contrast_oracle_beats_relmax(cohort):
    """Paired contrast: Oracle-RelMax at k=1 is positive with CI excluding 0
    (the machinery the paper uses for agent-vs-RelMax)."""
    from pwkbench.strategies import relmax_order, oracle_order
    pc = M.paired_contrast_ci(cohort, oracle_order(cohort), relmax_order(cohort), k=1, B=200)
    assert pc["diff"] > 0 and not pc["includes_zero"]


def test_domain_gap_sign_is_structural(cohort):
    """oracle >= relmax is an ALGEBRAIC identity (the oracle's subset family
    contains RelMax's budget-k subset), so gap >= 0 must hold for every domain
    on any data. This documents that the sign carries no information -- the
    informative quantity is the null-excess reported by
    consequence_stratified_gap, not this sign."""
    ds = M.domain_stratified_gap(cohort, k=1)
    assert len(ds) >= 2
    assert all(v["gap"] >= -1e-9 for v in ds.values())


def test_ablation_gap_nonneg_after_dropping_pp3(cohort):
    """Evidence-pool ablation: dropping PP3 leaves gap >= 0 -- structural, not
    a finding (see docstring). Kept-channel set must exclude PP3."""
    ab = M.evidence_pool_ablation(cohort, drop=["PP3"], k=1)
    assert "PP3" not in ab["kept_channels"] and ab["gap"] >= -1e-9


def test_consequence_gap_reports_null_excess_and_per_k(cohort):
    """The stratified gap must expose the cross-stratum-comparable statistics
    (gap_excess, z, per-k gaps) and a permutation p-value that is never exactly
    0 (the (1+c)/(B+1) convention), and must separate a true LoF stratum from
    the non-LoF 'other' stratum so PVS1 is never a dead first pick."""
    out = M.consequence_stratified_gap(cohort, n_perm=30)
    assert "missense" in out
    # true-LoF stratum, if present, keeps PVS1; 'other' never does
    if "other" in out:
        assert "PVS1" not in out["other"]["channels"]
    if "lof" in out:
        assert "PVS1" in out["lof"]["channels"]
    for v in out.values():
        assert isinstance(v["per_k_gap"], list) and len(v["per_k_gap"]) >= 1
        assert "gap_excess" in v and "z" in v
        assert v["p_value"] >= 1.0 / (30 + 1) - 1e-12   # never exactly 0


def test_holm_bonferroni_monotone():
    out = M.holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.30})
    assert out["a"]["reject"] and not out["c"]["reject"]
    assert out["a"]["p_adj"] <= out["b"]["p_adj"] <= out["c"]["p_adj"]


def test_paper_named_entry_points_exist_and_run(tmp_path):
    """The scripts the paper's Experiments section cites (run_gate_4slot.py,
    run_agent.py --replay, run_closed_book.py) must exist and run offline, so a
    reviewer following the paper does not hit a missing-file error."""
    import subprocess, sys, os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    # Ensure the repo root is importable in the child even on a plain checkout
    # (no editable install / no exported PYTHONPATH), else `import pwkbench`
    # fails and these entry-point tests spuriously error.
    env["PYTHONPATH"] = here + os.pathsep + env.get("PYTHONPATH", "")
    out = str(tmp_path / "out")
    fig = str(tmp_path / "fig")
    # Write every output under tmp_path via explicit flags so the test never
    # pollutes the repo's results/ and figures/ directories. (run_closed_book
    # takes only --outdir; run_gate/run_agent wrap run_benchmark and take both.)
    cases = [
        ["scripts/variant/run_gate_4slot.py", "--demo", "--domain", "variant",
         "--outdir", out, "--figdir", fig],
        ["scripts/variant/run_agent.py", "--replay", "--domain", "variant",
         "--outdir", out, "--figdir", fig],
        ["scripts/variant/run_closed_book.py", "--demo", "--domain", "variant",
         "--outdir", out],
    ]
    for cmd in cases:
        assert os.path.exists(os.path.join(here, cmd[0])), f"missing {cmd[0]}"
        r = subprocess.run([sys.executable] + cmd, cwd=here,
                           capture_output=True, text=True, env=env)
        assert r.returncode == 0, f"{cmd[0]} failed: {r.stderr[-500:]}"
    # Each script must have written its results UNDER the passed --outdir (proving
    # --outdir is honoured). We check the tmp_path got populated rather than that
    # the repo's results/ is absent -- a developer may have a real results/ dir
    # from an actual run, and this test must not depend on that being clean.
    # With an explicit --outdir, results.csv is written directly there (the
    # <domain> subdir only appears in the default results/<domain> path).
    assert os.path.exists(os.path.join(out, "results.csv")), \
        "entry points did not write results.csv under --outdir"


def test_cost_model_reports_deployable_ranks_and_oracle_reference(cohort):
    """Cost-model robustness (RQ4/appendix): the Oracle uses the global
    best-subset rule and is charged that subset's true cost, so it is a separate
    reference ceiling, NOT ranked against the deployable (first-k) strategies.
    We check the reported structure, and that `ranking_invariant` is the
    MEASURED invariance of the deployable ranks (not an assumed True): on this
    near-tied synthetic cohort the deployable top strategy can swap across
    calibers, and that honest outcome must be reflected, not forced."""
    from pwkbench.strategies import relmax_order, oracle_order, heuristic_order
    orders = {"Oracle": oracle_order(cohort), "RelMax": relmax_order(cohort),
              "Heuristic": heuristic_order(cohort)}
    cr = M.cost_robustness(cohort, orders)
    assert "full_ranking_invariant" in cr
    tops = set()
    for cal in ("per_query", "per_token", "per_monetary_cost"):
        rank = cr["per_caliber"][cal]["rank"]
        assert "Oracle" not in rank            # oracle excluded from the ranked set
        assert set(rank) == {"RelMax", "Heuristic"}
        assert cr["per_caliber"][cal]["oracle_reference_aucc"] is not None
        tops.add(rank[0])
    # ranking_invariant must equal the actually-measured invariance
    assert cr["ranking_invariant"] == (len(tops) == 1)


def test_confidence_stopping_saves_queries(cohort):
    """Confidence-based stopping: a positive margin stops before the full
    budget on average (fewer than K queries) while staying well above chance."""
    from pwkbench.strategies import oracle_order
    cs = M.confidence_based_stopping(cohort, oracle_order(cohort))
    for m, v in cs.items():
        assert v["mean_queries"] <= v["full_k"]
        assert v["auc_at_stop"] > 0.5


def test_trajectory_diversity_fixed_vs_adaptive(cohort):
    """Trajectory diversity: a fixed relevance strategy uses exactly one order;
    the per-variant oracle uses many (paper RQ2 diversity statistic)."""
    from pwkbench.strategies import relmax_order, oracle_order
    td_rel = M.trajectory_diversity(relmax_order(cohort), cohort.domain.channels)
    td_ora = M.trajectory_diversity(oracle_order(cohort), cohort.domain.channels)
    assert td_rel["n_distinct_orders"] == 1
    assert td_ora["n_distinct_orders"] > 1
    assert sum(td_ora["first_query_counts"].values()) == td_ora["n_variants"]


def test_constant_channel_is_scoring_noop(cohort):
    """A channel with a single distinct value carries no information: acquiring
    it must not shift the score (scorer centers on the channel's neutral value,
    not a hardcoded 0.5). Regression for the neutral_for() scorer fix."""
    from pwkbench.domains.base import load_real_cohort
    df = cohort.df.copy()
    df["PM1"] = float(df["PM1"].median())        # make PM1 constant
    df["PM1__defined"] = True
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # zero-info warning is expected here
        c = load_real_cohort(df, VARIANT_DOMAIN)
    j = c.domain.channels.index("PM1")
    only_pm1 = M._score_from_evidence(
        c, M._acquired_mask(np.tile([j] + [t for t in range(c.domain.K) if t != j],
                                    (len(c), 1)), 1))
    assert np.abs(only_pm1).max() < 1e-9


def _lof_stratum_view(cohort):
    """The exact row-subset view consequence_stratified_gap builds for the `lof`
    stratum: channels reduced to PVS1/PM2, rows reduced to PVS1==1 & non-missense,
    and the Domain (hence neutral_values) INHERITED from the parent cohort."""
    import dataclasses
    from pwkbench.domains.base import Cohort
    keep = ["PVS1", "PM2"]
    is_lof = (cohort.value("PVS1") == 1) & ~cohort.df["is_missense"].to_numpy(bool)
    dom2 = dataclasses.replace(
        cohort.domain, channels=keep,
        channel_defined_on={c: cohort.domain.channel_defined_on[c] for c in keep})
    return Cohort(cohort.df[is_lof].reset_index(drop=True), dom2), is_lof


def test_constant_channel_on_subset_view_is_noop(cohort):
    """Regression: on a ROW-SUBSET view lo/hi come from the subset while the
    neutral is inherited from the PARENT cohort, so `lo == hi != neutral` is
    reachable -- the `lof` stratum is PVS1==1 by construction while
    neutral_for('PVS1') is the full-cohort median 0. Without an explicit
    constant-channel branch the centering gives s_neutral = (0-1)/1e-12 ~ -1e12;
    that offset is uniform (so rank-preserving in exact arithmetic) but
    ulp(1e12) ~ 1.2e-4 quantizes the other channels' O(1) contributions into
    spurious ties. The constant channel must contribute EXACTLY 0."""
    view, is_lof = _lof_stratum_view(cohort)
    assert is_lof.sum() > 10, "need a non-trivial LoF stratum"
    # preconditions: PVS1 constant on the stratum, inherited neutral differs
    assert view.value("PVS1").min() == view.value("PVS1").max() == 1.0
    assert view.domain.neutral_for("PVS1") != 1.0
    n = len(view)
    only_pvs1 = M._score_from_evidence(view, np.array([[True, False]] * n))
    assert np.abs(only_pvs1).max() < 1e-9
    # acquiring it on top of PM2 must leave PM2's score BIT-identical
    both = M._score_from_evidence(view, np.ones((n, 2), bool))
    pm2_only = M._score_from_evidence(view, np.array([[False, True]] * n))
    assert np.array_equal(both, pm2_only)


def test_constant_channel_never_wins_the_heuristic_order(cohort):
    """Same blow-up, ordering side: |s - s_neutral| for a constant channel must
    be 0 so it sorts LAST. At ~1e12 a zero-information channel would be the
    heuristic's FIRST pick for every variant."""
    view, _ = _lof_stratum_view(cohort)
    order = S.heuristic_order(view)
    pm2 = view.value("PM2")
    lo, hi = pm2.min(), pm2.max()
    s = (pm2 - lo) / (hi - lo + 1e-12)
    s_neutral = (view.domain.neutral_for("PM2") - lo) / (hi - lo + 1e-12)
    informative = np.abs(s - s_neutral) > 1e-12      # PM2 carries signal here
    assert informative.sum() > 10
    # index 1 == PM2; wherever PM2 has any strength it must be queried first
    assert (order[informative, 0] == 1).all()


def test_zero_info_channel_warns(cohort):
    """load_real_cohort warns when a channel has no defined entries or a single
    distinct value, so a partial-real cohort cannot silently produce a
    plausible-looking table."""
    import warnings
    from pwkbench.domains.base import load_real_cohort
    df = cohort.df.copy()
    df["PM2"] = 0.5
    df["PM2__defined"] = True                    # constant -> zero information
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        load_real_cohort(df, VARIANT_DOMAIN)
    assert any("zero-information" in str(x.message) for x in w)


def test_reveal_buckets_on_defined_only(cohort):
    """reveal() must compute quantile edges on DEFINED entries only, so a
    missense-only channel keeps its full bucket resolution instead of collapsing
    to 2 when hundreds of undefined rows dominate the quantiles. Undefined
    entries return the -1 'no data' code."""
    r = cohort.reveal("PP3")
    undef = ~cohort.df["PP3__defined"].to_numpy(bool)
    assert (r[undef] == -1).all()                       # undefined -> -1
    assert len(set(r[~undef].tolist())) >= 4            # defined keep >=4 buckets


def test_synthetic_cohort_has_empirical_neutrals(cohort):
    """The synthetic/demo cohort must carry empirical per-channel neutrals (same
    path as real cohorts), not a scalar 0.5 -- otherwise PM2 (range ~[0,8]) gets
    a strong spurious offset."""
    assert cohort.domain.neutral_values is not None
    assert cohort.domain.neutral_for("PM2") > 1.0       # median of [0,8], not 0.5


def test_masked_auc_null_and_power():
    """The closed-book memorization probe must satisfy BOTH:
    (a) NULL: labels independent of gene identity -> AUC ~ 0.5 (unbiased), and
    (b) POWER: labels DETERMINED by gene identity -> AUC >> 0.5.
    (b) is the load-bearing half: a probe that returns 0.5 regardless of the
    labels (as a per-gene-macro variant once did) has zero power to detect
    memorization and silently passes any leak."""
    import numpy as np, copy
    from pwkbench.domains.variant import make_synthetic_cohort
    c = make_synthetic_cohort()
    genes = c.genes
    rng = np.random.default_rng(1)

    # (a) true null: labels iid, independent of gene
    c_null = copy.copy(c); dfn = c.df.copy()
    dfn["label"] = rng.integers(0, 2, len(dfn)); c_null.df = dfn
    assert abs(M.masked_auc(c_null) - 0.5) < 0.05

    # (b) power: label a deterministic function of gene identity
    gu = np.unique(genes); pos = {g: (i % 2 == 0) for i, g in enumerate(gu)}
    c_sig = copy.copy(c); dfs = c.df.copy()
    dfs["label"] = np.array([int(pos[g]) for g in genes]); c_sig.df = dfs
    assert M.masked_auc(c_sig) > 0.8


def test_orders_invariant_to_monotone_channel_rescale(cohort):
    """JUDGE test (not a regression net): every strategy's order must go through
    min-max normalization, so an affine-positive rescale of a channel (x -> a*x+b,
    a>0) -- which min-max is invariant to -- must not change ANY order. Catches
    the class of bug where a new consumer forgets to normalize (heuristic_order
    once compared raw magnitudes, always querying the widest-range channel).

    SCOPE: this covers AFFINE recodes (scale/offset) only. min-max is NOT
    rank-invariant, so a nonlinear monotone recode (e.g. -log10(AF) -> rank(AF))
    DOES change the curves. The benchmark must not claim "the channel encoding
    choice is irrelevant" -- only that scale/offset are."""
    import numpy as np
    from pwkbench.domains.base import attach_empirical_neutrals, Cohort
    from pwkbench import strategies as S
    df2 = cohort.df.copy()
    df2["PM2"] = 3.7 * df2["PM2"] - 1.2                  # monotone-increasing
    c2 = Cohort(df2, attach_empirical_neutrals(df2, VARIANT_DOMAIN))
    for fn in (S.relmax_order, S.oracle_order, S.heuristic_order, S.best_fixed_order):
        assert np.array_equal(fn(cohort), fn(c2)), fn.__name__


def test_bootstrap_preserves_cluster_multiplicity(cohort):
    """JUDGE test: a gene drawn r times in a cluster bootstrap must contribute r
    separate macro-average units, not be re-merged to 1. Without this the CI is
    systematically too narrow (degenerates to a random subset of ~7.6 genes)."""
    import numpy as np
    genes = list(np.unique(cohort.genes))
    grow = {g: np.where(cohort.genes == g)[0] for g in genes}
    pick = genes + [genes[0], genes[0]]                 # first gene drawn 3x
    cids = np.concatenate([np.full(len(grow[g]), rep) for rep, g in enumerate(pick)])
    assert len(np.unique(cids)) == len(pick)            # 14 units, not 12


def test_memorization_probe_quantities_are_same_unit(cohort):
    """DEFINITION test: every AUC in memorization_probe is a POOLED closed-book
    hold-out AUC (same unit), so G_acq = full_acquisition - full_hgvs is a
    coherent same-unit difference -- NOT the earlier bug that subtracted a
    per-gene-macro ranking AUC from a pooled AUC (which produced full_hgvs >
    full_acquisition).

    Asserted: (1) all four AUCs lie in [0,1]; (2) on this cohort the channel
    values carry real signal, so acquiring evidence beats the deployment-
    realistic (identity-disclosed) closed-book baseline. NOTE: hold-out AUC is
    NOT monotone in the feature set, so we do NOT assert coord <= full_hgvs <=
    full_acquisition -- adding a feature can lower a hold-out AUC, and that is a
    real property of the estimator, not a unit error."""
    p = M.memorization_probe(cohort)
    for kk in ("coord_masked", "full_hgvs", "pvs1_alone", "full_acquisition"):
        assert 0.0 <= p[kk] <= 1.0
    assert p["G_acq"] > 0.1                               # evidence beats full-HGVS baseline
    assert abs(p["G_acq"] - (p["full_acquisition"] - p["full_hgvs"])) < 1e-9
    # memory_floor is a separate validity-GATE diagnostic (chance-clamped
    # coordinate mask), not what G_acq is measured from -- see metrics.py.
    mem_floor = max(p["coord_masked"], 0.5)
    assert abs(p["memory_floor"] - mem_floor) < 1e-9


def test_llm_agent_parallel_matches_serial(cohort):
    """The real-agent order() may fan per-variant calls across threads. Parallel
    execution must produce EXACTLY the serial order (results written back by
    index) and the same step/parse-failure counts (lock-guarded), so the speedup
    never changes the science. Uses a deterministic fake _complete (no API key)."""
    import time
    from pwkbench.agents.llm import _StepwiseLLMAgent

    class _Fake(_StepwiseLLMAgent):
        def _complete(self, system, user):
            time.sleep(0.0005)                       # expose races if any
            for ch in ["PM2", "PVS1", "PP3", "PM1"]:
                if f"{ch}: not yet acquired" in user:
                    return f"take {ch}"
            return "PVS1"

    import copy
    c = copy.copy(cohort)
    c.df = cohort.df.iloc[:40].reset_index(drop=True)
    a1 = _Fake("fake", "f", max_workers=1); o1 = a1.order(c)
    a8 = _Fake("fake", "f", max_workers=8); o8 = a8.order(c)
    assert (o1 == o8).all()
    assert a1.total_steps == a8.total_steps
    assert a1.parse_failures == a8.parse_failures


def test_llm_agent_disk_cache_persists_across_instances(cohort, tmp_path):
    """A resumable daily driver (scripts/variant/warm_llm_cache.py) depends on the
    on-disk cache_path surviving a fresh process: a SECOND agent instance
    pointed at the same cache_path must reuse every prompt the first already
    answered and never call the model again, or a multi-day warm-up would
    re-ask (and re-spend scarce daily quota on) yesterday's prompts."""
    import copy
    from pwkbench.agents.llm import _StepwiseLLMAgent

    class _Fake(_StepwiseLLMAgent):
        def _complete(self, system, user):
            for ch in ["PM2", "PVS1", "PP3", "PM1"]:
                if f"{ch}: not yet acquired" in user:
                    return f"take {ch}"
            return "PVS1"

    c = copy.copy(cohort)
    c.df = cohort.df.iloc[:5].reset_index(drop=True)
    cache_path = str(tmp_path / "cache.json")

    a1 = _Fake("fake", "f", max_workers=1, cache_path=cache_path)
    o1 = a1.order(c)
    assert len(a1._cache) > 0

    class _MustNotBeCalled(_StepwiseLLMAgent):
        def _complete(self, system, user):
            raise AssertionError("disk cache miss: every prompt should already "
                                 "be cached from the first instance")

    a2 = _MustNotBeCalled("fake", "f", max_workers=1, cache_path=cache_path)
    assert len(a2._cache) == len(a1._cache)     # preloaded from disk at construction
    o2 = a2.order(c)                            # must not raise -> 100% cache hit
    assert (o1 == o2).all()


def test_llm_agent_daily_quota_stops_immediately_and_keeps_cache(tmp_path):
    """A DAILY free-tier quota error (Gemini's
    GenerateRequestsPerDayPerProjectPerModel-FreeTier) must raise
    DailyQuotaExhausted on the FIRST failing attempt, not after burning the
    full retry budget against a ~24h wall that will not move within this
    process -- and whatever completed before the failure must already be on
    disk, so a resumable driver never loses a day's partial progress."""
    import json
    from pwkbench.agents.llm import _StepwiseLLMAgent, DailyQuotaExhausted

    attempts = {"n": 0}

    class _QuotaCapped(_StepwiseLLMAgent):
        def _complete(self, system, user):
            attempts["n"] += 1
            if "PVS1: bucket" not in user:
                return "PVS1"           # first step: answers fine
            raise RuntimeError(
                "429 RESOURCE_EXHAUSTED: Quota exceeded for metric: "
                "generate_content_free_tier_requests, quotaId: "
                "GenerateRequestsPerDayPerProjectPerModel-FreeTier, "
                "quotaValue: 20. Please retry in 27s.")

    cache_path = str(tmp_path / "cache.json")
    a = _QuotaCapped("fake", "f", max_workers=1, retries=6, cache_path=cache_path)
    channels = ["PVS1", "PM2", "PP3", "PM1"]
    with pytest.raises(DailyQuotaExhausted):
        a._order_one(channels, [0, 0, 0, 0])
    assert attempts["n"] == 2                  # 1 success + 1 failure, NOT retried 6x
    with open(cache_path) as f:
        disk = json.load(f)
    assert len(disk) == 1                      # the one successful step survived to disk


def test_llm_agent_randomize_channel_order_remaps_to_canonical_indices(cohort):
    """randomize_channel_order=True presents a fresh per-variant random LISTING
    order (a position-bias diagnostic: does the model query whichever channel
    is listed first, or a specific channel regardless of listing?), but
    _order_one's returned picks are indices into THAT permuted list -- order()
    must remap them back to canonical cohort.domain.channels indices before
    storing the row, or every downstream metrics.py consumer (which assumes
    out[i, k] is a canonical index, consistent across rows) would silently
    score the wrong channel for permuted rows. Verified with a purely
    POSITION-biased fake model (always names whichever not-yet-acquired
    channel is listed first): under the fixed listing order every other run
    uses, it produces the IDENTICAL first pick for every variant; under the
    randomized listing order, its first pick must instead match that variant's
    OWN recorded permutation."""
    import copy
    from pwkbench.agents.llm import _StepwiseLLMAgent

    class _PositionBiased(_StepwiseLLMAgent):
        def _complete(self, system, user):
            for line in user.splitlines():
                if "not yet acquired" in line:
                    return line.split(":")[0].strip("- ").strip()
            return ""

    c = copy.copy(cohort)
    c.df = cohort.df.iloc[:20].reset_index(drop=True)
    K = c.domain.K

    fixed = _PositionBiased("fake", "f", max_workers=1)
    o_fixed = fixed.order(c)
    assert len(set(o_fixed[:, 0].tolist())) == 1        # same first pick for every variant, fixed listing

    rand = _PositionBiased("fake", "f", max_workers=1, randomize_channel_order=True)
    o_rand = rand.order(c)
    assert len(rand.channel_perms) == len(c)
    for i in range(len(c)):
        # first pick must match THIS variant's own permutation's first slot
        assert o_rand[i, 0] == rand.channel_perms[i][0]
        # every row must still be a valid permutation of canonical indices
        assert sorted(o_rand[i].tolist()) == list(range(K))
    # a purely position-biased model, under per-variant randomized listing,
    # must vary its first pick across variants (not collapse to one channel)
    assert len(set(o_rand[:, 0].tolist())) > 1


def test_oracle_curve_is_monotone_upper_bound(cohort):
    """The Oracle follows the paper's pi*(k)=argmax_{|pi|<=k} rule: acquiring AT
    MOST k categories (skipping label-hurtful ones) is a genuine upper bound, so
    A_Oracle(k) is monotone non-decreasing and dominates every deployable
    strategy at every budget. A deployable |pi|=k strategy (RelMax) may be
    non-monotone (forced to acquire a hurtful late channel); the oracle may not.
    """
    from pwkbench import metrics as M, strategies as S
    K = cohort.domain.K
    bud = list(range(1, K + 1))
    A_ora = M.curve_A(cohort, S.oracle_order(cohort), budgets=bud, leq_k=True)
    A_rel = M.curve_A(cohort, S.relmax_order(cohort), budgets=bud, leq_k=False)
    ks = sorted(A_ora)
    # monotone non-decreasing
    for a, b in zip(ks, ks[1:]):
        assert A_ora[b] >= A_ora[a] - 1e-9, f"oracle drops {a}->{b}: {A_ora}"
    # dominates the deployable reference at every budget
    for k in ks:
        assert A_ora[k] >= A_rel[k] - 1e-9, f"oracle below RelMax at k={k}"
    # the deployable |pi|=k rule is NOT skip-hurtful: identical order under both
    # flags would mean the flag is a no-op; confirm the oracle flag actually
    # changes the tail (else the fix is inert on this cohort).
    A_ora_forced = M.curve_A(cohort, S.oracle_order(cohort), budgets=bud, leq_k=False)
    assert A_ora[ks[-1]] >= A_ora_forced[ks[-1]] - 1e-9


def test_oracle_is_null_clean_under_shuffled_labels(cohort):
    """CRITICAL calibration guard: the oracle is a MEASUREMENT RULER, so when the
    evidence carries no information about the label (labels shuffled), the best
    attainable AUC must collapse to chance (~0.5), NOT stay high. A high oracle
    on shuffled labels means the label is leaking into the score rather than
    only selecting which subset to acquire -- which would make the whole
    planning-gap headline an artifact. This is the test that caught (and now
    prevents the return of) the per-variant signed-decisiveness leak: the oracle
    must use the GLOBAL best-subset argmax with the shared scorer, never a
    per-variant label gate.
    """
    import numpy as np
    from pwkbench import metrics as M, strategies as S
    bud = list(range(1, cohort.domain.K + 1))
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(3):
        df2 = cohort.df.copy()
        df2["label"] = rng.permutation(df2["label"].to_numpy())
        c2 = type(cohort)(df2, cohort.domain)
        # leq_k=True routes through the oracle best-subset path; pass the oracle
        # order explicitly so the test reads as (and stays) an oracle guard even
        # if curve_A's branch ever starts consuming `order`.
        A = M.curve_A(c2, S.oracle_order(c2), budgets=bud, leq_k=True)
        vals.append(float(np.mean(list(A.values()))))
    null_pe = float(np.mean(vals))
    # a null-clean ruler reads ~0.5 on pure noise; allow finite-sample slack.
    assert null_pe < 0.60, (
        f"oracle PE on shuffled labels = {null_pe:.3f} (should be ~0.5); "
        "label is leaking into the score, not just selecting the subset")


def test_multimodel_contrast_holm_corrects_family(cohort):
    """Multi-model agent-vs-RelMax contrasts are Holm-corrected across the model
    family (paper A4). p_holm >= p_raw for every model, and with >=2 models the
    correction is applied (adjusted p never below raw)."""
    from pwkbench import harness
    matrix = [
        {"kind": "mock", "slot": "m_hi", "adherence": 0.9, "capability_proxy": 0.9},
        {"kind": "mock", "slot": "m_mid", "adherence": 0.6, "capability_proxy": 0.6},
        {"kind": "mock", "slot": "m_lo", "adherence": 0.3, "capability_proxy": 0.3},
    ]
    df = harness.multimodel_contrast_holm(cohort, matrix, budget=1)
    assert len(df) == 3
    assert (df["p_holm"] >= df["p_raw"] - 1e-9).all()
    assert set(["diff_vs_relmax", "p_holm", "reject_holm"]).issubset(df.columns)


def test_harness_run_skips_daily_quota_exhausted_agent_without_crashing(cohort, monkeypatch):
    """A single agent hitting a provider's daily quota wall (DailyQuotaExhausted,
    e.g. a Gemini free-tier row mid-way through scripts/variant/warm_llm_cache.py's
    multi-day warm-up) must NOT crash the whole harness run: the reference
    strategies and any OTHER agent's row (already paid for in this same call)
    must still make it into the results table. Regression for harness.run()
    (and multimodel_contrast_holm) originally having no per-agent try/except
    around agent_and_order."""
    from pwkbench import harness
    from pwkbench.agents.llm import DailyQuotaExhausted
    from pwkbench.agents.base import Agent

    class _QuotaBoom(Agent):
        name = "boom"
        model_id = "boom-model"
        def order(self, cohort):
            raise DailyQuotaExhausted("simulated daily cap")

    real_build_agent = harness.build_agent
    def patched(kind, model_id, **kw):
        return _QuotaBoom() if kind == "boom" else real_build_agent(kind, model_id, **kw)
    monkeypatch.setattr(harness, "build_agent", patched)

    matrix = [
        {"slot": "ok_mock", "kind": "mock", "model_id": "mock-x", "adherence": 0.8},
        {"slot": "stuck", "kind": "boom", "model_id": "boom-model"},
    ]
    with pytest.warns(RuntimeWarning, match="daily quota exhausted"):
        df = harness.run(cohort, model_matrix=matrix)
    assert "agent:ok_mock" in df["strategy"].values     # unaffected agent survives
    assert "agent:stuck" not in df["strategy"].values   # quota-capped agent skipped, not crashed
    assert "Oracle" in df["strategy"].values            # reference rows still present


def test_clinical_yield_curve_is_a_valid_yield_accuracy_curve(cohort):
    """Clinical yield endpoint: resolved+VUS fractions partition to 1, called
    count is consistent, and accuracy is a valid probability. Abstention means
    resolved_frac must be <= 1 and never negative."""
    cy = M.clinical_yield_curve(cohort, S.relmax_order(cohort), confidence=0.90)
    for k, v in cy.items():
        assert abs(v["resolved_frac"] + v["vus_frac"] - 1.0) < 1e-9
        assert 0.0 <= v["resolved_frac"] <= 1.0
        assert v["n_called"] >= 0
        if v["n_called"] > 0:
            assert 0.0 <= v["call_accuracy"] <= 1.0


def test_acmg_points_curve_matches_guideline_scale_and_calls(cohort):
    """Guideline-native rule: point constants must equal the verified Tavtigian
    2020 scale, and every CALLED variant is one whose summed points crossed the
    documented thresholds (>=6 pathogenic / <=-1 benign). Yield must not exceed
    the calibrated-posterior rule's ceiling of 1."""
    # verified point scale (Tavtigian 2020 Table 2)
    assert (M._ACMG_PATH_MIN, M._ACMG_BEN_MAX) == (6.0, -1.0)
    ap = M.acmg_points_curve(cohort, S.relmax_order(cohort))
    for k, v in ap.items():
        assert abs(v["resolved_frac"] + v["vus_frac"] - 1.0) < 1e-9
        if v["n_called"] > 0:
            assert 0.0 <= v["call_accuracy"] <= 1.0
    # strict (>=10) threshold cannot resolve MORE than LP-inclusive (>=6)
    s = M.acmg_points_sensitivity(cohort, S.relmax_order(cohort), k=len(cohort.domain.channels))
    assert s["strict_path_ge10"]["resolved_frac"] <= s["LP_inclusive_ge6"]["resolved_frac"] + 1e-9


def test_cost_weighted_yield_charges_functional_channel_more(cohort):
    """Cost view: cumulative cost is non-decreasing in k, and acquiring the
    high-cost functional channel (PM1, cost 10) makes the last budget step's
    cost jump, lowering resolved_per_cost. resolved_per_cost must be finite and
    non-negative wherever cost>0."""
    cw = M.cost_weighted_yield(cohort, S.relmax_order(cohort))
    ks = sorted(cw)
    costs = [cw[k]["cum_cost"] for k in ks]
    assert all(costs[i + 1] >= costs[i] - 1e-9 for i in range(len(costs) - 1))
    for k in ks:
        assert cw[k]["cum_cost"] > 0 and cw[k]["resolved_per_cost"] >= 0.0
