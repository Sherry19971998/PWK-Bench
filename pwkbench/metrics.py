"""
Evaluation metrics — the three axes (paper's Evaluation suite).

A. Acquisition efficiency (what/how much):
     A(k) = per-gene-stratified ranking AUC of the scorer f(E_k) built from the
            first k acquired channels.
     PE          = (1/K) sum_k A(k)
     Delta_Oracle= PE_oracle - PE_method
     k*          = min{k : A(k) >= tau}
B. Order-optimality (which):
     per-variant Spearman rho of the strategy's query order against
     (i) the global relevance order and (ii) the per-variant oracle order.
C. Memorization control (validity):
     closed-book masked AUC -> 0.5 (evidence removed; label must not be
     recoverable from identifiers alone).

CIs: gene-clustered bootstrap, B=2000 (paper).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from .domains.base import Cohort
from .strategies import oracle_order, relmax_order
from .spec import BUDGETS, TAU, BOOTSTRAP_B, GLOBAL_SEED


def _score_from_evidence(cohort: Cohort, acquired_mask: np.ndarray) -> np.ndarray:
    """
    f(E_k): a pathogenicity score per variant from the acquired channels only.
    Undefined/un-acquired channels contribute the neutral value. We use a fixed
    linear combination in signed-decisiveness space so the scorer is monotone in
    evidence and needs no training (keeps A(k) a pure function of the order).
    """
    n, K = acquired_mask.shape
    contrib = np.zeros(n)
    for j, ch in enumerate(cohort.domain.channels):
        v = cohort.value(ch).astype(float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        # A channel that is CONSTANT on the population being scored carries zero
        # information -> skip it, contributing exactly 0. This must be an
        # explicit branch, not a consequence of the centering below: on a
        # row-subset VIEW the domain's neutral is inherited from the PARENT
        # cohort while lo/hi come from the subset, so `lo == hi != neutral` is
        # reachable and s_neutral would blow up to ~1e12. (Live case:
        # consequence_stratified_gap's `lof` stratum is PVS1==1 by construction,
        # while neutral_for("PVS1") is the full-cohort median 0.) That offset is
        # uniform, hence rank-preserving in exact arithmetic, but ulp(1e12) is
        # ~1.2e-4, which quantizes every other channel's O(1) contribution into
        # spurious ties.
        if hi - lo < 1e-12:
            continue
        s = (v - lo) / (hi - lo + 1e-12)            # [0,1], higher = more pathogenic
        # Center on the CHANNEL'S neutral value in the same normalized space,
        # not a hardcoded 0.5. An acquired-but-undefined entry carries the
        # neutral value (Cohort.value), so it now contributes EXACTLY 0.
        # Hardcoding 0.5 gave zero-information channels a spurious +/- offset
        # whose size scaled with the channel's range.
        s_neutral = (cohort.domain.neutral_for(ch) - lo) / (hi - lo + 1e-12)
        use = acquired_mask[:, j]
        contrib += np.where(use, s - s_neutral, 0.0)
    return contrib


def _acquired_mask(order: np.ndarray, k: int) -> np.ndarray:
    """Boolean (n, K): the first k channels of each variant's order are acquired."""
    n, K = order.shape
    mask = np.zeros((n, K), bool)
    rows = np.arange(n)[:, None]
    mask[rows, order[:, :k]] = True
    return mask


def _macro_auc_by_cluster(y: np.ndarray, score: np.ndarray,
                          cluster_ids: np.ndarray) -> float:
    """Macro-averaged ranking AUC over integer cluster ids (fast path for the
    bootstrap: avoids np.unique on string labels). Clusters with a single class
    are skipped, exactly like per_gene_auc."""
    aucs = []
    order = np.argsort(cluster_ids, kind="stable")
    cid_s, y_s, sc_s = cluster_ids[order], y[order], score[order]
    bounds = np.flatnonzero(np.diff(cid_s)) + 1
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(cid_s)]):
        ys, ss = y_s[a:b], sc_s[a:b]
        n_pos = int(ys.sum()); n = len(ys); n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        # Mann-Whitney U AUC via mid-ranks (ties averaged) -- same value as
        # roc_auc_score but ~100x faster than a per-cluster sklearn call in the
        # B=2000 bootstrap loop.
        r = _midranks(ss)
        auc = (r[ys == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        aucs.append(auc)
    return float(np.mean(aucs)) if aucs else 0.5


def _midranks(v: np.ndarray) -> np.ndarray:
    """1-based average ranks (ties share the mean rank) -- for the U-statistic AUC."""
    o = np.argsort(v, kind="stable")
    vs = v[o]
    ranks = np.empty(len(v), float)
    i = 0
    while i < len(vs):
        j = i
        while j + 1 < len(vs) and vs[j + 1] == vs[i]:
            j += 1
        ranks[o[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def per_gene_auc(y: np.ndarray, score: np.ndarray, genes: np.ndarray) -> float:
    """Ranking AUC stratified (macro-averaged) over genes (paper)."""
    aucs = []
    for g in np.unique(genes):
        m = genes == g
        if len(np.unique(y[m])) < 2:
            continue
        aucs.append(roc_auc_score(y[m], score[m]))
    return float(np.mean(aucs)) if aucs else 0.5


def _best_subset_auc(cohort: Cohort, k: int) -> tuple:
    """The paper's oracle pi*(k)=argmax_{|pi|<=k} AUC(f(E_pi), y).

    Enumerate every channel SUBSET of size 1..k, score the WHOLE cohort with
    the same shared scorer `_score_from_evidence` (the identical scoring path
    used for RelMax and the agents -- no label gate, no per-variant selection),
    and return the highest per-gene AUC together with the winning subset.

    The gold label enters through EXACTLY ONE global decision -- which single
    subset maximizes cohort AUC -- not through the scorer and not per variant.
    This is the classical value-of-information / active-feature-acquisition
    "best feature subset under budget" oracle (howard1966voi,
    saartsechansky2009afa), and it is null-clean: when the evidence carries no
    information about the label (e.g. shuffled labels), the best attainable AUC
    is ~0.5, as a calibrated ruler must be. It is monotone non-decreasing in k
    because the argmax is over a family that only grows with the budget.
    """
    import itertools
    K = cohort.domain.K
    y, genes = cohort.y, cohort.genes
    best_auc, best_subset = -np.inf, ()
    for size in range(1, k + 1):
        for subset in itertools.combinations(range(K), size):
            mask = np.zeros((len(cohort), K), dtype=bool)
            mask[:, list(subset)] = True
            a = per_gene_auc(y, _score_from_evidence(cohort, mask), genes)
            if a > best_auc:
                best_auc, best_subset = a, subset
    return best_auc, best_subset


def oracle_subset_curve(cohort: Cohort, budgets=BUDGETS) -> dict:
    """A_Oracle(k) = best-subset AUC at each budget (see `_best_subset_auc`)."""
    return {k: _best_subset_auc(cohort, k)[0] for k in budgets}


def _greedy_subset_auc(cohort: Cohort, k: int) -> dict:
    """Myopic-greedy approximation to `_best_subset_auc`: build the GLOBAL
    subset one channel at a time, at each step adding whichever unselected
    channel gives the largest per-gene AUC given the channels already locked
    in, and never revisiting that choice. Unlike `_best_subset_auc` (exhaustive
    argmax over every size-<=k subset), this can in principle get stuck below
    the true optimum -- it is the standard "AFA-style" myopic baseline E2 asks
    for. Returns {k: auc} for k=1..k so the whole curve costs one call.

    `_best_subset_auc`'s pi*(k)=argmax_{|pi|<=k} is over subsets of size AT
    MOST k, not exactly k (a weak/noisy channel can make a forced-size subset
    WORSE than a smaller one -- e.g. a linear scorer with no per-channel
    weighting can be diluted by a low-signal channel). Greedy only ever grows
    its subset, but reports the running best-so-far across the sizes it has
    built through, so it gets the same "<=k" option: it is compared on equal
    terms to the exact oracle, not penalized for a size restriction the exact
    oracle does not have.
    """
    K = cohort.domain.K
    y, genes = cohort.y, cohort.genes
    selected: list[int] = []
    remaining = list(range(K))
    out = {}
    best_so_far = -np.inf
    for step in range(1, k + 1):
        best_auc, best_ch = -np.inf, None
        for ch in remaining:
            mask = np.zeros((len(cohort), K), dtype=bool)
            mask[:, selected + [ch]] = True
            a = per_gene_auc(y, _score_from_evidence(cohort, mask), genes)
            if a > best_auc:
                best_auc, best_ch = a, ch
        selected.append(best_ch)
        remaining.remove(best_ch)
        best_so_far = max(best_so_far, best_auc)
        out[step] = best_so_far
    return out


def _lookahead_subset_auc(cohort: Cohort, k: int) -> dict:
    """1-step-lookahead greedy: at each step, score every (candidate now,
    best-response next) PAIR two moves ahead and commit only the first move of
    the winning pair -- a 2-ply search rather than pure 1-ply greedy. Still
    cheaper than full enumeration for large K, and a strictly better
    approximation than `_greedy_subset_auc`; at this benchmark's K=4 all three
    oracle variants are expected to coincide exactly (E2), because the whole
    subset lattice is small enough that greedy has nowhere to get trapped.
    """
    K = cohort.domain.K
    y, genes = cohort.y, cohort.genes
    selected: list[int] = []
    remaining = list(range(K))
    out = {}
    best_so_far = -np.inf

    def auc_of(subset):
        mask = np.zeros((len(cohort), K), dtype=bool)
        mask[:, subset] = True
        return per_gene_auc(y, _score_from_evidence(cohort, mask), genes)

    for step in range(1, k + 1):
        if len(remaining) == 1:
            best_ch = remaining[0]
            best_auc = auc_of(selected + [best_ch])
        else:
            best_pair_auc, best_ch = -np.inf, None
            for ch in remaining:
                rest = [r for r in remaining if r != ch]
                # best RESPONSE to `ch`: the second move that maximizes AUC
                # given `ch` is taken -- this is the "lookahead" over `ch`'s
                # 1-ply greedy successor, not just `ch` alone.
                two_ply = max(auc_of(selected + [ch, r]) for r in rest)
                if two_ply > best_pair_auc:
                    best_pair_auc, best_ch = two_ply, ch
            best_auc = auc_of(selected + [best_ch])
        selected.append(best_ch)
        remaining.remove(best_ch)
        # Same "<=k" running-best convention as `_greedy_subset_auc` -- see
        # its docstring for why this is required for a fair comparison.
        best_so_far = max(best_so_far, best_auc)
        out[step] = best_so_far
    return out


def oracle_greedy_curve(cohort: Cohort, budgets=BUDGETS) -> dict:
    """A(k) under the myopic-greedy subset oracle (E2 sensitivity check)."""
    kmax = max(budgets)
    full = _greedy_subset_auc(cohort, kmax)
    return {k: full[k] for k in budgets}


def oracle_lookahead_curve(cohort: Cohort, budgets=BUDGETS) -> dict:
    """A(k) under the 1-step-lookahead subset oracle (E2 sensitivity check)."""
    kmax = max(budgets)
    full = _lookahead_subset_auc(cohort, kmax)
    return {k: full[k] for k in budgets}


def oracle_approximation_sensitivity(cohort: Cohort, budgets=BUDGETS) -> dict:
    """E2 deliverable: exact global-subset oracle vs myopic-greedy vs
    1-step-lookahead, at every budget. `_best_subset_auc`
    (`oracle_subset_curve`) already enumerates every subset of size <=k
    exhaustively -- NOT a myopic greedy -- so at this benchmark's K=4 it is
    the true per-budget optimum, and the reviewer's "unbounded greedy
    approximation" concern does not apply. This function makes that
    falsifiable: if exact/greedy/lookahead disagree at any budget, the
    exactness claim above is wrong and must be corrected before it goes in the
    paper, rather than asserted from the docstring alone.
    """
    exact = oracle_subset_curve(cohort, budgets)
    greedy = oracle_greedy_curve(cohort, budgets)
    lookahead = oracle_lookahead_curve(cohort, budgets)
    rows = {k: {"exact": exact[k], "greedy": greedy[k], "lookahead": lookahead[k],
                "greedy_gap": exact[k] - greedy[k],
                "lookahead_gap": exact[k] - lookahead[k]}
            for k in budgets}
    rows["all_coincide"] = all(
        abs(rows[k]["greedy_gap"]) < 1e-9 and abs(rows[k]["lookahead_gap"]) < 1e-9
        for k in budgets)
    return rows


def curve_A(cohort: Cohort, order: np.ndarray, budgets=BUDGETS,
            leq_k: bool = False) -> dict:
    """A(k) for k in budgets.

    leq_k=False (default): the deployable |pi|=k rule -- acquire exactly the
    first k channels of `order`, keep all their contributions. Used for RelMax,
    agents, and every non-oracle strategy.

    leq_k=True: the oracle's paper rule pi*(k)=argmax_{|pi|<=k}, computed by
    global-subset enumeration (`oracle_subset_curve`). `order` is ignored in
    this branch -- the oracle is a per-budget best SUBSET, not an order-driven
    prefix. Only the Oracle reference is scored this way. The label enters only
    through the subset argmax, never the scorer, so the curve is null-clean.
    """
    y, genes = cohort.y, cohort.genes
    if leq_k:
        return oracle_subset_curve(cohort, budgets)
    return {k: per_gene_auc(y, _score_from_evidence(cohort, _acquired_mask(order, k)), genes)
            for k in budgets}


def planning_efficiency(A: dict) -> float:
    return float(np.mean(list(A.values())))


def k_star(A: dict, tau: float = TAU):
    for k in sorted(A):
        if A[k] >= tau:
            return k
    return None


def _mean_prefix_overlap(order_a: np.ndarray, order_b: np.ndarray, K: int) -> np.ndarray:
    """Per-variant mean prefix-set overlap between two acquisition orders:
    mean over k=1..K of |first_k(a) ∩ first_k(b)| / k.

    This is the correct alignment measure for order-optimality because A(k)
    depends ONLY on the SET of the first k acquired channels -- the internal
    ordering within a prefix, and the ordering of positions beyond k, do not
    change A(k). Rank correlations (Spearman) instead weight every position
    equally and reward matching positions that never affect any A(k); for K=4
    they also collapse to a handful of discrete values. Prefix overlap is the
    set-agreement at each budget, exactly what the budget curve rewards. It is
    1.0 at k=K by construction (both prefixes contain all channels).
    """
    n = order_a.shape[0]
    out = np.zeros(n)
    for i in range(n):
        a, b = order_a[i], order_b[i]
        vals = [len(set(a[:k]) & set(b[:k])) / k for k in range(1, K + 1)]
        out[i] = float(np.mean(vals))
    return out


def order_optimality(cohort: Cohort, order: np.ndarray) -> dict:
    """
    Per-variant alignment of `order` vs the global relevance order and vs the
    per-variant oracle order. Reported two ways:

    - PRIMARY: mean prefix-set overlap (`po_vs_relevance`, `po_vs_oracle`) --
      the set agreement at each budget k, which is what A(k) actually depends
      on (see `_mean_prefix_overlap`).
    - SECONDARY (kept for continuity): mean Spearman rank correlation
      (`rho_vs_relevance`, `rho_vs_oracle`).

    A variant-adaptive planner aligns with the oracle; a fixed-relevance
    strategy aligns only with the relevance order (paper).
    """
    rel = relmax_order(cohort)
    ora = oracle_order(cohort)
    K = cohort.domain.K
    # PRIMARY: prefix-set overlap (drives A(k))
    po_rel = _mean_prefix_overlap(order, rel, K)
    po_ora = _mean_prefix_overlap(order, ora, K)
    # SECONDARY: Spearman rho (position-weighted; kept for continuity)
    def to_rank(perm):
        r = np.empty_like(perm)
        rows = np.arange(perm.shape[0])[:, None]
        r[rows, perm] = np.arange(K)
        return r
    ro, rr, rq = to_rank(order), to_rank(rel), to_rank(ora)
    rho_rel = np.array([spearmanr(ro[i], rr[i]).statistic for i in range(len(cohort))])
    rho_ora = np.array([spearmanr(ro[i], rq[i]).statistic for i in range(len(cohort))])
    # Chance-correct the prefix-overlap: a RANDOM order has expected mean
    # prefix-overlap (K+1)/(2K) (0.625 at K=4), not 0, so raw po sits in
    # [(K+1)/2K, 1] and is easy to misread next to the zero-centred Spearman
    # (random rho ~ 0). Report the normalized (po - chance)/(1 - chance), which
    # is 0 for a random order and 1 for a perfectly aligned one -- on the same
    # scale as rho.
    chance = (K + 1) / (2 * K)
    def _norm(p):
        return float((np.nanmean(p) - chance) / (1 - chance + 1e-12))
    return {"po_vs_relevance": float(np.nanmean(po_rel)),
            "po_vs_oracle": float(np.nanmean(po_ora)),
            "po_chance_baseline": float(chance),
            "po_vs_relevance_adj": _norm(po_rel),
            "po_vs_oracle_adj": _norm(po_ora),
            "po_rel_per_variant": po_rel, "po_ora_per_variant": po_ora,
            "rho_vs_relevance": float(np.nanmean(rho_rel)),
            "rho_vs_oracle": float(np.nanmean(rho_ora)),
            "rho_rel_per_variant": rho_rel, "rho_ora_per_variant": rho_ora}


def _closed_book_auc(cohort: Cohort, feature_cols, seed: int = 0,
                     n_splits: int = 20) -> float:
    """POOLED hold-out AUC of a logistic scorer on the given closed-book
    features, averaged over n_splits 70/30 splits.

    This is the correct memorization estimator: under a TRUE null (labels
    independent of the identity feature) it returns ~0.5, and it retains full
    power -- a feature that genuinely determines the label scores ~0.9+. (An
    earlier per-gene-MACRO variant was reverted: gene one-hot is constant within
    a gene, so per_gene_auc was structurally 0.5 for ANY labels -- zero power to
    detect memorization, the very thing this probe exists to catch.)

    Note on block structure: on data where between-gene base rates are preserved
    but within-gene labels are permuted, this can dip slightly BELOW 0.5 -- a
    finite-sample CV inversion (train-fold gene rate anti-correlates with the
    held-out rate), not an estimator bias. Such a permutation is not a pure null
    for a gene-identity feature; the pure null (labels independent of gene)
    correctly gives ~0.5.
    """
    y = cohort.y
    X = np.asarray(feature_cols, float)
    aucs = []
    for s in range(n_splits):
        rng = np.random.default_rng(seed + s)
        idx = rng.permutation(len(y))
        cut = int(0.7 * len(y))
        tr, te = idx[:cut], idx[cut:]
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = LogisticRegression(max_iter=500).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)) if aucs else 0.5


def masked_auc(cohort: Cohort, seed: int = 0) -> float:
    """
    Memorization control: closed-book scorer seeing only gene identity (one-hot),
    no evidence. Pooled hold-out AUC -> ~0.5 under a true null (labels
    independent of gene), ~0.9+ if identity determines the label (full power).
    """
    import pandas as pd
    Xc = pd.get_dummies(cohort.df["gene"]).to_numpy(float)
    return _closed_book_auc(cohort, Xc, seed)


def _auc_cv(X: np.ndarray, y: np.ndarray, seed: int = 0, n_splits: int = 20) -> float:
    """Mean 70/30 hold-out AUC of a logistic scorer on features X. Thin wrapper
    that DELEGATES to `_closed_book_auc` (the two were once duplicated
    line-for-line and their docstrings both demand they stay identical, so a
    single implementation now backs both), adapting the (X, y) signature to the
    (cohort-with-.y, X) one via a minimal shim object."""
    class _YShim:
        pass
    shim = _YShim()
    shim.y = np.asarray(y)
    return _closed_book_auc(shim, X, seed=seed, n_splits=n_splits)


def memorization_probe(cohort: Cohort, seed: int = 0) -> dict:
    """
    Two-mask closed-book memorization control (paper RQ3, Table tab:closed).

    Returns the full validity-control panel the paper reports:
      - coord_masked : closed-book AUC from the COORDINATE mask (gene identity
        only, no evidence, no consequence). -> 0.5 means the label is not
        recoverable from parametric memory of the locus.
      - full_hgvs    : closed-book AUC from the FULL-HGVS mask. The molecular
        consequence a full HGVS name exposes (missense vs LoF-bearing) is added
        as a feature via the cohort's `is_missense` column -- on real data this
        column IS the consequence parsed from the HGVS/VEP annotation; the
        synthetic cohort carries it directly (its variant_id has no consequence
        to parse). This is NOT recall: using the consequence is in-context ACMG
        reasoning worth the PVS1 category, not a memorized label.
      - pvs1_alone   : AUC from the PVS1 (LoF) channel alone. On REAL data the
        full-HGVS mask should approach this ceiling, because the HGVS name
        exposes the same LoF consequence PVS1 encodes. On the SYNTHETIC cohort
        the `is_missense` flag is decoupled from the PVS1 value by construction,
        so full_hgvs stays well below pvs1_alone here — the gap is expected on
        the stand-in and is not a defect; it closes only with real annotations.
      - full_acquisition : POOLED closed-book AUC of gene identity + all (non-
        masked) channel VALUES -- the same pooled unit as coord/full_hgvs, so it
        nests them (full_hgvs uses a subset of these features and cannot exceed
        it). This is the memorization control's ceiling, NOT the efficiency-axis
        A(K) (which is the per-gene-macro ranking AUC in results.csv).
      - G_acq        : acquisition-attributable gain = full_acquisition - full_hgvs
        (the paper's final axis-C performance quantity, +0.077 in the paper).
        Measured from the full-HGVS floor, NOT the coordinate-mask floor:
        full_hgvs is the deployment-realistic closed-book baseline (identity
        disclosed, no evidence acquired yet), so this is the gain acquisition
        adds on top of what the agent already has for free, not the gain over
        a near-chance identity-only control (that quantity is still available
        separately as full_acquisition - memory_floor, a validity-gate
        diagnostic, not the headline number).

    The coordinate/full-HGVS distinction is the paper's argument that a high
    full-HGVS AUC reflects consequence-reading, not memorized labels.
    """
    import pandas as pd
    y = cohort.y

    # --- coordinate mask: gene one-hot only ---
    # pooled closed-book AUC (~0.5 under a true null, full power if identity
    # determines the label). Every quantity in this probe uses the SAME pooled
    # unit so the G_acq subtraction is coherent.
    Xc = pd.get_dummies(cohort.df["gene"]).to_numpy(float)
    coord = _closed_book_auc(cohort, Xc, seed)

    # --- full-HGVS mask: gene one-hot + molecular consequence ---
    # The consequence a full HGVS name exposes is the LoF status (a
    # nonsense/frameshift/splice name is visibly PVS1-equivalent), so the
    # closed-book feature is the TRUE LoF flag (PVS1 == 1), NOT ~is_missense.
    # The earlier ~is_missense proxy pooled pathogenic LoF with benign
    # synonymous/intronic variants into one indicator, which broke the intended
    # "does the HGVS name alone reveal the PVS1 consequence" reading. No
    # acquired evidence VALUE is used -- only the categorical LoF status a
    # reader would infer from the variant name.
    if "PVS1" in cohort.domain.channels:
        cons = (cohort.value("PVS1").astype(float) == 1).astype(float)  # true LoF
    elif "is_missense" in cohort.df.columns:
        cons = (~cohort.df["is_missense"].to_numpy(bool)).astype(float)
    else:
        cons = np.zeros(len(y))
    Xh = np.column_stack([Xc, cons])
    full_hgvs = _closed_book_auc(cohort, Xh, seed)

    # --- PVS1-alone in-context ceiling ---
    pvs1_ch = "PVS1" if "PVS1" in cohort.domain.channels else cohort.domain.channels[0]
    pv = cohort.value(pvs1_ch).astype(float).reshape(-1, 1)
    pvs1_alone = _closed_book_auc(cohort, pv, seed)

    # --- full-acquisition ceiling, SAME UNIT as coord/full_hgvs ---
    # The probe is a self-contained validity control: every quantity in it is a
    # POOLED hold-out logistic AUC, so a G_acq subtraction between any two of
    # them is like-for-like. (A prior version used per_gene_auc here -- a
    # per-gene MACRO ranking AUC -- while coord/full_hgvs were pooled;
    # subtracting two different AUC units produced incoherent numbers, e.g.
    # full_hgvs > full_acquisition. The open-book performance ceiling A(K)
    # still lives in results.csv as the efficiency axis; this is the
    # memorization control, a different measurement.)
    # EXCLUDE axis-C masked channels (e.g. PRIOR_CLASS): they are label-derived,
    # so including them would let the control read the held-out label back.
    masked = set(getattr(cohort.domain, "masked_channels", ()) or ())
    chan_cols = [cohort.value(ch).astype(float) for ch in cohort.domain.channels
                 if ch not in masked]
    Xfull = np.column_stack([Xc] + chan_cols)
    full_acq = _closed_book_auc(cohort, Xfull, seed)

    # memory_floor: a validity-GATE diagnostic, not the G_acq baseline. It is
    # the closed-book coordinate mask, floored at CHANCE (0.5): on this
    # benchmark's fixed per-gene positive counts a held-out gene-identity
    # classifier can land slightly BELOW 0.5 (a finite-sample CV inversion of
    # the real per-gene base rates), and a sub-chance value would misread as
    # "memorization is working". Memorization cannot do worse than chance for
    # this control's purpose, so the floor is max(coord, 0.5). Kept in the
    # output for that gate check; G_acq itself is measured from full_hgvs
    # below, not from this floor.
    mem_floor = max(coord, 0.5)
    # G_acq = acquisition gain over the DEPLOYMENT-REALISTIC closed-book
    # baseline (full_hgvs, identity disclosed, no evidence yet acquired), not
    # over the near-chance identity-only floor above. An agent that is shown
    # the real variant name already has full_hgvs's worth of performance for
    # free; acquisition's attributable contribution is what it adds beyond
    # that, which is what a deployed system actually starts from.
    out = {"coord_masked": coord, "full_hgvs": full_hgvs,
           "pvs1_alone": pvs1_alone, "full_acquisition": full_acq,
           "memory_floor": float(mem_floor),
           "G_acq": float(full_acq - full_hgvs)}
    # Diagnostic that the control actually bites: how well the masked channel
    # ALONE predicts the label (its leakage), and what the ceiling would have
    # been had we NOT masked it. A large prior_leak with masking ON confirms the
    # exclusion removed a genuine leak rather than a no-op.
    if masked:
        leak_cols = [cohort.value(ch).astype(float) for ch in masked]
        Xleak = np.column_stack(leak_cols)
        out["prior_leak_auc"] = _auc_cv(Xleak, y, seed)
        # same pooled unit as full_acquisition, but WITH the masked channel(s)
        # added back -- so full_acq_if_unmasked > full_acquisition confirms the
        # mask removed a real leak rather than being a no-op.
        chan_all = [cohort.value(ch).astype(float) for ch in cohort.domain.channels]
        out["full_acq_if_unmasked"] = _closed_book_auc(
            cohort, np.column_stack([Xc] + chan_all), seed)
    return out


def gene_bootstrap_ci(cohort: Cohort, order: np.ndarray, k: int,
                      B: int = BOOTSTRAP_B, seed: int = 0, leq_k: bool = False):
    """Gene-clustered bootstrap CI for A(k).

    leq_k=True scores with the oracle's global best-subset argmax_{|pi|<=k}
    (same shared scorer, no per-variant label gate) so the CI matches the
    monotone oracle curve; default False = deployable first-k.

    CAVEAT (leq_k=True): the winning subset is fixed ONCE on the full cohort and
    held constant across resamples (for speed), so the interval is CONDITIONAL
    on that subset choice -- it does not propagate the variability of subset
    selection and is therefore systematically NARROWER than a fully
    re-optimized bootstrap. Read it as "CI of A(k) given the chosen subset", not
    "CI of the oracle including which subset it would pick".
    """
    rng = np.random.default_rng(seed)
    genes = np.unique(cohort.genes)
    # The full-cohort score at budget k does NOT depend on the resampled genes,
    # so compute it ONCE outside the B-loop and index it per resample (was
    # recomputed B*len(budgets)*len(strategies) times -- ~64k redundant full
    # scorings on the --with-ci run). For the oracle we pick the winning subset
    # ONCE on the full cohort and score that fixed subset for every resample --
    # the label enters only through that single global subset choice.
    if leq_k:
        _, subset = _best_subset_auc(cohort, k)
        mask = np.zeros((len(cohort), cohort.domain.K), dtype=bool)
        mask[:, list(subset)] = True
        sc_full = _score_from_evidence(cohort, mask)
    else:
        sc_full = _score_from_evidence(cohort, _acquired_mask(order, k))
    gene_rows = [np.where(cohort.genes == g)[0] for g in genes]
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(genes), size=len(genes))
        # Proper CLUSTER bootstrap: a gene drawn r times must count as r separate
        # clusters, not be silently re-merged by per_gene_auc (np.unique).
        # Distinct integer cluster ids per draw (rep*n_genes + gene) keep
        # duplicates separate -- else B resamples degenerate to "a random subset
        # of ~7.6 distinct genes" and the CI is systematically ~24% too narrow.
        rows = np.concatenate([gene_rows[g] for g in pick])
        cids = np.concatenate([np.full(len(gene_rows[g]), rep * len(genes) + g)
                               for rep, g in enumerate(pick)])
        vals.append(_macro_auc_by_cluster(cohort.y[rows], sc_full[rows], cids))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_contrast_ci(cohort: Cohort, order_a: np.ndarray, order_b: np.ndarray,
                       k: int, B: int = BOOTSTRAP_B, seed: int = 0) -> dict:
    """
    PAIRED gene-clustered bootstrap of the A(k) difference between two
    strategies (paper RQ2: agent vs RelMax, curve_contrast_ci). Both curves are
    recomputed on the SAME resampled genes each iteration, so the difference is
    paired and the CI reflects the contrast, not two independent CIs.

    Returns point difference (a-b), its 95% CI, a bootstrap two-sided p-value
    for H0: diff=0, and a standardized effect size (mean diff / bootstrap SD).
    'CI includes 0' => a statistical tie at budget k (the paper's claim for
    agent vs RelMax).
    """
    rng = np.random.default_rng(seed)
    genes = np.unique(cohort.genes)
    gene_rows = {g: np.where(cohort.genes == g)[0] for g in genes}  # precompute once
    sc_a_full = _score_from_evidence(cohort, _acquired_mask(order_a, k))
    sc_b_full = _score_from_evidence(cohort, _acquired_mask(order_b, k))
    diffs = []
    genes_list = list(genes)
    gene_rows_l = [gene_rows[g] for g in genes_list]
    for _ in range(B):
        pick = rng.integers(0, len(genes_list), size=len(genes_list))
        # cluster-bootstrap multiplicity: a gene drawn r times = r clusters
        # (distinct int id), else per_gene_auc re-merges duplicates -> CI too narrow.
        rows = np.concatenate([gene_rows_l[g] for g in pick])
        cids = np.concatenate([np.full(len(gene_rows_l[g]), rep * len(genes_list) + g)
                               for rep, g in enumerate(pick)])
        y = cohort.y[rows]
        a = _macro_auc_by_cluster(y, sc_a_full[rows], cids)
        b = _macro_auc_by_cluster(y, sc_b_full[rows], cids)
        diffs.append(a - b)
    diffs = np.asarray(diffs)
    point = float(per_gene_auc(cohort.y, sc_a_full, cohort.genes)
                  - per_gene_auc(cohort.y, sc_b_full, cohort.genes))
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    # Degenerate case: the two orders acquire the SAME evidence set at this k
    # (e.g. k=K, every strategy holds all channels), so the difference is
    # structurally zero. That is a tie with no evidence against H0, not a
    # significant result -> p=1.
    if np.allclose(sc_a_full, sc_b_full):
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "p_value": 1.0,
                "z_stat": 0.0, "includes_zero": True}
    # Two-sided bootstrap p with the (1+count)/(B+1) convention (North et al.
    # 2002): never returns an exact 0 -- the floor is 1/(B+1), the resolution
    # the bootstrap can actually support. `diffs` is the resampled distribution
    # of (a-b); count how many land on each side of 0.
    B_eff = len(diffs)
    c_le = int(np.sum(diffs <= 0))
    c_ge = int(np.sum(diffs >= 0))
    p = float(min(1.0, 2 * (1 + min(c_le, c_ge)) / (B_eff + 1)))
    sd = float(np.std(diffs) + 1e-12)
    return {"diff": point, "lo": lo, "hi": hi, "p_value": p,
            # point / sd(bootstrap diffs) is a Z-STATISTIC (how many bootstrap
            # SDs the point difference sits from 0), NOT Cohen's d -- named
            # accordingly so a value of 6-12 is not misread as a huge effect size.
            "z_stat": point / sd, "includes_zero": lo <= 0 <= hi}


def domain_stratified_gap(cohort: Cohort, k: int = 1) -> dict:
    """
    Per-disease-domain planning gap (paper RQ4, Table tab:domain). For each
    disease domain (cohort.df['domain']), recompute the oracle-vs-relmax A(k)
    gap on that domain's variants only.

    CAVEAT -- the SIGN is structural, not empirical. The oracle is a max over
    subsets that CONTAINS RelMax's budget-k subset, so gap >= 0 is an algebraic
    identity for every domain and carries no information by its sign; and the
    raw magnitude grows with 1/n under the null (max-over-subsets is a selection
    statistic), so a domain with fewer genes shows a larger raw gap for that
    reason alone. Do NOT read "uniform positive sign" or "magnitude tracks
    gene-specificity" as findings. The informative, cross-domain-comparable
    quantity is the EXCESS over a within-gene shuffled-label null; use
    consequence_stratified_gap (which reports gap_excess/z/p) for any claim, and
    treat this function's raw output as descriptive only.

    Returns domain -> {n, oracle, relmax, gap} at budget k.
    """
    rel = relmax_order(cohort)
    sc_rel = _score_from_evidence(cohort, _acquired_mask(rel, k))
    # Oracle = global best-subset argmax_{|pi|<=k} (null-clean, no per-variant
    # label gate), consistent with the headline curve. The winning subset is
    # chosen PER DOMAIN on that domain's rows.
    import itertools
    K = cohort.domain.K
    y, genes, dom = cohort.y, cohort.genes, cohort.df["domain"].to_numpy()
    out = {}
    for d in sorted(set(dom.tolist())):
        m = dom == d
        if len(np.unique(y[m])) < 2:
            continue
        best = -np.inf
        for size in range(1, k + 1):
            for subset in itertools.combinations(range(K), size):
                mask = np.zeros((len(cohort), K), dtype=bool)
                mask[:, list(subset)] = True
                sc = _score_from_evidence(cohort, mask)
                best = max(best, per_gene_auc(y[m], sc[m], genes[m]))
        a_ora = best
        a_rel = per_gene_auc(y[m], sc_rel[m], genes[m])
        out[d] = {"n": int(m.sum()), "oracle": a_ora, "relmax": a_rel,
                  "gap": float(a_ora - a_rel)}
    return out


def domain_stratified_gap_excess(cohort: Cohort, budgets=BUDGETS,
                                 n_perm: int = 200, seed: int = GLOBAL_SEED) -> dict:
    """
    NULL-CORRECTED analogue of domain_stratified_gap, same method as
    consequence_stratified_gap: PE (oracle vs RelMax, averaged over `budgets`)
    per disease domain, plus its excess over a within-gene shuffled-label
    permutation null.

    WHY THIS EXISTS. domain_stratified_gap's own docstring warns that its raw
    gap is NOT cross-domain-comparable: "the raw magnitude grows with 1/n
    under the null (max-over-subsets is a selection statistic), so a domain
    with fewer genes shows a larger raw gap for that reason alone." On this
    cohort that caveat is not hypothetical -- lipid_disorder (n=69, the
    smallest domain) has the largest raw gap (0.195) and hereditary_cancer
    (n=254, the largest domain) has the smallest (0.000), which is exactly the
    pattern the artifact predicts. gap_excess (gap minus its own null's mean)
    and z divide that artifact out, the same way consequence_stratified_gap
    already does for the missense/lof/other split -- see that function's
    docstring for the full argument.

    UNLIKE consequence_stratified_gap, domain does not restrict which ACMG
    channels apply (that is a per-variant molecular-consequence property, not
    a per-domain one), so every domain is scored on the full channel set --
    no APPLICABLE-channel filtering, no reduced Cohort/Domain view.

    Returns domain -> {n, n_pos, oracle_pe, relmax_pe, gap, per_k_gap,
                       null_mean, null_sd, gap_excess, z, p_value}.
    """
    import hashlib
    dc = cohort.domain.channels
    bud = [b for b in budgets if b <= len(dc)] or [len(dc)]
    dom = cohort.df["domain"].to_numpy()
    out = {}
    for d in sorted(set(dom.tolist())):
        m = dom == d
        if m.sum() < 2:
            continue
        dsub = cohort.df[m].reset_index(drop=True)
        view = Cohort(dsub, cohort.domain)
        if len(np.unique(view.y)) < 2:
            continue

        def _pe_perk(v, bud=bud):
            rel = relmax_order(v)
            rel_k = np.array([per_gene_auc(
                v.y, _score_from_evidence(v, _acquired_mask(rel, k)), v.genes)
                for k in bud])
            ora_k = np.array([_best_subset_auc(v, k)[0] for k in bud])
            return ora_k, rel_k

        ora_k, rel_k = _pe_perk(view)
        ora_pe, rel_pe = float(ora_k.mean()), float(rel_k.mean())
        gap = ora_pe - rel_pe
        per_k_gap = [float(o - r) for o, r in zip(ora_k, rel_k)]

        genes = dsub["gene"].to_numpy()
        gene_idx = {g: np.where(genes == g)[0] for g in np.unique(genes)}
        offset = int(hashlib.sha256(d.encode()).hexdigest(), 16) % (2 ** 31)
        rng = np.random.default_rng(seed + offset)
        y0 = dsub["label"].to_numpy()
        nulls = np.empty(n_perm)
        for i in range(n_perm):
            yp = y0.copy()
            for g, idx in gene_idx.items():
                yp[idx] = rng.permutation(y0[idx])
            dperm = dsub.copy(); dperm["label"] = yp
            o, r = _pe_perk(Cohort(dperm, cohort.domain))
            nulls[i] = float(o.mean() - r.mean())
        p = float((1 + (nulls >= gap).sum()) / (n_perm + 1))     # (1+c)/(B+1)
        nm, nsd = float(nulls.mean()), float(nulls.std())
        out[d] = {
            "n": int(m.sum()), "n_pos": int(view.y.sum()),
            "oracle_pe": ora_pe, "relmax_pe": rel_pe, "gap": float(gap),
            "per_k_gap": per_k_gap,
            "null_mean": nm, "null_sd": nsd,
            "gap_excess": float(gap - nm),
            "z": float((gap - nm) / nsd) if nsd > 1e-12 else float("nan"),
            "p_value": p,
        }
    return out


def consequence_stratified_gap(cohort: Cohort, budgets=BUDGETS,
                               n_perm: int = 200, seed: int = GLOBAL_SEED) -> dict:
    """
    Planning gap stratified by molecular consequence, on the FULL cohort (no
    subset selection -- every variant is scored, then grouped by the biological
    consequence annotation `is_missense`). This answers "where does acquisition
    planning have value?" without any label- or evidence-based filtering that a
    reviewer could call selection bias: the split is `is_missense`, a fixed
    property of the variant known at deployment.

    STRATA (three, not two). The split is by molecular consequence, a fixed
    deployment-time property. Crucially we separate TRUE LoF from other
    non-missense variants, because PVS1 is a dead channel (always 0) for the
    latter and folding them together silently reintroduces the "RelMax wastes
    its first pick on a zero channel" artifact through the back door:
      - lof      : PVS1 == 1 (real loss-of-function)      -> channels PVS1, PM2
      - missense : is_missense                            -> channels PM2, PP3, PM1
      - other    : non-missense, non-LoF (synonymous /     -> channels PM2
                   intronic / inframe; PVS1 inapplicable)
    Each stratum is scored ONLY over the ACMG categories applicable to it, so
    the gap cannot be inflated by an inapplicable ("dead") channel. Channel
    order is taken from cohort.domain.channels (not hardcoded), so RelMax stays
    the domain's declared relevance order even if spec.CHANNELS is reordered.

    GAP IS STRUCTURALLY >= 0. The oracle enumerates every size<=k subset and
    takes the max; RelMax's budget-k acquisition is one such subset, so
    A_oracle(k) >= A_relmax(k) is an algebraic identity carrying no information
    by its SIGN. What is informative is the EXCESS over a shuffled-label null:
    we report gap_excess = gap - null_mean and z = (gap - null_mean)/null_sd,
    and these -- not the raw gap -- are the cross-stratum-comparable quantities
    (raw gap grows with 1/n under the null because max-over-subsets is a
    selection statistic). The permutation shuffles labels WITHIN gene (so the
    per-gene base rates that the macro-AUC statistic conditions on are
    preserved) and uses the (1+c)/(B+1) p-value convention (never exactly 0).

    PER-k GAPS. PE averages A(k) over a per-stratum budget grid, and the top
    budget k=len(channels) is a subset-selection point (RelMax already holds
    every channel, so the oracle's edge there is "drop a harmful channel using
    the label", not acquisition planning). We therefore also return the per-k
    gap vector so that point can be read separately from the planning budgets.

    SCOPE / VUS BOUNDARY. Every stratum here is scored only on variants that
    carry a gold Pathogenic/Benign label (the oracle and AUC require one).
    Variants of uncertain significance (VUS) -- clinically the evidence-sparse
    missense where acquisition planning is expected to matter most -- have no
    gold label and therefore CANNOT be scored by this function. Measuring
    planning on VUS requires a delayed gold standard (variants once submitted as
    VUS and later reclassified P/B in ClinVar); that reclassified cohort is
    future work, not something this function computes. Do not read the missense
    stratum here as "planning on VUS": it is planning on labeled missense, from
    which the VUS case is an extrapolation, not a measurement.

    Returns consequence -> {n, n_pos, channels, oracle_pe, relmax_pe, gap,
                            per_k_gap, null_mean, null_sd, gap_excess, z,
                            p_value}.
    """
    import dataclasses
    dc = cohort.domain.channels
    APPLICABLE = {
        "lof":      [c for c in dc if c in ("PVS1", "PM2")],
        "missense": [c for c in dc if c in ("PM2", "PP3", "PM1")],
        "other":    [c for c in dc if c in ("PM2",)],
    }
    is_mis = cohort.df["is_missense"].to_numpy(bool) if "is_missense" in cohort.df \
        else np.zeros(len(cohort), bool)
    pvs1 = cohort.value("PVS1").astype(float) if "PVS1" in dc else np.zeros(len(cohort))
    is_lof = (pvs1 == 1) & ~is_mis
    strata = {"lof": is_lof, "missense": is_mis, "other": (~is_mis) & ~is_lof}
    out = {}
    for name, mask in strata.items():
        keep = APPLICABLE[name]
        if mask.sum() < 2 or not keep:
            continue
        dsub = cohort.df[mask].reset_index(drop=True)
        dom2 = dataclasses.replace(
            cohort.domain, channels=keep,
            channel_defined_on={c: cohort.domain.channel_defined_on[c] for c in keep})
        view = Cohort(dsub, dom2)
        if len(np.unique(view.y)) < 2:
            continue
        bud = [b for b in budgets if b <= len(keep)] or [len(keep)]

        # `bud` is bound as a default so the closure captures THIS stratum's
        # budgets by value. It is currently only ever called inside the same
        # iteration, so late binding would give the same answer today -- but the
        # permutation loop below calls it n_perm times, and moving that loop out
        # would silently score every stratum against the last one's budgets.
        def _pe_perk(v, bud=bud):
            rel = relmax_order(v)
            rel_k = np.array([per_gene_auc(
                v.y, _score_from_evidence(v, _acquired_mask(rel, k)), v.genes) for k in bud])
            ora_k = np.array([_best_subset_auc(v, k)[0] for k in bud])
            return ora_k, rel_k

        ora_k, rel_k = _pe_perk(view)
        ora_pe, rel_pe = float(ora_k.mean()), float(rel_k.mean())
        gap = ora_pe - rel_pe
        per_k_gap = [float(o - r) for o, r in zip(ora_k, rel_k)]
        # WITHIN-GENE label permutation, independent seed per stratum so the
        # result does not depend on dict traversal order.
        genes = dsub["gene"].to_numpy()
        gene_idx = {g: np.where(genes == g)[0] for g in np.unique(genes)}
        # Deterministic per-stratum offset (built-in hash() of a str is
        # per-process randomized under PYTHONHASHSEED -> not reproducible).
        import hashlib
        offset = int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2 ** 31)
        rng = np.random.default_rng(seed + offset)
        y0 = dsub["label"].to_numpy()
        nulls = np.empty(n_perm)
        for i in range(n_perm):
            yp = y0.copy()
            for g, idx in gene_idx.items():
                yp[idx] = rng.permutation(y0[idx])
            dperm = dsub.copy(); dperm["label"] = yp
            o, r = _pe_perk(Cohort(dperm, dom2))
            nulls[i] = float(o.mean() - r.mean())
        p = float((1 + (nulls >= gap).sum()) / (n_perm + 1))     # (1+c)/(B+1)
        nm, nsd = float(nulls.mean()), float(nulls.std())
        out[name] = {
            "n": int(mask.sum()), "n_pos": int(view.y.sum()), "channels": keep,
            "oracle_pe": ora_pe, "relmax_pe": rel_pe, "gap": float(gap),
            "per_k_gap": per_k_gap,
            "null_mean": nm, "null_sd": nsd,
            "gap_excess": float(gap - nm),
            "z": float((gap - nm) / nsd) if nsd > 1e-12 else float("nan"),
            "p_value": p,
        }
    return out


def channel_complementarity(cohort: Cohort, restrict_missense: bool = False) -> dict:
    """
    Pairwise channel COMPLEMENTARITY diagnostic: does acquiring a second
    evidence channel add information the first does not already carry? This is
    the precondition for per-instance acquisition planning to have any value --
    if two channels are redundant (highly rank-correlated and adding one to the
    other lifts AUC by ~0), then a single fixed channel saturates and there is
    nothing to plan.

    For every unordered channel pair (A, B), on variants where BOTH are defined
    and both label classes are present, we report:
      - rho          : per-gene-averaged Spearman rank correlation of the two
                       channel values (averaged over genes so a between-gene
                       correlation cannot masquerade as within-gene redundancy,
                       Simpson-style). |rho| near 1 = redundant; near 0 =
                       orthogonal. None if a channel is constant on the stratum.
      - auc_a, auc_b : per-gene single-channel AUC of A and B alone.
      - auc_pair     : per-gene AUC of the summed (A+B) evidence score.
      - lift         : auc_pair - max(auc_a, auc_b).
      - n            : variants entering the comparison.
      - n_genes_used : genes with both classes present (the ones per_gene_auc
                       actually averages over; n counts all selected variants).

    INTERPRETATION IS ONE-DIRECTIONAL. The pair combiner is an equal-weight
    sum, so a strong channel diluted by a weak one gives NEGATIVE lift; lift<=0
    therefore does NOT imply redundancy (it can just mean one channel is weak).
    Only POSITIVE lift is evidence -- of genuine complementarity (the pair
    beats either alone). Use lift<=0 together with low single-channel AUC to
    read "weak channel", not "redundant channel". This is a label-conditioned
    DIAGNOSTIC that characterizes the evidence pool; it is not a strategy score
    and (unlike the gap axes) carries no permutation null.

    Set restrict_missense=True to run on the missense stratum, where the
    in-silico (PP3/AlphaMissense) vs functional-domain (PM1) and rarity (PM2)
    channels are the clinically relevant, non-saturated comparison. NOTE: with
    restrict_missense=False, different pairs are evaluated on different
    populations (a pair is scored wherever BOTH its channels are defined), so
    lift is not comparable row-to-row across pairs -- read each row on its own
    population (reported as n / n_genes_used).
    """
    import itertools
    df = cohort.df
    if restrict_missense and "is_missense" in df.columns:
        m = df["is_missense"].to_numpy(bool)
    else:
        m = np.ones(len(cohort), bool)
    y_all, genes_all = cohort.y, cohort.genes
    chs = cohort.domain.channels

    def _per_gene_rho(a, b, g):
        vals = []
        for gg in np.unique(g):
            idx = g == gg
            if idx.sum() < 3 or np.unique(a[idx]).size < 2 or np.unique(b[idx]).size < 2:
                continue
            vals.append(spearmanr(a[idx], b[idx]).statistic)
        return float(np.mean(vals)) if vals else None

    out = {}
    for A, B in itertools.combinations(chs, 2):
        va, vb = cohort.value(A).astype(float), cohort.value(B).astype(float)
        da = df[f"{A}__defined"].to_numpy(bool) if f"{A}__defined" in df else np.ones(len(cohort), bool)
        db = df[f"{B}__defined"].to_numpy(bool) if f"{B}__defined" in df else np.ones(len(cohort), bool)
        sel = m & da & db
        if sel.sum() < 10 or len(np.unique(y_all[sel])) < 2:
            continue
        ya, ga = y_all[sel], genes_all[sel]
        a, b = va[sel], vb[sel]
        def norm(x):
            lo, hi = x.min(), x.max()
            return (x - lo) / (hi - lo + 1e-12)
        na, nb = norm(a), norm(b)
        auc_a = per_gene_auc(ya, na, ga)
        auc_b = per_gene_auc(ya, nb, ga)
        auc_pair = per_gene_auc(ya, na + nb, ga)
        rho = _per_gene_rho(a, b, ga)
        n_genes_used = int(sum(len(np.unique(ya[ga == gg])) == 2 for gg in np.unique(ga)))
        out[f"{A}|{B}"] = {
            "n": int(sel.sum()), "n_genes_used": n_genes_used, "rho": rho,
            "auc_a": float(auc_a), "auc_b": float(auc_b),
            "auc_pair": float(auc_pair),
            "lift": float(auc_pair - max(auc_a, auc_b)),
        }
    return out


def evidence_pool_ablation(cohort: Cohort, drop: list, k: int = 1) -> dict:
    """
    Evidence-pool ablation (paper RQ4): recompute the oracle-vs-relmax A(k)
    gap after REMOVING one or more channels from the acquirable pool, e.g.
    dropping PP3 (AlphaMissense) to check the effect is not carried solely by
    the in-silico channel.

    CAVEAT -- as in domain_stratified_gap, gap >= 0 here is structural (the
    oracle's subset family contains RelMax's), so a "gap survives dropping PP3"
    statement is guaranteed by construction and is NOT evidence on its own. Only
    the excess over a shuffled-label null is informative; interpret the reduced
    pool's gap against that baseline, not by its sign.

    Returns {kept_channels, oracle, relmax, gap} at budget k over the reduced
    pool. Implemented by building a reduced-channel Domain/Cohort view.
    """
    import dataclasses
    keep = [c for c in cohort.domain.channels if c not in set(drop)]
    if not keep:
        raise ValueError("cannot drop all channels")
    dom2 = dataclasses.replace(
        cohort.domain, channels=keep,
        channel_defined_on={c: cohort.domain.channel_defined_on[c] for c in keep})
    view = Cohort(cohort.df.copy(), dom2)
    kk = min(k, len(keep))
    rel = relmax_order(view)
    a_rel = per_gene_auc(view.y, _score_from_evidence(view, _acquired_mask(rel, kk)), view.genes)
    # Oracle = global best-subset argmax over the REDUCED pool (null-clean).
    a_ora = _best_subset_auc(view, kk)[0]
    return {"kept_channels": keep, "oracle": a_ora, "relmax": a_rel,
            "gap": float(a_ora - a_rel), "k": kk}


def clinical_yield_curve(cohort: Cohort, order: np.ndarray, budgets=BUDGETS,
                         confidence: float = 0.90, seed: int = GLOBAL_SEED,
                         n_splits: int = 5) -> dict:
    """
    CLINICAL-ENDPOINT view of the acquisition curve. Instead of a ranking AUC,
    this reports what a diagnostic lab actually cares about at each acquisition
    budget k: how many variants can be CALLED (crossed a decision threshold to
    Pathogenic or Benign) versus left as a VUS, and whether those calls are
    correct.

    Decision rule (three-way, abstaining) -- CALIBRATED-POSTERIOR variant. This
    is the data-driven companion to the guideline-native `acmg_points_curve`
    (which applies the ACMG/ClinGen Bayesian point rule directly). Here, at
    budget k, the acquired evidence is turned into a calibrated posterior
    P(pathogenic) by an out-of-fold logistic model fit on the SAME budget-k
    evidence (label-stratified n_splits k-fold, so the posterior for a variant
    is never fit on that variant). A variant is:
      - called Pathogenic if posterior >= confidence,
      - called Benign     if posterior <= 1 - confidence,
      - left VUS          otherwise (abstain).
    `confidence` is a FIXED clinician-set operating point (default 0.90), NOT
    tuned to the data -- we report whatever accuracy that operating point yields.

    Per budget k we return:
      - resolved_frac : fraction of variants called (1 - VUS rate) = the lab's
                        diagnostic yield;
      - vus_frac      : fraction left uncertain;
      - call_accuracy : accuracy AMONG called variants (the safety number -- a
                        lab tolerates abstaining but not confident wrong calls);
      - balanced_call_accuracy : class-balanced version (guards against a
                        majority-class shortcut when calls are imbalanced);
      - n_called.
    This makes the efficiency/stopping axis clinically legible: "stopping at
    k=k* resolves R% of variants at A% call accuracy, and acquiring further
    changes yield by ..." -- a statement in the lab's own KPIs, not in AUC.

    CAVEATS. (1) `confidence` is an external operating point; the companion
    `clinical_yield_sensitivity` sweeps it so no single cutoff drives the
    conclusion. (2) Only gold-labeled P/B variants are scored (true VUS have no
    label -- see consequence_stratified_gap's VUS boundary note); "VUS" here
    means "this evidence budget is insufficient to call a variant that DOES have
    a gold answer", i.e. an under-acquisition measure, not a real-world VUS rate.
    """
    from sklearn.model_selection import StratifiedKFold
    y = cohort.y.astype(int)
    out = {}
    for k in budgets:
        mask = _acquired_mask(order, k)
        # feature = the acquired, oriented evidence values per channel (the same
        # evidence the scorer sees); undefined/unacquired slots are the neutral
        # value, so the model simply sees no signal there.
        X = _evidence_matrix_fallback(cohort, mask)
        post = np.full(len(y), 0.5)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr, te in skf.split(X, y):
            if len(np.unique(y[tr])) < 2:
                continue
            # Standardize on the TRAIN fold only. Channels live on very
            # different scales (PP3 in [0,1] vs PM2 = -log10 AF in ~[0,8]), so
            # the fit needs scaling to be stable -- but computing mean/sd over
            # the whole cohort lets the held-out rows leak their scale into the
            # model, which makes `post` no longer strictly out-of-fold. A
            # constant column keeps sd=1 so it centers to zero and drops out.
            mu = X[tr].mean(axis=0)
            sd = X[tr].std(axis=0)
            sd = np.where(sd > 1e-9, sd, 1.0)
            clf = LogisticRegression(max_iter=1000).fit((X[tr] - mu) / sd, y[tr])
            post[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
        call = np.full(len(y), -1)              # -1 = VUS/abstain
        call[post >= confidence] = 1
        call[post <= 1 - confidence] = 0
        called = call >= 0
        n_called = int(called.sum())
        if n_called:
            acc = float((call[called] == y[called]).mean())
            # balanced over the called subset's true classes
            accs = [float((call[called & (y == c)] == c).mean())
                    for c in (0, 1) if (called & (y == c)).sum() > 0]
            bacc = float(np.mean(accs)) if accs else float("nan")
        else:
            acc = bacc = float("nan")
        out[k] = {
            "resolved_frac": float(called.mean()),
            "vus_frac": float((~called).mean()),
            "call_accuracy": acc,
            "balanced_call_accuracy": bacc,
            "n_called": n_called,
            "call": call,   # per-variant: -1 VUS/abstain, 0 Benign, 1 Pathogenic
                            # (row order matches cohort.df) -- lets a caller do
                            # a paired significance test across strategies
                            # (see paired_resolved_test) without refitting.
        }
    return out


def _evidence_matrix_fallback(cohort: Cohort, mask: np.ndarray) -> np.ndarray:
    """Per-variant matrix of acquired channel values (neutral where not
    acquired/undefined). Mirrors the scorer's view but keeps channels separate
    (the scorer sums them) so the logistic model can weight them.

    Returns RAW values: standardization is the caller's job and must be done
    FOLD-WISE (see clinical_yield_curve), otherwise the held-out rows leak their
    scale into the fit."""
    cols = []
    for j, ch in enumerate(cohort.domain.channels):
        v = cohort.value(ch).astype(float).copy()
        v[~mask[:, j]] = cohort.domain.neutral_for(ch)   # per-channel neutral
        cols.append(v)
    return np.column_stack(cols)


def _acmg_channel_points(cohort: Cohort) -> dict:
    """Per-channel ACMG point contribution for every variant, as an
    {channel_name: (n,) float array} map.

    Factored out of `acmg_points_curve` so the fixed-budget curve and the
    agent-chosen-stop endpoint (`acmg_points_at_stop`) score through ONE copy of
    the guideline mapping -- two copies would silently drift apart the first time
    a threshold is revised. The mapping and thresholds are unchanged; see
    `acmg_points_curve`'s docstring for the primary-source citations.
    """
    def channel_points(ch, v, defined):
        p = np.zeros(len(v))
        if not defined.any():
            return p
        if ch == "PVS1":
            p[defined & (v >= 0.5)] = 8.0
        elif ch == "PM1":
            # functional-domain membership: strength encoded 0.2/0.6/1.0
            p[defined & (v >= _ACMG_PM1_DOMAIN_MIN)] = 2.0
        elif ch == "PM2":
            # rarity score (higher = rarer). Rare -> PM2_Supporting (+1);
            # common -> BS1 benign (-1). Neutral band contributes 0.
            p[defined & (v >= _ACMG_PM2_RARE_MIN)] = 1.0
            p[defined & (v <= _ACMG_PM2_COMMON_MAX)] = -1.0
        elif ch == "PP3":
            # AlphaMissense in [0,1]; ClinGen/Pejaver 2022 PP3/BP4 bins.
            d = defined
            p[d & (v >= _ACMG_PP3_STRONG)] = 4.0
            p[d & (v >= _ACMG_PP3_MOD) & (v < _ACMG_PP3_STRONG)] = 2.0
            p[d & (v >= _ACMG_PP3_SUP) & (v < _ACMG_PP3_MOD)] = 1.0
            p[d & (v <= _ACMG_BP4_SUP)] = -1.0
        return p

    chpts = {}
    for ch in list(cohort.domain.channels):
        v = cohort.value(ch).astype(float)
        dmask = cohort.df[f"{ch}__defined"].to_numpy(bool) \
            if f"{ch}__defined" in cohort.df.columns else np.ones(len(v), bool)
        chpts[ch] = channel_points(ch, v, dmask)
    return chpts


def acmg_points_at_stop(cohort: Cohort, order: np.ndarray, stop_k: np.ndarray,
                        path_min: float | None = None,
                        ben_max: float | None = None) -> dict:
    """
    Clinical endpoint under an AGENT-CHOSEN stopping rule: each variant is scored
    on the evidence it acquired up to ITS OWN stop point `stop_k[i]`, not a
    cohort-wide budget k.

    WHY THIS RULE AND NOT `confidence_based_stopping`
    -------------------------------------------------
    `confidence_based_stopping` is a retrospective DIAGNOSTIC with two documented
    limits (see its docstring): its per-budget min-max normalization is taken
    over the whole cohort, so (a) `auc_at_stop` is not comparable across budgets
    and can exceed the full-acquisition value as a pure scale artifact, and (b)
    the rule is transductive -- one variant's stop depends on the other variants'
    scores, so it cannot be evaluated at deployment time on a single case.

    The ACMG point rule has neither problem. Points are summed on an ABSOLUTE,
    guideline-fixed scale and the call thresholds (>= path_min -> LP/P,
    <= ben_max -> LB/B, else VUS) are constants, so a variant stopped at k=1 and
    one stopped at k=4 are classified by the same ruler, and each variant's call
    depends only on its own evidence. That makes this the endpoint an
    agent-chosen stop can actually be scored on.

    `stop_k[i]` is the number of channels variant i acquired (1..K). Returns
    resolved_frac / vus_frac / call_accuracy / balanced_call_accuracy / n_called
    -- the same fields as `acmg_points_curve` so a stopping row drops straight
    into the fixed-budget table -- plus `mean_queries` (the query-saving story)
    and the per-variant `call` and `stop_k` vectors.
    """
    y = cohort.y.astype(int)
    chans = list(cohort.domain.channels)
    K = cohort.domain.K
    path_min = _ACMG_PATH_MIN if path_min is None else float(path_min)
    ben_max = _ACMG_BEN_MAX if ben_max is None else float(ben_max)

    stop_k = np.asarray(stop_k, int)
    if stop_k.shape != (len(y),):
        raise ValueError(f"stop_k must be ({len(y)},), got {stop_k.shape}")
    if stop_k.min() < 1 or stop_k.max() > K:
        raise ValueError(f"stop_k entries must lie in 1..{K}, got "
                         f"{stop_k.min()}..{stop_k.max()}")

    chpts = _acmg_channel_points(cohort)
    total = np.zeros(len(y))
    for i in range(len(y)):
        for j in order[i, :stop_k[i]]:
            total[i] += chpts[chans[j]][i]

    call = np.full(len(y), -1)
    call[total >= path_min] = 1
    call[total <= ben_max] = 0
    called = call >= 0
    nc = int(called.sum())
    if nc:
        acc = float((call[called] == y[called]).mean())
        accs = [float((call[called & (y == c)] == c).mean())
                for c in (0, 1) if (called & (y == c)).sum() > 0]
        bacc = float(np.mean(accs)) if accs else float("nan")
    else:
        acc = bacc = float("nan")
    return {"resolved_frac": float(called.mean()),
            "vus_frac": float((~called).mean()),
            "call_accuracy": acc, "balanced_call_accuracy": bacc,
            "n_called": nc, "mean_queries": float(stop_k.mean()),
            "call": call, "stop_k": stop_k}


def acmg_points_curve(cohort: Cohort, order: np.ndarray, budgets=BUDGETS,
                      path_min: float | None = None,
                      ben_max: float | None = None) -> dict:
    """
    GUIDELINE-NATIVE clinical-endpoint view. Unlike `clinical_yield_curve`
    (which fits a logistic posterior), this applies the ACMG/ClinGen Bayesian
    POINT system directly as the decision rule -- the criterion strengths and
    classification thresholds come from the guideline, not from the data.

    Point scale (Tavtigian et al. 2020, Hum Mutat 41:1734, PMC8011844, Table of
      "Evidence Point Scale"; verified against primary source):
      Very Strong = +/-8, Strong = +/-4, Moderate = +/-2, Supporting = +/-1.
      Classification thresholds (same paper): Pathogenic >= 10 points, Likely
      Pathogenic >= 6 points (LP-inclusive call uses >=6; strict P uses >=10).

    Per-channel point contribution when ACQUIRED (guideline-sourced thresholds;
    the channel value is the evidence, the mapping is fixed a-priori, NOT tuned):
      PVS1 (LoF, Very Strong)          : value==1  -> +8         (else 0)
      PM2  (rarity, Supporting*)       : rarity high-> +1         (ClinGen 2020
                                         downgraded PM2 to Supporting)
      PP3  (AlphaMissense, calibrated) : ClinGen/Pejaver 2022 PP3/BP4 bins on
                                         the in-silico score -> +4/+2/+1 on the
                                         pathogenic side, -1 (BP4) on the benign
      PM1  (domain/hotspot, Moderate)  : in functional domain -> +2 (else 0)
    Thresholds (`_ACMG_*` module constants) are documented and swept by
    `acmg_points_sensitivity`; they are the guideline's, so no per-dataset fit.

    Classification from summed points (Tavtigian et al. 2020 Table 3 ranges,
    verified: P >=10, LP 6-9, VUS 0-5, LB -1 to -6, B <=-7):
      >= 6  -> pathogenic side (LP/P);  <= -1 -> benign side (LB/B);  else VUS.
    (>=6 rather than >=10 so that Likely Pathogenic counts as a clinical call;
    the pure-P >=10 cut is available via `acmg_points_sensitivity`.)

    Per budget k returns resolved_frac / vus_frac / call_accuracy /
    balanced_call_accuracy / n_called, matching clinical_yield_curve so the two
    decision rules (learned vs guideline) can be compared head-to-head.

    `path_min` / `ben_max` override the two combining cutoffs (default: the
    module constants `_ACMG_PATH_MIN` / `_ACMG_BEN_MAX`). They are explicit
    PARAMETERS rather than module state so `acmg_points_sensitivity` can sweep
    the threshold without mutating a global -- a global swap is not reentrant and
    is unsafe if two sweeps ever run concurrently.

    CAVEAT. On this cohort PP3/PM1 are defined only for the 80 missense variants
    and PM2 is a rarity proxy; the point mapping is therefore exercised mainly on
    PVS1 (LoF) and the missense channels. The thresholds are guideline values,
    but their applicability to these proxy channel encodings is an assumption we
    flag, not hide.
    """
    chans = list(cohort.domain.channels)
    y = cohort.y.astype(int)
    # resolved at CALL time (the constants are defined below this function)
    path_min = _ACMG_PATH_MIN if path_min is None else float(path_min)
    ben_max = _ACMG_BEN_MAX if ben_max is None else float(ben_max)

    chpts = _acmg_channel_points(cohort)

    out = {}
    for k in budgets:
        total = np.zeros(len(y))
        for i in range(len(y)):
            for j in order[i, :k]:
                total[i] += chpts[chans[j]][i]
        call = np.full(len(y), -1)
        call[total >= path_min] = 1
        call[total <= ben_max] = 0
        called = call >= 0
        nc = int(called.sum())
        if nc:
            acc = float((call[called] == y[called]).mean())
            accs = [float((call[called & (y == c)] == c).mean())
                    for c in (0, 1) if (called & (y == c)).sum() > 0]
            bacc = float(np.mean(accs)) if accs else float("nan")
        else:
            acc = bacc = float("nan")
        out[k] = {"resolved_frac": float(called.mean()),
                  "vus_frac": float((~called).mean()),
                  "call_accuracy": acc, "balanced_call_accuracy": bacc,
                  "n_called": nc,
                  "call": call}   # see clinical_yield_curve's `call` field
    return out


def paired_resolved_test(call_a: np.ndarray, call_b: np.ndarray) -> dict:
    """
    Paired significance test of whether two strategies differ in DIAGNOSTIC
    YIELD (resolved vs VUS) on the SAME variants at a fixed budget k. Inputs
    are the per-variant `call` arrays returned by `clinical_yield_curve` /
    `acmg_points_curve` (-1 = VUS/abstain, else called), aligned by cohort row
    order -- so this only makes sense comparing two calls computed on the same
    cohort (and, for clinical_yield_curve, the same confidence/seed/n_splits).

    b = #variants resolved by A but not B, c = #variants resolved by B but not
    A (a standard McNemar construction for paired binary outcomes). Two-sided
    exact binomial test on (b, c). b==c==0 (no discordant pairs) -> p=1.0,
    which means "nothing to test", not "confirmed no difference".
    """
    from scipy.stats import binomtest
    resolved_a = np.asarray(call_a) != -1
    resolved_b = np.asarray(call_b) != -1
    b = int(np.sum(resolved_a & ~resolved_b))
    c = int(np.sum(~resolved_a & resolved_b))
    n = b + c
    p = float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue) if n > 0 else 1.0
    return {"b_resolved_only_by_a": b, "c_resolved_only_by_b": c,
            "n_discordant": n, "p_value": p}


def clinical_yield_by_domain(cohort: Cohort, order: np.ndarray, k: int,
                             confidence: float = 0.90, seed: int = GLOBAL_SEED,
                             n_splits: int = 5) -> dict:
    """
    Diagnostic yield (see clinical_yield_curve) for a SPECIFIC agent/strategy
    order, broken down by disease domain. Answers "does this agent's real
    yield differ by disease domain", as distinct from domain_stratified_gap's
    Oracle-vs-RelMax structural-ceiling question (which is not agent-specific
    and whose sign is a structural identity, not a finding -- see its
    docstring). The calibrated-posterior classifier is fit ONCE on the full
    cohort (not re-fit per domain on a small subsample, which would be noisy);
    only the resulting per-variant calls are then split by domain.

    Returns domain -> {n, resolved_frac, call_accuracy, n_called}.
    """
    full = clinical_yield_curve(cohort, order, budgets=[k], confidence=confidence,
                                seed=seed, n_splits=n_splits)[k]
    call = full["call"]
    y = cohort.y.astype(int)
    dom = cohort.df["domain"].to_numpy()
    out = {}
    for d in sorted(set(dom.tolist())):
        m = dom == d
        c = call[m]
        called = c >= 0
        n_called = int(called.sum())
        acc = float((c[called] == y[m][called]).mean()) if n_called else float("nan")
        out[d] = {"n": int(m.sum()), "resolved_frac": float(called.mean()),
                  "call_accuracy": acc, "n_called": n_called}
    return out


# --- ACMG/ClinGen point-mapping thresholds (guideline-sourced, not fit) ---
_ACMG_PM1_DOMAIN_MIN = 0.6     # PM1 strength >=0.6 == curated functional domain
_ACMG_PM2_RARE_MIN = 6.0       # rarity score (~-log10 AF) rare enough for PM2
_ACMG_PM2_COMMON_MAX = 2.0     # common enough for BS1 benign
_ACMG_PP3_STRONG = 0.99        # AlphaMissense PP3_Strong (Pejaver 2022 calib.)
_ACMG_PP3_MOD = 0.972          # PP3_Moderate
_ACMG_PP3_SUP = 0.906          # PP3_Supporting
_ACMG_BP4_SUP = 0.07           # BP4 benign supporting
_ACMG_PATH_MIN = 6.0           # summed points >=6 -> LP/P (clinical call)
_ACMG_BEN_MAX = -1.0           # summed points <=-1 -> LB/B


def acmg_points_sensitivity(cohort: Cohort, order: np.ndarray, k: int) -> dict:
    """Robustness of the ACMG-points endpoint to the pathogenic-call threshold:
    report yield/accuracy at budget k under the LP-inclusive (>=6) and the
    strict-Pathogenic (>=10) combining cutoffs, so the conclusion does not rest
    on one threshold choice.

    The cutoff is passed as an argument; an earlier version swapped the module
    global `_ACMG_PATH_MIN` around the call, which was not reentrant."""
    return {name: acmg_points_curve(cohort, order, budgets=[k], path_min=thr)[k]
            for name, thr in (("LP_inclusive_ge6", 6.0),
                              ("strict_path_ge10", 10.0))}


def cost_weighted_yield(cohort: Cohort, order: np.ndarray,
                        confidence: float = 0.90,
                        cost_tiers: dict | None = None,
                        seed: int = GLOBAL_SEED) -> dict:
    """
    CLINICAL COST view of the acquisition curve. Combines the diagnostic yield
    (clinical_yield_curve) with the REAL acquisition cost of each ACMG evidence
    category, so the efficiency axis is expressed in the lab's true currency:
    how much diagnostic yield per unit of acquisition cost/turnaround.

    Cost tiers (ORDINAL turnaround/effort weights, not dollar prices; documented
    and swept, not tuned):
      - in-silico predictor (PP3 / AlphaMissense) : tier 1 -- seconds, free,
        already computed genome-wide.
      - population database (PM2 / gnomAD)         : tier 1 -- a database
        lookup, seconds-to-minutes, free.
      - LoF annotation (PVS1)                      : tier 1 -- computational
        consequence call.
      - functional / domain evidence (PM1, and any functional-assay channel)
                                                   : tier HIGH -- wet-lab or
        curated functional study, days-to-months, expensive.
    The default puts the three fast/cheap computational-or-database channels at
    cost 1 and the functional-domain channel (PM1) at cost 10, reflecting the
    orders-of-magnitude turnaround gap between an in-silico/database query and a
    functional assay. `cost_tiers` overrides per channel; the companion sweep
    reports robustness to the functional-channel multiplier.

    For each budget k we report cumulative acquisition cost (mean over variants
    of the summed tier costs of the first-k acquired channels) alongside the
    resolved fraction and call accuracy, and the yield efficiency
    resolved_per_cost = resolved_frac / cumulative_cost.

    Returns {k: {cum_cost, resolved_frac, call_accuracy, resolved_per_cost}}.
    """
    chans = cohort.domain.channels
    if cost_tiers is None:
        cost_tiers = {c: (10.0 if c == "PM1" else 1.0) for c in chans}
    yc = clinical_yield_curve(cohort, order, confidence=confidence, seed=seed)
    cvec = np.array([cost_tiers.get(c, 1.0) for c in chans], float)
    out = {}
    for k, y in yc.items():
        # mean cumulative cost of the first-k acquired channels across variants
        cum = float(cvec[order[:, :k]].sum(axis=1).mean())
        rf = y["resolved_frac"]
        out[k] = {"cum_cost": cum, "resolved_frac": rf,
                  "call_accuracy": y["call_accuracy"],
                  "resolved_per_cost": float(rf / cum) if cum > 0 else float("nan")}
    return out


def clinical_yield_sensitivity(cohort: Cohort, order: np.ndarray, k: int,
                               confidences=(0.80, 0.90, 0.95),
                               seed: int = GLOBAL_SEED) -> dict:
    """Sensitivity of the clinical-yield endpoint to the operating point: run
    clinical_yield_curve at budget k for several confidence cutoffs so the
    yield/accuracy trade-off is reported as a curve, not a single tuned point."""
    return {c: clinical_yield_curve(cohort, order, budgets=[k], confidence=c,
                                    seed=seed)[k] for c in confidences}


def cost_robustness(cohort: Cohort, orders: dict,
                    cost_models: dict | None = None) -> dict:
    """
    Cost-model robustness (paper Appendix Cost Robustness): the strategy
    ORDERING by area-under-the-cost-curve must be invariant to how a "query" is
    priced. We integrate each strategy's A(k) curve against the three cost
    calibers named in the appendix:
      - per_query : every channel costs one query (the default unit cost).
      - per_token : channels priced by an increasing per-position burden weight
        (1 + 0.5*position in the declared channel order) -- an arbitrary but
        strictly ordered ramp; the point is that the STRATEGY ranking survives
        even this monotone reweighting, not that the weights are calibrated
        prices.
      - per_monetary_cost : tiered price, a cheap first tier then a step-up
        (models a rate-limited or paywalled source acquired later).
    The Oracle is scored under the global best-subset (leq_k) rule -- a
    different acquisition rule than the deployable first-k strategies -- so its
    AUCC is reported per caliber as a separate reference ceiling
    (`oracle_reference_aucc`) and is NOT ranked against the deployable
    strategies.

    HONEST FINDING (do not overclaim invariance). Once the oracle is charged the
    true cost of its winning subset (fixing the earlier bug where oracle A came
    from the best subset but its cost was read off the per-variant order), the
    AUCC ordering is NOT robust across calibers: the deployable strategies swap
    top place between calibers (on the synthetic cohort RelMax vs Heuristic
    gaps range from ~4e-4 under per_token to ~2.7e-2 under per_query), and the
    oracle's higher A(k) does not guarantee a higher AUCC because its subset is
    more expensive. `ranking_invariant` therefore
    reports the ACTUAL computed invariance (often False on near-tied data) and
    should be read as a measured property, not an assumed one. Returns
    per-caliber deployable ranks, the oracle reference AUCC, and the measured
    invariance flags.

    cost_models maps caliber-name -> {channel: cost}; defaults are provided.
    The costs are ORDINAL burden weights, not calibrated prices: only their
    order matters, and the point of the analysis is that even the ranking is
    insensitive to them.
    """
    chans = cohort.domain.channels
    if cost_models is None:
        K = len(chans)
        cost_models = {
            "per_query": {c: 1.0 for c in chans},
            # first channel cheapest, ramp up by position (arbitrary but ordered)
            "per_token": {c: 1.0 + i * 0.5 for i, c in enumerate(chans)},
            # two-tier: first half cheap, second half 2x
            "per_monetary_cost": {c: (1.0 if i < K / 2 else 2.0) for i, c in enumerate(chans)},
        }

    def aucc(order, costs, leq_k=False):
        # Area under the A vs cumulative-cost curve. The cumulative cost at
        # budget k is averaged ACROSS VARIANTS, because a per-instance strategy
        # (e.g. Oracle) acquires a different channel order for each variant, so
        # a single row's order is not representative. The integral runs from
        # cost 0 (nothing acquired, A defined as chance-ish baseline via k=0) so
        # the FIRST channel's cost is included, and is normalized by the total
        # cost span 0..sum(costs).
        A = curve_A(cohort, order, budgets=list(range(1, cohort.domain.K + 1)),
                    leq_k=leq_k)
        ks = sorted(A)
        c = np.array([costs[ch] for ch in chans], float)
        if leq_k:
            # A comes from the global best-subset at each k, so the cost must be
            # the cost of THAT winning subset -- not order[:, :k], which is the
            # per-variant oracle ORDER (a different channel set under a
            # non-uniform caliber). Charge the winning subset's channels.
            cum_k = np.array([c[list(_best_subset_auc(cohort, k)[1])].sum()
                              for k in ks])
        else:
            # per-variant cumulative cost after its first k channels, averaged
            cum_k = np.array([c[order[:, :k]].sum(axis=1).mean() for k in ks])
        cum = np.concatenate([[0.0], cum_k])                     # include origin
        a = np.concatenate([[0.5], [A[k] for k in ks]])          # chance at cost 0
        trap = getattr(np, "trapezoid", None) or np.trapz        # numpy>=2 rename
        return float(trap(a, cum) / (cum[-1] - cum[0] + 1e-12))

    # The Oracle is scored under the leq_k (global best-subset) rule while the
    # deployable strategies use first-k; those are DIFFERENT acquisition rules,
    # so their AUCC values are not on a common footing and must not be ranked
    # against each other. We therefore rank only the DEPLOYABLE strategies for
    # the invariance claim and report the Oracle's AUCC as a separate reference
    # ceiling per caliber.
    ranks = {}
    for caliber, costs in cost_models.items():
        scores = {name: aucc(order, costs, leq_k=(name == "Oracle"))
                  for name, order in orders.items()}
        deployable = {m: s for m, s in scores.items() if m != "Oracle"}
        order_by = sorted(deployable, key=lambda m: -deployable[m])
        ranks[caliber] = {"scores": scores, "rank": order_by,
                          "oracle_reference_aucc": scores.get("Oracle")}
    rank_lists = [tuple(v["rank"]) for v in ranks.values()]
    # `ranking_invariant` = the TOP DEPLOYABLE strategy is the same under every
    # caliber (Oracle excluded -- it uses a different acquisition rule and is a
    # reference ceiling, not a competitor). The strict FULL deployable order is
    # reported separately and is NOT guaranteed: near-tied deployable strategies
    # can swap under different weightings.
    tops = {rl[0] for rl in rank_lists if rl}
    invariant = len(tops) == 1
    full_invariant = all(rl == rank_lists[0] for rl in rank_lists)
    return {"per_caliber": ranks, "ranking_invariant": invariant,
            "full_ranking_invariant": full_invariant}


def confidence_based_stopping(cohort: Cohort, order: np.ndarray,
                              margins=(0.05, 0.10, 0.20)) -> dict:
    """
    Confidence-based stopping (paper Appendix Downstream Analysis). Instead of a
    fixed budget k*, stop acquiring for a variant once the scorer's decision
    margin |f(E_k) - 0.5| exceeds a confidence threshold. Reports, per margin:
    the mean queries actually spent and the resulting A at that adaptive budget,
    versus always spending the full K. Shows acquisition can stop early on
    confident variants without losing ranking accuracy.

    Returns margin -> {mean_queries, auc_at_stop, full_k, full_auc}.

    CAVEAT (read before citing): the per-budget min-max normalization makes each
    k's scores span [0,1] but does NOT put different-k scores on a common scale,
    and the margin is taken around 0.5, which is the midpoint of the normalized
    range, not a calibrated decision boundary. So `auc_at_stop` can come out
    ABOVE `full_auc` (stopping early "beats" full acquisition) -- that is a
    scale/threshold artifact of this diagnostic, NOT evidence that acquiring less
    is more accurate. Report mean_queries as the query-saving story; treat
    auc_at_stop as a coarse "accuracy roughly held", not a headline number.

    SECOND CAVEAT (deployability): the min-max normalization uses the WHOLE
    cohort's score range, so a single variant's stop decision depends on the
    other variants' scores -- this rule is transductive and cannot be computed
    at deployment time on one variant in isolation. It is a retrospective
    diagnostic of the stopping OPPORTUNITY, not a deployable stopping policy.
    """
    n, K = len(cohort), cohort.domain.K
    # per-variant score after each budget k
    scores_by_k = {k: _score_from_evidence(cohort, _acquired_mask(order, k))
                   for k in range(1, K + 1)}
    # normalize scores to [0,1] per budget to read a margin around 0.5
    def norm(s):
        lo, hi = np.nanmin(s), np.nanmax(s)
        return (s - lo) / (hi - lo + 1e-12)
    normed = {k: norm(scores_by_k[k]) for k in scores_by_k}
    full_auc = per_gene_auc(cohort.y, scores_by_k[K], cohort.genes)
    out = {}
    for m in margins:
        stop_k = np.full(n, K, int)
        for i in range(n):
            for k in range(1, K + 1):
                if abs(normed[k][i] - 0.5) >= m:
                    stop_k[i] = k
                    break
        # score each variant at its own stop budget. Use the PER-BUDGET
        # NORMALIZED score, not the raw sum: the raw _score_from_evidence sums
        # (s-0.5) over acquired channels, so its range grows with k (~[-0.5,0.5]
        # at k=1 vs ~[-K/2,K/2] at k=K). Mixing variants that stopped at
        # different k on the raw scale would rank high-budget variants above
        # low-budget ones purely by scale, making auc_at_stop uninterpretable.
        stopped = np.array([normed[stop_k[i]][i] for i in range(n)])
        out[float(m)] = {
            "mean_queries": float(stop_k.mean()),
            "auc_at_stop": per_gene_auc(cohort.y, stopped, cohort.genes),
            "full_k": K, "full_auc": full_auc}
    return out


def trajectory_diversity(order: np.ndarray, channels: list) -> dict:
    """
    Acquisition-trajectory diversity (paper RQ2). Summarizes a strategy/agent's
    per-variant orders: how many DISTINCT full orders it uses, and the
    distribution of the FIRST acquired channel. A fixed strategy uses 1 order;
    a variant-adaptive planner uses many. On the current 491-variant real
    cohort under context:false, every strategy collapses to n_distinct_orders
    == 1 with PVS1 first for all 491 variants -- this is the structural
    finding this metric is built to surface, not a hypothetical.

    Returns {n_distinct_orders, first_query_counts, n_variants}.
    """
    n = len(order)
    seen = {tuple(order[i].tolist()) for i in range(n)}
    first = order[:, 0]
    counts = {}
    for j in first:
        counts[channels[int(j)]] = counts.get(channels[int(j)], 0) + 1
    return {"n_distinct_orders": len(seen),
            "first_query_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
            "n_variants": n}


def holm_bonferroni(pvals: dict, alpha: float = 0.05) -> dict:
    """
    Holm-Bonferroni step-down correction for a family of hypotheses (paper A4:
    multiple-model claims must be corrected). `pvals` maps a label -> raw p.
    Returns label -> {p_raw, p_adj, reject} controlling FWER at alpha.
    """
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (lab, p) in enumerate(items):
        p_adj = min(1.0, (m - i) * p)
        running = max(running, p_adj)          # enforce monotonicity
        out[lab] = {"p_raw": float(p), "p_adj": float(running),
                    "reject": running <= alpha}
    return out
