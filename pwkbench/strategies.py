"""
Acquisition strategies (paper, Acquisition Protocol section).

A strategy produces, for each variant, an ORDER over channels (a permutation).
At budget k, the first k channels of that order are "acquired". The scorer
f(E_k) then ranks variants by pathogenicity from the acquired evidence, and
A(k) is the per-gene-stratified ranking AUC of that scorer (see metrics.py).

References (paper):
  - RelMax  : ONE fixed global relevance order, identical for every variant
              (lower reference).
  - Oracle  : per-instance order that maximizes AUC at each k (upper reference),
              computed on gold labels — a ruler, not a deployable policy.
Baselines:
  - Heuristic: greedy "query the strongest observed category".
  - Random   : random order per variant.
Agent orders come from agents/ adapters.
"""
from __future__ import annotations
import numpy as np
from .domains.base import Cohort


def _signed_decisiveness(cohort: Cohort) -> np.ndarray:
    """
    d = (2y-1)(s - 1/2) per (variant, channel): how much a channel's value
    points the right way for that variant's gold label. Used for the oracle
    order and for order-optimality (paper).
    Channel value is min-max normalized to [0,1] so s-1/2 is comparable.
    """
    y = cohort.y
    K = cohort.domain.K
    d = np.zeros((len(cohort), K))
    for j, ch in enumerate(cohort.domain.channels):
        v = cohort.value(ch).astype(float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        # Constant on this population -> zero information, decisiveness stays 0.
        # Explicit branch for the same reason as _score_from_evidence: on a
        # row-subset view the inherited neutral can differ from the constant, and
        # the centering below would then produce a ~1e12 magnitude instead of 0.
        if hi - lo < 1e-12:
            continue
        s = (v - lo) / (hi - lo + 1e-12)
        # center on the channel's neutral value (same fix as _score_from_evidence):
        # an undefined entry has s == s_neutral -> decisiveness 0, so it never
        # spuriously orders ahead of an informative channel.
        s_neutral = (cohort.domain.neutral_for(ch) - lo) / (hi - lo + 1e-12)
        d[:, j] = (2 * y - 1) * (s - s_neutral)
    return d


def relmax_order(cohort: Cohort) -> np.ndarray:
    """Fixed, LABEL-FREE relevance order (deployable lower reference).

    The domain's declared channel order -- for the variant domain the ACMG
    evidence-strength prior [PVS1, PM2, PP3, PM1] (spec.py CHANNELS): a
    very-strong LoF category first, then rarity, in-silico, gene constraint.
    The SAME permutation for every variant, using NO gold labels, so it is a
    genuinely deployable relevance baseline -- the order a non-planning
    retriever following prior evidence strength would use. This now matches the
    spec.py declaration ("CHANNELS order = the fixed global RELEVANCE order used
    by RelMax"); the earlier implementation ignored CHANNELS and ranked by
    global signed decisiveness, which (a) contradicted the spec and (b) peeked
    at labels. RelMax is deployable and label-free; BestFixed (below) is the
    label-conditioned tight reference; the oracle is the per-instance ceiling.
    """
    K = cohort.domain.K
    return np.tile(np.arange(K), (len(cohort), 1))


def best_fixed_order(cohort: Cohort) -> np.ndarray:
    """Tightest FIXED (non-adaptive) order under the evaluation metric itself.

    Unlike RelMax (a label-free prior order) this is selected to maximize the
    benchmark's own score -- the per-gene ranking AUC -- over fixed orders, so
    no single global permutation beats it. It is LABEL-CONDITIONED, like the
    oracle, so ``Oracle - BestFixed`` isolates exactly the gain from
    PER-INSTANCE adaptation over the best possible NON-adaptive order (a
    stronger, cleaner decomposition than a gap to an arbitrary hand-picked
    order). The SAME permutation is used for every variant.

    Selection: for a small channel set (K! <= 5040, i.e. K<=7 -- covers the
    variant K=4 domain) we EXHAUSTIVELY pick the permutation
    with the highest planning efficiency PE = mean_k A(k); for larger K we fall
    back to the greedy single-slot order (channels by descending single-slot
    per-gene AUC), which is A(1)-optimal.

    NOTE: the order is fitted in-sample. For K<=6 the permutation space is tiny
    so overfitting is negligible; on large-K cohorts interpret the tightness
    with care (and, for strict CIs, the order should be re-fitted per bootstrap
    resample -- see gene_bootstrap_ci).
    """
    from . import metrics as M                 # lazy: metrics imports strategies
    import itertools, math
    K = cohort.domain.K
    n = len(cohort)
    if math.factorial(K) <= 5040:
        best_perm, best_pe = None, -1.0
        budgets = list(range(1, K + 1))
        for perm in itertools.permutations(range(K)):
            order = np.tile(perm, (n, 1))
            A = M.curve_A(cohort, order, budgets=budgets)
            pe = float(np.mean(list(A.values())))
            if pe > best_pe:
                best_pe, best_perm = pe, perm
        rank = np.array(best_perm)
    else:
        aucs = []
        for j in range(K):
            perm = [j] + [t for t in range(K) if t != j]
            order = np.tile(perm, (n, 1))
            aucs.append(M.curve_A(cohort, order, budgets=[1])[1])
        rank = np.argsort(-np.array(aucs))
    return np.tile(rank, (n, 1))


def definedness_baseline_order(cohort: Cohort) -> np.ndarray:
    """Schema-only baseline: order channels by DEFINEDNESS, no labels, no values.

    For each variant the defined channels come first (in the domain's declared
    order) and the undefined channels last. This encodes ONLY "which evidence
    categories even exist for this variant" -- e.g. a non-missense variant has
    no PP3/PM1, so those sink to the back -- information an agent reads for free
    from the variant name (missense vs not). It does NO per-instance value
    reasoning and peeks at NO gold labels.

    Diagnostic use: if ``Oracle - DefinednessBaseline`` is small, most of the
    apparent planning gap over RelMax is really schema lookup. See
    ``definedness_stratified_best_order`` for the strongest schema-only
    reference. On the real 491-variant cohort (PVS1 defined on all 491; PM2 on
    447; PP3 on 91; PM1 on 129) schema/definedness accounts for ~46% of the
    Oracle-RelMax gap and genuine per-instance planning for ~54% (reproduce with
    docs/variant/real_data.md; the bundled synthetic cohort gives a different, higher
    schema share because its channels are near-fully defined and label-tied).
    """
    K = cohort.domain.K
    dm = np.column_stack([cohort.df[f"{ch}__defined"].to_numpy().astype(bool)
                          for ch in cohort.domain.channels])
    order = np.zeros((len(cohort), K), dtype=int)
    for i in range(len(cohort)):
        defined = [j for j in range(K) if dm[i, j]]
        undef = [j for j in range(K) if not dm[i, j]]
        order[i] = defined + undef
    return order


def definedness_stratified_best_order(cohort: Cohort) -> np.ndarray:
    """Strongest SCHEMA-ONLY reference: best FIXED order within each definedness
    group (a much tighter, reviewer-defensible lower reference than RelMax when
    definedness varies across the cohort).

    Variants are grouped by definedness pattern (which channels exist). Within
    each group a single fixed order maximizes that group's planning efficiency
    PE (exhaustive for K<=7, else greedy by single-slot AUC). The order is
    constant within a group, so it exploits only "which channels are defined"
    (schema knowledge), doing NO per-instance value reasoning.
    ``Oracle - DefinednessStratifiedBest`` is therefore the part of the gap that
    genuinely requires reading each variant's channel VALUES -- the real
    order-optimality signal (+0.081 PE, ~54% of the gap, on the real cohort).

    NOTE: like BestFixed the per-group order is label-conditioned and fitted
    in-sample; interpret on large-K cohorts with the same caution.
    """
    from . import metrics as M
    import itertools, math
    K = cohort.domain.K
    n = len(cohort)
    y = cohort.y
    genes = cohort.genes
    dm = np.column_stack([cohort.df[f"{ch}__defined"].to_numpy().astype(bool)
                          for ch in cohort.domain.channels])
    order = np.zeros((n, K), dtype=int)
    groups = {}
    for i in range(n):
        groups.setdefault(tuple(dm[i]), []).append(i)
    budgets = list(range(1, K + 1))
    for pat, idx in groups.items():
        idx = np.array(idx)
        if math.factorial(K) <= 5040:
            best_perm, best_pe = None, -1.0
            for perm in itertools.permutations(range(K)):
                om = np.tile(perm, (n, 1))
                pe = np.mean([M.per_gene_auc(
                    y[idx], M._score_from_evidence(cohort, M._acquired_mask(om, k))[idx],
                    genes[idx]) for k in budgets])
                if pe > best_pe:
                    best_pe, best_perm = pe, perm
            order[idx] = best_perm
        else:
            aucs = []
            for j in range(K):
                perm = [j] + [t for t in range(K) if t != j]
                om = np.tile(perm, (n, 1))
                aucs.append(M.per_gene_auc(
                    y[idx], M._score_from_evidence(cohort, M._acquired_mask(om, 1))[idx],
                    genes[idx]))
            order[idx] = list(np.argsort(-np.array(aucs)))
    return order


def oracle_order(cohort: Cohort) -> np.ndarray:
    """Per-instance oracle order (upper REFERENCE, not a deployable policy):
    per variant, channels sorted by that variant's own signed decisiveness
    (2y-1)(s-1/2), most label-favourable first. It is label-conditioned by
    construction -- that is what makes it an upper bound -- so on a separable
    cohort it can reach AUC 1.0 at small budgets (each variant's single most
    decisive channel already ranks it correctly). It greedily orders each
    variant's own channels; it is NOT a search for the globally AUC-maximizing
    subset, and the two coincide only because the scorer is additive and
    monotone in signed decisiveness. The gap between a strategy's curve and this
    reference is the headroom still recoverable by re-planning what to acquire.
    """
    d = _signed_decisiveness(cohort).copy()
    # An UNDEFINED channel has s == neutral -> decisiveness exactly 0. For a
    # benign variant (y=0) any mildly-pathogenic DEFINED channel has d<0 and
    # would sort AFTER the 0-valued undefined channel, so the oracle would
    # "acquire" a no-data channel ahead of a real one. Acquiring a channel with
    # no data can never help, so force undefined entries to the very back.
    for j, ch in enumerate(cohort.domain.channels):
        dcol = f"{ch}__defined"
        if dcol in cohort.df.columns:
            undef = ~cohort.df[dcol].to_numpy(bool)
            d[undef, j] = -np.inf
    return np.argsort(-d, axis=1)


def heuristic_order(cohort: Cohort) -> np.ndarray:
    """Greedy: query the channel with the strongest observed |value-neutral|
    first, per variant (a naive baseline, not a reference)."""
    K = cohort.domain.K
    strength = np.zeros((len(cohort), K))
    for j, ch in enumerate(cohort.domain.channels):
        v = cohort.value(ch).astype(float)
        lo, hi = np.nanmin(v), np.nanmax(v)
        # Constant channel -> no observed strength at all, so it must stay 0 and
        # sort LAST. Without the branch an inherited neutral that differs from
        # the constant makes |s - s_neutral| ~1e12, i.e. a zero-information
        # channel would be queried FIRST for every variant.
        if hi - lo < 1e-12:
            continue
        s = (v - lo) / (hi - lo + 1e-12)                 # min-max like the scorer
        s_neutral = (cohort.domain.neutral_for(ch) - lo) / (hi - lo + 1e-12)
        strength[:, j] = np.abs(s - s_neutral)           # comparable across channels
    return np.argsort(-strength, axis=1)


def discriminative_greedy_order(cohort: Cohort, n_folds: int = 5) -> np.ndarray:
    """AFA discriminative-greedy baseline (E1): the
    textbook active-feature-acquisition myopic policy -- at each step, add
    whichever unacquired channel gives the largest per-gene AUC given the
    channels already selected, choosing OUT OF FOLD so the pick cannot overfit
    to which variants are in this cohort.

    Distinct from `best_fixed_order` (in-sample EXHAUSTIVE search over the
    entire K! permutation space, which that function's own docstring already
    flags as fitted in-sample): this is the weaker, more standard AFA
    baseline -- a single step-wise myopic choice per position, cross-validated
    -- so it answers the reviewer's "is the agent just doing simple
    discriminative-greedy?" question with a comparably-weak reference rather
    than the strongest possible fixed order.

    The channels are selected on a GLOBAL, gene-grouped k-fold split (so no
    gene's variants leak between the fold that picks a channel and the fold
    that scores it); the resulting order is the SAME permutation for every
    variant, like RelMax and BestFixed. `_score_from_evidence` has no
    trainable parameters (a fixed linear combination -- see its docstring), so
    only the CHANNEL-SELECTION decision is cross-validated, not a model fit.
    """
    from sklearn.model_selection import GroupKFold
    from . import metrics as M                 # lazy: metrics imports strategies
    K = cohort.domain.K
    n = len(cohort)
    y, genes = cohort.y, cohort.genes
    n_splits = min(n_folds, len(np.unique(genes)))
    gkf = GroupKFold(n_splits=n_splits)
    folds = list(gkf.split(np.zeros(n), y, groups=genes))

    selected: list[int] = []
    remaining = list(range(K))
    for _ in range(K):
        best_ch, best_cv_auc = None, -np.inf
        for ch in remaining:
            mask = np.zeros((n, K), dtype=bool)
            mask[:, selected + [ch]] = True
            score = M._score_from_evidence(cohort, mask)
            fold_aucs = [M.per_gene_auc(y[te], score[te], genes[te])
                         for _, te in folds]
            mean_auc = float(np.mean(fold_aucs))
            if mean_auc > best_cv_auc:
                best_cv_auc, best_ch = mean_auc, ch
        selected.append(best_ch)
        remaining.remove(best_ch)
    return np.tile(selected, (n, 1))


def random_order(cohort: Cohort, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    orders = np.tile(np.arange(cohort.domain.K), (len(cohort), 1))
    for i in range(len(cohort)):
        rng.shuffle(orders[i])
    return orders


STRATEGY_FNS = {
    "RelMax": relmax_order,
    "BestFixed": best_fixed_order,
    "DefinednessBaseline": definedness_baseline_order,
    "DefinednessStratifiedBest": definedness_stratified_best_order,
    "Oracle": oracle_order,
    "Heuristic": heuristic_order,
    "Random": random_order,
    "DiscriminativeGreedy": discriminative_greedy_order,
}
