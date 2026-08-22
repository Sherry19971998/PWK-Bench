"""
Evaluation harness (Plan A). Runs the reference strategies AND any number of
agents on ONE fixed cohort/environment (harness-invariance): the same retriever,
scorer, and cohort for every row, so only the acquisition ORDER differs. The
deployable strategies (RelMax, agents, Heuristic) are all scored by the
identical shared scorer `_score_from_evidence` under the |pi|=k rule. The Oracle
UPPER REFERENCE is the only exception: it is the global best-subset
argmax_{|pi|<=k} (Eq. oracle), where the gold label enters solely through the
single choice of which channel subset maximizes AUC -- never through the scorer,
which is the same shared function. Produces a tidy results table with one row
per strategy/agent, mirroring the paper's Table 2 structure.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from .domains.base import Cohort
from . import strategies as S, metrics as M
from .agents import build_agent
from .agents.llm import DailyQuotaExhausted

# Per-run cache of built agents and their computed acquisition orders, keyed by
# the full identity of the agent spec. A real LLM agent computes its order by
# calling the model 506x(K) times; the same (slot, model, adherence, seed,
# context) must therefore be built and queried ONCE per cohort, not re-elicited
# in the main loop, the CI pass, the robustness block, and figure generation
# (that was ~4x the API cost). Callers use `agent_and_order` instead of calling
# build_agent + agent.order directly. The cache hangs off the cohort object via
# a WeakKeyDictionary: keying by id(cohort) would be unsafe because CPython
# reuses an address after the object is collected, so a fresh cohort landing on
# a freed address could read a stale acquisition order. The weak map is dropped
# automatically when the cohort is garbage-collected (no unbounded growth).
import weakref
_AGENT_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _agent_key(m: dict) -> tuple:
    return (m.get("slot"), m.get("kind"), m.get("model_id", ""),
            m.get("adherence", 0.85), m.get("seed", 0), bool(m.get("context", False)))


def agent_and_order(cohort: Cohort, m: dict):
    """Build the agent for spec `m` on `cohort` and compute its acquisition
    order, memoized so repeated calls within a run reuse the first result
    (and, for a real LLM agent, the API calls behind it)."""
    per_cohort = _AGENT_CACHE.setdefault(cohort, {})
    key = _agent_key(m)
    if key not in per_cohort:
        agent = build_agent(kind=m["kind"], model_id=m.get("model_id", ""),
                            name=m.get("slot"), adherence=m.get("adherence", 0.85),
                            seed=m.get("seed", 0), context=m.get("context", False),
                            max_workers=m.get("workers", 1),
                            cache_path=m.get("cache_path"))
        per_cohort[key] = (agent, agent.order(cohort))
    return per_cohort[key]


def clear_agent_cache():
    """Drop cached agents/orders."""
    _AGENT_CACHE.clear()


def evaluate_order(cohort: Cohort, order: np.ndarray, label: str, model_id: str = "") -> dict:
    # Budget grid spans the domain's OWN channel count, not the hardcoded
    # variant BUDGETS=[1..4]; a domain with more channels would otherwise
    # silently drop budgets 5..K from PE, k*, and A(k*).
    # The Oracle reference follows the paper's pi*(k)=argmax_{|pi|<=k} rule
    # (acquire AT MOST k, skip label-hurtful categories) -> monotone upper
    # bound. Every deployable strategy uses the standard first-k (|pi|=k) rule.
    A = M.curve_A(cohort, order, budgets=list(range(1, cohort.domain.K + 1)),
                  leq_k=(label == "Oracle"))
    oo = M.order_optimality(cohort, order)
    row = {"strategy": label, "model_id": model_id,
           "PE": M.planning_efficiency(A), "k_star": M.k_star(A),
           "po_vs_relevance": oo["po_vs_relevance"],
           "po_vs_oracle": oo["po_vs_oracle"],
           "rho_vs_relevance": oo["rho_vs_relevance"],
           "rho_vs_oracle": oo["rho_vs_oracle"]}
    for k, v in A.items():
        row[f"A_k{k}"] = v
    return row


def multimodel_contrast_holm(cohort: Cohort, model_matrix: list[dict],
                             budget: int = 1) -> pd.DataFrame:
    """
    Multi-model family of paired agent-vs-RelMax contrasts with Holm-Bonferroni
    correction (paper A4: when several models each get an agent-vs-reference
    claim, the p-values must be corrected for multiple comparisons).

    For each agent in the matrix, compute the paired gene-clustered bootstrap
    contrast of its A(budget) against the RelMax reference, then Holm-correct
    the family of per-model p-values. Returns one row per model with the raw
    diff/CI/effect size and the Holm-adjusted p and reject decision.

    A single-model matrix is a family of size 1 (Holm reduces to the raw p);
    the correction only bites once >=2 models are compared, which is exactly the
    scaling sweep the paper's Plan A runs.
    """
    rel = S.relmax_order(cohort)
    contrasts = {}
    for m in model_matrix:
        try:
            ag, ag_order = agent_and_order(cohort, m)
        except DailyQuotaExhausted as e:
            # Same rationale as run(): a quota-capped row being incomplete on a
            # given day must not sink the whole multi-model contrast table.
            warnings.warn(
                f"agent {m.get('slot')}: daily quota exhausted -- excluded "
                f"from this contrast table ({e})", RuntimeWarning, stacklevel=2)
            continue
        slot = m.get("slot", ag.model_id)
        pc = M.paired_contrast_ci(cohort, ag_order, rel, k=budget)
        contrasts[slot] = pc
    adj = M.holm_bonferroni({s: c["p_value"] for s, c in contrasts.items()})
    rows = []
    for slot, c in contrasts.items():
        rows.append({"model": slot, "budget": budget, "diff_vs_relmax": c["diff"],
                     "lo": c["lo"], "hi": c["hi"], "z_stat": c["z_stat"],
                     "p_raw": adj[slot]["p_raw"], "p_holm": adj[slot]["p_adj"],
                     "reject_holm": adj[slot]["reject"], "includes_zero": c["includes_zero"]})
    return pd.DataFrame(rows)


def run(cohort: Cohort, model_matrix: list[dict] | None = None,
        with_ci: bool = False) -> pd.DataFrame:
    """
    model_matrix: list of rows from configs/models.yaml (kind, model_id,
    capability_proxy, [adherence]). If None, only the reference/baseline
    strategies are run.
    """
    rows = []
    orders = {name: S.STRATEGY_FNS[name](cohort) for name in S.STRATEGY_FNS}
    # references + baselines
    for name, order in orders.items():
        rows.append(evaluate_order(cohort, order, name))
    # oracle gap is defined relative to the oracle PE
    pe_oracle = next(r["PE"] for r in rows if r["strategy"] == "Oracle")

    # agents
    if model_matrix:
        for m in model_matrix:
            try:
                agent, order = agent_and_order(cohort, m)
            except DailyQuotaExhausted as e:
                # A quota-capped, cache_path-backed row (e.g. a Gemini free-tier
                # sweep spread over several days via scripts/variant/warm_llm_cache.py)
                # is EXPECTED to be incomplete some days. Losing this one row is
                # not a reason to also lose every reference/agent row already
                # computed above (and any real API cost already paid for them
                # in this same call) -- skip it, warn loudly, and keep going.
                warnings.warn(
                    f"agent {m.get('slot')}: daily quota exhausted before its "
                    "order could be computed -- skipping this row (not "
                    "aborting the run). Re-run once the cache_path is fuller "
                    f"or the provider's daily cap resets. ({e})", RuntimeWarning, stacklevel=2)
                continue
            r = evaluate_order(cohort, order, f"agent:{m.get('slot')}", agent.model_id)
            r["capability_proxy"] = m.get("capability_proxy")
            r["context"] = bool(m.get("context", False))
            # fraction of elicitation steps where the model named no valid
            # channel and we fell back to a fixed order. High -> results are a
            # fallback artifact (see agents/llm.py), not model behaviour.
            tot = getattr(agent, "total_steps", 0)
            fails = getattr(agent, "parse_failures", 0)
            r["parse_failure_rate"] = fails / tot if tot else 0.0
            # A failure on the LAST step (one channel left) cannot change the
            # order -- the fallback picks the only admissible channel. Gate the
            # reliability warning on the CONSEQUENTIAL rate, over the steps where
            # a real choice existed, or a model that merely babbles on the forced
            # final step is wrongly flagged as producing fabricated results.
            forced_fails = getattr(agent, "forced_parse_failures", 0)
            forced_steps = getattr(agent, "forced_steps", 0)
            free = tot - forced_steps
            r["parse_failure_rate_effective"] = (
                (fails - forced_fails) / free if free > 0 else 0.0)
            r["parse_failures_forced"] = forced_fails
            if r["parse_failure_rate_effective"] > 0.05:
                warnings.warn(
                    f"agent {m.get('slot')}: "
                    f"{r['parse_failure_rate_effective']:.0%} of CONSEQUENTIAL "
                    "steps fell back to a fixed order (model named no valid "
                    "channel where a real choice existed). Results are "
                    "unreliable — check max_tokens / API.",
                    RuntimeWarning, stacklevel=2)
            rows.append(r)

    # ---- Axis C: memorization control (was computed only in tests) ----------
    # A closed-book scorer that sees only gene identity; AUC -> 0.5 means the
    # task genuinely requires acquisition. Reported for every run so all three
    # axes (efficiency / order-optimality / memorization) appear in the table.
    masked = M.masked_auc(cohort)
    for r in rows:
        r["masked_auc"] = masked

    # ---- optional gene-clustered bootstrap CIs for each A(k) ----------------
    if with_ci:
        budget_grid = list(range(1, cohort.domain.K + 1))  # span domain's K
        strat_orders = dict(orders)
        if model_matrix:
            for m in model_matrix:
                try:
                    _, ag_order = agent_and_order(cohort, m)
                except DailyQuotaExhausted:
                    # Already warned about and skipped in the main loop above;
                    # this row has no entry in `rows` either, so skipping here
                    # too just avoids recomputing a doomed call.
                    continue
                strat_orders[f"agent:{m.get('slot')}"] = ag_order
        for r in rows:
            order = strat_orders.get(r["strategy"])
            if order is None:
                continue
            for k in budget_grid:
                lo, hi = M.gene_bootstrap_ci(cohort, order, k,
                                             leq_k=(r["strategy"] == "Oracle"))
                r[f"A_k{k}_lo"], r[f"A_k{k}_hi"] = lo, hi

    df = pd.DataFrame(rows)
    df["delta_oracle"] = pe_oracle - df["PE"]
    return df
