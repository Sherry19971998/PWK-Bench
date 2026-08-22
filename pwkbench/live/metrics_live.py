"""Metrics for the live tool-using agent: sufficiency, over-acquisition, yield.

THE YARDSTICK PROBLEM
---------------------
The frozen-pool benchmark could enumerate a per-instance oracle because the
evidence set was four fixed channels. A live tool space cannot be enumerated, so
the reference here is a RETROSPECTIVE SUFFICIENCY POINT: for each case, the
smallest number of acquirable evidence channels that already produces the correct
ACMG call. The agent should not need to exceed it, and how far it exceeds it is
the headline.

TOOL -> ACMG CHANNEL MAPPING, AND ITS LIMITS
--------------------------------------------
    get_population_freq          -> PM2   (rarity)
    get_insilico_pathogenicity   -> PP3   (AlphaMissense)
    get_domain_context           -> PM1   (functional domain)
    get_functional_assay         -> PS3/BS3, WHICH THE COHORT DOES NOT CARRY

Two consequences that must be stated wherever these numbers appear:

1. **PVS1 is free.** Loss-of-function status is readable from the HGVS name the
   agent is already shown (`p.Arg213Ter`, `...fs`), so no tool is needed for it.
   The sufficiency point is therefore computed over the three ACQUIRABLE channels
   with PVS1 already in hand -- otherwise every LoF case would look trivially
   "sufficient at zero tools" for a reason that has nothing to do with tool use.

2. **`get_functional_assay` cannot lower the sufficiency point**, because the
   reference classifier has no PS3 channel to score it with. Under this reference
   every functional-assay call is over-acquisition BY CONSTRUCTION. That is a
   limitation of the yardstick, not a finding about the agent, so
   `over_acquisition` is reported twice: `over_acq_mapped` counts only the three
   ACMG-mapped tools (interpretable), and `over_acq_all` counts every tool
   (what the agent actually spent). Never quote the second as if the reference
   had judged the assay call unnecessary.
"""
from __future__ import annotations
from itertools import combinations

import numpy as np
import pandas as pd

from ..metrics import _acmg_channel_points, _ACMG_PATH_MIN, _ACMG_BEN_MAX

TOOL_TO_CHANNEL = {
    "get_population_freq": "PM2",
    "get_insilico_pathogenicity": "PP3",
    "get_domain_context": "PM1",
    "get_functional_assay": None,      # PS3/BS3 -- not in this cohort
}
ACQUIRABLE = ("PM2", "PP3", "PM1")
FREE = ("PVS1",)


def _call_from_points(total, path_min=_ACMG_PATH_MIN, ben_max=_ACMG_BEN_MAX):
    call = np.full(np.shape(total), -1)
    call[total >= path_min] = 1
    call[total <= ben_max] = 0
    return call


def sufficiency_points(cohort, acquirable=ACQUIRABLE, free=FREE) -> pd.DataFrame:
    """Smallest number of acquirable channels that already gives the right call.

    Returns one row per variant: `k_star` (0..len(acquirable), or NaN when NO
    subset produces the correct call), plus `sufficient_at_zero` and the winning
    subset size's example membership. NaN is not a failure to compute -- it means
    the reference classifier never gets this variant right no matter what it
    acquires, and those cases must be excluded from over-acquisition rather than
    silently treated as "needs everything".
    """
    chpts = _acmg_channel_points(cohort)
    y = cohort.y.astype(int)
    n = len(y)
    base = np.zeros(n)
    for c in free:
        base += chpts[c]

    k_star = np.full(n, np.nan)
    best_subset = [None] * n
    for k in range(0, len(acquirable) + 1):
        todo = np.isnan(k_star)
        if not todo.any():
            break
        for subset in combinations(acquirable, k):
            tot = base.copy()
            for c in subset:
                tot += chpts[c]
            ok = (_call_from_points(tot) == y) & todo
            newly = ok & np.isnan(k_star)
            k_star[newly] = k
            for i in np.flatnonzero(newly):
                best_subset[i] = subset
            todo = np.isnan(k_star)
            if not todo.any():
                break
    return pd.DataFrame({
        "variant_id": cohort.df["variant_id"].to_numpy(),
        "label": y, "k_star": k_star,
        "sufficient_at_zero": k_star == 0,
        "k_star_example_subset": [",".join(s) if s else "" for s in best_subset],
    })


def over_acquisition(traj: pd.DataFrame, suff: pd.DataFrame) -> pd.DataFrame:
    """Join trajectories to sufficiency points and compute both over-acq columns.

    `traj.tools_called` may be a JSON string (as written by run_live.py) or a
    list; both are accepted so the function works on a freshly returned
    trajectory and on a reloaded CSV.
    """
    import json

    def _as_list(v):
        if isinstance(v, list):
            return v
        if isinstance(v, str) and v.strip().startswith("["):
            return json.loads(v)
        return []

    t = traj.copy()
    t["_tools"] = t["tools_called"].map(_as_list)
    # DISTINCT mapped channels: calling the same tool twice acquires nothing new,
    # so counting raw calls would inflate over-acquisition with repetition rather
    # than with breadth. Repetition is still visible in `n_tool_calls`.
    t["n_mapped"] = t["_tools"].map(
        lambda ts: len({TOOL_TO_CHANNEL[x] for x in ts
                        if TOOL_TO_CHANNEL.get(x)}))
    t["n_unmapped"] = t["_tools"].map(
        lambda ts: sum(1 for x in ts if TOOL_TO_CHANNEL.get(x) is None))
    t["n_tool_calls"] = t["_tools"].map(len)

    m = t.merge(suff[["variant_id", "k_star", "sufficient_at_zero"]],
                on="variant_id", how="left")
    m["over_acq_mapped"] = m["n_mapped"] - m["k_star"]
    m["over_acq_all"] = m["n_mapped"] + m["n_unmapped"] - m["k_star"]
    m["reference_undefined"] = m["k_star"].isna()
    return m.drop(columns=["_tools"])


def yield_curve(traj: pd.DataFrame) -> pd.DataFrame:
    """Accuracy as a function of how many tools the agent chose to call.

    This is DESCRIPTIVE, not causal: the agent picked its own budget, so cases
    where it called four tools are the cases it found hard. A flat or falling
    curve therefore does NOT show that acquiring more hurts -- it shows the agent
    spends more on cases it is more likely to get wrong. Read it next to the
    tool-ablation arm, which does vary the tool set exogenously.
    """
    # run_live.py writes `n_tools_called`; over_acquisition() adds `n_tool_calls`.
    # Accept either so the curve works on a raw trajectory file and on a joined
    # frame without the caller having to know which stage produced it.
    col = ("n_tool_calls" if "n_tool_calls" in traj.columns else "n_tools_called")
    g = traj.groupby(col).agg(
        n_cases=("correct", "size"),
        accuracy=("correct", "mean"),
        n_correct=("correct", "sum"))
    return g.reset_index().rename(columns={col: "n_tool_calls"})


def cost_weighted(traj: pd.DataFrame) -> pd.DataFrame:
    """Burden-rank spend per case, split by whether the reference needed it.

    Ranks are ORDINAL (see tools.TOOL_COST_RANK): they are summed here only to
    order cases against each other, and the sum must not be read as "this agent
    spent 7 units of money". Reported alongside the raw per-tool counts so the
    ordinal claim is checkable.
    """
    import json
    from .tools import TOOL_COST_RANK

    def _as_list(v):
        if isinstance(v, list):
            return v
        if isinstance(v, str) and v.strip().startswith("["):
            return json.loads(v)
        return []

    rows = []
    for _, r in traj.iterrows():
        ts = _as_list(r["tools_called"])
        rows.append({
            "variant_id": r["variant_id"],
            "n_tool_calls": len(ts),
            "rank_sum": sum(TOOL_COST_RANK[t] for t in ts),
            "n_rank3_calls": sum(1 for t in ts if TOOL_COST_RANK[t] == 3),
            "called_empty_assay": any(
                t == "get_functional_assay" for t in ts) and
                r.get("gene") in ("MYH7", "MYBPC3", "PCSK9"),
        })
    return pd.DataFrame(rows)
