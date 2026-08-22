"""E5 -- reference stopping rule for the live agent.

WHY THIS IS "NO NEW API CALLS", AND WHAT IT IS NOT
---------------------------------------------------
The experiment plan asks for a stopping rule driven by "the live agent's own
per-step posterior" -- run_live.py does not log a posterior at every step
(only the final `confidence`), so that literal version needs a re-run that
logs per-step state. What CAN be built from what is already recorded
(`tools_called`, in the order the agent actually called them) is a
REFERENCE rule: replay each case's own acquisition order and ask, after each
tool, whether the guideline's own ACMG-points classifier
(`pwkbench.metrics._acmg_channel_points`, the SAME reference used for
`sufficiency_points` and every other ACMG-points number in this repo) has
already reached a definitive (non-VUS) call. This is a principled,
guideline-anchored stopping rule, not the model's own subjective belief --
state it that way, do not call it "the agent's calibrated posterior".

It answers the question E5 is really after: given the EVIDENCE THE AGENT
ITSELF CHOSE TO ACQUIRE, in the order it acquired it, would a rule with no
free parameters have stopped earlier at no worse accuracy? A "yes" says the
over-acquisition gap is a STOPPING-POLICY deficiency (the agent keeps
querying evidence it already has enough of); a "no" would say the rule offers
nothing the agent isn't already doing.
"""
from __future__ import annotations
import json

import numpy as np
import pandas as pd

from ..metrics import _acmg_channel_points, _ACMG_PATH_MIN, _ACMG_BEN_MAX
from .metrics_live import TOOL_TO_CHANNEL


def _call_from_points(total, path_min=_ACMG_PATH_MIN, ben_max=_ACMG_BEN_MAX):
    if total >= path_min:
        return 1
    if total <= ben_max:
        return 0
    return -1


def _as_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str) and v.strip().startswith("["):
        return json.loads(v)
    return []


def reference_stop(cohort, traj: pd.DataFrame) -> pd.DataFrame:
    """One row per (case, run) in `traj`: where the reference rule would have
    stopped vs where the agent actually stopped, replaying the SAME order the
    agent used. PVS1 is free (per metrics_live's ACQUIRABLE/FREE split) so it
    always contributes before the first tool call.
    """
    chpts = _acmg_channel_points(cohort)
    idx = {vid: i for i, vid in enumerate(cohort.df["variant_id"])}
    y = np.asarray(cohort.y, dtype=int)

    rows = []
    for _, r in traj.iterrows():
        vid = r["variant_id"]
        if vid not in idx:
            continue
        i = idx[vid]
        mapped_order, seen = [], set()
        for t in _as_list(r["tools_called"]):
            ch = TOOL_TO_CHANNEL.get(t)
            if ch and ch not in seen:
                mapped_order.append(ch)
                seen.add(ch)

        total = float(chpts["PVS1"][i])
        rule_n, rule_call = np.nan, -1
        for n, ch in enumerate(mapped_order, start=1):
            total += float(chpts[ch][i])
            c = _call_from_points(total)
            if c != -1:
                rule_n, rule_call = n, c
                break

        # bool(...) is required, not cosmetic: `rule_call == y[i]` compares a
        # Python int to a numpy int64 and yields numpy.bool_, and numpy.bool_
        # values stored in an object-dtype column SUM VIA LOGICAL OR, not
        # arithmetic addition -- pandas .mean() over such a column silently
        # returns ~1/n instead of the true fraction. Verified: a column of 428
        # numpy.bool_(True) gave .mean() == 0.0023, not 1.0. Native Python
        # bool does not have this failure mode.
        rows.append(dict(
            variant_id=vid, label=int(y[i]),
            rule_n_tools=rule_n, rule_call=rule_call,
            rule_resolved=bool(rule_call != -1),
            rule_correct=bool(rule_call == y[i]) if rule_call != -1 else np.nan,
            agent_n_tools=len(mapped_order),
            agent_correct=bool(r["correct"]) if "correct" in r else np.nan,
        ))
    return pd.DataFrame(rows)
