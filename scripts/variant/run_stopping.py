#!/usr/bin/env python3
"""Agent-chosen stopping: does the agent know when it has seen enough?

WHY THIS EXISTS
---------------
Every other experiment in this repo hands the agent a budget k and asks which
categories it acquires. That measures ORDERING. It cannot measure the other half
of an acquisition policy -- knowing when to stop -- because the harness's step
loop runs until every channel is acquired, so "stop" was never an action the
agent could take.

This script turns stopping into an agent decision (`allow_stop=True`): at each
step after its first acquisition the model may reply STOP instead of naming a
category. The agent's own `stop_k[i]` (categories acquired for variant i) is then
scored as a clinical endpoint.

WHY THE ACMG POINT RULE AND NOT `confidence_based_stopping`
-----------------------------------------------------------
The repo already has `metrics.confidence_based_stopping`, but its own docstring
disqualifies it as a headline: it min-max normalizes scores per budget over the
WHOLE cohort, so (a) `auc_at_stop` is not comparable across budgets and can
exceed the full-acquisition number as a pure scale artifact, and (b) the rule is
transductive -- a variant's stop depends on the other variants -- so it is a
retrospective diagnostic, not a policy anyone could deploy on one case.

`metrics.acmg_points_at_stop` has neither problem: ACMG/ClinGen points are summed
on an absolute, guideline-fixed scale with constant call thresholds, so a variant
stopped at k=1 and one stopped at k=4 are judged by the same ruler, and each
call depends only on that variant's own evidence. That is what makes an
agent-chosen stop scoreable at all.

READ THE OUTPUT THIS WAY
------------------------
The comparison of interest is the agent's row against the FIXED-BUDGET rows on
the SAME acquisition order: at mean_queries = q, does the agent resolve as many
variants (and call them as accurately) as simply spending ceil(q) queries on
every variant? If it does, stopping is adaptive and saves queries; if it lands
below the fixed budget nearest its own mean spend, the agent is stopping on the
wrong variants.

USAGE
-----
    export OPENAI_API_KEY=...
    python3 scripts/variant/run_stopping.py \
        --cohort data/sample/cohort_full_real.parquet \
        --config configs/models.yaml --models real \
        --slot frontier_B --outdir results/variant/stopping

    # cheap smoke first: add --max-variants 40
    # offline plumbing check with no API key: --models demo (mock agent has no
    #   STOP action, so it reports the no-stop baseline only)

Output: results/variant/stopping/stopping_<slot>.csv + a console table.
"""
import argparse, os, sys, copy
import numpy as np
import pandas as pd
import yaml

# Scripts live one level deeper than the repo root (scripts/<block>/x.py),
# so the root is three dirnames up, not two.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from pwkbench.domains.variant import VARIANT_DOMAIN
from pwkbench.domains.base import load_real_cohort
from pwkbench.agents import build_agent
from pwkbench.agents.llm import _STOP_WORDINGS
from pwkbench import metrics as M
from pwkbench import strategies as S


def _row(rule, res, mean_q):
    return {"rule": rule, "mean_queries": mean_q,
            "resolved_frac": res["resolved_frac"],
            "vus_frac": res["vus_frac"],
            "call_accuracy": res["call_accuracy"],
            "balanced_call_accuracy": res["balanced_call_accuracy"],
            "n_called": res["n_called"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--models", default="real")
    ap.add_argument("--slot", required=True)
    ap.add_argument("--max-variants", type=int, default=None)
    ap.add_argument("--outdir", default="results/variant/stopping")
    ap.add_argument("--stop-wording", default="default",
                    choices=sorted(_STOP_WORDINGS),
                    help="phrasing of the STOP option. Control for whether the "
                         "observed stop points are the agent's judgement or an "
                         "artifact of how the option is worded.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = yaml.safe_load(open(args.config))
    rows = [m for m in cfg.get(args.models, []) if m.get("slot") == args.slot]
    if not rows:
        raise SystemExit(f"slot '{args.slot}' not found under '{args.models}:' "
                         f"in {args.config}")
    spec = copy.deepcopy(rows[0])

    df = pd.read_parquet(args.cohort)
    if args.max_variants:
        # Gene-stratified subsample. Written as an explicit concat rather than
        # `groupby(...).apply(...)`: that form is deprecated in pandas and, once
        # the deprecation lands, the grouping column is dropped from the result --
        # here that silently removes `gene`, which `load_real_cohort` requires, so
        # the failure would surface far from its cause. Verified to produce
        # row-identical output to the old form at several --max-variants values.
        df = pd.concat([
            g.sample(max(1, round(len(g) * args.max_variants / len(df))),
                     random_state=0)
            for _, g in df.groupby("gene")])
    cohort = load_real_cohort(df, VARIANT_DOMAIN)
    K = cohort.domain.K

    kind = spec.get("kind", "mock")
    agent = build_agent(
        kind=kind, model_id=spec.get("model_id", "unspecified"),
        name=spec.get("slot"), context=bool(spec.get("context", False)),
        seed=int(spec.get("seed", 0)), max_workers=int(spec.get("workers", 1)),
        cache_path=spec.get("cache_path"),
        allow_stop=(kind != "mock"),     # MockAgent has no step loop to stop in
        stop_wording=args.stop_wording)

    print(f"[stopping] slot={args.slot} kind={kind} "
          f"model_id={spec.get('model_id')} n={len(cohort)} K={K} "
          f"stop_wording={args.stop_wording}", flush=True)
    order = agent.order(cohort)
    stop_k = getattr(agent, "stop_k", None)
    if stop_k is None:                    # mock path: no STOP action exists
        stop_k = np.full(len(cohort), K, int)

    out = []
    # 1) the agent's own stopping decision
    at_stop = M.acmg_points_at_stop(cohort, order, stop_k)
    out.append(_row(f"agent_stop:{args.slot}", at_stop, at_stop["mean_queries"]))
    # 2) the same agent order under every FIXED budget (the like-for-like control:
    #    same ordering, only the stopping rule differs)
    fixed = M.acmg_points_curve(cohort, order, budgets=list(range(1, K + 1)))
    for k in range(1, K + 1):
        out.append(_row(f"agent_fixed_k{k}", fixed[k], float(k)))
    # 3) the label-free reference ordering under the same fixed budgets
    rel = S.relmax_order(cohort)
    relc = M.acmg_points_curve(cohort, rel, budgets=list(range(1, K + 1)))
    for k in range(1, K + 1):
        out.append(_row(f"relmax_fixed_k{k}", relc[k], float(k)))

    res = pd.DataFrame(out)
    res["slot"] = args.slot
    res["model_id"] = spec.get("model_id")
    res["stop_requests"] = getattr(agent, "stop_requests", 0)
    res["parse_failures"] = getattr(agent, "parse_failures", 0)
    # Split FORCED from CONSEQUENTIAL fallbacks. A forced failure happens with a
    # single channel left, where the fallback picks the only admissible answer
    # and cannot bias anything; a consequential one fabricates a choice the model
    # did not make. Only the latter threatens the stopping result, so reporting
    # the raw rate alone would leave a reader unable to tell a benign
    # final-step quirk from real contamination.
    res["forced_parse_failures"] = getattr(agent, "forced_parse_failures", 0)
    res["consequential_parse_failures"] = (
        getattr(agent, "parse_failures", 0)
        - getattr(agent, "forced_parse_failures", 0))
    res["forced_steps"] = getattr(agent, "forced_steps", 0)
    res["total_steps"] = getattr(agent, "total_steps", 0)
    # NEVER overwrite a previous run. gpt-5.x samples at temperature=1, so every
    # invocation is an independent draw and clobbering the last one destroys
    # evidence that cannot be regenerated (the same prompts re-queried later are
    # a NEW sample, not a recovery of the old one). Each run lands in its own
    # indexed file; `stopping_<slot>_all.csv` is rebuilt from whatever indexed
    # files exist, so it is always a faithful concatenation rather than an
    # append-log that could double-count on a re-run.
    # Each wording arm keeps its own file series so arms never mix. The default
    # arm keeps the unsuffixed name the first runs already wrote, so previously
    # reported paths stay valid and its run counter continues rather than
    # restarting.
    tag = "" if args.stop_wording == "default" else f"_{args.stop_wording}"
    stem = os.path.join(args.outdir, f"stopping_{args.slot}{tag}")
    idx = 0
    while os.path.exists(f"{stem}_run{idx}.csv"):
        idx += 1
    res["run"] = idx
    res["stop_wording"] = args.stop_wording
    out_csv = f"{stem}_run{idx}.csv"
    res.to_csv(out_csv, index=False)

    parts = []
    for j in range(idx + 1):
        p = f"{stem}_run{j}.csv"
        if os.path.exists(p):
            parts.append(pd.read_csv(p))
    all_csv = f"{stem}_all.csv"
    pd.concat(parts, ignore_index=True).to_csv(all_csv, index=False)

    print(f"\n=== agent-chosen stopping ({args.slot}) ===")
    print(res[["rule", "mean_queries", "resolved_frac", "call_accuracy",
               "n_called"]].to_string(index=False,
                                      float_format=lambda v: f"{v:.4f}"))
    uniq, cnt = np.unique(stop_k, return_counts=True)
    print("\nstop_k distribution:",
          {int(u): int(c) for u, c in zip(uniq, cnt)})
    pf = getattr(agent, "parse_failures", 0)
    fpf = getattr(agent, "forced_parse_failures", 0)
    steps = getattr(agent, "total_steps", 0) or 1
    print(f"STOP replies: {getattr(agent, 'stop_requests', 0)} / {steps} steps")
    print(f"parse failures: {pf} ({100*pf/steps:.1f}% of steps) = "
          f"{fpf} forced (only one channel left -> cannot bias) + "
          f"{pf-fpf} CONSEQUENTIAL ({100*(pf-fpf)/steps:.1f}% of steps)")
    print("written:", out_csv, f"(+ combined: {all_csv})")


if __name__ == "__main__":
    main()
