#!/usr/bin/env python
"""
Resumable cache warmer for a real LLM agent stuck behind a provider's DAILY
free-tier request cap (e.g. Gemini's Google AI Studio free tier: quotaId
GenerateRequestsPerDayPerProjectPerModel-FreeTier, 20 requests/day).

The problem this solves: a full cohort pass needs ~90 distinct prompts (491
variants x 4 steps, collapsed by the prompt cache), but a 20/day free-tier cap
cannot cover that in one run, and a plain re-run starts from an EMPTY in-memory
cache (pwkbench/agents/llm.py's `_StepwiseLLMAgent._cache` does not otherwise
persist across process restarts) -- so without this script, each new attempt
would re-ask prompts already answered on a prior day and never converge.

This script builds the agent with its models.yaml `cache_path` set, so already-
answered prompts load from disk and are never re-asked, then calls `.order()`
until either it finishes (cache complete) or the provider's daily cap raises
`DailyQuotaExhausted` -- caught here, reported, and exited 0 (not an error) so
a cron/launchd job invoking this daily does not alert on an expected stop.

Usage (run once a day until it reports COMPLETE):
    python scripts/variant/warm_llm_cache.py --models real --slot frontier_C \
        --cohort data/sample/cohort_full_real.parquet

Once COMPLETE, the row's cache_path holds every prompt the real run needs, so
`python scripts/variant/run_benchmark.py --cohort <same file> --models real` (or
--robustness) reuses it directly -- a 100% cache hit, no further API calls,
exactly like the already-finished OpenAI rows in this matrix.
"""

import os as _os, sys as _sys
# Make the package importable when the script is run directly from a
# checkout (scripts/<block>/x.py -> repo root is three dirnames up).
# Without this the script only works under `PYTHONPATH=.` or an
# installed package, which silently looks like the layout is broken.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))

import argparse
import sys
import pandas as pd
import yaml
from pwkbench.agents import build_agent
from pwkbench.agents.llm import DailyQuotaExhausted
from pwkbench.domains.base import load_real_cohort
from pwkbench.domains.variant import VARIANT_DOMAIN


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", required=True, help="real cohort parquet")
    ap.add_argument("--models", default="real", help="key in --config")
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--slot", default=None,
                    help="only warm this slot (default: every row in the "
                         "matrix that has a cache_path set)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.models not in cfg:
        sys.exit(f"'{args.models}' is not a key in {args.config}. "
                 f"Available: {', '.join(cfg)}")
    matrix = cfg[args.models]
    rows = [m for m in matrix if m.get("cache_path")
            and (args.slot is None or m.get("slot") == args.slot)]
    if not rows:
        sys.exit(f"no row in '{args.models}' has cache_path set"
                 + (f" for slot '{args.slot}'" if args.slot else "")
                 + " -- nothing to warm (this script is only for resumable, "
                   "quota-capped real agents; a row without cache_path should "
                   "just be run directly via run_benchmark.py).")

    cohort = load_real_cohort(pd.read_parquet(args.cohort), VARIANT_DOMAIN)

    any_incomplete = False
    for m in rows:
        slot = m.get("slot", m["kind"])
        agent = build_agent(kind=m["kind"], model_id=m.get("model_id", ""),
                            name=slot, adherence=m.get("adherence", 0.85),
                            seed=m.get("seed", 0), context=m.get("context", False),
                            max_workers=m.get("workers", 1),
                            cache_path=m["cache_path"])
        before = len(agent._cache)
        print(f"[{slot}] {m.get('model_id')}: {before} distinct prompts cached "
              f"so far ({m['cache_path']})")
        try:
            agent.order(cohort)
        except DailyQuotaExhausted as e:
            after = len(agent._cache)
            any_incomplete = True
            print(f"[{slot}] daily quota hit after caching {after - before} new "
                  f"prompt(s) today (total {after}). Re-run this script "
                  f"tomorrow to continue. ({e})")
            continue
        after = len(agent._cache)
        print(f"[{slot}] COMPLETE -- {after} distinct prompts cached, "
              f"{after - before} new today. Run scripts/variant/run_benchmark.py "
              f"--cohort {args.cohort} --models {args.models} normally now "
              "(this row will be a 100% cache hit, no further API calls).")

    if any_incomplete:
        print("\nnot all rows finished today -- re-run this script again "
              "tomorrow (or after billing/quota changes) to keep going.")


if __name__ == "__main__":
    main()
