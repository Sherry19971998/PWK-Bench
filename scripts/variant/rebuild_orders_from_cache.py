#!/usr/bin/env python
"""
Rebuild a frozen-pool agent's per-variant acquisition orders from a released
prompt->completion cache, with NO model call, and write them in the
`run_agent.py --replay` format.

Every elicitation is looked up in the cache. A cache miss is an error (the
script never calls the API, and no API key is needed), so a successful rebuild
means the orders are exactly what the model returned during the original run.

Cache key formats. Current caches key on model, sampling profile, system
prompt, and user prompt. Caches written before the sampling profile was added
to the key (the released variance runs) key on model, system, and user only.
Both are tried.

Example (the paper's three frozen gpt-5.5 runs):
    for r in 0 1 2; do
      python scripts/variant/rebuild_orders_from_cache.py \
        --kind openai --model gpt-5.5-2026-04-23 --seed $((20260101 + r)) \
        --cache results/variant/variance/.cache_frontier_B_run$r.json \
        --out results/variant/variance/agent_orders/gpt55_run$r.csv
    done
    python scripts/variant/run_agent.py --replay results/variant/variance/agent_orders \
        --cohort data/sample/cohort_full_real.parquet --outdir results/variant/replay_gpt55
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))

import argparse, hashlib, json, os
import pandas as pd
from pwkbench.domains.base import load_real_cohort
from pwkbench.domains.variant import VARIANT_DOMAIN
from pwkbench.agents import build_agent
from pwkbench.agents.recorded import write_orders


class CacheMiss(RuntimeError):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["openai", "gemini", "anthropic"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, required=True,
                    help="seed of the original run (part of the current cache key)")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cohort", default="data/sample/cohort_full_real.parquet")
    ap.add_argument("--context", action="store_true",
                    help="the original run supplied variant context")
    args = ap.parse_args()

    # Adapters construct a client at init; a placeholder key keeps that offline.
    for k in ("OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        os.environ.setdefault(k, "offline-rebuild-no-network")

    cohort = load_real_cohort(pd.read_parquet(args.cohort), VARIANT_DOMAIN)
    with open(args.cache) as f:
        cache = json.load(f)
    agent = build_agent(args.kind, args.model, seed=args.seed, context=args.context)
    profile = agent._sampling_profile()

    def lookup(system, user):
        for raw in (f"{args.model}\x00{profile}\x00{system}\x00{user}",
                    f"{args.model}\x00{system}\x00{user}"):
            key = hashlib.sha256(raw.encode()).hexdigest()
            if key in cache:
                return cache[key]
        raise CacheMiss("prompt not in cache -- this cache does not match the "
                        "given model/seed/context; refusing to call the API")

    agent._complete_cached = lookup
    order = agent.order(cohort)
    free = agent.total_steps - agent.forced_steps
    eff = (agent.parse_failures - agent.forced_parse_failures) / free if free else 0.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_orders(args.out, cohort, order, args.model)
    print(f"wrote {args.out}: {len(order)} variants, "
          f"{len({tuple(r) for r in order})} distinct orders, "
          f"consequential parse-failure rate {eff:.3f}")


if __name__ == "__main__":
    main()
