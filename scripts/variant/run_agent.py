#!/usr/bin/env python
"""
Paper entry point for RQ2 (agent order-optimality): run the tool-augmented LLM
agent(s) through the fixed frozen-pool harness and score their acquisition
orders. Thin wrapper over scripts/variant/run_benchmark.py.

Three modes:

  --replay PATH   Re-score RECORDED agent orders offline (no model, no API
                  call). PATH is a CSV or a directory of CSVs as written by a
                  real run to <outdir>/agent_orders/<slot>.csv. Needs --cohort
                  (the same cohort the orders were recorded on). Every cohort
                  variant must have a recorded order, or the run fails -- it
                  never falls back to mock or fixed orders.
  --models KEY    Run real models from configs/models.yaml (needs API keys).
                  The run also writes <outdir>/agent_orders/ so it can be
                  replayed later.
  --demo          Synthetic toy cohort + mock agents. Pipeline check only;
                  NOT a replay of any real model run.

The paper's three frozen gpt-5.5 runs and three frozen gemini-2.5-pro runs
(no context) are released as order files, rebuilt offline from their prompt
caches (scripts/variant/rebuild_orders_from_cache.py):
results/variant/variance/agent_orders/ and
results/variant/gemini_pro/variance_gemini_noctx/agent_orders/.

Usage:
    python scripts/variant/run_agent.py --replay results/variant/variance/agent_orders \
        --cohort data/sample/cohort_full_real.parquet --outdir results/variant/replay
    python scripts/variant/run_agent.py --models real_all_api \
        --cohort data/sample/cohort_full_real.parquet
    python scripts/variant/run_agent.py --demo
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import run_benchmark

if __name__ == "__main__":
    argv = sys.argv
    if "--replay" in argv:
        i = argv.index("--replay")
        if i + 1 >= len(argv) or argv[i + 1].startswith("--"):
            sys.exit("run_agent.py: --replay needs a path to recorded agent orders "
                     "(<outdir>/agent_orders/ from a real run). For the synthetic "
                     "pipeline check use --demo instead.")
        if "--cohort" not in argv:
            sys.exit("run_agent.py: --replay needs --cohort <parquet> (the cohort "
                     "the orders were recorded on).")
        argv[i] = "--replay-orders"
    elif not any(a in argv for a in ("--demo", "--cohort")):
        sys.exit("run_agent.py: pass --replay PATH --cohort ..., --cohort ... "
                 "--models KEY, or --demo. See --help / module docstring.")
    run_benchmark.main()
