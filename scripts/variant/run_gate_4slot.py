#!/usr/bin/env python
"""
Paper entry point for RQ1 (Table tab:main + Figure fig:gate): the four-slot
budget--accuracy gate. This is the script named in the paper's Experiments
section ("running run_gate_4slot.py ... regenerates every number below").

It is a thin wrapper over scripts/variant/run_benchmark.py so the repo has exactly the
entry points the paper cites. It runs the offline synthetic demo by default
(no API key, no downloads) and writes results/<domain>/results.csv +
figures/<domain>/{budget_curve,single_slot,...}.png.

Usage (identical flags to run_benchmark.py):
    python scripts/variant/run_gate_4slot.py --demo --domain variant
    python scripts/variant/run_gate_4slot.py --cohort data/real/cohort_full.parquet --models real
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import run_benchmark

if __name__ == "__main__":
    # default to the offline demo if no data/model flags were given
    if not any(a in sys.argv for a in ("--demo", "--cohort")):
        sys.argv.append("--demo")
    run_benchmark.main()
