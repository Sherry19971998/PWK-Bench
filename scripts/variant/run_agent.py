#!/usr/bin/env python
"""
Paper entry point for RQ2 (agent order-optimality): run the tool-augmented LLM
agent(s) through the fixed harness and score their acquisition orders. This is
the script the paper cites as "run_agent.py --replay".

--replay reproduces the committed agent trajectories OFFLINE (the demo mock
matrix, no API call), matching the paper's claim that released trajectories
re-derive every number with no live model endpoint. Without --replay it runs
the real model matrix from configs/models.yaml (needs API keys).

Thin wrapper over scripts/variant/run_benchmark.py so the repo carries the exact entry
points the paper names.

Usage:
    python scripts/variant/run_agent.py --replay --domain variant     # offline
    python scripts/variant/run_agent.py --models real --cohort ...     # live models
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import run_benchmark

if __name__ == "__main__":
    argv = sys.argv
    if "--replay" in argv:
        argv.remove("--replay")                       # replay == offline demo mocks
        if "--demo" not in argv:
            argv.append("--demo")
    if not any(a in argv for a in ("--demo", "--cohort")):
        argv.append("--demo")
    run_benchmark.main()
