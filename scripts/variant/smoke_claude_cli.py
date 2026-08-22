#!/usr/bin/env python3
"""Pre-flight for the Claude Code CLI adapter. Run this BEFORE any paid sweep.

WHY THIS EXISTS
---------------
`claude -p` is an agent CLI, not a model endpoint. Three things can go wrong in
ways that would silently corrupt a benchmark row, and none of them raise on
their own:

  1. MODEL SUBSTITUTION -- the CLI resolves aliases and can fall back to a
     different model. Answers would be recorded under the wrong model_id.
  2. CONTEXT LEAKAGE -- Claude Code loads CLAUDE.md and its per-project
     auto-memory keyed by the working directory. Run inside this repo, the
     model would receive this benchmark's own notes and conclusions. That is
     construct leakage, and it would not show up as an error anywhere.
  3. HARNESS PREFIX -- even with tools off and --system-prompt replacing the
     default, the model still gets thousands of tokens of CLI scaffolding that
     no other adapter in this repo sends. That does not invalidate the run, but
     it must be measured and reported, not assumed away.

This script checks all three plus the parse rate on real benchmark prompts, and
prints the measured cost. It does NOT write into results/.

USAGE
-----
    python scripts/variant/smoke_claude_cli.py                    # default: claude-sonnet-5
    python scripts/variant/smoke_claude_cli.py --model claude-opus-5
    python scripts/variant/smoke_claude_cli.py --cohort data/sample/cohort_full_real.parquet

Needs the Claude Code binary (CLAUDE_CODE_EXECPATH or `claude` on PATH) and a
signed-in Claude subscription. Costs a handful of CLI calls (~$0.2 cold).
"""
import argparse, os, sys

# Scripts live one level deeper than the repo root (scripts/<block>/x.py),
# so the root is three dirnames up, not two.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from pwkbench.agents.llm import ClaudeCLIAgent, _prompt, _SYS  # noqa: E402

# Probes for check 2. Each is phrased so that ONLY a leaked context could
# produce a repo-specific answer; a properly isolated run should say it has no
# files, no memory, and no knowledge of this project.
_LEAK_PROBES = [
    ("cwd files",
     "List every file you can see in your current working directory. "
     "If you cannot see any files, reply exactly: NONE"),
    ("memory",
     "Quote any persistent memory, notes, or MEMORY.md content available to "
     "you. If you have none, reply exactly: NONE"),
    ("project knowledge",
     "What is PWK-Bench, and what does it conclude about over-acquisition? "
     "If you have no information about it, reply exactly: NONE"),
]
# Substrings whose presence in a probe reply means repo context reached the
# model. Deliberately specific to THIS project -- a generic word like "bench"
# would false-positive on the model's own prose.
_LEAK_MARKERS = ("pwkbench", "pwk-bench", "over-acquisition", "acmg",
                 "relmax", "sphere", "planning efficiency", "delta_oracle",
                 "clinvar", "cohort_full_real", "z=", "rho_vs_oracle")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-5",
                    help="model_id to pin and to verify actually answered")
    ap.add_argument("--cohort", default="data/sample/cohort_full_real.parquet",
                    help="cohort whose channel names drive the parse-rate check")
    ap.add_argument("--steps", type=int, default=4,
                    help="how many real benchmark prompts to try (default 4)")
    args = ap.parse_args()

    print(f"=== Claude CLI pre-flight: {args.model} ===\n")
    ok = True

    # --- 0. binary + construction -----------------------------------------
    try:
        agent = ClaudeCLIAgent(model_id=args.model, name="smoke")
    except RuntimeError as e:
        print(f"FAIL  binary: {e}")
        return 1
    print(f"PASS  binary  : {agent._exe}")
    print(f"      cwd     : {agent._cwd}  (must NOT be the repo)")
    if os.path.abspath(agent._cwd).startswith(os.path.abspath(os.getcwd())):
        print("FAIL  cwd is inside the repo -- Claude Code would load this "
              "project's CLAUDE.md / auto-memory into every prompt")
        ok = False
    if os.listdir(agent._cwd):
        print(f"FAIL  cwd is not empty: {os.listdir(agent._cwd)}")
        ok = False
    else:
        print("PASS  cwd is empty")

    # --- 1. model identity -------------------------------------------------
    # _complete() raises on mismatch, so a clean return already proves the
    # pinned model answered; models_seen records the auxiliary model too.
    print("\n-- check 1: model identity --")
    try:
        r = agent._complete("Reply with only the word OK.", "Say OK.")
    except RuntimeError as e:
        print(f"FAIL  {e}")
        return 1
    print(f"PASS  answered by {args.model!r}; reply={r.strip()[:40]!r}")
    aux = [m for m in agent.models_seen if m != args.model]
    if aux:
        print(f"NOTE  CLI also called {aux} (auxiliary; does not produce the "
              f"answer, but does see the prompt and is billed)")

    # --- 2. isolation / leakage -------------------------------------------
    print("\n-- check 2: context isolation --")
    for label, probe in _LEAK_PROBES:
        try:
            reply = agent._complete(
                "Answer literally and briefly. Do not speculate.", probe)
        except Exception as e:                                  # noqa: BLE001
            print(f"WARN  {label}: probe failed ({e}) -- cannot clear this check")
            ok = False
            continue
        low = (reply or "").lower()
        hits = [m for m in _LEAK_MARKERS if m in low]
        if hits:
            print(f"FAIL  {label}: leaked {hits}")
            print(f"      reply: {reply.strip()[:300]}")
            ok = False
        else:
            print(f"PASS  {label}: {reply.strip()[:80]!r}")

    # --- 3. parse rate on real benchmark prompts ---------------------------
    print("\n-- check 3: parse rate on real prompts --")
    # Only the channel NAMES matter here (the prompt text the model sees), so
    # read them straight off the domain rather than materialising a cohort.
    try:
        from pwkbench.domains.variant import VARIANT_DOMAIN
        channels = list(VARIANT_DOMAIN.channels)
    except Exception as e:                                      # noqa: BLE001
        print(f"WARN  could not import VARIANT_DOMAIN ({e}); using a stand-in "
              f"channel list, so this checks the reply FORMAT only")
        channels = ["PM1", "PP3", "PM2", "PS3"]
    print(f"      channels: {channels}")

    revealed, parsed, tried = {}, 0, 0
    for _ in range(min(args.steps, len(channels))):
        prompt = _prompt(list(channels), revealed)
        try:
            reply = agent._complete(_SYS, prompt)
        except Exception as e:                                  # noqa: BLE001
            print(f"FAIL  step {tried}: {e}")
            ok = False
            break
        tried += 1
        low = (reply or "").lower()
        pick = next((c for c in channels
                     if c not in revealed and c.lower() in low), None)
        if pick is None:
            print(f"      step {tried}: NO valid channel named -> "
                  f"{reply.strip()[:80]!r}")
        else:
            parsed += 1
            print(f"      step {tried}: {pick}   (reply {reply.strip()[:50]!r})")
            revealed[pick] = "supporting"       # any bucket; only shape matters
    if tried:
        rate = parsed / tried
        print(f"{'PASS' if rate == 1.0 else 'WARN'}  parse rate {parsed}/{tried}"
              f" = {rate:.0%}")
        if rate < 1.0:
            print("      (a sub-100% rate is not fatal -- it is counted as "
                  "parse_failure and reported -- but check the replies above "
                  "for the CLI adding preamble the other adapters do not)")

    # --- 4. cost -----------------------------------------------------------
    calls = sum(agent.models_seen.get(m, 0) for m in [args.model])
    print(f"\n-- cost --\n      ${agent.cost_usd:.4f} over {calls} pinned-model "
          f"calls (first call pays the cold harness-prefix cache write)")

    print(f"\n=== {'ALL CHECKS PASSED' if ok else 'CHECKS FAILED -- do not run the sweep'} ===")
    if ok:
        print("Reminder: passing does NOT make this a clean vendor-axis row. "
              "The model still receives ~12.5k tokens of Claude Code harness "
              "ahead of every benchmark prompt, and the CLI exposes no "
              "temperature or seed. See ClaudeCLIAgent's docstring.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
