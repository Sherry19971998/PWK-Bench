#!/usr/bin/env python3
"""Run the live tool-using agent over the real cohort and record its trajectories.

    export OPENAI_API_KEY=...
    python3 scripts/live/run_live.py --cohort data/sample/cohort_full_real.parquet \
        --model gpt-5.5-2026-04-23 --max-variants 8 --outdir results/live

    # or, same protocol, a different vendor (see --vendor below):
    export GEMINI_API_KEY=...     && python3 scripts/live/run_live.py --vendor gemini    --tag gemini
    export ANTHROPIC_API_KEY=...  && python3 scripts/live/run_live.py --vendor anthropic --tag anthropic

Writes `trajectories.csv` (one row per case: which tools, in what order, where it
stopped, what it answered) and `tool_calls.csv` (the raw call log).

READ THE ACCURACY COLUMN WITH THE CEILING IN MIND
--------------------------------------------------
The agent is shown the real HGVS name, and this repo has already measured that
the name alone yields closed-book AUC 0.865 against 0.942 with all evidence
(`results/variant/real_trusted/memorization_probe.csv`). Accuracy here is therefore
mostly recall, and the tool layer has ~0.077 of room to move it. The intended
headline is the OVER-ACQUISITION distribution -- how many tools an agent that
already knows the answer still chooses to order -- which is a behavioural
measurement and does not inherit that ceiling. `--no-tools` runs the closed-book
arm on the same cases so the gap is measured rather than assumed.
"""
import argparse, json, os, sys, threading, time
import pandas as pd

# Scripts live one level deeper than the repo root (scripts/<block>/x.py),
# so the root is three dirnames up, not two.
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from pwkbench.live.agent import LiveAgent, GeminiLiveAgent, AnthropicLiveAgent
from pwkbench.live.tools import ToolCache, TOOLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/sample/cohort_full_real.parquet")
    # No cross-vendor default: `--vendor gemini` with OpenAI's default model id
    # would send "gpt-5.5-..." to Gemini and fail with a confusing 404, or worse
    # be silently recorded as the wrong model. Resolved after parsing, per vendor.
    ap.add_argument("--model", default=None,
                    help="model id. Defaults per vendor: gpt-5.5-2026-04-23 "
                         "(openai) / gemini-2.5-pro (gemini).")
    ap.add_argument("--outdir", default="results/live")
    ap.add_argument("--max-variants", type=int, default=None,
                    help="gene-stratified subsample for a cheap smoke run")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-tools", action="store_true",
                    help="closed-book arm: withhold every evidence tool, so the "
                         "agent can only call final_answer. Measures how much of "
                         "the accuracy is recall rather than acquisition.")
    ap.add_argument("--ablate", default=None,
                    help="comma-separated tools to WITHHOLD (tool-ablation arm)")
    ap.add_argument("--tool-order", default=None,
                    help="permute the order tools are PRESENTED in: 'reverse', "
                         "or an explicit comma-separated list. Tests whether "
                         "'population frequency is always the first query' is a "
                         "judgement or an artefact of it being listed first. "
                         "final_answer stays last either way.")
    ap.add_argument("--tag", default=None, help="suffix for the output files")
    ap.add_argument("--vendor", choices=["openai", "gemini", "anthropic"],
                    default="openai",
                    help="which agent loop to drive. 'openai' is the arm every "
                         "existing results/live/* file was produced with and is "
                         "the default so those stay reproducible; 'gemini' runs "
                         "the same tools/prompt/stopping rule over Gemini's "
                         "function-calling protocol (needs GEMINI_API_KEY); "
                         "'anthropic' runs it over the Messages API directly "
                         "(needs ANTHROPIC_API_KEY) -- this is the API-mode "
                         "equivalent of run_claude_code_live.py's CLI-based "
                         "arm, with no Claude Code CLI/subscription dependency "
                         "and no harness-prompt scaffolding, so it IS a clean "
                         "vendor-axis control (see AnthropicLiveAgent's "
                         "docstring). Set --tag so different vendors' outputs "
                         "do not overwrite each other -- the filenames key on "
                         "the arm, not the model.")
    args = ap.parse_args()
    # gemini-2.5-pro is the Gemini default because it is the only STABLE
    # (non-preview) pro-tier Gemini, i.e. the closest available capability match
    # to gpt-5.5. The 3.x pro models are all preview and can change under a
    # submitted paper; gemini-pro-latest is a moving alias. It has NO free tier
    # (measured 2026-08-02: 429 on the first call, quotaId
    # GenerateContentInputTokensPerModelPerDay-FreeTier), so this default
    # requires a billed project.
    if args.model is None:
        args.model = {"gemini": "gemini-2.5-pro",
                      "anthropic": "claude-sonnet-5"}.get(
                          args.vendor, "gpt-5.5-2026-04-23")

    os.makedirs(args.outdir, exist_ok=True)
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
    df = df.reset_index(drop=True)

    if args.no_tools:
        allow = ()
    elif args.ablate:
        drop = {t.strip() for t in args.ablate.split(",")}
        unknown = drop - set(TOOLS)
        if unknown:
            raise SystemExit(f"unknown tool(s) to ablate: {sorted(unknown)}")
        allow = tuple(t for t in TOOLS if t not in drop)
    else:
        allow = tuple(TOOLS)

    # The tool cache is shared across vendors ON PURPOSE: the evidence payloads
    # are external facts, not model output, so both arms must see byte-identical
    # tool results or a "vendor difference" could just be a gnomAD/UniProt fetch
    # that changed between the two runs.
    cache = ToolCache(os.path.join(args.outdir, ".tool_cache.json"))
    cls = {"gemini": GeminiLiveAgent,
           "anthropic": AnthropicLiveAgent}.get(args.vendor, LiveAgent)
    agent = cls(args.model, cache, max_steps=args.max_steps,
                allow_tools=allow, tool_order=args.tool_order)

    tag = args.tag or ("notools" if args.no_tools else
                       ("ablate_" + args.ablate.replace(",", "_") if args.ablate
                        else ("order_" + args.tool_order.replace(",", "_")
                              if args.tool_order else "full")))
    print(f"[live] vendor={args.vendor} model={args.model} cases={len(df)} tools={list(allow) or 'NONE'}"
          f" tag={tag}", flush=True)

    # Incremental checkpoint. A full sweep is ~30-60 paid minutes; writing only
    # at the end means one exception at case 400 throws away every dialogue that
    # was already paid for. Completed trajectories are appended to a JSONL as
    # they land, and a rerun skips whatever is already in it.
    ckpt = os.path.join(args.outdir, f"_ckpt_{tag}.jsonl")
    done = {}
    if os.path.exists(ckpt):
        with open(ckpt) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    done[d["variant_id"]] = d
                except json.JSONDecodeError:
                    pass          # a torn final line from a hard kill
        if done:
            print(f"  resuming: {len(done)} case(s) already in {ckpt}", flush=True)

    ck_lock = threading.Lock()

    def _one(r):
        """Never let a single case abort the sweep: an API error on one variant
        must cost that variant, not the 400 dialogues already paid for."""
        if r.variant_id in done:
            return done[r.variant_id]
        try:
            res = agent.run_one(r.variant_id, r.gene, r.clinvar_name)
        except Exception as e:                     # noqa: BLE001
            res = {"variant_id": r.variant_id, "gene": r.gene,
                   "clinvar_name": r.clinvar_name, "n_tools_called": 0,
                   "tools_called": [], "distinct_tools": [], "cost_rank_sum": 0,
                   "answer": None, "confidence": None,
                   "stop_reason": f"error:{type(e).__name__}",
                   "allowed_tools": list(allow)}
        with ck_lock:
            with open(ckpt, "a") as fh:
                fh.write(json.dumps(res) + "\n")
        return res

    rows, t0 = [], time.time()
    todo = list(df.itertuples())
    if args.workers <= 1:
        for i, r in enumerate(todo, 1):
            rows.append(_one(r))
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_one, r) for r in todo]
            for i, f in enumerate(futs, 1):
                rows.append(f.result())
                if i % 25 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)

    out = pd.DataFrame(rows)
    n_err = int(out.stop_reason.astype(str).str.startswith("error:").sum())
    if n_err:
        print(f"  WARNING: {n_err} case(s) errored and carry no answer; they are "
              f"kept as rows so the denominator stays honest.")
    lab = df.set_index("variant_id").label
    out["label"] = out.variant_id.map(lab)
    out["predicted"] = out.answer.map({"Pathogenic": 1, "Benign": 0})
    out["correct"] = (out.predicted == out.label)
    out["tools_called"] = out.tools_called.map(json.dumps)
    out["distinct_tools"] = out.distinct_tools.map(json.dumps)
    out["allowed_tools"] = out.allowed_tools.map(json.dumps)
    out["model_id"], out["arm"] = args.model, tag
    path = os.path.join(args.outdir, f"trajectories_{tag}.csv")
    out.to_csv(path, index=False)
    pd.DataFrame(cache.calls).to_csv(
        os.path.join(args.outdir, f"tool_calls_{tag}.csv"), index=False)

    el = time.time() - t0
    n_ans = int(out.predicted.notna().sum())
    print(f"\n=== {tag} ({len(out)} cases, {el:.0f}s) ===")
    print(f"  answered            {n_ans}/{len(out)}")
    if n_ans:
        # Accuracy over ANSWERED cases. `correct` is False for an unanswered
        # case (NaN != label), so averaging over every row would silently charge
        # API errors to the model's accuracy instead of reporting them as the
        # separate failure they are.
        ans = out[out.predicted.notna()]
        print(f"  accuracy (answered) {ans.correct.mean():.3f}")
        if n_ans < len(out):
            print(f"  accuracy (all rows) {out.correct.mean():.3f}  "
                  f"<- unanswered counted as wrong")
    print(f"  tools called/case   mean {out.n_tools_called.mean():.2f}  "
          f"dist {out.n_tools_called.value_counts().sort_index().to_dict()}")
    print(f"  stop_reason         {out.stop_reason.value_counts().to_dict()}")
    print(f"  tokens              in={agent.usage['prompt_tokens']} "
          f"out={agent.usage['completion_tokens']}")
    print(f"  per case            in={agent.usage['prompt_tokens']/max(len(out),1):.0f} "
          f"out={agent.usage['completion_tokens']/max(len(out),1):.0f}")
    print("written:", path)


if __name__ == "__main__":
    main()
