#!/usr/bin/env python3
"""Closed-book arm run via the Claude Code CLI (claude-sonnet-5), not the
Anthropic API. `--tools ""` disables every built-in tool, so this is a clean
text-in/text-out closed-book call directly comparable in spirit to
results/live/trajectories_notools.csv (gpt-5.5) -- same system prompt shape,
same "Variant / Gene" user message, same JSON classification+confidence
output. Cost is read directly from the CLI's own `total_cost_usd` per call
(no token-rate guessing needed, unlike the OpenAI/Gemini arms).

CAVEAT stated once here rather than per-row: this is a different trust
boundary from the API-based arms even with zero tools -- Claude Code's own
harness (memory, hooks, model routing) sits around the model call. Report
this as "Claude Sonnet 5 via Claude Code CLI, closed-book", not as
interchangeable with an "Anthropic API" arm.

USAGE
    python3 scripts/live/run_claude_code_notools.py \
        --cohort data/masked/cohort_full_real_masked_coords.parquet \
        --outdir results/live/claude_code --tag masked_coords_notools
"""
import argparse, json, os, subprocess, sys, threading
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLAUDE_BIN = os.environ.get("CLAUDE_CODE_EXECPATH", "claude")

SYSTEM = ("You are a clinical variant scientist. Classify the variant as "
          "Pathogenic or Benign under ACMG/AMP criteria. No evidence tools "
          "are available; answer from the variant identifier and gene alone.")
SCHEMA = json.dumps({"type": "object", "properties": {
    "classification": {"type": "string", "enum": ["Pathogenic", "Benign"]},
    "confidence": {"type": "number"}}, "required": ["classification", "confidence"]})


def classify_one(variant_id: str, gene: str, model: str) -> dict:
    user = f"Variant: {variant_id}\nGene: {gene}\n\nClassify this variant as Pathogenic or Benign."
    cmd = [CLAUDE_BIN, "-p", "--model", model, "--tools", "",
           "--output-format", "json", "--no-session-persistence",
           "--system-prompt", SYSTEM, "--json-schema", SCHEMA, user]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return {"variant_id": variant_id, "gene": gene, "predicted": None,
                "confidence": None, "cost_usd": 0.0, "stop_reason": "error:nonzero_exit",
                "raw_stderr": proc.stderr[:2000]}
    d = json.loads(proc.stdout)
    so = d.get("structured_output") or {}
    cls = so.get("classification")
    return {"variant_id": variant_id, "gene": gene,
            "predicted": {"Pathogenic": 1, "Benign": 0}.get(cls),
            "confidence": so.get("confidence"),
            "cost_usd": d.get("total_cost_usd", 0.0),
            "stop_reason": "final_answer" if cls else "error:no_structured_output"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/masked/cohort_full_real_masked_coords.parquet")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--outdir", default="results/live/claude_code")
    ap.add_argument("--tag", default="masked_coords_notools")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    os.chdir(_ROOT)
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_parquet(args.cohort)[["variant_id", "gene", "label"]].reset_index(drop=True)
    ckpt = os.path.join(args.outdir, f"_ckpt_{args.tag}.jsonl")
    done = {}
    if os.path.exists(ckpt):
        with open(ckpt) as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                    done[d["variant_id"]] = d
                except json.JSONDecodeError:
                    pass
        if done:
            print(f"resuming: {len(done)} case(s) already in {ckpt}")

    lock = threading.Lock()

    def _one(row):
        if row.variant_id in done:
            return done[row.variant_id]
        try:
            res = classify_one(row.variant_id, row.gene, args.model)
        except Exception as e:
            res = {"variant_id": row.variant_id, "gene": row.gene, "predicted": None,
                   "confidence": None, "cost_usd": 0.0,
                   "stop_reason": f"error:{type(e).__name__}"}
        with lock:
            with open(ckpt, "a") as fh:
                fh.write(json.dumps(res) + "\n")
        return res

    rows = []
    todo = list(df.itertuples())
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_one, r) for r in todo]
        for i, f in enumerate(futs, 1):
            rows.append(f.result())
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    out = pd.DataFrame(rows).merge(df[["variant_id", "label"]], on="variant_id")
    out["correct"] = out["predicted"] == out["label"]
    out["model_id"] = args.model
    out["arm"] = args.tag
    path = os.path.join(args.outdir, f"trajectories_{args.tag}.csv")
    out.to_csv(path, index=False)

    n_ans = int(out["predicted"].notna().sum())
    total_cost = out["cost_usd"].sum()
    print(f"\n=== {args.tag} ({len(out)} cases) ===")
    print(f"  answered            {n_ans}/{len(out)}")
    if n_ans:
        ans = out[out["predicted"].notna()]
        print(f"  accuracy (answered) {ans['correct'].mean():.3f}")
    print(f"  stop_reason         {out['stop_reason'].value_counts().to_dict()}")
    print(f"  total cost          ${total_cost:.2f}")
    print("written:", path)


if __name__ == "__main__":
    main()
