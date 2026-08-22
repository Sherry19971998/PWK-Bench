#!/usr/bin/env python3
"""Live (4-evidence-tool) arm run via the Claude Code CLI + a hand-rolled MCP
server (pwkbench/live/claude_mcp_server.py), matching LiveAgent's protocol
(same system prompt, same four tools + final_answer, same shared tool
cache) so this is directly comparable to results/live/trajectories_full.csv
(gpt-5.5) and the Gemini arm.

REQUIRES: --dangerously-skip-permissions, because in `-p` (non-interactive)
mode there is no user to approve MCP tool calls and they are denied by
default (verified: `permission_denials` populated, model fell back to
closed-book "prior knowledge" and said so in its own answer, without this
flag). Every BUILT-IN Claude Code tool is separately hard-blocked via
--disallowed-tools, so skipping permission checks only affects the four
MCP evidence tools this script itself defines.

Each case gets its own MCP server SUBPROCESS (env-scoped to that one
variant -- see claude_mcp_server.py) and its own tool-call log file, which
is read back after the CLI call to reconstruct `tools_called` order (the
CLI's own --output-format json result has no tool-call transcript).

USAGE
    python3 scripts/live/run_claude_code_live.py \
        --cohort data/sample/cohort_full_real.parquet \
        --outdir results/live/claude_code --tag full --workers 4
"""
import argparse, json, os, subprocess, sys, threading, uuid
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLAUDE_BIN = os.environ.get("CLAUDE_CODE_EXECPATH", "claude")
MCP_SERVER = os.path.join(_ROOT, "pwkbench", "live", "claude_mcp_server.py")

SYSTEM = (
    "You are a clinical variant scientist. Classify the variant as Pathogenic or "
    "Benign under ACMG/AMP criteria. Evidence tools are available; each call "
    "costs time and money, so acquire only what you actually need. When you are "
    "ready, call final_answer. You may call final_answer at any point, including "
    "immediately."
)
_DISALLOWED = ["Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebFetch",
               "WebSearch", "Task", "Agent", "TodoWrite", "NotebookEdit",
               "Artifact", "Skill", "ToolSearch", "Workflow",
               "AskUserQuestion", "Monitor"]


def run_one(variant_id: str, gene: str, clinvar_name: str, model: str,
            cache_path: str, work_root: str) -> dict:
    case_dir = os.path.join(work_root, uuid.uuid4().hex)
    os.makedirs(case_dir, exist_ok=True)
    log_path = os.path.join(case_dir, "log.jsonl")

    mcp_config = json.dumps({"mcpServers": {"pwkbench-evidence": {
        "command": "python3", "args": [MCP_SERVER],
        "env": {"PWK_VARIANT_ID": variant_id, "PWK_GENE": gene,
                "PWK_CACHE_PATH": cache_path, "PWK_LOG_PATH": log_path,
                "PWK_CLINVAR_NAME": clinvar_name}}}})

    user = f"Variant: {clinvar_name}\nGene: {gene}\n\nClassify this variant as Pathogenic or Benign."
    cmd = [CLAUDE_BIN, "-p", "--model", model,
           "--disallowed-tools", *_DISALLOWED,
           "--strict-mcp-config", "--mcp-config", mcp_config,
           "--dangerously-skip-permissions",
           "--output-format", "json", "--no-session-persistence",
           "--system-prompt", SYSTEM, user]
    proc = subprocess.run(cmd, cwd=case_dir, capture_output=True, text=True, timeout=300)

    tools_called, predicted, confidence = [], None, None
    if os.path.exists(log_path):
        with open(log_path) as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("tool") == "final_answer":
                    args = e.get("args", {})
                    predicted = {"Pathogenic": 1, "Benign": 0}.get(args.get("classification"))
                    confidence = args.get("confidence")
                elif e.get("tool"):
                    tools_called.append(e["tool"])

    cost_usd, stop_reason = 0.0, "final_answer" if predicted is not None else "error:no_final_answer"
    if proc.returncode != 0:
        stop_reason = "error:nonzero_exit"
    else:
        try:
            d = json.loads(proc.stdout)
            cost_usd = d.get("total_cost_usd", 0.0)
        except json.JSONDecodeError:
            stop_reason = "error:non_json_output"

    return {"variant_id": variant_id, "gene": gene, "tools_called": json.dumps(tools_called),
            "n_tools_called": len(tools_called), "predicted": predicted,
            "confidence": confidence, "cost_usd": cost_usd, "stop_reason": stop_reason}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/sample/cohort_full_real.parquet")
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--outdir", default="results/live/claude_code")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-variants", type=int, default=None)
    args = ap.parse_args()
    os.chdir(_ROOT)
    os.makedirs(args.outdir, exist_ok=True)
    work_root = os.path.join(args.outdir, f".work_{args.tag}")
    os.makedirs(work_root, exist_ok=True)
    cache_path = os.path.join(args.outdir, f".tool_cache_{args.tag}.json")

    df = pd.read_parquet(args.cohort)[["variant_id", "gene", "clinvar_name", "label"]].reset_index(drop=True)
    if args.max_variants:
        df = df.groupby("gene", group_keys=False).apply(
            lambda g: g.sample(max(1, round(len(g) * args.max_variants / len(df))), random_state=0))
        df = df.reset_index(drop=True)

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
            res = run_one(row.variant_id, row.gene, row.clinvar_name, args.model, cache_path, work_root)
        except Exception as e:
            res = {"variant_id": row.variant_id, "gene": row.gene, "tools_called": "[]",
                   "n_tools_called": 0, "predicted": None, "confidence": None,
                   "cost_usd": 0.0, "stop_reason": f"error:{type(e).__name__}"}
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
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    out = pd.DataFrame(rows).merge(df[["variant_id", "label"]], on="variant_id")
    out["correct"] = out["predicted"] == out["label"]
    out["model_id"] = args.model
    out["arm"] = args.tag
    path = os.path.join(args.outdir, f"trajectories_live_{args.tag}.csv")
    out.to_csv(path, index=False)

    n_ans = int(out["predicted"].notna().sum())
    print(f"\n=== live/{args.tag} ({len(out)} cases) ===")
    print(f"  answered            {n_ans}/{len(out)}")
    if n_ans:
        ans = out[out["predicted"].notna()]
        print(f"  accuracy (answered) {ans['correct'].mean():.3f}")
        print(f"  tools called/case   mean {out['n_tools_called'].mean():.2f}")
    print(f"  stop_reason         {out['stop_reason'].value_counts().to_dict()}")
    print(f"  total cost          ${out['cost_usd'].sum():.2f}")
    print("written:", path)


if __name__ == "__main__":
    main()
