#!/usr/bin/env python3
"""Run the live evidence-acquisition loop through the local Codex CLI.

This is intentionally a separate arm from ``run_live.py``. Codex CLI is an
agent harness, not the OpenAI API chat-completions function-calling endpoint
used for the paper's gpt-5.5-2026-04-23 rows, so these outputs must be treated
as a Codex trial rather than a continuation of that model's official run.

Example, to ask Codex only for cases missing from the original checkpoint:

    python3 scripts/live/run_codex_cli.py \
        --cohort data/masked/cohort_full_real_masked_coords.parquet \
        --outdir results/live/masked \
        --tag codex_masked_coords_full \
        --skip-ckpt results/live/masked/_ckpt_masked_coords_full.jsonl

For the interrupted E4 run, an even more explicit filter is:

    python3 scripts/live/run_codex_cli.py \
        --cohort data/masked/cohort_full_real_masked_coords.parquet \
        --outdir results/live/masked \
        --tag codex_masked_coords_full \
        --only-errors-from results/live/masked/trajectories_masked_coords_full.csv
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from pwkbench.live.agent import SYSTEM  # noqa: E402
from pwkbench.live.tools import TOOL_COST_RANK, TOOLS, ToolCache  # noqa: E402


CHOICE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "action": {"type": "string", "enum": ["tool", "final_answer"]},
        "tool_name": {
            "type": ["string", "null"],
            "enum": list(TOOLS) + [None],
        },
        "classification": {
            "type": ["string", "null"],
            "enum": ["Pathogenic", "Benign", None],
        },
        "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "rationale": {"type": ["string", "null"]},
    },
    "required": ["action", "tool_name", "classification", "confidence", "rationale"],
}


def _load_jsonl_by_variant(path: str | None) -> dict:
    done = {}
    if not path or not os.path.exists(path):
        return done
    with open(path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = row.get("variant_id")
            if vid:
                done[vid] = row
    return done


class CodexCLILiveAgent:
    """LiveAgent-compatible loop that delegates each decision to ``codex exec``."""

    def __init__(self, codex_bin: str, tool_cache: ToolCache,
                 max_steps: int = 8, codex_model: str | None = None,
                 allow_tools: tuple[str, ...] | None = None, timeout: int = 300):
        self.codex_bin = codex_bin
        self.codex_model = codex_model
        self.cache = tool_cache
        self.max_steps = max_steps
        self.allow_tools = tuple(allow_tools) if allow_tools is not None else tuple(TOOLS)
        self.timeout = timeout
        self.model_id = "codex-cli" + (f":{codex_model}" if codex_model else ":default")

    def _dispatch(self, name: str, variant_id: str, gene: str,
                  clinvar_name: str, called: list[str], order: list[str]) -> dict:
        fn = TOOLS.get(name)
        if fn is None or name not in self.allow_tools:
            return {"error": f"tool {name} is not available"}
        kw = {"clinvar_name": clinvar_name} if name == "get_domain_context" else {}
        result = fn(variant_id, gene, self.cache, **kw)
        called.append(name)
        order.append(name)
        return result

    def _trajectory(self, variant_id: str, gene: str, clinvar_name: str,
                    called: list[str], order: list[str], answer, confidence,
                    stop_reason: str) -> dict:
        return {"variant_id": variant_id, "gene": gene,
                "clinvar_name": clinvar_name,
                "n_tools_called": len(called),
                "tools_called": order,
                "distinct_tools": sorted(set(called)),
                "cost_rank_sum": sum(TOOL_COST_RANK[t] for t in called),
                "answer": answer, "confidence": confidence,
                "stop_reason": stop_reason,
                "allowed_tools": list(self.allow_tools)}

    def _prompt(self, variant_id: str, gene: str, clinvar_name: str,
                observations: list[dict]) -> str:
        available = "\n".join(f"- {name}" for name in self.allow_tools)
        history = json.dumps(observations, indent=2, sort_keys=True)
        return f"""{SYSTEM}

You are running inside a benchmark harness. Do not inspect files, use shell
commands, browse the web, or call any tools outside this prompt. Your only job
is to choose the next benchmark action from the JSON schema.

Variant: {clinvar_name}
Gene: {gene}
Coordinate id: {variant_id}

Available evidence tools:
{available}

Evidence already acquired:
{history}

Return exactly one JSON object matching the supplied schema.
- To acquire evidence: action="tool", tool_name=<one available tool>,
  classification=null, confidence=null.
- To stop: action="final_answer", tool_name=null, classification either
  "Pathogenic" or "Benign", confidence 0-1.
Acquire only evidence you actually need.
"""

    def _codex_choice(self, prompt: str) -> dict:
        with tempfile.TemporaryDirectory(prefix="pwkbench_codex_") as td:
            schema = Path(td) / "choice_schema.json"
            out = Path(td) / "last_message.json"
            schema.write_text(json.dumps(CHOICE_SCHEMA), encoding="utf-8")

            cmd = [self.codex_bin, "exec", "--ephemeral", "--ignore-rules",
                   "--sandbox", "read-only", "--ask-for-approval", "never",
                   "--output-schema", str(schema), "-o", str(out)]
            if self.codex_model:
                cmd.extend(["--model", self.codex_model])
            cmd.append("-")
            proc = subprocess.run(
                cmd, input=prompt, text=True, capture_output=True,
                timeout=self.timeout, check=False)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "").strip().splitlines()
                msg = err[-1] if err else f"codex exited {proc.returncode}"
                raise RuntimeError(msg[:500])
            raw = out.read_text(encoding="utf-8").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"codex returned non-JSON final message: {raw[:500]}") from e

    def run_one(self, variant_id: str, gene: str, clinvar_name: str) -> dict:
        called, order, observations = [], [], []
        answer, confidence, stop_reason = None, None, "max_steps"
        for _ in range(self.max_steps):
            choice = self._codex_choice(
                self._prompt(variant_id, gene, clinvar_name, observations))
            action = choice.get("action")
            if action == "final_answer":
                answer = choice.get("classification")
                confidence = choice.get("confidence")
                stop_reason = "final_answer"
                break
            if action != "tool":
                observations.append({"error": f"invalid action: {action!r}"})
                continue
            name = choice.get("tool_name")
            result = self._dispatch(name, variant_id, gene, clinvar_name, called, order)
            observations.append({"tool_name": name, "result": result})
        return self._trajectory(variant_id, gene, clinvar_name, called, order,
                                answer, confidence, stop_reason)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="data/sample/cohort_full_real.parquet")
    ap.add_argument("--outdir", default="results/live")
    ap.add_argument("--tag", default="codex_cli")
    ap.add_argument("--max-variants", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--codex-bin", default=shutil.which("codex") or "codex")
    ap.add_argument("--codex-model", default=None,
                    help="optional model passed to `codex exec --model`; omitted uses your Codex default")
    ap.add_argument("--timeout", type=int, default=300,
                    help="seconds allowed for each individual `codex exec` decision")
    ap.add_argument("--skip-ckpt", default=None,
                    help="JSONL checkpoint whose variant_ids should not be run by Codex")
    ap.add_argument("--only-errors-from", default=None,
                    help="CSV whose error stop_reason rows define the variants to run")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_parquet(args.cohort)

    skip = _load_jsonl_by_variant(args.skip_ckpt)
    if skip:
        df = df[~df.variant_id.isin(skip)].reset_index(drop=True)
        print(f"[codex-cli] skipping {len(skip)} case(s) from {args.skip_ckpt}", flush=True)

    if args.only_errors_from:
        prev = pd.read_csv(args.only_errors_from)
        err_ids = set(prev.loc[
            prev.stop_reason.astype(str).str.startswith("error"),
            "variant_id",
        ])
        df = df[df.variant_id.isin(err_ids)].reset_index(drop=True)
        print(f"[codex-cli] restricted to {len(err_ids)} error case(s) from "
              f"{args.only_errors_from}", flush=True)

    if args.max_variants:
        if len(df) > args.max_variants:
            df = pd.concat([
                g.sample(max(1, round(len(g) * args.max_variants / len(df))),
                         random_state=0)
                for _, g in df.groupby("gene")])
            if len(df) > args.max_variants:
                df = df.sample(args.max_variants, random_state=0)
        df = df.reset_index(drop=True)

    ckpt = os.path.join(args.outdir, f"_ckpt_{args.tag}.jsonl")
    done = _load_jsonl_by_variant(ckpt)
    if done:
        print(f"[codex-cli] resuming {len(done)} case(s) from {ckpt}", flush=True)

    cache = ToolCache(os.path.join(args.outdir, ".tool_cache.json"))
    agent = CodexCLILiveAgent(args.codex_bin, cache, max_steps=args.max_steps,
                              codex_model=args.codex_model, timeout=args.timeout)
    ck_lock = threading.Lock()

    print(f"[codex-cli] model={agent.model_id} cases={len(df)} tag={args.tag}", flush=True)

    def _one(r):
        if r.variant_id in done:
            return done[r.variant_id]
        try:
            res = agent.run_one(r.variant_id, r.gene, r.clinvar_name)
        except Exception as e:  # noqa: BLE001
            res = {"variant_id": r.variant_id, "gene": r.gene,
                   "clinvar_name": r.clinvar_name, "n_tools_called": 0,
                   "tools_called": [], "distinct_tools": [], "cost_rank_sum": 0,
                   "answer": None, "confidence": None,
                   "stop_reason": f"error:{type(e).__name__}",
                   "allowed_tools": list(agent.allow_tools)}
        with ck_lock:
            with open(ckpt, "a") as fh:
                fh.write(json.dumps(res) + "\n")
        return res

    rows, t0 = [], time.time()
    todo = list(df.itertuples())
    if args.workers <= 1:
        for i, r in enumerate(todo, 1):
            rows.append(_one(r))
            if i % 10 == 0:
                print(f"  {i}/{len(todo)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_one, r) for r in todo]
            for i, f in enumerate(futs, 1):
                rows.append(f.result())
                if i % 10 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)

    out = pd.DataFrame(rows)
    if out.empty:
        print("No Codex cases to write.")
        return
    lab = pd.read_parquet(args.cohort).set_index("variant_id").label
    out["label"] = out.variant_id.map(lab)
    out["predicted"] = out.answer.map({"Pathogenic": 1, "Benign": 0})
    out["correct"] = (out.predicted == out.label)
    out["tools_called"] = out.tools_called.map(json.dumps)
    out["distinct_tools"] = out.distinct_tools.map(json.dumps)
    out["allowed_tools"] = out.allowed_tools.map(json.dumps)
    out["model_id"], out["arm"] = agent.model_id, args.tag

    path = os.path.join(args.outdir, f"trajectories_{args.tag}.csv")
    out.to_csv(path, index=False)
    pd.DataFrame(cache.calls).to_csv(
        os.path.join(args.outdir, f"tool_calls_{args.tag}.csv"), index=False)

    elapsed = time.time() - t0
    print(f"\n=== {args.tag} ({len(out)} Codex case(s), {elapsed:.0f}s) ===")
    print(f"  stop_reason         {out.stop_reason.value_counts().to_dict()}")
    print(f"  tools called/case   mean {out.n_tools_called.mean():.2f}  "
          f"dist {out.n_tools_called.value_counts().sort_index().to_dict()}")
    print("written:", path)


if __name__ == "__main__":
    main()
