#!/usr/bin/env python3
"""Minimal hand-rolled MCP (Model Context Protocol) stdio server exposing
exactly the four real evidence tools plus `final_answer`, for driving the
Claude Code CLI through the SAME live-agent protocol as `LiveAgent`
(pwkbench/live/agent.py) / `GeminiLiveAgent`.

WHY HAND-ROLLED
---------------
The official `mcp` Python SDK requires Python >=3.10; this environment is
3.9. MCP's stdio transport is newline-delimited JSON-RPC 2.0 with exactly
three methods needed for a tools-only server (`initialize`,
`notifications/initialized`, `tools/list`, `tools/call`), so implementing it
directly is a few dozen lines, not a real dependency risk.

ONE SERVER PROCESS PER CASE
----------------------------
The variant this server answers for is fixed at process start via
environment variables (PWK_VARIANT_ID, PWK_GENE, PWK_CACHE_PATH,
PWK_LOG_PATH), not passed as tool arguments -- matching TOOL_SCHEMAS in
agent.py, where every evidence tool takes zero parameters (the agent under
test is not supposed to be able to name which variant it's asking about;
that would be a different, uncontrolled capability from the other vendors'
arms).

TOOL CACHE IS SHARED, ON PURPOSE
---------------------------------
Uses the same `pwkbench.live.tools.ToolCache` and the same evidence
functions as LiveAgent/GeminiLiveAgent, pointed at a cache path passed in by
the caller -- so identical (variant, tool) pairs return byte-identical
payloads across vendors, per the cache-sharing rule already documented in
tools.py.

LOGGING FOR OFFLINE ANALYSIS
------------------------------
Every tool call (including final_answer) is appended as one JSON line to
PWK_LOG_PATH, since the Claude Code CLI's own `--output-format json` result
does not include a tool-call transcript. This log is what
`run_claude_code_live.py` reads back to reconstruct `tools_called` order for
over-acquisition / tool-order analysis.
"""
from __future__ import annotations
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from pwkbench.live.tools import (                                    # noqa: E402
    ToolCache, get_population_freq, get_insilico_pathogenicity,
    get_domain_context, get_functional_assay,
)

VARIANT_ID = os.environ["PWK_VARIANT_ID"]
GENE = os.environ["PWK_GENE"]
CACHE_PATH = os.environ["PWK_CACHE_PATH"]
LOG_PATH = os.environ["PWK_LOG_PATH"]
# Full HGVS name, for get_domain_context's AA-position parse. Defaults to
# VARIANT_ID (bare coordinates) so this server also works unmodified for a
# masked-consequence run, where there is no HGVS name to give it.
CLINVAR_NAME = os.environ.get("PWK_CLINVAR_NAME", VARIANT_ID)

_cache = ToolCache(CACHE_PATH)

TOOLS = [
    {"name": "get_population_freq",
     "description": "Population allele frequency from gnomAD (ACMG PM2/BA1/BS1).",
     "inputSchema": {"type": "object", "properties": {}, "required": []}},
    {"name": "get_insilico_pathogenicity",
     "description": "AlphaMissense in-silico pathogenicity (ACMG PP3/BP4). "
                     "Missense substitutions only.",
     "inputSchema": {"type": "object", "properties": {}, "required": []}},
    {"name": "get_domain_context",
     "description": "Whether the affected residue lies in an annotated UniProt "
                     "functional domain/site (ACMG PM1).",
     "inputSchema": {"type": "object", "properties": {}, "required": []}},
    {"name": "get_functional_assay",
     "description": "Availability of curated deep-mutational-scan functional "
                     "data for this gene in MaveDB (ACMG PS3/BS3).",
     "inputSchema": {"type": "object", "properties": {}, "required": []}},
    {"name": "final_answer",
     "description": "Commit to a classification and stop acquiring evidence.",
     "inputSchema": {"type": "object", "properties": {
         "classification": {"type": "string", "enum": ["Pathogenic", "Benign"]},
         "confidence": {"type": "number", "description": "0-1 confidence in the call."}},
         "required": ["classification"]}},
]

_DISPATCH = {
    "get_population_freq": lambda: get_population_freq(VARIANT_ID, GENE, _cache),
    "get_insilico_pathogenicity": lambda: get_insilico_pathogenicity(VARIANT_ID, GENE, _cache),
    "get_domain_context": lambda: get_domain_context(VARIANT_ID, GENE, _cache, clinvar_name=CLINVAR_NAME),
    "get_functional_assay": lambda: get_functional_assay(VARIANT_ID, GENE, _cache),
}


def _log(entry: dict):
    entry = {"variant_id": VARIANT_ID, **entry}
    with open(LOG_PATH, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def _handle(msg: dict) -> dict | None:
    method = msg.get("method")
    msg_id = msg.get("id")

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "pwkbench-evidence", "version": "1.0.0"}}}
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {}) or {}
        if name == "final_answer":
            _log({"tool": "final_answer", "args": args})
            return {"jsonrpc": "2.0", "id": msg_id, "result": {
                "content": [{"type": "text", "text": "Recorded."}]}}
        fn = _DISPATCH.get(name)
        if fn is None:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {
                "code": -32601, "message": f"unknown tool {name!r}"}}
        try:
            result = fn()
        except Exception as e:                      # noqa: BLE001
            _log({"tool": name, "error": str(e)})
            return {"jsonrpc": "2.0", "id": msg_id, "result": {
                "content": [{"type": "text", "text": f"error: {e}"}], "isError": True}}
        _log({"tool": name, "result": result})
        return {"jsonrpc": "2.0", "id": msg_id, "result": {
            "content": [{"type": "text", "text": json.dumps(result)}]}}
    if msg_id is not None:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {
            "code": -32601, "message": f"unknown method {method!r}"}}
    return None


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = _handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
