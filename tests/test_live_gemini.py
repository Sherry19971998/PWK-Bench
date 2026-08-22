"""Offline pre-flight for GeminiLiveAgent.

WHY OFFLINE TESTS AND NOT JUST A SMOKE RUN
------------------------------------------
The live loop spends real money per case, and its failure modes are quiet: a
mis-wired message history does not raise, it just makes the agent behave as if
it acquired nothing, which is indistinguishable from the low-acquisition result
the study is trying to measure. The OpenAI arm's pre-flight caught a cached-
failure bug that would have fabricated cohort-wide MaveDB negatives; this file
is the equivalent gate for the Gemini port, driven by a stub client so every
branch is exercised for $0.

Everything here uses the REAL google.genai types (not dicts), so a shape the SDK
would reject fails at construction rather than in production.
"""
from __future__ import annotations
import os
import types as pytypes

import pytest

genai_types = pytest.importorskip("google.genai.types",
                                  reason="[gemini] extra not installed")

from pwkbench.live.agent import (  # noqa: E402
    GeminiLiveAgent, LiveAgent, MAX_STEPS, TOOL_SCHEMAS)
from pwkbench.live.tools import TOOLS  # noqa: E402


class _FakeCache:
    """Stands in for ToolCache. The evidence tools are not under test here --
    the wire protocol is -- so every lookup is a hit with a fixed payload,
    which also keeps these tests offline (a miss would trigger a real fetch).
    Must mirror ToolCache's full surface: get/put/log."""
    def __init__(self):
        self.logged = []

    def get(self, key):
        return {"stub": True}

    def put(self, key, value):
        pass

    def log(self, variant_id, tool, cached, ok):
        self.logged.append((variant_id, tool, cached, ok))


def _resp(parts, prompt_tokens=10, out_tokens=5, thoughts=7):
    """Build a response object shaped like GenerateContentResponse enough for
    the loop, using real Content/Part objects so field names are validated."""
    content = genai_types.Content(role="model", parts=parts) if parts is not None else None
    cand = pytypes.SimpleNamespace(content=content)
    usage = pytypes.SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=out_tokens,
        thoughts_token_count=thoughts)
    return pytypes.SimpleNamespace(candidates=[cand], usage_metadata=usage)


def _agent(scripted, allow_tools=None, monkeypatch=None):
    """GeminiLiveAgent whose client replays `scripted` (a list of responses)."""
    os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
    from google import genai
    calls = []

    class _FakeModels:
        def generate_content(self, *, model, contents, config):
            calls.append({"model": model, "contents": list(contents),
                          "config": config})
            return scripted[min(len(calls) - 1, len(scripted) - 1)]

    class _FakeClient:
        models = _FakeModels()

    monkeypatch.setattr(genai, "Client", lambda **kw: _FakeClient())
    a = GeminiLiveAgent("gemini-stub", _FakeCache(), allow_tools=allow_tools)
    a._calls = calls
    return a


# --------------------------------------------------------------------------
# 1. Tool-schema translation
# --------------------------------------------------------------------------

def test_schema_translation_matches_openai_arm(monkeypatch):
    """Both arms must advertise the SAME tools with the SAME descriptions --
    otherwise a 'vendor difference' is really a prompt difference."""
    a = _agent([_resp([])], monkeypatch=monkeypatch)
    tools = a._gemini_tools()
    assert len(tools) == 1
    names = [d.name for d in tools[0].function_declarations]
    assert names == [s["function"]["name"] for s in TOOL_SCHEMAS]
    descs = {d.name: d.description for d in tools[0].function_declarations}
    for s in TOOL_SCHEMAS:
        assert descs[s["function"]["name"]] == s["function"]["description"]


def test_parameterless_tools_declare_no_schema(monkeypatch):
    """The four evidence tools take no arguments. An empty
    {"properties": {}} is rejected by some backends, so they must be declared
    with no schema at all -- while final_answer, which DOES take arguments,
    must keep its."""
    a = _agent([_resp([])], monkeypatch=monkeypatch)
    decls = {d.name: d for d in a._gemini_tools()[0].function_declarations}
    for name in TOOLS:
        d = decls[name]
        assert not d.parameters and not d.parameters_json_schema, name
    fa = decls["final_answer"]
    assert fa.parameters_json_schema, "final_answer lost its argument schema"
    assert "classification" in fa.parameters_json_schema["properties"]


def test_ablation_removes_tool_from_declarations(monkeypatch):
    a = _agent([_resp([])],
               allow_tools=tuple(t for t in TOOLS if t != "get_functional_assay"),
               monkeypatch=monkeypatch)
    names = [d.name for d in a._gemini_tools()[0].function_declarations]
    assert "get_functional_assay" not in names
    assert "final_answer" in names


# --------------------------------------------------------------------------
# 2. The loop
# --------------------------------------------------------------------------

def test_tool_call_then_final_answer(monkeypatch):
    """The happy path: one acquisition, then a commit."""
    a = _agent([
        _resp([genai_types.Part.from_function_call(
            name="get_population_freq", args={})]),
        _resp([genai_types.Part.from_function_call(
            name="final_answer",
            args={"classification": "Pathogenic", "confidence": 0.8})]),
    ], monkeypatch=monkeypatch)
    tr = a.run_one("v1", "BRCA1", "NM_x:c.1A>T")
    assert tr["tools_called"] == ["get_population_freq"]
    assert tr["n_tools_called"] == 1
    assert tr["answer"] == "Pathogenic"
    assert tr["confidence"] == 0.8
    assert tr["stop_reason"] == "final_answer"


def test_function_response_is_sent_back_as_user_turn(monkeypatch):
    """Gemini has no `tool` role and no tool_call_id: results go back as a
    user Content of function_response parts, matched BY NAME. If this is wrong
    the model never sees its evidence and the run silently measures nothing."""
    a = _agent([
        _resp([genai_types.Part.from_function_call(
            name="get_domain_context", args={})]),
        _resp([genai_types.Part.from_function_call(
            name="final_answer", args={"classification": "Benign"})]),
    ], monkeypatch=monkeypatch)
    a.run_one("v1", "BRCA1", "NM_x:c.1A>T")
    sent = a._calls[1]["contents"]          # history as of the 2nd request
    roles = [c.role for c in sent]
    assert roles == ["user", "model", "user"], roles
    fr = [p.function_response for p in sent[-1].parts if p.function_response]
    assert len(fr) == 1
    assert fr[0].name == "get_domain_context"


def test_immediate_final_answer_acquires_nothing(monkeypatch):
    """Stopping at zero tools is the single most important behaviour this study
    measures; it must not be swallowed by the nudge branch."""
    a = _agent([_resp([genai_types.Part.from_function_call(
        name="final_answer", args={"classification": "Benign", "confidence": 0.6})])],
        monkeypatch=monkeypatch)
    tr = a.run_one("v1", "TP53", "NM_y:c.2G>C")
    assert tr["n_tools_called"] == 0
    assert tr["tools_called"] == []
    assert tr["cost_rank_sum"] == 0
    assert tr["stop_reason"] == "final_answer"


def test_withheld_tool_returns_error_and_is_not_counted(monkeypatch):
    """The ablation gate. A withheld tool must produce an error payload the
    model can react to, and must NOT enter `called` -- otherwise the ablation
    arm would report acquisitions it never made."""
    a = _agent([
        _resp([genai_types.Part.from_function_call(
            name="get_functional_assay", args={})]),
        _resp([genai_types.Part.from_function_call(
            name="final_answer", args={"classification": "Benign"})]),
    ], allow_tools=tuple(t for t in TOOLS if t != "get_functional_assay"),
        monkeypatch=monkeypatch)
    tr = a.run_one("v1", "BRCA1", "NM_x:c.1A>T")
    assert tr["n_tools_called"] == 0
    assert tr["tools_called"] == []
    fr = [p.function_response for p in a._calls[1]["contents"][-1].parts
          if p.function_response]
    assert "error" in fr[0].response


def test_empty_candidate_does_not_crash(monkeypatch):
    """A thinking model can spend its whole output budget on thoughts and
    return a candidate with no parts (content may even be None). That must
    nudge, not raise -- and must not be recorded as an acquisition."""
    a = _agent([
        _resp(None),                              # content is None
        _resp([]),                                # content with zero parts
        _resp([genai_types.Part.from_function_call(
            name="final_answer", args={"classification": "Benign"})]),
    ], monkeypatch=monkeypatch)
    tr = a.run_one("v1", "BRCA1", "NM_x:c.1A>T")
    assert tr["stop_reason"] == "final_answer"
    assert tr["n_tools_called"] == 0


def test_prose_reply_is_nudged_not_parsed(monkeypatch):
    """Free-text replies must be nudged through the loop, never parsed by an
    earliest-mention heuristic (the fragile path the frozen study had to
    caveat)."""
    a = _agent([
        _resp([genai_types.Part(text="I think this is probably pathogenic.")]),
        _resp([genai_types.Part.from_function_call(
            name="final_answer", args={"classification": "Pathogenic"})]),
    ], monkeypatch=monkeypatch)
    tr = a.run_one("v1", "BRCA1", "NM_x:c.1A>T")
    assert tr["answer"] == "Pathogenic"
    assert tr["n_tools_called"] == 0
    last = a._calls[1]["contents"][-1]
    assert last.role == "user" and "final_answer" in last.parts[0].text


def test_max_steps_terminates(monkeypatch):
    """A model that never commits must stop at max_steps, not loop forever on
    a paid endpoint."""
    a = _agent([_resp([genai_types.Part(text="thinking...")])],
               monkeypatch=monkeypatch)
    tr = a.run_one("v1", "BRCA1", "NM_x:c.1A>T")
    assert tr["stop_reason"] == "max_steps"
    assert len(a._calls) == MAX_STEPS


# --------------------------------------------------------------------------
# 3. Cross-vendor comparability
# --------------------------------------------------------------------------

def test_trajectory_schema_identical_to_openai_arm(monkeypatch):
    """Every downstream metric reads this dict. If the two arms' keys diverge,
    the cross-vendor comparison breaks in analysis rather than here."""
    a = _agent([_resp([genai_types.Part.from_function_call(
        name="final_answer", args={"classification": "Benign"})])],
        monkeypatch=monkeypatch)
    got = a.run_one("v1", "BRCA1", "NM_x:c.1A>T")
    expected = LiveAgent._trajectory(
        a, "v1", "BRCA1", "NM_x:c.1A>T", [], [], "Benign", None, "final_answer")
    assert set(got) == set(expected)


def test_shares_vendor_neutral_core_with_openai_arm():
    """These three must be the SAME function object, not copies -- that is what
    stops the two arms from drifting on what counts as an acquisition."""
    for m in ("_dispatch", "_trajectory", "_schemas"):
        assert getattr(GeminiLiveAgent, m) is getattr(LiveAgent, m), m


def test_thinking_tokens_are_billed_as_output(monkeypatch):
    """Gemini reports thinking tokens separately from candidate tokens. Both
    are billed as output; counting only candidates would understate the run's
    cost by most of its actual output spend."""
    a = _agent([_resp([genai_types.Part.from_function_call(
        name="final_answer", args={"classification": "Benign"})],
        prompt_tokens=100, out_tokens=5, thoughts=300)],
        monkeypatch=monkeypatch)
    a.run_one("v1", "BRCA1", "NM_x:c.1A>T")
    assert a.usage["prompt_tokens"] == 100
    assert a.usage["completion_tokens"] == 305


def test_automatic_function_calling_is_disabled(monkeypatch):
    """The SDK will execute Python callables itself if left on, which would
    bypass this loop's allow_tools gate entirely."""
    a = _agent([_resp([genai_types.Part.from_function_call(
        name="final_answer", args={"classification": "Benign"})])],
        monkeypatch=monkeypatch)
    a.run_one("v1", "BRCA1", "NM_x:c.1A>T")
    cfg = a._calls[0]["config"]
    assert cfg.automatic_function_calling.disable is True


def test_runs_at_vendor_defaults_like_the_openai_arm(monkeypatch):
    """Deliberate design decision, pinned here so it cannot regress silently.

    The OpenAI arm sends neither temperature nor max_tokens, so this arm must
    not either. Gemini WOULD accept temperature=0 (gpt-5.x rejects it), and an
    earlier revision set it along with max_output_tokens=4000. Both were
    removed: pinning one vendor's sampler and capping one vendor's output while
    the other runs unconstrained folds a configuration choice into what is
    supposed to be a vendor comparison, and a 4000-token cap on a thinking
    model can truncate a turn into an empty candidate -- which reads as "the
    agent chose to acquire nothing", the exact quantity under study.

    If a future change needs either knob, it must be applied to BOTH arms, and
    this test updated to say so.
    """
    a = _agent([_resp([genai_types.Part.from_function_call(
        name="final_answer", args={"classification": "Benign"})])],
        monkeypatch=monkeypatch)
    a.run_one("v1", "BRCA1", "NM_x:c.1A>T")
    cfg = a._calls[0]["config"]
    assert cfg.temperature is None, "temperature pinned on Gemini but not OpenAI"
    assert cfg.max_output_tokens is None, "output capped on Gemini but not OpenAI"


def test_tool_choice_is_explicitly_auto(monkeypatch):
    """Mirrors the OpenAI arm's explicit tool_choice="auto" rather than relying
    on Gemini's default, so neither arm depends on an unstated default."""
    a = _agent([_resp([genai_types.Part.from_function_call(
        name="final_answer", args={"classification": "Benign"})])],
        monkeypatch=monkeypatch)
    a.run_one("v1", "BRCA1", "NM_x:c.1A>T")
    cfg = a._calls[0]["config"]
    assert (cfg.tool_config.function_calling_config.mode
            == genai_types.FunctionCallingConfigMode.AUTO)


def test_system_and_user_prompt_are_identical_to_openai_arm(monkeypatch):
    """The two arms must differ ONLY in wire protocol. A drifted prompt would
    make any measured vendor difference uninterpretable."""
    from pwkbench.live.agent import SYSTEM
    a = _agent([_resp([genai_types.Part.from_function_call(
        name="final_answer", args={"classification": "Benign"})])],
        monkeypatch=monkeypatch)
    a.run_one("v1", "BRCA1", "NM_x:c.1A>T")
    cfg, contents = a._calls[0]["config"], a._calls[0]["contents"]
    assert cfg.system_instruction == SYSTEM
    # Byte-identical to the f-string LiveAgent.run_one builds.
    assert contents[0].parts[0].text == (
        "Variant: NM_x:c.1A>T\nGene: BRCA1\n\n"
        "Classify this variant as Pathogenic or Benign.")
