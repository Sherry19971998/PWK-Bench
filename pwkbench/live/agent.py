"""Tool-using agent loop: the agent decides WHICH evidence to acquire and WHEN to stop.

The whole experiment is that second decision. The frozen-pool benchmark had to
inject a STOP action because its harness ran the step loop to exhaustion; here
stopping is native -- the agent emits `final_answer` whenever it wants, and the
number of tools it called before doing so is the measurement.

WHAT THE AGENT SEES
-------------------
Gene + HGVS name (the "real case" presentation). This is a deliberate choice with
a measured consequence: the repo's own closed-book probe
(`results/variant/real_trusted/memorization_probe.csv`) scores AUC 0.865 from the full
HGVS name alone, against 0.942 with all evidence acquired. So the model very
often already knows the answer, and the tool layer can move accuracy by at most
~0.077. Read the output as a BEHAVIOURAL result -- does an agent that already
knows still order tests? -- and do not claim the tools improved accuracy.
"""
from __future__ import annotations
import json
import threading

from .tools import TOOLS, TOOL_COST_RANK

MAX_STEPS = 8

SYSTEM = (
    "You are a clinical variant scientist. Classify the variant as Pathogenic or "
    "Benign under ACMG/AMP criteria. Evidence tools are available; each call "
    "costs time and money, so acquire only what you actually need. When you are "
    "ready, call final_answer. You may call final_answer at any point, including "
    "immediately."
)

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_population_freq",
        "description": "Population allele frequency from gnomAD (ACMG PM2/BA1/BS1).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_insilico_pathogenicity",
        "description": "AlphaMissense in-silico pathogenicity (ACMG PP3/BP4). "
                       "Missense substitutions only.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_domain_context",
        "description": "Whether the affected residue lies in an annotated UniProt "
                       "functional domain/site (ACMG PM1).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_functional_assay",
        "description": "Availability of curated deep-mutational-scan functional "
                       "data for this gene in MaveDB (ACMG PS3/BS3).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "final_answer",
        "description": "Commit to a classification and stop acquiring evidence.",
        "parameters": {"type": "object", "properties": {
            "classification": {"type": "string",
                               "enum": ["Pathogenic", "Benign"]},
            "confidence": {"type": "number",
                           "description": "0-1 confidence in the call."}},
            "required": ["classification"]}}},
]


class LiveAgent:
    """One ReAct dialogue per variant, over the four evidence tools."""

    def __init__(self, model_id: str, tool_cache, max_steps: int = MAX_STEPS,
                 seed: int = 0, allow_tools: tuple | None = None,
                 tool_order: str | None = None):
        from openai import OpenAI
        self._client = OpenAI()
        self.model_id = model_id
        self.cache = tool_cache
        self.max_steps = max_steps
        self.tool_order = tool_order
        # Kept only so callers can record what was requested. It is NOT passed to
        # the API: gpt-5.x reasoning models reject a pinned temperature and treat
        # `seed` as best-effort, so accepting one here and forwarding it would
        # imply a reproducibility this loop cannot deliver. Repeated runs are the
        # only honest way to bound run-to-run spread (see the frozen-pool
        # variance study).
        self.seed = seed
        # allow_tools drives the tool-ablation control: withholding one tool and
        # re-running answers "did the agent's accuracy actually depend on it?".
        # None = all four.
        self.allow_tools = (tuple(allow_tools) if allow_tools is not None
                            else tuple(TOOLS))
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        # One agent object is shared across the runner's worker threads, so the
        # token counters need guarding: `+=` on a dict value is read-modify-write
        # and concurrent updates silently lose counts, which would understate the
        # measured cost of the sweep.
        self._usage_lock = threading.Lock()

    def _schemas(self):
        """Tool schemas offered to the model, in presentation order.

        `tool_order` permutes the ACQUISITION tools. It exists to test a
        confound in the headline behavioural result: `get_population_freq` is
        the first query in 100% of cases, but it is also the first entry in
        TOOL_SCHEMAS, so those runs cannot tell "the agent judges rarity to be
        the right opening" apart from "the agent takes the first tool it was
        shown". Re-running with the order reversed separates them.

        `final_answer` is pinned last whatever the permutation: it is not an
        acquisition, and moving it would change the stopping affordance rather
        than the acquisition order — a second variable in a one-variable test.
        """
        avail = [s for s in TOOL_SCHEMAS
                 if s["function"]["name"] in self.allow_tools
                 or s["function"]["name"] == "final_answer"]
        if not self.tool_order:
            return avail
        acq = [s for s in avail if s["function"]["name"] != "final_answer"]
        fin = [s for s in avail if s["function"]["name"] == "final_answer"]
        if self.tool_order == "reverse":
            acq = acq[::-1]
        else:
            rank = {n: i for i, n in enumerate(self.tool_order.split(","))}
            unknown = [s["function"]["name"] for s in acq
                       if s["function"]["name"] not in rank]
            if unknown:
                raise ValueError(f"tool_order does not mention {unknown}")
            acq.sort(key=lambda s: rank[s["function"]["name"]])
        return acq + fin

    # ---- vendor-neutral core -------------------------------------------
    # _dispatch and _trajectory are shared with GeminiLiveAgent on purpose.
    # They carry the parts that MUST be byte-identical across vendors for the
    # two rows to be comparable at all: which tool runs, how allow_tools gates
    # it, what counts as an acquisition, and the trajectory schema every
    # downstream metric reads. Two copies would drift the first time a tool
    # signature or a cost rank changes, and the resulting cross-vendor
    # comparison would be silently wrong rather than broken.

    def _retrying(self, call, retries: int = 6):
        """Run `call()`, retrying transient API failures with backoff.

        The live loop originally had NO retry at all, which was survivable on
        OpenAI's limits and fatal on Gemini's free tier: one 429 killed the
        case, `run_live.py` recorded stop_reason='error:ClientError' with zero
        tools called, and that row is INDISTINGUISHABLE in every downstream
        metric from "the agent chose to acquire nothing" -- the exact quantity
        this study measures. Measured 2026-08-02: a 40-case smoke on
        gemini-3.5-flash-lite lost 31/37 cases this way (free-tier limit is 15
        requests/minute).

        Honours the server's own retry delay, because the generic exponential
        ramp caps far below what a quota-limited endpoint asks for (observed:
        "Please retry in 47.3s"). A DAILY quota wall is re-raised immediately --
        no in-process backoff can wait out a ~24h reset, and burning the retry
        budget against it just delays the inevitable stop.

        Reuses the frozen block's helpers rather than reimplementing them, so
        the two blocks cannot disagree about what counts as retryable.
        """
        import time
        from ..agents.llm import _is_daily_quota_error, _suggested_retry_delay
        last = None
        for attempt in range(max(1, retries)):
            try:
                return call()
            except Exception as e:                        # noqa: BLE001
                if _is_daily_quota_error(e):
                    raise
                last = e
                if attempt == retries - 1:
                    break
                d = _suggested_retry_delay(e)
                time.sleep(d if d is not None else min(5 * 2 ** attempt, 60))
        raise last

    def _dispatch(self, name, variant_id, gene, clinvar_name, called, order):
        """Run one evidence tool and record the acquisition. A withheld tool
        (the ablation arm) returns an error payload instead of raising, so the
        model can react to the refusal the same way it would to a real API
        failure -- and, importantly, it is NOT appended to `called`, so the
        ablation cannot inflate the acquisition count it is meant to reduce."""
        fn = TOOLS.get(name)
        if fn is None or name not in self.allow_tools:
            return {"error": f"tool {name} is not available"}
        kw = {"clinvar_name": clinvar_name} if name == "get_domain_context" else {}
        result = fn(variant_id, gene, self.cache, **kw)
        called.append(name)
        order.append(name)
        return result

    def _trajectory(self, variant_id, gene, clinvar_name, called, order,
                    answer, confidence, stop_reason):
        return {"variant_id": variant_id, "gene": gene,
                "clinvar_name": clinvar_name,
                "n_tools_called": len(called),
                "tools_called": order,
                "distinct_tools": sorted(set(called)),
                "cost_rank_sum": sum(TOOL_COST_RANK[t] for t in called),
                "answer": answer, "confidence": confidence,
                "stop_reason": stop_reason,
                "allowed_tools": list(self.allow_tools)}

    def run_one(self, variant_id: str, gene: str, clinvar_name: str) -> dict:
        """Return the full trajectory for one case."""
        user = (f"Variant: {clinvar_name}\nGene: {gene}\n\n"
                f"Classify this variant as Pathogenic or Benign.")
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user}]
        called, order = [], []
        answer, confidence, stop_reason = None, None, "max_steps"

        for _step in range(self.max_steps):
            resp = self._retrying(lambda: self._client.chat.completions.create(
                model=self.model_id, messages=messages,
                tools=self._schemas(), tool_choice="auto"))
            u = resp.usage
            if u:
                with self._usage_lock:
                    self.usage["prompt_tokens"] += u.prompt_tokens
                    self.usage["completion_tokens"] += u.completion_tokens
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                # The model replied in prose instead of calling a tool. Nudge it
                # once via the loop rather than parsing free text: an
                # earliest-mention parse over prose is exactly the fragile
                # heuristic the frozen-pool study had to caveat.
                messages.append({"role": "user",
                                 "content": "Call a tool, or call final_answer."})
                continue

            for tc in msg.tool_calls:
                name = tc.function.name
                if name == "final_answer":
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    answer = args.get("classification")
                    confidence = args.get("confidence")
                    stop_reason = "final_answer"
                    break
                result = self._dispatch(name, variant_id, gene, clinvar_name,
                                        called, order)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result)})
            if stop_reason == "final_answer":
                break

        return self._trajectory(variant_id, gene, clinvar_name, called, order,
                                answer, confidence, stop_reason)


class GeminiLiveAgent(LiveAgent):
    """Gemini port of LiveAgent. Same tools, same system prompt, same stopping
    rule, same trajectory schema -- only the wire protocol differs.

    Inherits _dispatch/_trajectory/_schemas so the parts that decide WHAT the
    agent measures cannot drift from the OpenAI arm. Only run_one is rewritten,
    because Gemini's function calling differs from OpenAI's in four ways that
    each need explicit handling:

      1. NO TOOL-CALL IDs. OpenAI pairs a result to a call via tool_call_id;
         Gemini pairs by FUNCTION NAME. If the model emits two calls to the
         same function in one turn, the pairing is ambiguous by construction --
         the loop below therefore answers every call it was given, in order,
         and does not attempt to synthesise ids.
      2. ARGS ARE ALREADY A DICT. `part.function_call.args` is parsed, unlike
         OpenAI's `tc.function.arguments` JSON string, so there is no
         json.loads and no JSONDecodeError branch here. Do not "restore" one.
      3. NO `tool` ROLE. Results go back as a `user` Content whose parts are
         function_response parts.
      4. AUTOMATIC FUNCTION CALLING. The SDK will try to invoke Python
         callables itself; it is explicitly DISABLED below so this loop stays
         the only thing that executes tools. Left on, the ablation arm's
         allow_tools gate would be bypassed silently.

    Determinism: NEITHER temperature nor seed is forwarded, and no
    max_output_tokens is set, so this arm runs at Gemini's defaults exactly as
    the OpenAI arm runs at OpenAI's. Gemini would accept temperature=0 (gpt-5.x
    will not), but pinning one arm's sampler and not the other would mix a
    configuration choice into the vendor comparison. Repeated runs remain the
    only honest way to bound run-to-run spread, for both arms.
    """

    def __init__(self, model_id: str, tool_cache, max_steps: int = MAX_STEPS,
                 seed: int = 0, allow_tools: tuple | None = None,
                 tool_order: str | None = None):
        # Skip LiveAgent.__init__ (it constructs an OpenAI client) but keep
        # every field it sets, so inherited methods behave identically.
        import os
        from google import genai                # requires [gemini] extra
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model_id = model_id
        self.cache = tool_cache
        self.max_steps = max_steps
        self.tool_order = tool_order
        self.seed = seed
        self.allow_tools = (tuple(allow_tools) if allow_tools is not None
                            else tuple(TOOLS))
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self._usage_lock = threading.Lock()

    def _gemini_tools(self):
        """Translate the shared TOOL_SCHEMAS into Gemini FunctionDeclarations.

        Derived from the SAME list the OpenAI arm uses, never a hand-copied
        second list: a tool description that differs between arms would be a
        prompt difference masquerading as a vendor difference.

        `parameters_json_schema` takes the OpenAI-style JSON schema verbatim.
        The four evidence tools declare no parameters, and an empty
        `{"properties": {}}` is rejected by some backends, so those are
        declared with no schema at all rather than an empty one.
        """
        from google.genai import types
        decls = []
        for s in self._schemas():
            f = s["function"]
            params = f.get("parameters") or {}
            kw = {}
            if params.get("properties"):
                kw["parameters_json_schema"] = params
            decls.append(types.FunctionDeclaration(
                name=f["name"], description=f.get("description", ""), **kw))
        return [types.Tool(function_declarations=decls)]

    def run_one(self, variant_id: str, gene: str, clinvar_name: str) -> dict:
        from google.genai import types
        user = (f"Variant: {clinvar_name}\nGene: {gene}\n\n"
                f"Classify this variant as Pathogenic or Benign.")
        contents = [types.Content(role="user",
                                  parts=[types.Part(text=user)])]
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=self._gemini_tools(),
            # Explicit AUTO to mirror the OpenAI arm's tool_choice="auto".
            # Gemini defaults to AUTO when tools are present, so this changes
            # nothing -- it just removes the need to trust a default for a
            # parameter the other arm states outright.
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.AUTO)),
            # DELIBERATELY no temperature and no max_output_tokens: the OpenAI
            # arm sends neither, so both arms run at their vendor's default.
            # Pinning only this side's sampler would fold a configuration
            # choice into what is supposed to be a vendor comparison -- if
            # Gemini then showed a different acquisition spread, there would be
            # no way to tell that apart from "I froze one sampler and not the
            # other". The cost is that neither arm is reproducible from a seed;
            # repeated runs are the only honest way to bound spread, which is
            # already true of the OpenAI arm. See the max_output_tokens note in
            # the empty-candidate branch of run_one for what this trades away.
            # See docstring item 4: this loop must be the only tool executor.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True))

        called, order = [], []
        answer, confidence, stop_reason = None, None, "max_steps"

        for _step in range(self.max_steps):
            resp = self._retrying(lambda: self._client.models.generate_content(
                model=self.model_id, contents=contents, config=config))
            um = getattr(resp, "usage_metadata", None)
            if um:
                with self._usage_lock:
                    self.usage["prompt_tokens"] += um.prompt_token_count or 0
                    # thoughts_token_count is billed as output but is reported
                    # separately from candidates_token_count; omitting it would
                    # understate the cost of a thinking model by most of its
                    # actual output spend.
                    self.usage["completion_tokens"] += (
                        (um.candidates_token_count or 0)
                        + (getattr(um, "thoughts_token_count", 0) or 0))

            cand = (resp.candidates or [None])[0]
            content = getattr(cand, "content", None) if cand else None
            parts = list(getattr(content, "parts", None) or [])
            calls = [p.function_call for p in parts
                     if getattr(p, "function_call", None)]

            if not calls:
                # Prose instead of a call, or an EMPTY CANDIDATE. The empty case
                # is specific to thinking models: the output budget covers
                # thinking + the function call, so a turn that thinks to the
                # cap returns a candidate with no parts. Because no
                # max_output_tokens is sent (see the config above), that cap is
                # the model's own default -- 65536 on the current Gemini
                # models, far above anything this small task should need, so
                # truncation should be rare rather than routine. It is still
                # handled here rather than assumed away, because the failure is
                # silent and self-serving: an empty turn looks exactly like "the
                # agent chose not to acquire anything", which is the very
                # quantity this study measures. If a run shows an unexpected
                # number of zero-tool cases, check the smoke run's reported
                # thinking-token usage before believing them.
                #
                # Nudge via the loop rather than parsing free text -- same
                # reasoning as the OpenAI arm. `content` may be None on an empty
                # candidate, so echo a model turn only when there is one.
                if content is not None:
                    contents.append(content)
                contents.append(types.Content(role="user", parts=[types.Part(
                    text="Call a tool, or call final_answer.")]))
                continue

            contents.append(content)            # echo the model turn verbatim
            responses = []
            for fc in calls:
                name = fc.name
                if name == "final_answer":
                    args = dict(fc.args or {})   # already parsed; see docstring
                    answer = args.get("classification")
                    confidence = args.get("confidence")
                    stop_reason = "final_answer"
                    break
                result = self._dispatch(name, variant_id, gene, clinvar_name,
                                        called, order)
                responses.append(types.Part.from_function_response(
                    name=name, response=result))
            if stop_reason == "final_answer":
                break
            if responses:
                contents.append(types.Content(role="user", parts=responses))

        return self._trajectory(variant_id, gene, clinvar_name, called, order,
                                answer, confidence, stop_reason)


# Output-token budget for one Anthropic turn. Generous for the same reason as
# the frozen-pool loop's _MAX_TOKENS (pwkbench/agents/llm.py): on a model with
# extended thinking, max_tokens caps thinking + text/tool-use together, so a
# tight budget can return an empty turn that this loop would otherwise misread
# as "the agent chose to call nothing".
_ANTHROPIC_MAX_TOKENS = 2000


class AnthropicLiveAgent(LiveAgent):
    """Anthropic port of LiveAgent, driven through the Messages API directly --
    no Claude Code CLI, no MCP subprocess. This is the API-mode counterpart to
    scripts/live/run_claude_code_live.py's CLI-based arm.

    Inherits _dispatch/_trajectory/_schemas/_retrying from LiveAgent so the
    parts that decide WHAT the agent measures cannot drift from the OpenAI or
    Gemini arms (same reasoning as GeminiLiveAgent's docstring). Only run_one
    is rewritten, because the Messages API's tool-use protocol differs from
    both:

      1. TOOL RESULTS ARE PAIRED BY tool_use_id, carried in a `user` message
         whose content is a list of {"type": "tool_result", ...} blocks --
         closer to OpenAI's tool_call_id than to Gemini's name-based pairing,
         but a different message shape from either.
      2. THE ASSISTANT TURN'S TOOL ARGS ARE ALREADY A DICT: `tc.input` on an
         Anthropic tool_use block is parsed, so there is no json.loads and no
         JSONDecodeError branch here, matching Gemini's `fc.args` rather than
         OpenAI's `tc.function.arguments` (a JSON string).
      3. EVERY tool_use BLOCK IN A TURN NEEDS A tool_result, even when one of
         them is `final_answer` and the loop is about to stop: an unanswered
         tool_use in message history makes the NEXT API call malformed, so
         final_answer still gets a (throwaway) tool_result before the loop
         breaks.

    Unlike the CLI arm (see ClaudeCLIAgent's docstring in
    pwkbench/agents/llm.py), this sends ONLY the ~200-token benchmark system
    prompt -- no Claude Code harness scaffolding ahead of it -- so it IS a
    clean vendor-axis control, comparable prompt-for-prompt to the OpenAI and
    Gemini arms. Requires ANTHROPIC_API_KEY.

    Determinism: the Messages API exposes no `seed` (same as AnthropicAgent in
    the frozen-pool loop), so `self.seed` is accepted for signature
    compatibility and not forwarded. Repeated runs are the only honest way to
    bound run-to-run spread, exactly as for the other two vendor arms.
    """

    def __init__(self, model_id: str, tool_cache, max_steps: int = MAX_STEPS,
                 seed: int = 0, allow_tools: tuple | None = None,
                 tool_order: str | None = None):
        import os
        import anthropic                         # requires [anthropic] extra
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model_id = model_id
        self.cache = tool_cache
        self.max_steps = max_steps
        self.tool_order = tool_order
        self.seed = seed
        self.allow_tools = (tuple(allow_tools) if allow_tools is not None
                            else tuple(TOOLS))
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self._usage_lock = threading.Lock()

    def _anthropic_tools(self):
        """Translate the shared TOOL_SCHEMAS (via self._schemas(), so a
        withheld/reordered tool set stays consistent with the other arms)
        into Anthropic's tool format. `parameters` and `input_schema` are the
        same JSON-schema shape under a different key name, so this is a
        rename, not a re-specification -- never a hand-copied second schema
        that could drift from TOOL_SCHEMAS."""
        out = []
        for s in self._schemas():
            f = s["function"]
            out.append({"name": f["name"], "description": f.get("description", ""),
                        "input_schema": f.get("parameters") or
                                        {"type": "object", "properties": {}}})
        return out

    def run_one(self, variant_id: str, gene: str, clinvar_name: str) -> dict:
        from ..agents.llm import RefusalError
        user = (f"Variant: {clinvar_name}\nGene: {gene}\n\n"
                f"Classify this variant as Pathogenic or Benign.")
        messages = [{"role": "user", "content": user}]
        tools = self._anthropic_tools()
        called, order = [], []
        answer, confidence, stop_reason = None, None, "max_steps"

        for _step in range(self.max_steps):
            resp = self._retrying(lambda: self._client.messages.create(
                model=self.model_id, max_tokens=_ANTHROPIC_MAX_TOKENS,
                system=SYSTEM, messages=messages, tools=tools))
            if getattr(resp, "stop_reason", None) == "refusal":
                raise RefusalError("model refused")
            u = getattr(resp, "usage", None)
            if u:
                with self._usage_lock:
                    self.usage["prompt_tokens"] += u.input_tokens
                    self.usage["completion_tokens"] += u.output_tokens

            raw_content = resp.content
            calls = [b for b in raw_content if getattr(b, "type", None) == "tool_use"]
            # Echo the assistant turn verbatim (text + tool_use blocks) as
            # plain dicts, mirroring the OpenAI arm's msg.model_dump(...) --
            # the model needs its own prior turn in context to correctly pair
            # tool_result blocks next round.
            messages.append({"role": "assistant",
                             "content": [b.model_dump(exclude_none=True)
                                         for b in raw_content]})

            if not calls:
                # Prose instead of a tool call. Nudge via the loop rather than
                # parsing free text -- same reasoning as the other two arms.
                messages.append({"role": "user",
                                 "content": "Call a tool, or call final_answer."})
                continue

            results = []
            for tc in calls:
                name = tc.name
                if name == "final_answer":
                    args = dict(tc.input or {})
                    answer = args.get("classification")
                    confidence = args.get("confidence")
                    stop_reason = "final_answer"
                    # A throwaway result: every tool_use block issued this
                    # turn needs a matching tool_result or the next API call
                    # is malformed, even though the loop is about to end.
                    results.append({"type": "tool_result", "tool_use_id": tc.id,
                                    "content": "Recorded."})
                    continue
                result = self._dispatch(name, variant_id, gene, clinvar_name,
                                        called, order)
                results.append({"type": "tool_result", "tool_use_id": tc.id,
                                "content": json.dumps(result)})
            messages.append({"role": "user", "content": results})
            if stop_reason == "final_answer":
                break

        return self._trajectory(variant_id, gene, clinvar_name, called, order,
                                answer, confidence, stop_reason)
