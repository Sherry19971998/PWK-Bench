"""Agent adapters. Real LLM adapters import lazily so the offline demo needs
no optional deps."""
# Required for the `X | None` annotations below: pyproject declares
# requires-python >=3.10, but this checkout runs on 3.9, where that syntax is
# evaluated at runtime and raises TypeError on import. Every other module in the
# package carries the same import for the same reason.
from __future__ import annotations

from .base import Agent, MockAgent

def build_agent(kind: str, model_id: str, name: str | None = None,
                adherence: float = 0.85, seed: int = 0, context: bool = False,
                max_workers: int = 1, cache_path: str | None = None,
                randomize_channel_order: bool = False,
                allow_stop: bool = False, stop_wording: str = "default"):
    """Factory. kind in {mock, anthropic, claude_cli, openai, gemini, hf}.
    `claude_cli` reaches Claude through the Claude Code CLI instead of the
    Messages API (no ANTHROPIC_API_KEY needed) -- read ClaudeCLIAgent's
    docstring before citing it: the model sees ~12.5k tokens of CLI harness
    ahead of the benchmark prompt, so it is NOT a clean vendor-axis control.
    For mock, `adherence`/`seed`/`model_id` come from the models.yaml row so
    each demo slot behaves differently (capability-scale sweep).
    `context` toggles the P1(b) variant-context ablation: when True the agent
    is told each variant's gene/consequence/domain and can plan
    instance-adaptively; when False every variant looks identical at k=1.
    `cache_path` (real agents only): disk-backed prompt->completion cache, so a
    sweep that dies mid-run (e.g. a provider's daily quota wall) resumes on the
    next invocation instead of re-asking everything from an empty cache.
    `allow_stop` (real agents only) offers the agent an explicit STOP action, so
    the number of categories acquired becomes the agent's decision rather than a
    budget the harness imposes; the agent then exposes `stop_k` alongside its
    order. Default False, which reproduces every previously reported number."""
    if kind == "mock":
        return MockAgent(adherence=adherence, seed=seed, model_id=model_id,
                         context=context)
    from .llm import (AnthropicAgent, ClaudeCLIAgent, OpenAIAgent, GeminiAgent,
                      HFLocalAgent)
    cls = {"anthropic": AnthropicAgent, "claude_cli": ClaudeCLIAgent,
           "openai": OpenAIAgent, "gemini": GeminiAgent, "hf": HFLocalAgent}[kind]
    return cls(model_id=model_id, name=name or kind, context=context,
               max_workers=max_workers, seed=seed, cache_path=cache_path,
               randomize_channel_order=randomize_channel_order,
               allow_stop=allow_stop, stop_wording=stop_wording)

__all__ = ["Agent", "MockAgent", "build_agent"]
