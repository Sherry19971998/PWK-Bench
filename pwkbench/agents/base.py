"""
Agent interface (Plan A). An Agent maps each variant to an acquisition ORDER
over channels (same output shape as the reference strategies), so the harness
scores agents and references through the identical metric path
(harness-invariance: the environment is fixed across all agents).

Adapters:
  - MockAgent      : deterministic, no API key. Drives the offline demo + CI.
  - AnthropicAgent : real, needs `anthropic` extra + ANTHROPIC_API_KEY.
  - OpenAIAgent    : real, needs `openai` extra + OPENAI_API_KEY.
  - HFLocalAgent   : real, needs `hf` extra (transformers/torch) + local model.

Real adapters are thin: they present each variant's revealed evidence to the
model and parse the model's chosen next channel, step by step, until the budget
is spent or the model stops. Model IDs are NEVER hard-coded — they come from
configs/models.yaml and are recorded verbatim in the results.
"""
from __future__ import annotations
import numpy as np
from ..domains.base import Cohort


class Agent:
    name: str = "agent"
    model_id: str = "unspecified"

    def order(self, cohort: Cohort) -> np.ndarray:
        """Return an (n_variants, K) array; row i = channel order for variant i."""
        raise NotImplementedError

    # convenience metadata for the results table
    def meta(self) -> dict:
        return {"agent": self.name, "model_id": self.model_id}


class MockAgent(Agent):
    """
    Deterministic offline agent. Simulates the paper's headline behaviour: on a
    fraction `adherence` of variants it follows the GLOBAL relevance order
    (relevance-greedy), and on the rest it follows that variant's oracle order.
    Higher `adherence` => closer to the relevance reference and farther from the
    oracle (rho high vs relevance, low vs oracle). Lower `adherence` mixes in
    more oracle-aligned orders, so a capability-scale sweep (frontier -> small
    open model) can be simulated by lowering `adherence`.

    This is a STAND-IN so the pipeline runs with no API key; it is NOT a real
    model and its numbers are illustrative, not the paper's results.
    """
    name = "mock"

    def __init__(self, adherence: float = 0.85, seed: int = 0,
                 model_id: str = "mock-relevance-greedy-v0",
                 context: bool = False):
        self.adherence = adherence      # P(follow global relevance order per variant)
        self.seed = seed
        self.model_id = model_id
        self.context = context          # P1(b) ablation: variant context supplied?

    def order(self, cohort: Cohort) -> np.ndarray:
        from ..strategies import relmax_order, oracle_order
        rel = relmax_order(cohort)
        ora = oracle_order(cohort)
        rng = np.random.default_rng(self.seed)
        follow = rng.random(len(cohort)) < self.adherence
        # per variant: use the whole relevance order (relevance-greedy) or the
        # whole oracle order — a full per-variant swap, not a local perturbation.
        out = np.where(follow[:, None], rel, ora)
        if self.context and "is_missense" in cohort.df.columns:
            # P1(b) ablation, context ON: a rational agent told the variant is
            # missense knows PP3/PM1 (missense-only channels) are decisive and
            # can follow the per-variant oracle there. Stand-in for the real
            # "context lets the agent plan instance-adaptively" effect.
            mis = cohort.df["is_missense"].to_numpy(bool)
            out = np.where(mis[:, None], ora, out)
        return out
