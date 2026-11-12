"""
Replay of RECORDED agent acquisition orders (no model, no API call).

This is what `scripts/variant/run_agent.py --replay` scores. It reads a
per-variant order file written by a real run (`run_benchmark.py` saves one per
agent as `<outdir>/agent_orders/<slot>.csv`) and hands those exact orders to the
harness, so every frozen-pool metric for that agent is recomputed offline from
what the model actually chose.

File format (CSV, one row per variant):
    variant_id, model_id, order[, stop_k]
`order` lists channel names separated by "|" (e.g. "PVS1|PM2|PP3|PM1").

The replay is strict on purpose: a variant in the cohort with no recorded order,
an unknown channel, or an order that is not a permutation of the domain's
channels is an ERROR, never a silent fallback -- a silently filled-in order
would be a fabricated trajectory.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import Agent
from ..domains.base import Cohort

ORDER_SEP = "|"


def write_orders(path: str, cohort: Cohort, order: np.ndarray, model_id: str,
                 stop_k=None) -> None:
    """Persist an (n, K) channel-index order array as a replayable CSV."""
    chans = list(cohort.domain.channels)
    df = pd.DataFrame({
        "variant_id": cohort.df["variant_id"].astype(str).to_numpy(),
        "model_id": model_id,
        "order": [ORDER_SEP.join(chans[int(j)] for j in row) for row in order],
    })
    if stop_k is not None:
        df["stop_k"] = np.asarray(stop_k)
    df.to_csv(path, index=False)


class RecordedOrderAgent(Agent):
    name = "recorded"

    def __init__(self, orders_path: str, name: str | None = None,
                 model_id: str | None = None):
        if not orders_path:
            raise ValueError("RecordedOrderAgent needs orders_path")
        self.orders_path = orders_path
        self.name = name or "recorded"
        self._df = pd.read_csv(orders_path, dtype={"variant_id": str})
        missing = {"variant_id", "order"} - set(self._df.columns)
        if missing:
            raise ValueError(f"{orders_path}: missing columns {sorted(missing)}")
        if self._df["variant_id"].duplicated().any():
            raise ValueError(f"{orders_path}: duplicate variant_id rows")
        ids = (self._df["model_id"].dropna().unique()
               if "model_id" in self._df.columns else [])
        self.model_id = model_id or (str(ids[0]) if len(ids) == 1 else "recorded")
        # Bookkeeping attributes the harness reads off live agents. Parse
        # failures are not recoverable from an order file, so they are 0 here;
        # the original run's results.csv is where those are reported.
        self.total_steps = self.parse_failures = 0
        self.forced_steps = self.forced_parse_failures = 0
        self.stop_k = None

    def order(self, cohort: Cohort) -> np.ndarray:
        chans = list(cohort.domain.channels)
        idx = {c: i for i, c in enumerate(chans)}
        rec = self._df.set_index("variant_id")
        want = cohort.df["variant_id"].astype(str)
        absent = want[~want.isin(rec.index)]
        if len(absent):
            raise ValueError(
                f"{self.orders_path}: no recorded order for {len(absent)} cohort "
                f"variant(s), e.g. {absent.iloc[0]!r}. The order file must come "
                "from a run on this same cohort.")
        out = np.empty((len(want), len(chans)), dtype=int)
        for i, vid in enumerate(want):
            names = str(rec.at[vid, "order"]).split(ORDER_SEP)
            try:
                row = [idx[n] for n in names]
            except KeyError as e:
                raise ValueError(f"{self.orders_path}: variant {vid!r} has "
                                 f"unknown channel {e.args[0]!r}") from None
            if sorted(row) != list(range(len(chans))):
                raise ValueError(f"{self.orders_path}: variant {vid!r} order "
                                 f"{names} is not a permutation of {chans}")
            out[i] = row
        if "stop_k" in rec.columns:
            self.stop_k = rec.loc[want, "stop_k"].to_numpy()
        return out
