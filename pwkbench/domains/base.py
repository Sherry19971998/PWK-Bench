"""
Domain abstraction. A BudgetedAcquisitionDomain is a set of discrete evidence
channels over instances with a non-proxy gold label, plus a bucketized reveal.
Variant interpretation (paper) is an instantiation of this single interface —
the oracle, references, and metrics never change across domains.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np
import pandas as pd


@dataclass
class Domain:
    name: str
    channels: Sequence[str]          # ordered = fixed global relevance order (lower ref)
    channel_defined_on: dict         # channel -> "all" | "missense"-like subset flag column
    neutral_value: float = 0.5       # scalar fallback (fine for 0-1 channels)
    neutral_values: dict = None      # optional per-channel neutral (real data)
    masked_channels: tuple = ()      # channels HELD OUT of the axis-C leakage
                                     # control: a prior expert classification
                                     # (e.g. PRIOR_CLASS) is label-derived, so the
                                     # memorization probe must exclude it or the
                                     # "closed-book" AUC would just read the held-
                                     # out label back. Not excluded from planning
                                     # itself -- an agent may still acquire it --
                                     # only from the validity control.

    @property
    def K(self) -> int:
        return len(self.channels)

    def neutral_for(self, channel: str) -> float:
        """Per-channel neutral value. Channels live on different scales
        (PP3 in [0,1], PM2 = -log10 AF in ~[0,8], PM1 = mis_z), so a single
        0.5 would inject a spurious 'benign' signal into undefined entries of
        the wide-range channels. When `neutral_values` is set (real cohorts),
        use the channel's empirical midpoint; else fall back to the scalar."""
        if self.neutral_values and channel in self.neutral_values:
            return float(self.neutral_values[channel])
        return float(self.neutral_value)


class Cohort:
    """
    Holds a variant/instance table. Required columns:
      - 'variant_id', 'gene', 'domain', 'label' (1=positive/pathogenic, 0=negative)
      - one column per channel with the (already numeric) evidence value
      - one boolean column '<channel>__defined' per channel
    """
    def __init__(self, df: pd.DataFrame, domain: Domain):
        self.df = df.reset_index(drop=True)
        self.domain = domain
        self._validate()

    def _validate(self):
        # Use explicit raises, not assert: `python -O` strips asserts, which
        # would silently disable all cohort validation in an optimized run.
        need = {"variant_id", "gene", "domain", "label"}
        missing = need - set(self.df.columns)
        if missing:
            raise ValueError(f"cohort missing columns: {missing}")
        for c in self.domain.channels:
            if c not in self.df.columns:
                raise ValueError(f"missing channel column {c}")
            if f"{c}__defined" not in self.df.columns:
                raise ValueError(f"missing {c}__defined")
        if not set(self.df["label"].unique()) <= {0, 1}:
            raise ValueError("label must be 0/1")

    @property
    def y(self) -> np.ndarray:
        return self.df["label"].to_numpy(int)

    @property
    def genes(self) -> np.ndarray:
        return self.df["gene"].to_numpy()

    def value(self, channel: str) -> np.ndarray:
        """Raw numeric channel value; undefined entries carry the (per-channel)
        neutral value so wide-range channels don't get a spurious benign fill."""
        v = self.df[channel].to_numpy(float).copy()
        undef = ~self.df[f"{channel}__defined"].to_numpy(bool)
        v[undef] = self.domain.neutral_for(channel)
        return v

    def reveal(self, channel: str, n_buckets: int = 5) -> np.ndarray:
        """
        Bucketized qualitative reveal (paper: no raw score leaks). Returns
        integer bucket ids 0..n_buckets-1 for DEFINED entries; undefined entries
        return -1 (a distinct 'no data' code the agent can tell apart from a
        real low bucket).

        The quantile edges are computed on the DEFINED entries ONLY. Computing
        them on the neutral-filled vector let the many undefined rows (e.g. 411
        non-missense PP3 rows all == neutral) dominate the quantiles and collapse
        a 5-bucket channel to 2 -- destroying exactly the resolution the agent
        needs on the missense-only channels (PP3/PM1) that carry the
        per-instance planning signal.
        """
        raw = self.df[channel].to_numpy(float)
        defined = self.df[f"{channel}__defined"].to_numpy(bool)
        out = np.full(len(raw), -1, int)             # undefined -> -1
        vd = raw[defined]
        if vd.size == 0:
            return out
        edges = np.quantile(vd, np.linspace(0, 1, n_buckets + 1))
        # Collapse duplicated interior edges (low-cardinality channels, e.g.
        # binary PVS1) so digitize yields contiguous ids, not far-apart ones.
        interior = np.unique(edges[1:-1])
        out[defined] = np.digitize(vd, interior)
        return out

    def __len__(self):
        return len(self.df)


def load_real_cohort(df, domain):
    """Wrap a real (fetched) cohort DataFrame in a Cohort whose Domain carries
    an EMPIRICAL per-channel neutral value: the median over each channel's
    DEFINED entries. Channels live on different scales (PP3 in [0,1], PM2 =
    -log10 AF, PM1 = mis_z), so filling undefined entries with a flat 0.5 would
    inject a spurious benign signal into the wide-range channels. This is the
    single source of truth for real-cohort loading so every script
    (run_benchmark, run_closed_book, ...) scores the same parquet identically.
    """
    dom = attach_empirical_neutrals(df, domain, warn=True)
    return Cohort(df, dom)


def attach_empirical_neutrals(df, domain, warn=False):
    """Return a copy of `domain` whose neutral_values hold the EMPIRICAL
    per-channel neutral (median over each channel's defined entries). Channels
    live on different scales (PP3 in [0,1], PM2 = -log10 AF, PM1 = mis_z), so a
    flat scalar 0.5 injects a spurious signal into the wide-range channels --
    this is the single source of truth for BOTH real cohorts and the synthetic
    demo, so `--demo`, the tests, and real runs all score identically. Set
    warn=True to also flag zero-information channels (real-data path)."""
    import dataclasses, warnings
    neutrals, dead = {}, []
    for ch in domain.channels:
        dcol = f"{ch}__defined"
        if ch in df.columns and dcol in df.columns:
            n_def = int(df[dcol].astype(bool).sum())
            defined = df.loc[df[dcol].astype(bool), ch].astype(float)
            if len(defined):
                neutrals[ch] = float(defined.median())
            if warn and (n_def == 0 or defined.nunique() < 2):
                dead.append(f"{ch} (defined={n_def}, distinct={defined.nunique()})")
    if dead:
        warnings.warn(
            "zero-information channel(s) -- no variation among defined entries, "
            "so acquiring them is a no-op and the planning gap is not meaningful "
            "on this cohort: " + "; ".join(dead) + ". Expected for a PARTIAL real "
            "cohort (e.g. only PVS1 fetched); run fetch_annotations.py to fill "
            "PM2/PP3/PM1 before drawing conclusions.", stacklevel=2)
    return dataclasses.replace(domain, neutral_values=neutrals)
