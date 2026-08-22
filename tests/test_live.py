"""Tests for the live tool-using agent stack.

All offline: `_http_json` is monkeypatched everywhere a tool would reach the
network, so the suite never spends money and never depends on Ensembl/UniProt/
MaveDB being up. The API-shaped behaviour that IS worth pinning is the behaviour
that bit during development -- caching a transient failure, leaking one tool's
field into another, and reporting a failed lookup as evidence.
"""
import json
import threading

import pandas as pd
import pytest

from pwkbench.live import tools as T
from pwkbench.live.metrics_live import (TOOL_TO_CHANNEL, over_acquisition,
                                        sufficiency_points, yield_curve,
                                        cost_weighted)


# --------------------------------------------------------------------------
# tool layer
# --------------------------------------------------------------------------

def test_no_clinvar_assertion_tool_exists():
    """The removed leak must stay removed.

    The task label IS the ClinVar classification, so a tool returning prior
    ClinVar assertions hands the agent the answer at inference time -- a leak no
    memorization control catches. This guards against it being reintroduced
    under any name.
    """
    for name in T.TOOLS:
        assert "clinvar" not in name.lower(), (
            f"{name} looks like a ClinVar-assertion tool; the label is the "
            f"ClinVar classification, so such a tool serves the answer")
    assert set(T.TOOLS) == set(T.TOOL_COST_RANK) == set(T.TOOL_COST_BASIS)


def test_region_parsing():
    assert T._region("17:43049188:A>G") == "17 43049188 . A G"
    assert T._region("chr17:43049188:A>G") == "17 43049188 . A G"
    # the cohort really does contain one of these; it must not silently become
    # a queryable region
    assert T._region("17:43097244:na>na") is None
    assert T._region("garbage") is None


def test_failed_fetch_is_not_cached(monkeypatch):
    """A transient upstream failure must not become a permanent cached negative.

    MaveDB availability was measured at roughly 1 success in 4, so caching an
    empty index would make every later variant report "no functional data for
    <gene>" -- a fabricated cohort-wide negative from one timeout.
    """
    cache = T.ToolCache(None)
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        return None if calls["n"] == 1 else {"BRCA1": ["urn:mavedb:0000-a-1"]}

    monkeypatch.setattr(T, "_http_json", lambda *a, **k: None)
    first = T.get_functional_assay("1:1:A>G", "BRCA1", cache)
    assert first["n_score_sets"] is None
    assert "lookup failed" in first["interpretation"]
    assert cache.get("_mavedb_index") is None, "a failed index was cached"
    assert cache.get("get_functional_assay::1:1:A>G") is None

    # once upstream recovers, the same call must succeed rather than inherit
    monkeypatch.setattr(T, "_http_json", lambda *a, **k: [
        {"name": "BRCA1", "mappedHgncName": "BRCA1",
         "scoreSetUrn": "urn:mavedb:0000-a-1"}])
    second = T.get_functional_assay("1:1:A>G", "BRCA1", cache)
    assert second["n_score_sets"] == 1


def test_vep_tools_do_not_leak_each_others_field(monkeypatch):
    """One upstream call, two tools -- but each must expose only its own field.

    The measurement is which evidence the agent CHOOSES to acquire, so calling
    the frequency tool must not reveal the in-silico score as a side effect.
    """
    monkeypatch.setattr(T, "_http_json", lambda *a, **k: [{
        "colocated_variants": [{"frequencies": {"G": {"gnomade": 0.001}}}],
        "transcript_consequences": [
            {"alphamissense": {"am_pathogenicity": 0.97}}],
        "most_severe_consequence": "missense_variant"}])
    cache = T.ToolCache(None)
    freq = T.get_population_freq("1:1:A>G", "BRCA1", cache)
    assert "gnomad_af" in freq
    assert not any("missense" in str(k).lower() or "alpha" in str(k).lower()
                   for k in freq), freq
    assert 0.97 not in freq.values()

    ins = T.get_insilico_pathogenicity("1:1:A>G", "BRCA1", cache)
    assert ins["alphamissense"] == 0.97
    assert "gnomad_af" not in ins


def test_unqueryable_variant_reports_failure_not_rarity(monkeypatch):
    """'Could not query' must never be rendered as 'absent from gnomAD'.

    Absence from gnomAD is affirmative PM2 evidence; a failed lookup is not, and
    presenting one as the other fabricates evidence for the agent.
    """
    monkeypatch.setattr(T, "_http_json", lambda *a, **k: None)
    cache = T.ToolCache(None)
    r = T.get_population_freq("17:43097244:na>na", "BRCA1", cache)
    assert r["gnomad_af"] is None
    assert "lookup failed" in r["interpretation"]
    assert "PM2" not in r["interpretation"]
    assert "ultra-rare" not in r["interpretation"]


def test_cache_survives_concurrent_writes(tmp_path):
    """The runner fans cases across threads; concurrent puts must not lose data."""
    path = tmp_path / "c.json"
    cache = T.ToolCache(str(path))

    def worker(i):
        for j in range(20):
            cache.put(f"k{i}_{j}", {"v": i * 100 + j})

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]

    on_disk = json.loads(path.read_text())
    assert len(on_disk) == 120, f"lost writes: {len(on_disk)}/120"
    assert on_disk["k3_7"] == {"v": 307}


def test_protein_position_parsing():
    assert T._protein_pos("NM_007294.4(BRCA1):c.5339T>C (p.Leu1780Pro)") == 1780
    assert T._protein_pos("NM_007294.4(BRCA1):c.5478_5479dup (p.Met1827fs)") == 1827
    assert T._protein_pos("NM_000257.4(MYH7):c.2524A>G") is None


def test_cost_ranks_are_ordinal_and_documented():
    """Ranks order tools; they are never a price. Every rank needs a basis."""
    assert set(T.TOOL_COST_RANK.values()) <= {1, 2, 3}
    assert T.TOOL_COST_RANK["get_functional_assay"] > T.TOOL_COST_RANK["get_population_freq"]
    for name, basis in T.TOOL_COST_BASIS.items():
        assert basis and isinstance(basis, str)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cohort():
    from pwkbench.domains.variant import VARIANT_DOMAIN
    from pwkbench.domains.base import load_real_cohort
    return load_real_cohort(
        pd.read_parquet("data/sample/cohort_full_real.parquet"), VARIANT_DOMAIN)


def test_sufficiency_marks_unreachable_cases_as_nan(cohort):
    """k* must be NaN where the reference is never right, not silently maximal.

    The ACMG point rule calls most of this cohort VUS, which equals neither
    label, so it resolves only ~39%. Treating those as "needed everything" would
    invent an over-acquisition denominator out of the reference's own blind spot.
    """
    s = sufficiency_points(cohort)
    assert len(s) == len(cohort)
    assert s.k_star.isna().any(), "expected unreachable cases on this cohort"
    defined = s.k_star.dropna()
    assert defined.min() >= 0 and defined.max() <= 3
    assert (s.sufficient_at_zero == (s.k_star == 0)).all()


def test_over_acquisition_counts_distinct_channels_and_splits_unmapped(cohort):
    s = sufficiency_points(cohort)
    vid = s.variant_id.iloc[0]
    traj = pd.DataFrame([{
        "variant_id": vid, "gene": "BRCA1",
        # same tool twice acquires nothing new; the assay maps to no channel
        "tools_called": json.dumps(["get_population_freq", "get_population_freq",
                                    "get_domain_context", "get_functional_assay"]),
        "correct": True}])
    m = over_acquisition(traj, s)
    assert m.n_tool_calls.iloc[0] == 4          # raw calls, repetition visible
    assert m.n_mapped.iloc[0] == 2              # PM2 + PM1, deduplicated
    assert m.n_unmapped.iloc[0] == 1            # the assay
    assert m.over_acq_all.iloc[0] - m.over_acq_mapped.iloc[0] == 1


def test_functional_assay_maps_to_no_channel():
    """Pinned because it changes how over_acq_all must be read: the reference has
    no PS3 channel, so every assay call is over-acquisition by construction --
    a limit of the yardstick, not a judgement about the agent."""
    assert TOOL_TO_CHANNEL["get_functional_assay"] is None
    assert TOOL_TO_CHANNEL["get_population_freq"] == "PM2"
    assert TOOL_TO_CHANNEL["get_insilico_pathogenicity"] == "PP3"
    assert TOOL_TO_CHANNEL["get_domain_context"] == "PM1"


def test_yield_curve_accepts_either_tool_count_column():
    base = pd.DataFrame({"n_tools_called": [0, 1, 1, 2],
                         "correct": [True, True, False, True]})
    a = yield_curve(base)
    b = yield_curve(base.rename(columns={"n_tools_called": "n_tool_calls"}))
    assert list(a.n_tool_calls) == [0, 1, 2] == list(b.n_tool_calls)
    assert a.equals(b)


def test_cost_weighted_flags_assay_calls_on_genes_without_data():
    """An expensive call on a gene MaveDB has no data for is over-testing caught
    in the act; it must be counted, not hidden."""
    traj = pd.DataFrame([
        {"variant_id": "v1", "gene": "MYBPC3",
         "tools_called": json.dumps(["get_functional_assay"])},
        {"variant_id": "v2", "gene": "BRCA1",
         "tools_called": json.dumps(["get_functional_assay"])}])
    cw = cost_weighted(traj)
    assert cw.n_rank3_calls.tolist() == [1, 1]
    assert cw.called_empty_assay.tolist() == [True, False]
