# PWK-Bench

PWK-Bench (Planning What to Know Benchmark) is the benchmark and reference
code release for **"More Evidence Is Not Always Better: Benchmarking Evidence
Acquisition in Biomedical Agents."**

Core question: under a query budget, does an agent acquire the right evidence,
in the right order, and stop when enough evidence has been acquired?

Submitted-paper core reproduction path is variant + live; sphere is kept only
as a future-work extension track.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Model credentials are read only from environment variables:
`OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`.

## Quick Sanity Check

```bash
pytest -q
python scripts/variant/run_benchmark.py --demo
```

## Data Scope

| Data source | Role |
|---|---|
| `data/sample/cohort_variant_synthetic.parquet` | toy demo only (`--demo`), not paper headline numbers |
| `data/sample/cohort_full_real.parquet` | main real-cohort reproduction (paper core) |
| `data/sphere/` | future-work extension track only |

## Main Reproduction Path (paper core)

Run these blocks for the paper's main real-cohort claims (491-variant ClinVar
cohort).

1. **Frozen evidence pool (variant)**

```bash
export OPENAI_API_KEY=... GEMINI_API_KEY=...
python scripts/variant/run_benchmark.py \
  --cohort data/sample/cohort_full_real.parquet \
  --models real_all_api \
  --robustness \
  --outdir results/variant/real_full
```

2. **Live tool-using loop (live)**

```bash
python scripts/live/run_live.py --vendor openai --tag openai
python scripts/live/run_live.py --vendor gemini --tag gemini
```

Outputs are written to `results/variant/*`, `results/live/*`,
`figures/variant/*`, and `figures/live/*`. A frozen-pool run also writes
`<outdir>/agent_orders/<slot>.csv` (each agent's per-variant acquisition
order), which can be re-scored later with no model call:

```bash
python scripts/variant/run_agent.py \
  --replay results/variant/real_full/agent_orders \
  --cohort data/sample/cohort_full_real.parquet \
  --outdir results/variant/replay
```

`--replay` only scores recorded order files and fails if any cohort variant
has none. `--demo` is the synthetic mock pipeline check. It is not a replay
of any real model run.

## Reproducibility Scope

**Recomputable offline from this release (no API call):** the 491-variant
cohort and its labels, the reference strategies (RelMax, Heuristic, Greedy,
Oracle) with their A(k), AE, and clinical-yield numbers, the sufficiency
reference, and the RQ3 logistic probes.

**Recomputable offline from the released run outputs in `results/`:** the
original LLM trajectories and tool-call logs from the paper's runs. These
include the three live gpt-5.5 runs (`results/live/trajectories_full*.csv`),
the three live gemini-2.5-pro runs (`results/live/gemini_vertex_pro/`), the
closed-book and masking arms (`results/live/masked/`,
`results/live/claude_code/`), the tool ablations, and the live tool cache
(`results/live/.tool_cache.json`). From these, the figure and analysis
scripts reproduce live exceedance (62.0% ± 2.6%), the 3-run disagreement
(61/491), and the masked accuracies (0.794, 0.621, Claude 0.616 on 485/491
valid) with no API call. The tool cache covers every tool call in the
gpt-5.5 live runs. For the three gemini-2.5-pro live runs it covers about
98% of calls (34 distinct variant–tool responses are missing). Gemini
metrics are fully recomputable from the released trajectories, but a cached
re-run of Gemini would re-fetch those 34 responses live. Removed from the
release: per-case Claude CLI
scratch directories (`.work_full/`), a 12-case smoke run, and the cloud
project ID in two run summaries.

**Frozen-pool agent orders.** The per-variant orders of the paper's
three frozen gpt-5.5 runs and three frozen gemini-2.5-pro runs (no context)
were rebuilt offline from the released prompt caches with
`scripts/variant/rebuild_orders_from_cache.py`. Every prompt hit the cache,
and no API call was made. They are in
`results/variant/variance/agent_orders/` and
`results/variant/gemini_pro/variance_gemini_noctx/agent_orders/`. Replaying
them reproduces the reported values (gpt-5.5 A(k) 0.807/0.917/0.837/0.807 in
every run; gemini A(2) 0.810 ± 0.093, ρ_dec 0.373 ± 0.190):

```bash
python scripts/variant/run_agent.py --replay results/variant/variance/agent_orders \
  --cohort data/sample/cohort_full_real.parquet --outdir results/variant/replay_gpt55
```

Not replayable from cache: the single `real_trusted` frozen run (no cache
was kept; its summaries are in `results/variant/real_trusted/`) and the
context-augmentation runs (two of their three caches are missing 3–10
prompts). For these, the released summary CSVs are the record. A fresh paid
re-run will not be bit-identical (see decoding below).

**Decoding.** Temperature is never set. Frozen-pool API adapters send a seed
where the endpoint accepts one and allow up to 2,000 output tokens per step.
The live gpt-5.5 and Gemini runs use each vendor's default decoding with no
seed.

**Claude.** The paper's only Claude result, claude-sonnet-5 on the strongest
masking condition (HGVS name replaced by genomic coordinates, gene
retained), was run through the **Claude Code CLI**
(`scripts/live/run_claude_code_notools.py`). The Anthropic API path
(`run_live.py --vendor anthropic`) is a separate route and is not
guaranteed to reproduce the CLI numbers.

## Paper Metrics Map (RQ1-RQ4)

| RQ | Focus | Primary metrics |
|---|---|---|
| **RQ1** | Marginal value of added evidence; stopping vs retrospective sufficiency | `A(k)` (per-gene macro AUC), `max_k A(k)-A(K)`, clinical yield/resolution, exceedance rate, over-acquisition |
| **RQ2** | Efficiency and ordering quality | `AE`, `DeltaOracle`, `PO_rel`, `PO_dec`, `rho_rel`, `rho_dec` |
| **RQ3** | Acquisition signal vs closed-book signal | `G_acq = AUC_acq - AUC_HGVS`, masking controls |
| **RQ4** | Reliability and sensitivity | run variance, provenance/context/cost/domain sensitivity, calibration checks |

Reference bounds are the fixed global relevance order (RelMax, lower
reference) and the exactly enumerable cohort-level measurement ceiling (upper
reference).

## Output Mapping (Code -> Paper-facing artifacts)

- Frozen core curves/strategy metrics: `results/variant/*/results.csv`
- Live trajectories/stopping behavior: `results/live/trajectories_*.csv`
- Clinical yield and paired significance: `results/variant/*/clinical_yield.csv`, `results/variant/*/yield_significance.csv`
- Reliability/sensitivity probes: `results/variant/*/circularity_robustness.csv`, `results/variant/*/clinvar_circularity_probe.csv`
- Masking/closed-book controls: `figures/live/consequence_mask.png` and related masked outputs

## More Details

This root README is the shortest run path; block-level READMEs contain full
command matrices, caveats, and complete output inventories.

- Variant details and full output list: [scripts/variant/README.md](scripts/variant/README.md)
- Live details and run caveats: [scripts/live/README.md](scripts/live/README.md)
- Supplementary analysis helpers: [scripts/analysis/README.md](scripts/analysis/README.md)
- Data breakdown: [data/README.md](data/README.md)
- SPHERE future-work extension track: [scripts/sphere/README.md](scripts/sphere/README.md)

To rebuild the real 491-variant cohort from public sources instead of using the
bundled file, see [docs/variant/real_data.md](docs/variant/real_data.md).
