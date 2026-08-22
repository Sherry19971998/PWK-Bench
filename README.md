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
`figures/variant/*`, and `figures/live/*`.

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
