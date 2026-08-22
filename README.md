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

## Quick Sanity Check (no API spend)

```bash
pytest -q
python scripts/variant/run_benchmark.py --demo
```

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

## More Details

- Variant details and full output list: [scripts/variant/README.md](scripts/variant/README.md)
- Live details and run caveats: [scripts/live/README.md](scripts/live/README.md)
- Supplementary analysis helpers: [scripts/analysis/README.md](scripts/analysis/README.md)
- Data breakdown: [data/README.md](data/README.md)
- SPHERE future-work extension track: [scripts/sphere/README.md](scripts/sphere/README.md)

To rebuild the real 491-variant cohort from public sources instead of using the
bundled file, see [docs/variant/real_data.md](docs/variant/real_data.md).
