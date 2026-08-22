# PWK-Bench

Submission snapshot for IEEE BigData 2026:
`v1.0-ieee-bigdata-2026-submission`

This repository is the public, fixed-code snapshot referenced by the paper
submission. The manuscript is self-contained for core claims; this repo provides
the executable code paths, complete output tables, and extended robustness
artifacts used to reproduce those claims.

PWK-Bench (Planning What to Know Benchmark) is the benchmark and
reference code release for **"More Evidence Is Not Always Better:
Benchmarking Evidence Acquisition in Biomedical Agents."** Given a query
budget, does a biomedical agent acquire
the *right* evidence, in the *right order*, and stop once it has enough? This
repo implements the paper's methodology end-to-end: two complementary
evaluation paradigms (a frozen evidence pool with an exactly enumerable
measurement ceiling, and live tool use with agent-controlled stopping) on a
real 491-variant ClinVar cohort. The paper's central finding: **more evidence
is not always better** — acquisition can degrade ranking performance and
agents keep querying past the point of retrospective sufficiency.

Model credentials are read only from environment variables
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`). No API key is
stored in this repository.

## Repository Layout

```text
.
├── pwkbench/               shared library: metrics, strategies, harness, domains, agents
├── scripts/
│   ├── variant/            frozen evidence pool -- run + reproduce (README here)
│   ├── live/               live tool-using agent + RQ3 masking check (README here)
│   ├── sphere/             second domain, synthetic ADRC cohort (README here)
│   └── analysis/           supplementary robustness checks (README here)
├── configs/models.yaml     model matrix (demo / real_all_api / ...)
├── data/                   bundled cohorts -- real vs. synthetic (README here)
├── docs/                   methodology reference (real-data recipe, masking, controls audit)
└── tests/                  test_pwkbench (variant) + test_sphere + test_live
```

Scripts are runnable directly from a checkout (`python3 scripts/<block>/x.py`);
each inserts the repo root on `sys.path` itself, so `PYTHONPATH=.` is not
needed. Running any script writes `results/<block>/` and `figures/<block>/`,
not bundled here — they are the output of reproduction, not an input.

## Data

Three distinct cohorts are bundled; "synthetic" does not mean the same thing
for all three — see [`data/README.md`](data/README.md) for the full
breakdown (including the RQ3 masking-check cohorts). In short:

| File | What it is | Used for |
|---|---|---|
| `data/sample/cohort_variant_synthetic.parquet` | Fully synthetic toy cohort | `--demo` only — a pipeline check, never the paper's numbers |
| `data/sample/cohort_full_real.parquet` | Fully real, already-annotated 491-variant cohort — no fetch step needed | Real runs — this is what the paper's headline numbers come from |
| `data/sphere/` | Stanford ADRC's own SPHERE release (officially synthetic) | Second-domain transfer check only, no clinical claim |

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Checks

No API key, no cost — verify the package before spending anything:

```bash
pytest -q                                    # correctness tests: oracle bound, fixed-order
                                              # RelMax, order-optimality signature, memorization control
python scripts/variant/run_benchmark.py --demo
```

The demo writes `results/variant/results.csv` + four figures to
`figures/variant/` from the fully synthetic toy cohort in ~30s. Add
`--robustness` to also exercise the RQ3/RQ4 analysis scripts (see
[`scripts/variant/README.md`](scripts/variant/README.md) for the full output
list). **Note on `scaling_gap.png` in the demo:** the offline mock agent ties
its capability proxy to how often it follows the relevance order, so the
demo scaling trend is a constructive artifact (it can even invert) and the
panel is stamped "SYNTHETIC — illustrative only"; real models replace it.

## What it computes

| Axis | Question | Metric |
|---|---|---|
| **A. Acquisition efficiency** | what / how much | `A(k)` per-gene ranking AUC; PE, oracle gap `Δ`, budget-to-target `k*` |
| **B. Order alignment** | which to acquire next | prefix-set overlap (primary) and Spearman `ρ` (secondary) vs. relevance and per-variant decisiveness orders |
| **C. Memorization control** | is acquisition necessary? | closed-book masked AUC vs. chance |

Two references bracket every agent: **RelMax** (one fixed global relevance
order, lower reference) and the cohort-level **measurement ceiling**
(exactly enumerable with 4 evidence categories, upper reference).

## Reproducing the paper's real numbers

Three run targets, one per block — each links to the block's own README for
the full command set, output files, and caveats:

- **Frozen evidence pool** — budget-AUC curves, order alignment, RQ3/RQ4
  robustness analyses. `gpt-5.5` (headline) + `gemini-2.5-pro` (second
  vendor), pure API mode, ready-to-run config:
  ```bash
  export OPENAI_API_KEY=... GEMINI_API_KEY=...
  python scripts/variant/run_benchmark.py \
      --cohort data/sample/cohort_full_real.parquet --models real_all_api \
      --robustness --outdir results/variant/real_full
  ```
  See [`scripts/variant/README.md`](scripts/variant/README.md) (smoke run
  first, run-to-run variance, context ablation).

- **Live tool-using agent** — over-acquisition, native stopping. Same two
  vendors:
  ```bash
  python scripts/live/run_live.py --vendor openai --tag openai
  python scripts/live/run_live.py --vendor gemini --tag gemini
  ```
  See [`scripts/live/README.md`](scripts/live/README.md) — this is also
  where `claude-sonnet-5` appears, but **only** for the RQ3 closed-book
  masking check, not this main loop.

- **SPHERE (second domain)** — synthetic multi-modal transfer check:
  ```bash
  python3 scripts/sphere/run_sphere.py --data data/sphere --task ad_vs_hc \
      --outdir results/sphere --figdir figures/sphere
  ```
  See [`scripts/sphere/README.md`](scripts/sphere/README.md).

- **Supplementary analysis** — five smaller checks against already-produced
  results (Oracle exactness, sufficiency selection bias, the
  `DiscriminativeGreedy` baseline, calibration, a reference stopping rule).
  See [`scripts/analysis/README.md`](scripts/analysis/README.md).

To build the real 491-variant cohort from scratch instead of using the
bundled one, see [`docs/variant/real_data.md`](docs/variant/real_data.md)
(exact public sources: ClinVar ≥2-star, gnomAD v4, gnomAD constraint,
AlphaMissense).
