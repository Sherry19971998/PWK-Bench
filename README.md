# PWK-Bench

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

The release is intentionally split into a short root README (paper-level
navigation) plus block-level READMEs (exact commands, output files, caveats).
This mirrors the execution structure and keeps method details close to the
scripts that implement them.

For scope clarity: the current submitted PDF's main reproduction path is
variant + live only. SPHERE is kept in this repo as an optional extra domain
experiment, not part of the submitted-paper headline claims.

## Repository Layout

```text
.
├── pwkbench/               shared library: metrics, strategies, harness, domains, agents
├── scripts/
│   ├── variant/            frozen evidence pool -- run + reproduce (README here)
│   ├── live/               live tool-using agent + RQ3 masking check (README here)
│   ├── sphere/             optional extra-domain sandbox (not in submitted-paper core path)
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
| `data/sphere/` | Stanford ADRC's own SPHERE release (officially synthetic) | Optional extra-domain sandbox only; not required for submitted-paper reproduction |

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

## Paper-aligned Metrics (RQ1-RQ4)

| RQ | Question | Primary metrics |
|---|---|---|
| **RQ1** | Marginal value of additional evidence; live stopping vs retrospective sufficiency | Frozen: `A(k)` (per-gene macro AUC), peak-to-final decline `max_k A(k)-A(K)`, clinical yield/resolution. Live: exceedance rate, mean over-acquisition |
| **RQ2** | Acquisition efficiency and order adaptation | `AE = mean_k A(k)`, `ΔOracle`, prefix overlap (`PO_rel`, `PO_dec`), secondary Spearman alignment (`ρ_rel`, `ρ_dec`) |
| **RQ3** | Acquisition-attributable signal vs closed-book signal | `G_acq = AUC_acq - AUC_HGVS`, plus masking controls (consequence mask, strongest mask) |
| **RQ4** | Reliability and sensitivity | run-to-run disagreement, cross-model live comparison, provenance/domain/context/cost sensitivity, calibration checks |

Two references bracket every agent: **RelMax** (one fixed global relevance
order, lower reference) and the cohort-level **measurement ceiling**
(exactly enumerable with 4 evidence categories, upper reference).

## Output Mapping (Code -> Paper Tables/Figures)

The repository does not treat GitHub text as a replacement for core paper
results; key claims remain in the manuscript. For reproducibility, the scripts
emit structured outputs that map to those claims:

- Frozen core curves and strategy metrics:
  `results/variant/*/results.csv`
- Clinical yield and paired significance:
  `results/variant/*/clinical_yield.csv`,
  `results/variant/*/yield_significance.csv`,
  `results/variant/*/yield_sensitivity.csv`
- Run-to-run variance (frozen):
  `results/variant/variance/variance_*.csv`
- Live trajectories and stopping metrics:
  `results/live/trajectories_*.csv`
- Provenance/circularity sensitivity:
  `results/variant/*/circularity_robustness.csv`,
  `results/variant/*/clinvar_circularity_probe.csv`
- Domain-level sensitivity:
  `results/variant/*/domain_gap*.csv`,
  `results/live/domain_*.png`
- Closed-book and masking controls:
  `figures/live/consequence_mask.png`, related masked trajectory outputs

## Reproducing the paper's main real-cohort results

This section is only for the paper's main real-cohort claims (491-variant
ClinVar cohort): frozen evidence pool + live agent loop.

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

To build the real 491-variant cohort from scratch instead of using the
bundled one, see [`docs/variant/real_data.md`](docs/variant/real_data.md)
(exact public sources: ClinVar ≥2-star, gnomAD v4, gnomAD constraint,
AlphaMissense).

## Optional extras not required for submitted-paper reproduction

SPHERE is provided as an optional extra-domain sandbox. It is not in the
submitted-paper core reproduction path.

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
