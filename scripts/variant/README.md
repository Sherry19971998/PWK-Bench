# Frozen evidence pool (variant domain)

Fixes the evidence pool and the acquisition budget so every strategy is
compared under identical, pre-fetched evidence. See the root [`README.md`](../../README.md#data) or
[`data/README.md`](../../data/README.md) for which cohort file is real vs.
demo — this page assumes you already know that and covers the run commands
and output files.

## Real-model sweep

`configs/models.yaml` defines the model matrix. `real_all_api` is the
ready-to-run block — `gpt-5.5` (headline) and `gemini-2.5-pro` (second
vendor), pure API mode, nothing to edit. `claude-sonnet-5` is deliberately
not in this block; see [`scripts/live/README.md`](../live/README.md) for why
and for the one experiment the paper does use it in.

```bash
pip install -e ".[openai,gemini]"
export OPENAI_API_KEY=... GEMINI_API_KEY=...

# 1) cheap smoke run first (gene-stratified subsample, ~a few variants/gene):
python scripts/variant/run_benchmark.py \
    --cohort data/sample/cohort_full_real.parquet --models real_all_api \
    --robustness --max-variants 60 --outdir results/variant/real_smoke

# 2) full paid sweep once the smoke run looks right:
python scripts/variant/run_benchmark.py \
    --cohort data/sample/cohort_full_real.parquet --models real_all_api \
    --robustness --outdir results/variant/real_full
```

Per-model `workers: N` in `models.yaml` fans the independent per-variant API
calls across N threads (identical result to serial, just overlaps latency).
`real` (edit `configs/models.yaml` → `real:` model_id fields) is the block to
use instead if you want to swap in different model versions.

`results/variant/real_full/results.csv` is what fills the paper's LLM-agent
rows. Model IDs are **not** hard-coded; whatever you set is recorded
verbatim in `results/`.

**Reliability guards on real-model runs:**
- Elicitation uses `max_tokens=256` and extracts the model's **text** block
  explicitly (not `content[0]`), so models with extended thinking don't
  return empty replies. A `refusal` stop reason raises rather than silently
  degrading.
- Every step where the model names no valid channel is **counted**, not
  silently absorbed: the runner reports `parse_failure_rate` per agent and
  warns if it exceeds 5%. A high rate means the numbers are a fixed-order
  fallback artifact, not model behaviour.
- Identical prompts are cached (the 491 identical no-context `k=1` calls hit
  the cache) and transient API errors are retried with backoff.
- Add `--with-ci` for gene-clustered bootstrap CIs on every `A(k)` (slower;
  this is the run that populates the paper's CI columns).

## `--robustness` output files (RQ3/RQ4 analyses)

- `memorization_probe.csv` — two-mask closed-book probe (RQ3): coordinate
  mask, full-HGVS mask, PVS1-alone ceiling, full-acquisition ceiling, and the
  acquisition-attributable gain `G_acq = full_acquisition − full_hgvs`
  (measured from the deployment-realistic identity-disclosed baseline, not
  from the near-chance coordinate-only floor — the latter is still reported
  separately as `memory_floor`, a validity-gate diagnostic).
- `curve_contrast_ci.csv` — **paired** gene-clustered bootstrap of the
  agent−RelMax A(k) gap per budget, with 95% CI, bootstrap p, and effect
  size (RQ2: "CI includes 0" = statistical tie).
- `domain_gap.csv` — per-disease-domain planning gap (RQ4 stratification).
- `ablation.csv` — evidence-pool ablation, dropping each channel in turn
  (RQ4: the gap survives dropping PP3). `metrics.holm_bonferroni` provides
  the multiple-comparison correction for multi-model claims (paper A4).
- `cost_robustness.csv` — strategy ranking under three cost calibers
  (per-query / per-token / per-monetary-cost); `ranking_invariant` confirms
  the ordering is insensitive to how a query is priced.
- `stopping.csv` — confidence-based stopping: mean queries spent and A at an
  adaptive per-variant stop budget vs. the full budget.
- `trajectory.csv` — acquisition-trajectory diversity: number of distinct
  orders and the first-query channel distribution (RQ2: a fixed strategy
  uses one order, an adaptive planner many).
- `complementarity.csv` — pairwise channel complementarity on the missense
  stratum (Spearman `ρ`, pair AUC, lift over the stronger single channel):
  is any evidence pair non-redundant, the precondition for order to matter.
- `clinical_yield.csv` / `yield_significance.csv` — per-budget resolved
  fraction, VUS fraction, and ACMG-points call accuracy per strategy, with
  paired McNemar-style significance against both RelMax and BestFixed at
  every budget.
- `yield_sensitivity.csv` — confidence-threshold sensitivity of the
  clinical-yield call at k=3 (resolved fraction and call accuracy across
  0.80/0.90/0.95 thresholds), one row per (strategy, threshold).
- `yield_by_domain.csv` — the same k=3 clinical-yield call, broken out by
  disease domain, one row per (strategy, domain).

`results.csv` (written even without `--robustness`) is the main per-budget
`A(k)` table across strategies/agents — this is what the paper's
per-budget-AUC table is read from.

Run offline (no key) with `--demo` in place of `--cohort ... --models ...`.

## Run-to-run variance

The main sweep runs each model once. To report run-to-run variance for one
model (e.g. the `gpt-5.5` `frontier_B` slot), `run_variance.py` re-runs the
full benchmark R times, giving each repeat its own cache so it genuinely
re-queries the API. Reasoning models (gpt-5.x) sample at temperature=1 —
these models reject a pinned temperature — so the spread across repeats is
real:

```bash
export OPENAI_API_KEY=...
pip install -e ".[openai]"

# smoke first (cheap): --max-variants 40
python scripts/variant/run_variance.py \
    --cohort data/sample/cohort_full_real.parquet \
    --config configs/models.yaml --models real \
    --slot frontier_B --runs 3 --outdir results/variant/variance

# -> results/variant/variance/variance_frontier_B.csv (one row per run + MEAN + SD rows)
# -> console prints mean +/- SD for A(k), PE, rho_vs_oracle, stopping_drop
```

If SD is near zero, that IS the finding — the greedy run is stable and a
single run suffices; state it as such rather than hiding the single-run
design.

`gpt-5.4` is the paper's other evaluated frozen-pool model (Table tab:agents:
"evaluated, 3 runs") and is reproduced the same way, using the
`real_frontier_D_only` block and its `frontier_D` slot instead of
`frontier_B`:

```bash
python scripts/variant/run_variance.py \
    --cohort data/sample/cohort_full_real.parquet \
    --config configs/models.yaml --models real_frontier_D_only \
    --slot frontier_D --runs 3 --outdir results/variant/variance
```

## Variant-context ablation (P1b)

At `k=1` with no per-variant context, every variant's prompt is identical,
so an agent cannot plan instance-adaptively and `rho_vs_oracle` is
structurally capped. The ablation runs the **same** model twice — with and
without variant context (gene / consequence / disease domain) — and reports
`rho_vs_oracle` for both:

```bash
python scripts/variant/run_benchmark.py --demo --domain variant --models ablation_demo \
    --outdir results/variant/ablation --figdir figures/variant/ablation
```

(the `--outdir`/`--figdir` overrides keep the ablation output out of
`results/variant`, so it doesn't overwrite the plain variant run.)

Offline (mock stand-in) this shows the mechanism: context raises
`rho_vs_oracle` (≈0.47 → 0.62) and shrinks the oracle gap. For real models
set `ablation_real` in `configs/models.yaml` (one `model_id`, two prompt
conditions). Reading:
- context **on ≫ off** → planning is limited by *context supply*, not model
  capability;
- context **on ≈ off** → the paper's strong claim, now with a control that
  answers "the gap is just a harness artifact."

Note: the oracle uses signed decisiveness (actual channel values), which the
agent cannot see at `k=1`, so `rho_vs_oracle = 1` is unreachable by
construction — the oracle is a *ruler, not a deployable policy*.

**On the bundled synthetic demo cohort the oracle reaches `A(k)=1.000` at
small budgets.** This is not a bug or a label leak: the oracle is *defined*
to use gold labels (it is an upper reference, not a strategy), and the
synthetic data is near-separable (almost every positive has an
above-neutral channel and every negative a below-neutral one), so a single
label-favourable channel already ranks each variant correctly. On this
clean data `Δ_oracle`, `domain_gap`, and the ablation gap therefore collapse
toward `1 − RelMax` and **do not reproduce the paper's real-data numbers**
(e.g. the paper's `oracle 0.900 vs RelMax 0.780` ablation). Those numbers
require the real cohort (`--models real_all_api` + `--cohort`), where the
oracle is below 1.0. The synthetic demo shows the *pipeline and metric
behaviour*, not the paper's headline gap.

## Provenance sensitivity (RQ4, Table tab:provenance)

Two-step pipeline — the first step makes real, rate-limited NCBI eutils
calls (one per variant's ClinVar submission comments; no bulk download,
no API key needed), the second is pure computation:

```bash
python scripts/variant/probe_clinvar_circularity.py
# -> results/variant/real_trusted/clinvar_circularity_probe.csv
#    (reads data/sample/cohort_full_real.parquet directly, no --cohort flag;
#    --n-sample N probes a random subsample first for a quick pilot)

python scripts/variant/make_circularity_robustness_check.py
# -> results/variant/real_trusted/circularity_robustness.csv
#    (the Oracle-RelMax k=1 gap recomputed under each provenance filter;
#    takes no arguments, reads the cohort and the probe CSV above directly)
```

`make_domain_gap_figure.py` (no new calls; ~2-3 min, re-enumerates the
exact oracle per domain per budget) writes the null-corrected,
cross-domain-comparable `domain_gap_excess.csv` — read this one, not the
raw `domain_gap.csv` from `--robustness` above, for any z/p-value claim
about a specific domain (the raw gap is confounded with domain size; see
the script's own docstring):

```bash
python scripts/variant/make_domain_gap_figure.py
```

## Reproducing the cohort from scratch

See [`docs/variant/real_data.md`](../../docs/variant/real_data.md) for the
exact public sources (ClinVar ≥2-star, gnomAD v4, gnomAD constraint,
AlphaMissense) and the fetch → annotate → run recipe. Not required for the
above — the annotated cohort is already bundled.
