# Does the 0.865 closed-book AUC reflect memorization? A holdout investigation

**Question this answers.** The paper's closed-book baseline (full-HGVS-name AUC =
0.8651 on the 491-variant real cohort, `results/variant/real_trusted/memorization_probe.csv`)
is high enough that a reviewer could reasonably ask: is this because the 12
spec genes (BRCA1/2, TP53, PTEN, MLH1, MSH2, MYH7, MYBPC3, KCNQ1, SCN5A, LDLR,
PCSK9) are textbook-famous, heavily-published, and their ≥2-star ClinVar labels
sit in every frontier model's training data? This document is the record of
testing that directly, across four escalating approaches, ending in a real
LLM closed-book run (1,144 paid API calls) rather than a proxy.

**Bottom line up front.** No approach below found a memorization effect that
survives scrutiny. The two proxy-based tests were first underpowered, then
(at larger scale) confounded by two composition shifts that mimic a
memorization signal. The one real-LLM test that isolated gene identity and
class balance by construction found a suggestive single-gene gap (ATM) that
did not replicate across three additional genes and is not significant when
properly pooled (Cochran–Mantel–Haenszel χ²=1.625, p=0.202, n=1,144). This
result now substantiates — rather than merely asserts — the paper's existing
interpretation that `full_hgvs` AUC reflects LoF/missense consequence-reading
from the HGVS name, not recall of a memorized label.

---

## 0. A prerequisite finding: the 491-variant cohort cannot be re-sampled, only re-joined

`scripts/variant/fetch_clinvar.py` was re-run against a freshly downloaded
`variant_summary.txt.gz` (2026-08-10 snapshot) with the same seed
(`spec.GLOBAL_SEED = 20260101`) that produced the bundled
`data/sample/cohort_clinvar_real.parquet`. **It reproduced only 219/491 (45%)
of the original variant set.** ClinVar has grown/changed since the original
fetch, so the pool `.sample(random_state=seed)` draws from has a different
size and row order — the same seed now picks different rows. Labels agreed
on all 219 overlapping variants (0 mismatches), so the sampling logic itself
is not at fault; the underlying pool moved.

**Consequence for any future re-fetch:** do not regenerate the paper's cohort
by re-running `fetch_clinvar.py`. Instead join the existing 491 `variant_id`
values directly against a fresh `variant_summary.txt.gz` to recover metadata
(this is what was done here — all 491/491 matched with no ambiguity, one row
each).

---

## 1. Approach A — date-holdout on the original 12-gene panel (underpowered)

**Method.** `submission_summary.txt.gz` (NCBI, keyed on `VariationID`) gives
a *first-submission-date proxy* per variant: `min(DateLastEvaluated)` across
every submitted record for that `VariationID`. This is weaker than a true
first-appearance date (a submitter's `DateLastEvaluated` is when *they*
evaluated it, not when the variant entered ClinVar) but is the best signal
available without a full historical snapshot archive.

| first submission ≥ | n | P/LP | B/LB |
|---|---|---|---|
| 2023-06-01 | 57 | 23 | 34 |
| 2024-01-01 | 37 | 12 | 25 |
| 2024-06-01 | 27 | 7 | 20 |
| 2025-01-01 | 12 | 3 | 9 |

**Result.** `pwkbench.metrics.memorization_probe` (a logistic-regression
closed-book AUC proxy over gene-identity + LoF-flag features; confirmed to
reproduce the paper's exact numbers on the full 491-cohort: `coord_masked
0.4307, full_hgvs 0.8651, full_acquisition 0.9422`) gave `full_hgvs = 0.8211`
(n=57) and `0.8010` (n=37) — lower than 0.8651, superficially supportive of
the memorization concern.

**But:** 200 random size-matched subsamples of the full 491-cohort gave
`full_hgvs` AUC of **0.832 ± 0.063** (n=57) and **0.841 ± 0.081** (n=37); the
observed low-memorization values sit at the **43rd** and **30th percentile**
of that null — indistinguishable from ordinary sampling noise at this size.

**Verdict:** underpowered. The 12-gene, ≥2-star panel cannot supply enough
post-cutoff variants to test the hypothesis at all; the noise floor (SD
≈0.06–0.08) dwarfs any plausible effect size.

---

## 2. Approach B — expanded 31-gene panel, same proxy (confounded)

**Method.** Widened the gene list within the same four clinical domains the
paper already uses (adding genes to hereditary_cancer, cardiomyopathy,
arrhythmia, lipid_disorder — see script for the full list), same ≥2-star
P/LP+B/LB filter. This used data already on disk (`variant_summary.txt.gz`
covers all genes; the original fetch only *filtered* to 12) — no new
download was needed. Yield: **37,007 candidate variants across 31 genes**,
giving low-memorization slices of n=782 to n=4,673 depending on cutoff — two
orders of magnitude more power than Approach A.

**First look (looked like a real effect):** `hgvs_gain` (the acquisition-
attributable AUC gain, `full_hgvs − coord_masked`) dropped monotonically as
the cutoff tightened (0.264 full-panel → 0.230 → 0.194 → 0.171), and every
cutoff from 2024-01-01 onward sat at the **0th percentile** of 100 random
size-matched null draws — a seemingly unambiguous, large, significant signal.

**Diagnosed two confounds before trusting it:**

1. **PVS1 (loss-of-function) rate drops with recency**: 34.8% (full panel) →
   23.6% (≥2024-01) → 20.9% (≥2024-06). Unambiguous LoF variants get
   classified early and rarely need re-review; ambiguous missense VUS
   accumulate submission activity over years before resolving — so a
   "recently submitted" filter mechanically selects harder-to-classify
   variants, independent of any memorization question.
2. **Gene composition drifts with recency**: BRCA1's share falls from 13.2%
   (full panel) to 4.4% (≥2024-06 slice); BRCA2 falls 18.2%→9.8%. These are
   the two most extreme-base-rate, most-published genes in the panel — their
   disappearance alone inflates `coord_masked` (gene-identity-only AUC) in
   the low-memorization slice, which mechanically depresses `hgvs_gain` with no
   memorization involved. Genes that entered clinical multi-gene panels more
   recently (BRIP1, RAD51C, RAD51D, APOB) gain share instead.

**Re-tested against a PVS1-rate-matched null** (bootstrap constrained to draw
the same count of LoF and non-LoF variants as the observed slice): `full_hgvs`
is **no longer significant at any cutoff** (62nd–92nd percentile). `hgvs_gain`
still showed a 0th-percentile gap at 2 of 3 cutoffs, but that residual tracks
the *un*-matched gene-composition shift (point 2 above), not memorization.

**Verdict:** the large, "clean-looking" effect from the first pass was an
artifact of two composition confounds that both correlate with submission
recency in ClinVar for reasons that have nothing to do with what an LLM has
memorized. This is the negative control that made Approach C necessary.

---

## 3. Why the proxy metric cannot settle this — and what does

`memorization_probe`'s `full_hgvs` feature set is `[gene one-hot, PVS1==1
flag]` (`pwkbench/metrics.py:321`, docstring: "the consequence a full HGVS
name exposes ... is added as a feature"). **Restricted to a single gene, the
gene one-hot becomes a constant column carrying zero information, so
`full_hgvs` degenerates to the LoF flag alone — i.e., it collapses onto
`pvs1_alone` by construction.** The proxy was never built to detect
variant-level memorization (recall of *this specific* variant's ClinVar
label); it only detects gene-identity and coarse consequence-class signal.
Adding more variants cannot fix this — the ceiling is the feature
resolution, not the sample size.

**The only direct test is the paper's own live closed-book LLM arm**
(`scripts/live/run_live.py --no-tools`): the real model (`gpt-5.5-2026-04-23`,
OpenAI, no tools, shown only `Variant: <clinvar_name>` / `Gene: <gene>`) is
asked for a real answer, so anything the model actually recalls or infers
from the full HGVS name is captured — not a coarse stand-in for it.

---

## 4. Approach C — real LLM closed-book, within-gene matched design (the direct test)

**Design.** For each gene, take every "recent" (first-submission ≥
2024-06-01) variant, then sample an equal-sized, **class-balanced** ("old")
comparison set from that *same gene's* pre-2024-06-01 pool. Holding gene
identity and P/LP:B/LB ratio fixed by construction removes both confounds
found in Approach B by design, not by statistical adjustment. All labels are
the same ClinVar ≥2-star P/LP-vs-B/LB gold standard used throughout the paper
(`norm_sig` mapping, same as `fetch_clinvar.py`) — these are not a separately
invented ground truth.

### 4a. Pilot: ATM only (424 real API calls)

| arm | n | correct | accuracy |
|---|---|---|---|
| old | 212 | 189 | **89.15%** |
| recent | 212 | 179 | **84.43%** |

Fisher exact: OR = 1.515, **p = 0.196** (not significant). Directionally
consistent with the memorization hypothesis, but not distinguishable from
noise at n=212/arm. Both arms' errors were overwhelmingly the same direction
(true Benign called Pathogenic: 22/23 old errors, 30/33 recent errors) — a
systematic conservative bias in ATM calls, present in both arms equally, and
not evidence of a memorization difference by itself.

*Run cost: 776s, 108,939 input / 268,976 output tokens (257 in / 634 out per
case). Actual USD cost was not available to the assistant for this model/date
and should be checked on the OpenAI account's usage dashboard directly.*

### 4b. Scale-up: + BRCA2, PALB2, BRCA1 (720 more real API calls)

| gene | old accuracy | recent accuracy | gap |
|---|---|---|---|
| ATM | 89.15% (189/212) | 84.43% (179/212) | −4.7pp |
| BRCA2 | 97.16% (171/176) | **97.73%** (172/176) | **+0.6pp (reversed)** |
| PALB2 | 93.33% (98/105) | 92.38% (97/105) | −1.0pp |
| BRCA1 | 98.73% (78/79) | 97.47% (77/79) | −1.3pp |

*Run cost: 1,232s, 187,441 input / 359,115 output tokens.*

BRCA1/BRCA2/PALB2 all sit near ceiling (92–99%) in both arms with tiny or
reversed gaps — the ATM effect did not generalize.

### 4c. Combined analysis, 4 genes, 1,144 real API calls

- **Naive pooled:** old 93.71% (n=572) vs recent 91.78% (n=572). Fisher exact
  **p = 0.254**.
- **Cochran–Mantel–Haenszel, stratified by gene** (the correct test here —
  it removes each gene's very different base difficulty rather than pooling
  through it): **χ² = 1.625, p = 0.202.**

**Verdict: no statistically significant memorization effect, pooled or
per-gene, after controlling for gene identity and class balance by design.**

---

## 5. Overall conclusion

Four independent approaches were run, each addressing the failure mode of
the one before it:

| approach | what it controlled for | result |
|---|---|---|
| A: 12-gene proxy | — | underpowered (n≤57), no signal detectable |
| B: 31-gene proxy | sample size | signal appeared, then explained away by LoF-rate and gene-composition confounds |
| C-pilot: 1-gene real LLM | gene identity, class balance | suggestive gap, not significant (p=0.196, n=424) |
| C-full: 4-gene real LLM, CMH | + generalization across genes | not significant (p=0.202, n=1,144) |

**No approach here produced evidence that the paper's 0.865/0.866
closed-book AUC baseline is driven by memorization of these specific ClinVar
labels.** This is now an empirically tested claim, not just the mechanistic
argument already in `memorization_probe`'s docstring (that `full_hgvs`
approaching `pvs1_alone` reflects consequence-reading, not recall) — 1,144
real, paid model calls were spent trying to find the effect and it did not
show up at a level that survives gene-stratified pooling.

## 6. Caveats to carry with this result

- **The first-submission-date proxy is approximate**, not a true
  first-appearance date (same limitation `fetch_clinvar.py`'s docstring
  already flags for `LastEvaluated`; this document uses `min` over
  `submission_summary.txt.gz` records instead, which is tighter but still not
  exact).
- **The training-cutoff date is unknown.** 2024-06-01 was chosen pragmatically
  to balance sample size against plausibility, not derived from a published
  cutoff for `gpt-5.5-2026-04-23`. A different cutoff could shift the ATM
  result in particular.
- **Single vendor, single model.** Only OpenAI's `gpt-5.5-2026-04-23` was
  tested closed-book here; `gemini-2.5-pro` was not run through this holdout.
- **Single run per condition**, not repeated — consistent with this
  project's general caution that a single run's apparent effect can collapse
  under repetition. A second seed/run of the 4-gene comparison would
  strengthen this further and was not done here (cost/time tradeoff,
  discussed and accepted).
- **The ClinVar snapshot is a moving target** (see §0) — variant_summary.txt.gz
  changes over time, so exact reproduction of any slice built here requires
  the archived data files (see below), not a re-fetch.
- **ATM's non-significant gap is worth a footnote, not a claim.** It is the
  only gene of four that showed a directionally-consistent effect; whether
  that is gene-specific noise or a real but small effect neither this design
  nor its sample size can distinguish.

## 7. Data and reproducibility

Raw NCBI downloads (`variant_summary.txt.gz`, `submission_summary.txt.gz`) and
the derived per-VariationID first-submission-date lookup used for this
investigation are not bundled in this repo (NCBI ClinVar's bulk release files
run into the hundreds of MB); re-fetch them with
`scripts/variant/fetch_clinvar.py` and re-derive the date lookup from the
`submission_summary` file's per-VariationID earliest `submission_date`. The
491-variant cohort itself is bundled at
`data/sample/cohort_full_real.parquet`.

Key source files referenced: `scripts/variant/fetch_clinvar.py`,
`pwkbench/metrics.py::memorization_probe` (`_closed_book_auc`, line 263),
`pwkbench/domains/base.py::load_real_cohort`, `scripts/live/run_live.py
--no-tools`, `pwkbench/live/agent.py`.
