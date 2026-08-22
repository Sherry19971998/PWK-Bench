# SPHERE (second domain)

A second instantiation of the budgeted-acquisition framework, used to show it
**transfers** to a heterogeneous-cost, multi-modal problem outside variant
interpretation — not to say anything about Alzheimer's disease.

> **SPHERE is a fully SYNTHETIC cohort** (Stanford ADRC SPHERE release),
> generated from the real ADRC cohort and intended by its authors for methods
> development, pilot analysis, grant applications, and teaching — **not for
> clinical findings**. Participant ids are literally `Synthetic537`,
> `Synthetic236`, ... **No clinical conclusion about AD is drawn or
> supported.** Access to the *real* ADRC cohort requires data-access approval
> and is future work.

## Run

```bash
# one-time: build the lightweight bundle from a full SPHERE download
# (drops scrna.csv at 1.1 GB; truncates each proteomics file to the top-500
#  highest-variance columns -- a label-free cut, so it cannot leak the outcome)
python3 scripts/sphere/prep_sphere.py --src /path/to/ADRC_SPHERE_all_data --out data/sphere

python3 scripts/sphere/run_sphere.py --data data/sphere --task ad_vs_hc \
    --outdir results/sphere --figdir figures/sphere
```

The bundled `data/sphere/` already contains the prepped output, so
`prep_sphere.py` only needs to run again if rebuilding from a fresh SPHERE
download.

## Cost model

Channels are acquired in **increasing acquisition burden**, encoded as an
**ordinal rank only** — `cognitive` (1, in-clinic battery) → `biomarkers` (2,
one blood draw) → `wgs` and `plasma` (both 3, blood draw + send-out assay;
tied, no order claimed between them). No price, ratio, or cumulative cost is
reported: the *order* is defensible from what each modality physically
requires, the *magnitudes* are not, and summing ranks would publish an
invented ratio. The curve runs on a **fixed cohort** — the participants
covered by *every* channel in the order (n=221 for AD vs. HC) — so a change
along the curve reflects the added modality, not a change of population.
`tau` is excluded from the curve because adding it collapses that
intersection to 37.

Two single-modality tables are written on purpose. `sphere_single_auc.csv`
scores each channel on its own covered participants, which is **not** a
like-for-like ranking (tau covers 102 participants, cognitive 358, and the
apparent winner flips with coverage alone); `sphere_single_auc_fixed.csv`
holds the participants fixed. Read the fixed one for any claim about
modalities.

Scoring uses a cross-validated logistic model over the cumulative feature
block rather than `metrics.curve_A`: a SPHERE channel is a *matrix* with no
per-feature calibrated direction, so the ACMG-shaped rank score used in the
variant domain is undefined here. The two domains' AUC columns are
comparable in meaning, not produced by identical code.

## Two guards to read before trusting any number

**Circularity.** `cognitive_scores.csv` carries the CDR staging block
(`b4_cdrglob`, `b4_cdrsum`, the six CDR boxes, `c2_cogstat`) — the instrument
the consensus diagnosis is *made with*. Those 11 columns alone score AUC
0.875, higher than all 73 cognitive columns together; the loader drops them
by default (`exclude_staging=True`, reported in
`sphere_staging_sensitivity.csv`), which moves the cognitive channel from
0.875 to **0.865**. This is not leakage the label-column check can catch —
the label really is absent from the file — it is the diagnosis being quoted
back to itself.

**Uncertainty.** Three spreads are reported and are not interchangeable:
CV-fold SD ≈ 0.010 (how much the estimate moves when the *same* people are
split differently — **not** a confidence interval), participant-bootstrap SD
≈ 0.047 (**4.4x larger**; how much it would move on a different 221 people),
and the paired between-budget contrast, which is the only one that answers
"did adding this modality change anything."

## What this domain does and does not show

The modality gap is large and unambiguous: on the fixed cohort the cheapest
channel scores 0.865 while both rank-3 channels sit near chance (0.534,
0.499), and no single modality saturates (max 0.865, against the variant
domain's 0.99 single channel). But **every paired between-budget contrast
has a 95% interval spanning zero** (e.g. adding WGS: −0.036 [−0.124,
+0.049]), so at n=221 with 46 positives this cohort cannot resolve the
budget points from one another. Read the result as *the framework
transfers*, not as evidence of over-acquisition.

## Supplement: differential-diagnosis pairs

AD-vs-HC is nearly saturated by one cheap channel, so its budget points sit
close together. `run_sphere.py` also scores the harder overlapping-dementia
pairs (`results/sphere/sphere_differential_gain.csv`; no figure is produced
for it): cognitive alone versus cognitive + biomarkers + wgs + plasma, on
each pair's own four-channel common cohort, averaged over 25 CV-fold seeds.
Still the same SYNTHETIC cohort — **no clinical claim about AD**, framework
transfer only.

| pair | n (pos/neg) | cognitive alone | +3 modalities | gain | seeds in which it falls |
|------|-------------|-----------------|---------------|------|-------------------------|
| AD vs LBD | 59 (46/13)  | 0.758 | 0.596 | **−0.162** | 25/25 |
| AD vs MCI | 101 (46/55) | 0.747 | 0.773 | **+0.026** | 6/25 |
| MCI vs HC | 230 (55/175)| 0.671 | 0.638 | −0.033 | 22/25 |
| AD vs PD  | 105 (46/59) | 0.896 | 0.813 | **−0.083** | 25/25 |
| AD vs PD+Dementia | 49 (46/3) | not scored — smallest class has 3 < 8 members | | | |

The direction is **not uniform**: adding the costly modalities hurts in
three of the four resolvable pairs and *helps* in AD-vs-MCI, so the honest
summary is a range (−0.162 to +0.026) with per-pair seed stability, not a
single headline decline. Where the gain is negative, part of the drop is
**statistical, not information-theoretic** — plasma contributes 500 columns
and wgs 167 against a smaller class of 13–55 people, so a linear model
over-fits and dilutes a strong cheap signal. The supportable claim is scoped
to that regime ("appending costly high-dimensional modalities to an
already-sufficient cheap channel degrades discrimination *here*"), **not**
"more evidence always hurts."

`sphere_modality_bootstrap` intervals each modality separately on the fixed
AD-vs-HC cohort with the same participant-level resampling
(`results/sphere/sphere_modality_bootstrap.csv`). At B=400 the cognitive
interval [0.740, 0.935] clears wgs [0.327, 0.636] and plasma [0.388, 0.687]
outright, so the cheap-versus-expensive gap *is* resolvable at n=221 — but
it overlaps biomarkers [0.611, 0.823], so not every pairwise gap is. These
intervals are 0.10–0.16 wide, several times the CV-fold SD; the fold SD is
not a confidence interval and must not be quoted as one.
