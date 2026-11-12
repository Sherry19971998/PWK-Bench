# Phase / gap-anchor analysis (variant domain)

**These are PRESERVED OUTPUTS, not reproducible artifacts.** No script in this
tree regenerates them — they were recovered on 2026-08-01 from an unreferenced
top-level dump, where an automated sweep had parked them because nothing in the
repo cited them. Label them as preserved outputs wherever they are cited, and do
not present them as something a reader can rerun.

They are kept because they carry the mechanistic explanation for the null results
in `results/variant/real_trusted/`, and their underlying data survived intact.

| file | content |
|---|---|
| `phase_grid.json` | 15-cell sweep: scorer capacity {additive, logistic, nonlinear GBM} x feature-interaction strength {0, .25, .5, .75, 1.0}. Each cell carries `gap_excess` (planning gap over a shuffled-label null) and `z`. |
| `phase_anchors.json` | where two REAL domains land on that grid |
| `pm1_ablation.csv` | A(k) for Oracle / RelMax(4ch) / Heuristic / RelMax-with-PM1-masked |

## What the grid says

Only the **nonlinear scorer at maximum feature interaction** shows a positive
planning gap over the null (+0.054, z>1.5). Every other cell is within +/-0.018.

The real-domain anchors sit far from that corner:

| domain | interaction | scorer | gap_excess | z |
|---|---|---|---|---|
| variant (real, ACMG) | 0.05 | additive | **-0.056** | -0.98 |
| heart disease (real) | 0.15 | logistic | -0.001 | -0.15 |

So real clinical domains fall in the additive / low-interaction region where
greedy relevance ordering is already near-optimal. **The planning gap is not
missing because the agents are good; it is missing because the domain does not
reward per-instance planning.** That `z = -0.98` is the same statistic as
`real_trusted/consequence_gap.csv`, computed independently here.

## What pm1_ablation.csv says, and its one hard caveat

| k | Oracle(<=k) | RelMax 4ch | Heuristic | RelMax PM1-masked |
|---|---|---|---|---|
| 1 | 0.9943 | 0.8074 | 0.7854 | 0.8074 |
| 2 | 0.9943 | 0.9172 | 0.8017 | 0.9172 |
| 3 | 0.9943 | 0.8668 | 0.8071 | 0.8668 |
| 4 | 0.9943 | **0.8071** | 0.8071 | **0.8668** |

Masking PM1 removes the full-budget collapse (0.807 -> 0.867), so acquiring PM1
specifically is what costs the tail.

**CAVEAT: the Oracle here is 0.9943, but the real 491-variant cohort's Oracle is
0.9172** (`real_trusted/results.csv`). These rows are therefore computed on a
different, near-separable cohort. Use the SHAPE and the masked-vs-unmasked
contrast; do not quote the absolute values as real-cohort numbers.

## Figures drawn from these

- `figures/variant/analysis/phase_diagram.png` — the grid above
- `figures/variant/analysis/_view_001_four_case_gap.png` — no non-circular real
  cohort shows a significant planning gap
