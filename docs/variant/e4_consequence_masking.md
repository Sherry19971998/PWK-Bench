# Appendix: Consequence-token masking (response to reviewer Q4)

*Motivation.* The closed-book baseline reported in the main text (0.866 AUC
on full HGVS names; §Evidence Validity) is high in part because
loss-of-function status is free-readable from the variant name shown to the
agent (`p.Arg213Ter`, `...fs`) — PVS1 costs no tool call. To test whether
this number reflects consequence-parsing rather than memorization of
ClinVar labels, we re-ran the closed-book arm with the consequence
information progressively masked, and — to check whether any resulting
effect is idiosyncratic to one model family — replicated it on a second,
independently-trained frontier model (Claude Sonnet 5).

*Method — two masking levels, one revealing a leakage channel.* An initial
attempt redacted only the parenthetical p.-level annotation
(`(p.Arg213Ter)` → `(p.?)`), leaving the c.-level HGVS notation intact.
This did not remove the LoF signal: indel/frameshift status remains
readable from c.-level keywords (`c.5478_5479dup`, `c.2477del`) independent
of the p.-level annotation, and closed-book accuracy on the loss-of-function
subset (n=151) stayed at 1.000 (identical to unmasked). We report this as a
methods finding rather than discard it: it shows the closed-book ceiling is
anchored to *any* consequence-bearing token, not specifically the p.-level
one. The variant identifier was then replaced entirely with bare genomic
coordinates (`chr:pos:ref>alt`, the same format used as `variant_id`
elsewhere in this cohort), removing all HGVS-level annotation while leaving
gene identity available, unchanged from every other arm.

*Result — closed-book accuracy falls where the free-read hypothesis
predicts, in both models.*

| | gpt-5.5, unmasked | gpt-5.5, p.-masked | gpt-5.5, coords-only | Claude Sonnet 5, coords-only |
|---|---|---|---|---|
| overall | 0.866 | 0.794 | 0.621 | 0.616 |
| LoF (n=151) | 1.000 | 1.000 | 0.914 | 0.894 |
| missense (n=80) | 0.875 | 0.863 | 0.800 | 0.782 |
| other (n=256–260) | 0.785 | 0.654 | 0.396 | 0.402 |

Coordinate-only masking drops overall closed-book accuracy by 0.245
(gpt-5.5) and to a closely matching 0.616 (Claude Sonnet 5) — obtained
independently, on a different model family. Both models retain
above-chance accuracy on the LoF subset even with no HGVS annotation at all
(0.914 / 0.894), which we attribute to residual memorization of specific,
highly-studied variant *coordinates* in these 12 well-characterized genes
(BRCA1/2 positions in particular) — a distinct leakage channel from
consequence-parsing, and one that coordinate-level masking cannot itself
remove. Both models also show a systematic bias toward predicting
Pathogenic once stripped of annotation (gpt-5.5: 369/491 Pathogenic calls;
Claude Sonnet 5: 356/491) on a cohort that is not majority-pathogenic
(229/491 P/LP), which is why the "other" stratum falls *below* chance
(~0.40) in both models rather than settling toward it.

*Reading this result.* The closed-book ceiling reported in the main text is
not primarily memorization of ClinVar labels by variant identity: it
substantially depends on consequence information being legible in the name
shown to the agent, and removing that legibility drops accuracy by roughly
a quarter in two independently-trained frontier models. The residual signal
that remains (0.62 vs. a 0.5 chance floor) is concentrated in
loss-of-function variants in a small number of extensively-published genes,
consistent with coordinate-level rather than consequence-level
memorization, and is itself informative about where this benchmark's
closed-book floor comes from.

*Limitations.* This experiment is closed-book only (zero tools); it does
not speak to whether masking changes tool-acquisition behavior, which we
leave to future work. (A partial live/full-tool run under coordinate
masking was attempted — gpt-5.5 completed 348/491 cases across 9 of 12
genes before the run was stopped by API credit exhaustion, showing a large
rise in over-acquisition rate, 59.1%→94.0%, and mean tools/case,
1.838→3.526, on that subset — but coverage is incomplete (missing SCN5A,
LDLR, PCSK9) and is not reported as a headline result here.) The Claude
Sonnet 5 results were obtained via the Claude Code CLI in a zero-tool,
single-turn configuration rather than a direct Anthropic API call — a
different serving harness from the gpt-5.5 arms above. Because no tools are
involved, this distinction should not affect the reported numbers, but it
is named explicitly rather than presented as an interchangeable third API
arm.

## Reproducibility notes

- gpt-5.5 unmasked closed-book: `results/live/trajectories_notools.csv` (pre-existing, 491/491)
- gpt-5.5 p.-level masked closed-book: `results/live/masked/trajectories_masked_notools.csv` (491/491)
- gpt-5.5 coordinates-only masked closed-book: `results/live/masked/trajectories_masked_coords_notools.csv` (491/491)
- Claude Sonnet 5 coordinates-only masked closed-book: `results/live/claude_code/trajectories_masked_coords_notools.csv` (485/491, 6 CLI-level errors not yet retried)
- Masking utility: `pwkbench.live.tools.mask_consequence` (p.-level regex mask)
- Coordinates-only masked cohort: `data/masked/cohort_full_real_masked_coords.parquet`
- p.-level masked cohort: `data/masked/cohort_full_real_masked_p_level.parquet`
- Claude Code CLI runner: `scripts/live/run_claude_code_notools.py`
- Partial gpt-5.5 live/full-tools run (348/491, not a headline result): `results/live/masked/trajectories_masked_coords_full.csv`

A LaTeX version of this section (formatted for `\section{Closed-Book
Memorization Probe}\label{sec:closedbook}` as a new
`\subsection{Consequence-token masking}`) was drafted in conversation and
should be pasted into the paper source directly; it is not duplicated here
to avoid the two copies drifting apart.
