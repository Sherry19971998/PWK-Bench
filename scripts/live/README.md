# Live (tool-using) agent

The second evaluation paradigm: the agent decides both *which* of four real
evidence tools to call (gnomAD population frequency, AlphaMissense, UniProt
domain context, MaveDB functional assay) and *when* to stop, over the real
491-variant cohort.

**The paper's live comparison uses `gpt-5.5` (headline) and `gemini-2.5-pro`
(second vendor) only** — both run through their native API, no CLI
dependency:

```bash
export OPENAI_API_KEY=...  GEMINI_API_KEY=...
pip install -e ".[openai,gemini]"

python scripts/live/run_live.py --vendor openai --tag openai --max-variants 20   # smoke first
python scripts/live/run_live.py --vendor gemini --tag gemini
```

Writes `results/live/trajectories_<tag>.csv` (per-case: tools called, order,
stop reason, answer) and `tool_calls_<tag>.csv` (raw call log). `--no-tools`
runs the closed-book arm (withholds every tool) so the memorization floor can
be measured on the same cases; `--ablate` withholds one named tool.

`--vendor anthropic` also works (`AnthropicLiveAgent`, added for API-mode
parity with the other two vendors) but **the paper never runs Claude through
this full tool-calling loop** — Table tab:agents lists `claude-sonnet-5` as
"masking check only." Use it here only if you want to extend the live
tool-use comparison beyond what the paper reports. Claude's actual,
paper-reported role is the closed-book masking check below.

## Reproducing the RQ3 closed-book masking check (the paper's only claude-sonnet-5 experiment)

The masking check (§Evaluation Metrics / RQ3) withholds every tool AND masks
the variant identity itself (consequence and/or genomic coordinates), then
asks the model to classify from gene + masked identifier alone — a
closed-book memorization probe, not a tool-use trajectory. The paper's
reported claude-sonnet-5 number (masked accuracy on the strongest,
coordinate-masked condition, n=485/491 valid) was produced via the Claude
Code CLI:

```bash
python3 scripts/live/run_claude_code_notools.py \
    --cohort data/masked/cohort_full_real_masked_coords.parquet \
    --outdir results/live/claude_code --tag masked_coords_notools
```

The same masked cohort works through `run_live.py --vendor anthropic
--no-tools` (the API-mode path added in this repo) and sends a
prompt-identical closed-book query — a from-scratch-reproducible alternative
to the CLI arm, not guaranteed bit-identical (different trust boundary; see
`ClaudeCLIAgent`'s docstring in `pwkbench/agents/llm.py`):

```bash
export ANTHROPIC_API_KEY=...
python scripts/live/run_live.py --vendor anthropic --no-tools \
    --cohort data/masked/cohort_full_real_masked_coords.parquet \
    --tag claude_masked_coords --outdir results/live
```

`gpt-5.5`'s side of this same comparison uses the identical masked cohort
through the ordinary `openai` vendor arm with `--no-tools`. See
[`docs/variant/e4_consequence_masking.md`](../../docs/variant/e4_consequence_masking.md)
for the full masking methodology and the progressive (consequence-only vs.
consequence+coordinate) ablation.

## Reproducing the 3-run reliability data (12.4% flip rate, domain figures)

**This one costs 3x a normal live sweep** — it is NOT free/cached like the
figures above. The paper's run-to-run reliability headline (12.4% of cases
flip their pathogenic/benign call across 3 identical repeats) and two
figures derived from it — `figures/live/domain_reliability.png` and
`figures/live/domain_overacquisition.png` — read three independent full
live sweeps (`results/live/trajectories_full.csv`, `_r1.csv`, `_r2.csv`),
not a single run. Produce them with:

```bash
export OPENAI_API_KEY=...
python scripts/live/run_live_variance.py --vendor openai \
    --model gpt-5.5-2026-04-23 --runs 3 --tag full --outdir results/live
```

Then, no further API calls:

```bash
python scripts/live/make_domain_reliability_figure.py
python scripts/live/make_domain_overacquisition_figure.py
```

`run_live_variance.py --summarize-only --tags full,full_r1,full_r2 --label
openai_full` re-summarizes an existing set of three runs without spending
anything further.
