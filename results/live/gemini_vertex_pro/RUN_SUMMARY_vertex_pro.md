# gemini-2.5-pro via Vertex AI — 3-run live-agent sweep

Run 2026-08-02. Full cohort (491 variants), all four evidence tools, 16 workers.
Total wall-clock ~18 min for all three runs.

## Model verification

The point of this column: `model_id` in every CSV is the string that was
*requested*. A run served by a different model looks identical in every
downstream artifact. `served_model` is what the server said answered.

| | |
|---|---|
| requested | `gemini-2.5-pro` |
| **served (verified, all 3 runs)** | **`gemini-2.5-pro`** |
| backend | `vertex` (project `<your-gcp-project>`, location `global`) |
| auth | gcloud Application Default Credentials — no API key involved |

Checked at three points: a pre-flight call before spending
(`GeminiLiveAgent._verify_served`, raises `ModelMismatch` rather than warning),
and across the three repeats before the variance table was computed
(`run_live_variance.py::_verify_models`, which is also what stops three tags
from two different models being averaged into one SD).

Vertex is not a preference here — it is the only way this row runs at all. On
the AI Studio key, `gemini-2.5-pro` returns 429 on call #1 with a **per-day**
quota id; the free tier's Pro allowance appears to be 0. See
`configs/models.yaml::real_gemini_pro`.

## Variance across the 3 runs

| run | n | mean_tools | zero_tool_frac | all_tools_frac | cost_rank_sum | answered_frac | error_frac | accuracy |
|---|---|---|---|---|---|---|---|---|
| vertex_pro_r0 | 491 | 1.5255 | 0.2627 | 0.1589 | 2.4969 | 0.9980 | 0.0020 | 0.9388 |
| vertex_pro_r1 | 491 | 1.5662 | 0.2240 | 0.1589 | 2.5519 | 1.0000 | 0.0000 | 0.9369 |
| vertex_pro_r2 | 491 | 1.5092 | 0.2444 | 0.1568 | 2.4379 | 1.0000 | 0.0000 | 0.9267 |
| **MEAN** | 491 | **1.5336** | 0.2437 | 0.1582 | 2.4956 | 0.9993 | 0.0007 | 0.9341 |
| **SD** | 0 | **0.0294** | 0.0194 | 0.0012 | 0.0570 | 0.0012 | 0.0012 | 0.0064 |

`mean_tools` CV is 1.9% — the aggregate replicates.

**One errored case in r0** (`error:ClientError`, 1/491). It is recorded as a row
with `n_tools_called=0`, which is indistinguishable from a genuine
zero-acquisition decision in every aggregate above. At 0.2% it does not move
anything, but it is the reason `error_frac` is reported separately at all.

## Per-case stability — read this before quoting any single variant

Identical tool count in **all three** runs: **45.0%** (270 of 491 variants differ
in at least one run).

So the aggregate numbers above replicate while individual trajectories largely do
not. Do not quote a per-case anecdote from this data. This matches what the three
existing gpt-5.5 runs showed (47%) — it is a property of the setup, not of Gemini.

## Comparison with the existing gpt-5.5 runs

Same cohort, same tools, same prompt, same stopping rule; only the vendor and
wire protocol differ.

| | gemini-2.5-pro (Vertex) | gpt-5.5-2026-04-23 |
|---|---|---|
| mean tools/case | **1.53** | 1.93 |
| zero-tool fraction | **0.244** | 0.111 |
| accuracy (answered) | **0.934** | 0.901 |
| per-case stability | 45.0% | 47% |

2.5 Pro acquires **less** evidence (1.53 vs 1.93 tools/case) and stops with zero
tools more than twice as often (24.4% vs 11.1%), while scoring slightly higher.

Two caveats before this becomes a claim:

1. **The accuracy ceiling.** The agent is shown the real HGVS name, and this
   repo's own closed-book probe scores AUC 0.865 from the name alone against
   0.942 with all evidence. Accuracy here is mostly recall; the tool layer has
   ~0.077 of room. The over-acquisition distribution is the intended headline,
   not the accuracy delta.
2. **Vertex vs AI Studio is a different serving stack**, and the gpt-5.5 rows
   came from OpenAI's API. This is a vendor comparison with a backend difference
   folded in on one side. The `backend` column exists so that cannot be lost.

Both arms run at their vendor's default sampler (no temperature, no seed, no
max_output_tokens on either side) — repeated runs are the only honest way to
bound spread, which is what this file is.

## Cost / tokens

Measured per run (r1 and r2; r0's totals were clipped from the console log by a
`tail`, and the two measured runs agree to within 0.6%):

| | per run | ×3 runs |
|---|---|---|
| input tokens | ~660k | ~2.0M |
| output tokens | ~464k | ~1.4M |
| wall-clock | ~360s | ~18 min |

Output tokens are ~98% **thinking** tokens, which Vertex bills as output. The
totals above already include them (the SDK reports `thoughts_token_count`
separately from `candidates_token_count`; omitting it would understate a
thinking model's spend by most of its output). At the list rate for 2.5 Pro this
is roughly $17–19 for the three runs — confirm the current rate rather than
trusting that number.

No 429s at 16 workers on this project.

## Files

- `trajectories_vertex_pro_r{0,1,2}.csv` — one row per case, incl. `served_model` / `backend`
- `tool_calls_vertex_pro_r{0,1,2}.csv` — raw tool-call log
- `variance_vertex_pro.csv` — the table above, incl. `requested_model` / `served_model`
- `variance_vertex_pro_stability.json` — per-case stability + the model-verification record
- `_ckpt_vertex_pro_r{0,1,2}.jsonl` — resumable checkpoints (safe to delete)

## Reproduce

```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=<your-gcp-project>
python scripts/live/run_live_variance.py --vendor gemini \
    --model gemini-2.5-pro --runs 3 --tag vertex_pro
```

`--model` has no default in either script: forgetting it used to silently run a
different model and write a complete, plausible results file labelled with it.
