# Experimental controls — what the live benchmark holds constant

Audit of the eight protocol controls a controlled agent comparison is normally
expected to fix, checked against the code rather than against intent. Every row
cites the file and line that decides it.

**Verdict: 5 hold in full, 1 is a deliberate departure, 2 are partial.** The
deliberate departure and the two partials belong in the paper's limitations;
they are not defects that invalidate the comparison, but a reviewer will ask
about each one and the answers should be ready.

Scope: the live (unblinded) tool-using arm, `gpt-5.5` vs `gemini-2.5-pro`, as
run for `results/live/trajectories_full{,_r1,_r2}.csv`.

---

## Summary

| # | Control | Status | Where |
|---|---|---|---|
| 1 | Identical system prompt | ✅ full | `live/agent.py:27` (`SYSTEM`), used at `:178` and `:305` |
| 2 | Identical tool definitions and permissions | ✅ full | `_schemas` / `_dispatch` inherited, `live/agent.py:102,226,230`; `allow_tools` on both constructors (`:71`, `:259`) |
| 3 | Same maximum step count | ✅ full | `MAX_STEPS = 8` (`live/agent.py:25`), both loops `:183`, `:331` |
| 4 | Same context and observation | ✅ full | same `_dispatch` return fed back; wire format differs by protocol only |
| 5 | Same tool-call timeout | ✅ full | shared `live/tools.py:113` — 60 s, 4 retries (90 s for the MaveDB target list, `:310`) |
| 6 | Fixed temperature, or reported seeds | ⚠️ **deliberate departure** | neither is sent; `live/agent.py:78-83`, `:250-252`, `:314-320` |
| 7 | Full trajectory, tool calls and error logs | ⚠️ partial | `_trajectory` (`live/agent.py:163`) → columns below |
| 8 | Model / tool / environment failure distinguished | ⚠️ partial | `stop_reason` only; tool errors returned as payloads (`live/agent.py:155`) |

---

## 6 — Sampling: no fixed temperature, no seed

Neither arm forwards a temperature or a seed. This is deliberate and documented
in the code:

* `gpt-5.x` reasoning models **reject** a pinned temperature and treat `seed`
  as best-effort, so accepting one and forwarding it would misrepresent the
  guarantee (`live/agent.py:78-83`).
* Gemini *would* accept `temperature=0`, and setting it there while the OpenAI
  arm ran at its own default would have made the two arms differ in sampling as
  well as in model (`live/agent.py:314-320`).

So both arms run at their vendor's default sampler, and variability is
**measured rather than suppressed**: three independent repeats per case.

> This is the right call and should be defended, not apologised for. Freezing a
> sampler does not make a result robust — it hides the spread. The 12.4%
> answer-flip finding only exists because the sampler was left alone and the
> repeats were run.

Cost: **neither arm is reproducible from a seed.** State this plainly. The
reproducible artefact is the distribution over three runs, not any single run.

## 7 — What is and is not logged

`trajectories_*.csv` records, per case:

```
variant_id, gene, clinvar_name, n_tools_called, tools_called, distinct_tools,
cost_rank_sum, answer, confidence, stop_reason, allowed_tools, label,
predicted, correct, model_id, arm
```

That is the full **ordered tool-call sequence**, the answer, and the termination
reason — enough to recompute every acquisition metric in the paper.

Not saved: raw model messages, reasoning text, and tool **response payloads**.
A finding that depends on *what a tool returned* or on *what the model said*
cannot be audited from these artefacts; it would need a re-run with transcript
capture.

## 8 — Failure taxonomy is coarse

`stop_reason` takes three shapes: `final_answer`, `max_steps`, and
`error:<ExceptionClass>`. Tool-level failures never reach it — an unavailable
tool returns `{"error": ...}` as a normal observation (`live/agent.py:155`), and
a dead upstream returns `None` rather than raising (`live/tools.py:115-117`), by
design, so that a failed lookup appears to the agent as *evidence that did not
arrive* rather than as a crashed sweep.

There is therefore no systematic split between model failure, tool failure and
environment failure.

**In these runs it never bound.** All 491 cases in all three repeats terminated
with `final_answer`:

| run | `final_answer` | `max_steps` | `error:*` |
|---|---|---|---|
| full | 491 | 0 | 0 |
| full_r1 | 491 | 0 | 0 |
| full_r2 | 491 | 0 | 0 |

So no result in the paper rests on a failure that was silently miscategorised.
The taxonomy is a gap for *future* runs — particularly ablation arms and any
vendor with tighter rate limits — not a caveat on the present numbers.

---

## What to write in the paper

1. **Claim the five that hold.** One sentence: identical system prompt, tool
   definitions, permissions, step limit and tool timeout across both arms.
2. **State the sampler choice as a choice.** Vendor-default sampling with three
   repeats, because freezing the sampler would hide the run-to-run variability
   that is itself a finding. Note that single runs are not seed-reproducible.
3. **Bound the logging claim.** Acquisition behaviour is fully auditable from
   the released trajectories; model text and tool payloads are not retained.
4. **Do not claim a failure taxonomy.** Say instead that every case in the
   reported runs terminated normally, which is the stronger and verifiable
   statement.

## What would close the gaps

* Per-request timeout on the **model** call, matched across SDKs. Tool-side
  timeouts already match; the model call currently inherits each SDK's default,
  which is the one asymmetry left in the control set.
* Transcript capture behind a flag (off by default — it multiplies artefact
  size), so that claims about tool payloads become auditable.
* A `failure_kind` column separating model, tool and environment failures,
  populated at the point the exception or error payload is produced.

---

## Two limits on the tool-coverage reading

The poster states that population frequency answers 490 of 491 variants while
the in-silico tool answers 80. Both numbers are correct, and neither supports a
causal claim on its own. These two caveats were on the board and were moved here
to keep it readable; they belong in the paper.

### 1. "Always the first query" is confounded with list order

`get_population_freq` is the FIRST entry in `TOOL_SCHEMAS`
(`pwkbench/live/agent.py`). It is therefore impossible to tell from these runs
whether the agent opens with it because rarity is the right first filter,
because the tool answers nearly every variant, or simply because it is the first
tool it was shown. Three candidate explanations, one observation.

Two things argue against *pure* list-following, though neither settles it:

* Beyond position 1 the agent departs from list order — at position 2 gpt-5.5
  prefers domain context (45%) over the second-listed in-silico tool (35%), and
  gemini splits differently again.
* The ablation result is independent of ordering: removing the tool changes HOW
  MUCH is acquired (1.96 -> 0.80 queries per case), which list position cannot
  explain.

**We ran it.** `--tool-order reverse` (2026-08-04) presents the four tools in
the opposite order, so `get_population_freq` is shown LAST. `final_answer` stays
pinned last either way, so the stopping affordance is unchanged and order is the
only variable. Matched on the same 23 cases:

| arm | pop_freq is the first query | queries/case |
|---|---|---|
| original order, run 1 | 100.0% | 2.35 |
| original order, run 2 | 95.7% | 1.91 |
| original order, run 3 | 100.0% | 2.26 |
| **reversed order** | **95.7%** | 2.65 |

**The confound does not explain the result.** Shown last instead of first,
population frequency is still the opening query in 95.7% of cases — inside the
run-to-run range of the original order (95.7-100%). The opening move survives
the permutation, so "the agent opens with rarity" is a property of the agent's
behaviour, not of the list it was handed.

Two bounds on this. It is **23 cases, one run** — a smoke test, not the full
sweep; and queries/case came out at 2.65 against an original-order range of
1.91-2.35, which may be noise at this n or may be a real order effect on
QUANTITY (as distinct from opening move). Both are settled by re-running
`--tool-order reverse` over all 491 with three repeats:

```
python3 scripts/live/run_live.py --tool-order reverse --outdir results/live
```

Cost from the smoke run: ~1.2k input + 1.3k output tokens per case, so ~600k in
/ ~640k out for a full 491-case sweep, and roughly 60-100 min at 4 workers.

### 2. The coverage asymmetry is a property of THIS cohort's consequence mix

| | | |
|---|---|---|
| cohort | 491 variants, 12 genes | 262 benign / 229 pathogenic |
| missense | **80 (16%)** | the only variants AlphaMissense can score |
| loss-of-function (PVS1 = 1) | 151 (31%) | |

Population frequency reaches 490/491 because rarity applies to **any** variant
class — every variant either is or is not present in gnomAD, whether it is a
stop-gain, a frameshift, or synonymous. AlphaMissense scores **missense
substitutions only**, and this cohort is 16% missense; the UniProt domain
criterion is computed on the same 80. So 80/491 is not a defect in the tool or a
sampling error — it is what the criterion means, applied to this cohort.

The limitation that follows is real and should be stated: **"the in-silico tool
is fungible" is a statement about a loss-of-function-heavy cohort, not a general
property of in-silico evidence.** The 12 genes were filtered to
high-review-status pathogenic/benign calls, and in genes like BRCA1/BRCA2/TP53
the confidently classified variants skew to truncating (pathogenic) and
common-or-synonymous (benign). A missense-enriched cohort would give the
in-silico tool far more to do, and the ablation could come out the other way.
