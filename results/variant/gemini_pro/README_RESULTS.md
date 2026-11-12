# pwkbench Gemini results — gemini-2.5-pro (frozen block, Vertex ADC)

Sampler: DEFAULT temperature (no deliberate temp=0) + seed. SAMPLING_POLICY §2.1/§2.4/§2.5 applied
(temperature dropped, `_sampling_profile()` added, sampling profile in cache key).

## Files
| Path | What |
|---|---|
| variance_gemini_ctx/variance_frontier_F.csv   | context:TRUE  · 3-run mean±SD (quotable per §4) |
| variance_gemini_noctx/variance_frontier_F.csv | context:FALSE · 3-run mean±SD (quotable per §4) |
| gemini_noctx_default/results.csv              | context:FALSE · single run (post-fix) |
| gemini_ctx_notemp/results.csv                 | context:TRUE  · single run (pre-cache-key-fix; valid default-temp sample) |
| */.cache*.json                                | prompt caches (reproducibility; each variance run has its own) |

## Headline (gemini-2.5-pro, 3-run mean ± SD)
|              | rho_vs_oracle   | A_k1          |
|--------------|-----------------|---------------|
| context:false| 0.373 ± 0.190   | 0.807 ± 0.000 |
| context:true | 0.598 ± 0.007   | 0.834 ± 0.003 |

Context raises rank-correlation AND collapses its run-to-run variance (±0.19 → ±0.007).
A_k accuracy is stable across runs; rho is the noisy metric — quote it as mean±SD, not a single run.

## Caveats
- capability_proxy=0.80 is a PLACEHOLDER (don't read the Google point's ordering on the scaling plot).
- model_id pinned to gemini-2.5-pro (stable), run via Vertex ADC on project <your-gcp-project>.
- This is gemini-2.5-pro (the earlier handoff target), NOT the SAMPLING_POLICY doc's frozen row
  (that doc targets gemini-3.5-flash-lite / budget_C in a different repo copy).
