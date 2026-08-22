# Supplementary analysis

Five scripts, each answering one reviewer-facing question about the frozen
or live results. All read already-produced artifacts (cohort + prior run
outputs); none make new API calls unless noted.

| Script | Question it answers | Output |
|---|---|---|
| `oracle_sensitivity.py` | Is the frozen Oracle an exact argmax over every ≤k-subset, or a myopic approximation? | `oracle_sensitivity.csv` |
| `sufficiency_coverage.py` | Do the 193 sufficiency-defined / 298 undefined live cases differ by gene or class? Is the 62% exceedance rate stable across strata? | `sufficiency_coverage.csv` |
| `afa_baselines.py` | Adds the `DiscriminativeGreedy` row (out-of-fold, gene-grouped AFA baseline) to `results.csv` | new row in `results.csv` |
| `calibration_pareto.py` | ECE / Brier score for the closed-book and full-acquisition live arms | `calibration_pareto_summary.csv` |
| `conformal_stop.py` | How does a reference stopping rule compare to the agent's own native stop? | `conformal_stop.csv` |

```bash
python scripts/analysis/oracle_sensitivity.py --cohort data/sample/cohort_full_real.parquet
python scripts/analysis/sufficiency_coverage.py --cohort data/sample/cohort_full_real.parquet
python scripts/analysis/afa_baselines.py --cohort data/sample/cohort_full_real.parquet \
    --results results/variant/real_full/results.csv
python scripts/analysis/calibration_pareto.py
python scripts/analysis/conformal_stop.py --cohort data/sample/cohort_full_real.parquet \
    --runs results/live/trajectories_full.csv results/live/trajectories_full_r1.csv \
           results/live/trajectories_full_r2.csv \
    --out results/live/conformal_stop.csv
```

`calibration_pareto.py` needs no `--cohort`; the others do. Run any script
with `--help` for its exact flags — most default to the paths the main
sweep (see [`scripts/variant/README.md`](../variant/README.md) and
[`scripts/live/README.md`](../live/README.md)) already produces.
