#!/bin/zsh
# Daily wrapper around scripts/warm_llm_cache.py for the Gemini frontier_C row,
# invoked by the launchd job com.pwkbench.warm-gemini-cache (see
# ~/Library/LaunchAgents/com.pwkbench.warm-gemini-cache.plist) at 15:00 and
# 16:00 every day. Idempotent: if the cache is already complete, or today's
# free-tier quota is already spent, this just logs that and exits 0 -- so the
# extra 16:00 trigger (a catch-up in case the machine was asleep at 15:00, or
# the 15:00 run hit the quota wall before finishing) never does anything
# harmful by running twice.
set -e
cd "$(dirname "$0")/.."
REPO="$PWD"
LOG="$REPO/logs/warm_gemini_cache.log"

{
  echo "===== $(date) ====="
  set -a
  source "$REPO/.env"
  set +a
  PYTHONPATH="$REPO" python3 "$REPO/scripts/warm_llm_cache.py" \
      --models real --slot frontier_C \
      --cohort "$REPO/data/sample/cohort_full_real.parquet"
  echo
} >> "$LOG" 2>&1
