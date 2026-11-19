#!/bin/bash
set -e

cd /home/manit/Desktop/fun_projects/tsa

set -a
source .env
set +a

# Usage: run_daily_pipeline.sh [WEEKLY_BANKROLL [DAILY_BANKROLL]]
#   no args          -> kalshi.py dry-run (weekly 150, daily 100)
#   WEEKLY           -> live weekly trading only
#   WEEKLY DAILY     -> live weekly + daily trading
if [ $# -eq 0 ]; then
    BANKROLL=150
    KALSHI_ARGS=(--dry-run --bankroll "$BANKROLL" --daily-bankroll 100)
else
    BANKROLL="$1"
    KALSHI_ARGS=(--bankroll "$BANKROLL")
    [ -n "$2" ] && KALSHI_ARGS+=(--daily-bankroll "$2")
fi
export KALSHI_BANKROLL="$BANKROLL"

# Cron's PATH excludes pyenv shims, so bare `python3` resolves to
# /usr/bin/python3 — which lacks autogluon and was silently breaking
# train_models, autogluon_predict, and the daily_predict subprocess inside
# kalshi.py. Pin the interpreter so cron matches an interactive shell.
PY="${PY:-/home/manit/.pyenv/shims/python3}"

# Shared run timestamp — all pipeline scripts use this to correlate DB rows.
export PIPELINE_RUN_TS=$(date -u +%Y%m%d_%H%M%S)
echo "[pipeline] Run ID: $PIPELINE_RUN_TS  PY=$PY"

# tsa.gov can 403 automated clients; keep going on the last data and let
# send_update.py warn. Backfill with: get_new_tsa.py --html <saved page>
"$PY" get_new_tsa.py || export TSA_FETCH_FAILED=1
# "$PY" get_weather.py
"$PY" build_features.py

# Retrain Prophet daily / TS3 weekly (skips if not stale).
# OOF update is excluded — run `python3 train_models.py --update-oof` manually
# when you want to recalibrate ensemble weights or Platt sigma.
#"$PY" train_models.py --no-oof || echo "[pipeline] model training failed — continuing"

"$PY" autogluon_predict.py

echo "[pipeline] kalshi.py ${KALSHI_ARGS[*]}"
"$PY" kalshi.py "${KALSHI_ARGS[@]}"
"$PY" send_update.py
