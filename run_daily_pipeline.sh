#!/bin/bash
set -e

cd /home/manit/Desktop/fun_projects/tsa

set -a
source .env
set +a

# Optional first arg overrides the weekly-market bankroll (default 150).
BANKROLL="${1:-150}"
export KALSHI_BANKROLL="$BANKROLL"

# Cron's PATH excludes pyenv shims, so bare `python3` resolves to
# /usr/bin/python3 — which lacks autogluon and was silently breaking
# train_models, autogluon_predict, and the daily_predict subprocess inside
# kalshi.py. Pin the interpreter so cron matches an interactive shell.
PY="${PY:-/home/manit/.pyenv/shims/python3}"

# Shared run timestamp — all pipeline scripts use this to correlate DB rows.
export PIPELINE_RUN_TS=$(date -u +%Y%m%d_%H%M%S)
echo "[pipeline] Run ID: $PIPELINE_RUN_TS  PY=$PY"

"$PY" get_new_tsa.py
"$PY" get_weather.py
"$PY" build_features.py

# Retrain Prophet daily / TS3 weekly (skips if not stale).
# OOF update is excluded — run `python3 train_models.py --update-oof` manually
# when you want to recalibrate ensemble weights or Platt sigma.
#"$PY" train_models.py --no-oof || echo "[pipeline] model training failed — continuing"

"$PY" autogluon_predict.py

#"$PY" kalshi.py --bankroll "$BANKROLL" --daily-bankroll 100
"$PY" send_update.py
