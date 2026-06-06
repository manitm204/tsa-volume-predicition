#!/bin/bash
set -e

cd /home/manit/Desktop/fun_projects/tsa

set -a
source .env
set +a

# Shared run timestamp — all pipeline scripts use this to correlate DB rows.
export PIPELINE_RUN_TS=$(date -u +%Y%m%d_%H%M%S)
echo "[pipeline] Run ID: $PIPELINE_RUN_TS"

python3 get_new_tsa.py
python3 get_weather.py
python3 build_features.py

# Retrain Prophet daily / TS3 weekly (skips if not stale).
# OOF update is excluded — run `python3 train_models.py --update-oof` manually
# when you want to recalibrate ensemble weights or Platt sigma.
python3 train_models.py --no-oof || echo "[pipeline] model training failed — continuing"

python3 autogluon_predict.py

python3 kalshi.py --bankroll 250 --daily-bankroll 100
python3 send_update.py
