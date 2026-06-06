# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Full daily pipeline (what cron runs)
bash run_daily_pipeline.sh

# Individual steps
python3 get_new_tsa.py          # scrape tsa.gov → data/tsa_volume.csv
python3 get_weather.py          # pull Open-Meteo → data/weather_national_features_with_lags.csv
python3 build_features.py       # rebuild master_features.csv
python3 autogluon_predict.py    # generate weekly forecast + probability summary
python3 kalshi.py               # snapshot markets + place orders (live)
python3 kalshi.py --dry-run     # simulate trading without placing orders
python3 send_update.py          # send Telegram message
python3 send_update.py --dry-run  # print message without sending

# Retrain production model (required on a fresh machine or after adding data)
python3 autogluon_full.py       # → output_autogluon_best/ag_final/ + OOF CSV

# Peer-comparison evaluation (fixed holdout, not used for inference)
python3 autogluon_evaluate.py
```

Environment variables are loaded from `.env` by `run_daily_pipeline.sh` via `source .env`. Required vars: `KALSHI_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`, `KALSHI_ENV` (prod/demo), `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`.

## Architecture

### Data flow (daily pipeline)

```
tsa.gov  ──► data/tsa_volume.csv ──┐
Open-Meteo ► data/weather_*.csv ───┤
data/google_trends.csv ────────────┴─► build_features.py ─► master_features.csv
                                                                      │
                                              output_autogluon_best/ag_final
                                                                      │
                                              autogluon_predict.py ◄──┘
                                                      │
                                    ┌─────────────────┴───────────────────┐
                              weekly_forecast.csv               weekly_summary.csv
                            (daily actuals+preds)           (avg, std, p_over_XM cols)
                                         │                          │
                              kalshi.py ◄┘                          │
                                  │                                 │
                      market_snapshot_latest.csv                    │
                                  │                                 │
                              send_update.py ◄──────────────────────┘
                                  │
                            Telegram message
```

### Two-model separation

| File | Model dir | Purpose |
|---|---|---|
| `autogluon_full.py` | `output_autogluon_best/ag_final` | **Production model** — trains on all available data; generates `oof_predictions.csv` used for uncertainty estimation |
| `autogluon_evaluate.py` | `output_autogluon_evaluate/ag_model` | Peer-comparison evaluation on a fixed holdout (2025-01-06 → 2026-01-04); never used for live inference |
| `autogluon_predict.py` | loads from `output_autogluon_best/ag_final` | Runs daily inference; also loads OOF residuals from `output_autogluon_best/` for uncertainty |

**The evaluate model is not the production model.** Retraining is done with `autogluon_full.py`.

### Feature pipeline (`build_features.py`)

This file is both a runnable script and a shared module imported by all training/inference scripts. The pipeline builds ~100 engineered features, then prunes to the `KEEP_FEATURES` list (77 features fed to the model). Key feature families:
- Calendar + holiday proximity (nearest holiday, rel-day, DOW interactions)
- Volume lag/rolling (lag 1/3/7/14/28, same-DOW lags, DOW 4-week means)
- Regime anchors — `anchor_master`, `lag365_blend_anchor`, `weather_penalized_anchor` — blended year-over-year same-DOW references adjusted by recent momentum
- Within-week cumulative volume and DOW share
- Weather penalty features (weighted snow/storm across 16 hub airports)
- Google Trends signals (lagged 7 days to avoid leakage)

To add a new feature: add it to `build_features_from_df()`, add its name to `KEEP_FEATURES`, then re-run `build_features.py` and `autogluon_full.py`.

### Autoregressive inference (`autogluon_predict.py`)

For future days in the current week, `build_features_for_date()` appends prior days' predictions to the TSA series before running the full feature pipeline. This means each day's prediction is fed as input to the next — prediction error compounds through the week.

### Kalshi trading (`kalshi.py`)

- Auth uses RSA-PSS-SHA256 (Elections API at `api.elections.kalshi.com`); keys loaded from `.env`
- `fetch_portfolio_summary()` returns normalised positions/orders consumed by `send_update.py`
- Market snapshots are written to `output_kalshi/market_snapshot_TIMESTAMP.csv` on every run; `market_snapshot_latest.csv` is always overwritten
- Trading logic: reads `p_over_XM` columns from `weekly_summary.csv`, runs a Kelly ladder to size YES/NO positions, places aggressive market orders or passive limit orders depending on price

### Telegram update (`send_update.py`)

Assembles the daily message from six sections (yesterday actual vs. predicted, weekly forecast summary, per-day forecast table, model-vs-Kalshi probability tables for YES and NO sides, positions, open orders, risk/bankroll). The `<pre>` blocks render as monospace tables in Telegram HTML parse mode. `save_prev_snapshots()` copies today's summary/forecast to `prev_*` files for tomorrow's change comparison — this is called after a successful send, not before.

### Output directories

| Directory | Contents |
|---|---|
| `output_autogluon_best/` | Production model (`ag_final/`), OOF predictions, CV fold metrics |
| `output_autogluon_evaluate/` | Evaluation-only model, holdout metrics |
| `output_autogluon_predict/` | `weekly_forecast.csv`, `weekly_summary.csv`, `prev_*` copies |
| `output_kalshi/` | Per-run market snapshots, action logs, `market_snapshot_latest.csv`, `positions.json` |
| `data/` | Raw inputs: `tsa_volume.csv`, weather CSVs, `google_trends.csv` |
