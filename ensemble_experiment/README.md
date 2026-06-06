# ensemble_experiment/

Four independent OOF forecasters on the same ~12-month window so their
per-day predictions can be compared 1:1.

## Models

| Script | Model | Budget | Notes |
|---|---|---|---|
| `ag_tabular.py` | AutoGluon Tabular, KEEP_FEATURES (34) | best_quality, 5min/fold × 4 folds + 10min final | Final full-data fit goes to `output/ag_final/` (not part of OOF) |
| `ag_timeseries.py` | AutoGluon TimeSeries, TS3 covariates | best_quality, 10min/fold × 4 folds | Day-by-day walk inside each fold (predict_length=1) |
| `prophet_oof.py` | Univariate Prophet (no regressors, no holidays) | weekly expanding folds (~52 fits) | Default yearly + weekly seasonality |
| `anchor_master_oof.py` | `anchor_master` feature from build_features | — | No training; reads column directly |

Walk-forward dates come from `sklearn.model_selection.TimeSeriesSplit(n_splits=4, test_size=91)` inside `ag_tabular.py`, then the other three scripts read `output/ag_tabular_oof.csv` to align their OOF dates 1:1.

## Run order

```bash
# 1. AG Tabular first — defines the OOF date range used by the others.
python3 ensemble_experiment/ag_tabular.py

# 2. The other three can run in any order.
python3 ensemble_experiment/ag_timeseries.py
python3 ensemble_experiment/prophet_oof.py
python3 ensemble_experiment/anchor_master_oof.py

# 3. Merge into one CSV + print MAE table and residual correlations.
python3 ensemble_experiment/combine_oof.py
```

All four model scripts cache per-fold predictions to `output/fold_predictions/`; reruns skip completed folds. Pass `--no-cache` to force retrain, `--clear-cache` to nuke the cache, `--report-only` to recompute summaries from the existing OOF CSV.

## Outputs (all in `output/`)

- `ag_tabular_oof.csv`, `ag_timeseries_oof.csv`, `prophet_oof.csv`, `anchor_master_oof.csv` — per-model OOF
- `combined_oof.csv` — merged on Date with one prediction column per model
- `*_summary.json` — fold metrics + overall MAE per script
- `ag_final/` — production-style AG Tabular model trained on all data (10min, best_quality)
