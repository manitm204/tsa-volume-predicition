#!/usr/bin/env python3
"""
feature_select.py
=================
Two modes, same fixed holdout as autogluon_evaluate.py:

  FORWARD  (default)
    1. Train initial model with all features → rank by importance
    2. Seed set = top --seed-size features
    3. Greedy add: test each remaining feature; keep if MAE drops

  BACKWARD  (--backward)
    1. Start with BACKWARD_START (41 features) as the baseline
    2. Shuffle them into a random order
    3. Greedy remove: test each feature's removal; remove if MAE drops

Checkpoint after every step → safe to Ctrl-C and resume with --resume.

Evaluators
----------
  --evaluator autogluon   AutoGluon ensemble (stochastic, ~200-400s/eval)
  --evaluator lgbm        LightGBM fixed seed (~2-5s/eval)
  --evaluator xgb         XGBoost fixed seed (~2-5s/eval)
  --evaluator catboost    CatBoost fixed seed (~3-6s/eval)
  --evaluator ensemble    XGB + LGBM + CatBoost average (~6-15s/eval) [recommended]

Usage
-----
  python feature_select.py                                           # forward, autogluon
  python feature_select.py --evaluator ensemble                     # forward, ensemble (recommended)
  python feature_select.py --evaluator ensemble --resume            # resume
  python feature_select.py --backward --evaluator ensemble          # backward, ensemble
  python feature_select.py --evaluator ensemble --skip-importance   # skip importance ranking step
  python feature_select.py --seed-file output_feature_select/selected_features.txt --evaluator ensemble
  python feature_select.py --evaluator ensemble --seed-top-n 20
                                                                     # rank checkpoint's 40 features,
                                                                     # seed with top 20, test all ~80 others
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

try:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    _FAST_MODELS_AVAILABLE = True
except ImportError:
    _FAST_MODELS_AVAILABLE = False

from autogluon.tabular import TabularPredictor

from build_features import (
    load_master, load_and_merge_data, get_feature_columns, TARGET_COL,
    transform_target, inverse_transform_target, USE_LOG_TARGET, KEEP_FEATURES,
)

try:
    from joblib import Parallel, delayed as joblib_delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

# ── Fixed holdout (mirrors autogluon_evaluate.py) ─────────────────────────
TRAIN_END  = pd.Timestamp("2025-01-05")
TEST_START = pd.Timestamp("2025-01-06")
TEST_END   = pd.Timestamp("2026-01-04")

OUT_DIR   = Path("./output_feature_select")
TMP_DIR   = OUT_DIR / "ag_tmp"
CKPT_FILE = OUT_DIR / "checkpoint.json"
LOG_FILE  = OUT_DIR / "selection_log.csv"

RANDOM_STATE    = 42
FAST_EVALUATORS = {"lgbm", "xgb", "catboost", "ensemble"}

# ── Starting set for backward elimination ─────────────────────────────────
BACKWARD_START = [
    "nearest_holiday_expected_vol",
    "anchor_master",
    "blend_anchor_tight",
    "adaptive_anchor",
    "Volume_lag3",
    "lag365_2yr_avg",
    "dow_month_expanding_mean",
    "lag7_x_dow_share",
    "vol_diff_1",
    "lag365_blend_anchor",
    "anchor_x_dow_share",
    "is_holiday",
    "recent_max_vs_anchor",
    "vol_diff_7",
    "Volume_lag7",
    "nearest_holiday_aligned_lag",
    "anchor_x_holiday_dist",
    "dow_expanding_mean",
    "log_Volume_lag7",
    "streak_above_rm7",
    "anchor_accel_adjusted",
    "days_to_holiday_x_dow",
    "week_vol_cumsum",
    "dow_weekly_share",
    "holiday_expected_vs_normal",
    "weekend_return_pressure",
    "recent_trend_vs_ly",
    "volume_accel_3d",
    "dow_cos",
    "dayofweek",
    "lag730_same_dow",
    "weather_penalized_anchor",
    "vol_same_dow_vs_8w",
    "nearest_holiday_dow",
    "decline_magnitude",
    "regime_gap_3d_7d",
    # borderline features
    "sunday_weekend_regime",
    "volume_accel_pct_3d",
    "holiday_decay_shape",
    "last_week_avg",
    "fri_vs_expected",
]


# ==========================================================================
# Fast deterministic model builders (same configs as mini_ensemble.py)
# ==========================================================================

def build_lgbm():
    return LGBMRegressor(
        n_estimators=1500, learning_rate=0.02,
        num_leaves=64, min_child_samples=40,
        subsample=0.85, colsample_bytree=0.85,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
    )

def build_xgb():
    return XGBRegressor(
        n_estimators=1000, learning_rate=0.03,
        max_depth=4, min_child_weight=4,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.1,
        random_state=RANDOM_STATE, tree_method="hist", n_jobs=-1,
        verbosity=0,
    )

def build_catboost():
    return CatBoostRegressor(
        iterations=1000, learning_rate=0.03, depth=6,
        l2_leaf_reg=3.0, subsample=0.85, colsample_bylevel=0.85,
        random_seed=RANDOM_STATE, verbose=0,
    )

FAST_MODEL_BUILDERS = {
    "lgbm":     build_lgbm,
    "xgb":      build_xgb,
    "catboost": build_catboost,
}


# ==========================================================================
# Helpers
# ==========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Feature selection (forward or backward)")
    p.add_argument("--backward",        action="store_true",
                   help="Run backward elimination starting from BACKWARD_START")
    p.add_argument("--seed-size",       type=int, default=15,
                   help="(forward only) Top-N features to start with (default: 30)")
    p.add_argument("--time-limit",      type=int, default=200,
                   help="AutoGluon time limit per eval in seconds (default: 200; ignored for fast evaluators)")
    p.add_argument("--preset",          default="good_quality",
                   help="AutoGluon preset (default: good_quality; ignored for fast evaluators)")
    p.add_argument("--evaluator",       default="autogluon",
                   choices=["autogluon", "lgbm", "xgb", "catboost", "ensemble"],
                   help="Evaluator to use for each candidate (default: autogluon)")
    p.add_argument("--swap",            action="store_true",
                   help="Swap mode: for each feature in the current set, try replacing it with any candidate")
    p.add_argument("--swap-start",      default=None,
                   help="(swap only) Path to selected_features.txt to start from; default: KEEP_FEATURES")
    p.add_argument("--n-jobs",          type=int, default=1,
                   help="(swap only) Parallel inner-loop workers; -1=all cores (default: 1=sequential)")
    p.add_argument("--candidates-keep-only", action="store_true",
                   help="(swap only) Restrict candidate pool to KEEP_FEATURES (default: all features from full pipeline)")
    p.add_argument("--resume",          action="store_true",
                   help="Resume from checkpoint.json")
    p.add_argument("--skip-importance", action="store_true",
                   help="(forward only) Skip importance step; use KEEP_FEATURES order")
    p.add_argument("--seed-file", default=None,
                   help="Load seed from a selected_features.txt file; test all remaining master features")
    p.add_argument("--seed-top-n",      type=int, default=None,
                   help="Read checkpoint's selected list, rank by importance, seed with top N; "
                        "all other master features become candidates")
    p.add_argument("--evaluate-only",  action="store_true",
                   help="Train once on the seed set, print MAE, and exit (no forward selection)")
    return p.parse_args()


def _fit_fast(train_df, test_df, features, evaluator):
    """Train deterministic model(s) on features; return (MAE, model_or_dict)."""
    X_tr = train_df[features]
    y_tr_raw = train_df[TARGET_COL]
    y_tr = transform_target(y_tr_raw) if USE_LOG_TARGET else y_tr_raw
    X_te = test_df[features]
    y_te = test_df[TARGET_COL].values

    builders = FAST_MODEL_BUILDERS if evaluator == "ensemble" else {evaluator: FAST_MODEL_BUILDERS[evaluator]}

    preds  = []
    fitted = {}
    for name, builder in builders.items():
        model = builder()
        model.fit(X_tr, y_tr)
        raw  = model.predict(X_te)
        pred = inverse_transform_target(raw) if USE_LOG_TARGET else raw
        preds.append(pred)
        fitted[name] = model

    mae = mean_absolute_error(y_te, np.mean(preds, axis=0))
    # Return a single model for single-model evaluators, dict for ensemble
    return mae, fitted if evaluator == "ensemble" else list(fitted.values())[0]


def _fit(train_df, test_df, features, time_limit, preset, evaluator="autogluon"):
    """Train model on features; return (MAE, model/predictor)."""
    if evaluator in FAST_EVALUATORS:
        return _fit_fast(train_df, test_df, features, evaluator)

    # ── AutoGluon path ────────────────────────────────────────────────────
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)

    train_in = train_df[features + [TARGET_COL]].copy()
    label = TARGET_COL

    if USE_LOG_TARGET:
        train_in["_log_vol"] = transform_target(train_in[TARGET_COL])
        train_in = train_in.drop(columns=[TARGET_COL])
        label = "_log_vol"

    predictor = TabularPredictor(
        label=label,
        path=str(TMP_DIR),
        eval_metric="mean_absolute_error",
        verbosity=0,
    ).fit(
        train_data=train_in,
        time_limit=time_limit,
        presets=preset,
    )

    pred_raw = predictor.predict(test_df[features])
    pred = inverse_transform_target(pred_raw.values) if USE_LOG_TARGET else pred_raw.values
    mae  = mean_absolute_error(test_df[TARGET_COL].values, pred)
    return mae, predictor


def rank_by_importance(model_or_predictor, test_df, features):
    """Return features sorted by importance (best first)."""
    # Ensemble dict: average importances across models
    if isinstance(model_or_predictor, dict):
        all_imp = np.array([m.feature_importances_ for m in model_or_predictor.values()])
        importances = all_imp.mean(axis=0)
        return [features[i] for i in np.argsort(-importances)]

    # Single sklearn-compatible model (LGBM / XGB / CatBoost)
    if hasattr(model_or_predictor, "feature_importances_"):
        importances = model_or_predictor.feature_importances_
        return [features[i] for i in np.argsort(-importances)]

    # AutoGluon: permutation importance on test sample
    try:
        sample = test_df[features + [TARGET_COL]].copy()
        if len(sample) > 250:
            sample = sample.sample(250, random_state=42)
        imp = model_or_predictor.feature_importance(
            data=sample,
            num_shuffle_sets=2,
            silent=True,
        )
        return imp["importance"].sort_values(ascending=False).index.tolist()
    except Exception as e:
        print(f"  [warning] feature_importance failed ({e}); using KEEP_FEATURES order")
        return [f for f in KEEP_FEATURES if f in features] + \
               [f for f in features if f not in KEEP_FEATURES]


def parse_feature_file(path):
    """Parse a selected_features.txt (KEEP_FEATURES = [...] format) → list of names."""
    features = []
    for line in Path(path).read_text().splitlines():
        line = line.strip().rstrip(',')
        if line.startswith('"') and line.endswith('"'):
            features.append(line[1:-1])
    return features


def save_checkpoint(mode, selected, remaining, current_mae, log_rows,
                    seed_size=None, evaluator=None):
    ckpt = {
        "mode":        mode,
        "selected":    selected,
        "remaining":   remaining,
        "current_mae": current_mae,
        "log":         log_rows,
    }
    if seed_size is not None:
        ckpt["seed_size"] = seed_size
    if evaluator is not None:
        ckpt["evaluator"] = evaluator
    CKPT_FILE.write_text(json.dumps(ckpt, indent=2))


def write_output(selected, log_rows, mode, seed_size=None):
    """Write selection_log.csv and selected_features.txt."""
    pd.DataFrame(log_rows).to_csv(LOG_FILE, index=False)

    lines = ["KEEP_FEATURES = ["]
    for f in selected:
        lines.append(f'    "{f}",')
    lines.append("]")
    (OUT_DIR / "selected_features.txt").write_text("\n".join(lines))

    if mode == "backward":
        removed = [r for r in log_rows if r["status"] == "REMOVE"]
        kept    = [r for r in log_rows if r["status"] == "KEEP"]
        print(f"\n{'=' * 72}")
        print(f"DONE  —  backward elimination")
        print(f"{'=' * 72}")
        print(f"  Started with : {len(BACKWARD_START)} features")
        print(f"  Removed      : {len(removed)}  (noise)")
        print(f"  Kept         : {len(kept)}  (useful)")
        print(f"  Final set    : {len(selected)} features")
        if removed:
            print(f"\n  Removed features (noise, ordered by MAE improvement):")
            for r in sorted(removed, key=lambda x: x["delta"])[:20]:
                print(f"    {r['delta']:>+9,.0f}  {r['feature']}")
    else:
        added   = [r for r in log_rows if r["status"] == "KEEP"]
        dropped = [r for r in log_rows if r["status"] == "DROP"]
        print(f"\n{'=' * 72}")
        print(f"DONE  —  forward selection")
        print(f"{'=' * 72}")
        print(f"  Seed         : {seed_size} features")
        print(f"  Added        : {len(added)}")
        print(f"  Dropped      : {len(dropped)}")
        print(f"  Final set    : {len(selected)} features")
        if added:
            print(f"\n  Top added features (by MAE improvement):")
            for r in sorted(added, key=lambda x: x["delta"])[:10]:
                print(f"    {r['delta']:>+9,.0f}  {r['feature']}")

    print(f"\n  Outputs → {OUT_DIR}/")
    print(f"    selection_log.csv      — full per-feature trial log")
    print(f"    selected_features.txt  — KEEP_FEATURES snippet (copy-paste ready)")


# ==========================================================================
# Backward elimination
# ==========================================================================

def run_backward(args, train_df, test_df, all_features):
    mode = "backward"

    if args.resume and CKPT_FILE.exists():
        ckpt = json.loads(CKPT_FILE.read_text())
        if ckpt.get("mode") != mode:
            print("  [warning] checkpoint is from a different mode — ignoring --resume")
        else:
            ckpt_evaluator = ckpt.get("evaluator")
            if ckpt_evaluator and ckpt_evaluator != args.evaluator:
                print(f"  [warning] checkpoint used evaluator '{ckpt_evaluator}' "
                      f"but --evaluator is '{args.evaluator}' — results will be mixed")
            selected    = ckpt["selected"]
            remaining   = ckpt["remaining"]
            current_mae = ckpt["current_mae"]
            log_rows    = ckpt["log"]
            print(f"\n  Resumed: {len(selected)} kept | {len(remaining)} left to test "
                  f"| current MAE {current_mae:,.0f}")
            _run_backward_loop(args, train_df, test_df, selected, remaining,
                               current_mae, log_rows, mode)
            return

    if args.seed_file:
        pool = parse_feature_file(args.seed_file)
        print(f"  Starting set: {args.seed_file} ({len(pool)} features)")
    else:
        pool = KEEP_FEATURES
        print(f"  Starting set: KEEP_FEATURES ({len(pool)} features)")

    missing = [f for f in pool if f not in all_features]
    if missing:
        print(f"  [warning] {len(missing)} features not in master — skipping:")
        for f in missing:
            print(f"    {f}")
    start_features = [f for f in pool if f in all_features]

    order = list(start_features)
    random.shuffle(order)

    print(f"\n{'─' * 72}")
    print(f"Step 1  Baseline with all {len(start_features)} features …")
    t0 = time.time()
    current_mae, _ = _fit(train_df, test_df, start_features,
                          args.time_limit, args.preset, args.evaluator)
    print(f"  Baseline MAE : {current_mae:>10,.0f}  ({time.time()-t0:.0f}s)")

    selected = list(start_features)
    log_rows = []
    save_checkpoint(mode, selected, order, current_mae, log_rows,
                    evaluator=args.evaluator)

    print(f"\n{'─' * 72}")
    print(f"Step 2  Greedy backward elimination  ({len(order)} features to test, random order)")
    _run_backward_loop(args, train_df, test_df, selected, order, current_mae, log_rows, mode)


def _run_backward_loop(args, train_df, test_df, selected, remaining,
                       current_mae, log_rows, mode):
    print(f"{'─' * 72}")
    print(f"{'#':>4}  {'Status':<6}  {'MAE-feat':>10}  {'Delta':>9}  {'#Feat':>5}  Feature")
    print(f"{'─' * 72}")

    already_done = {row["feature"] for row in log_rows}

    for feature in remaining:
        if feature in already_done:
            row = next(r for r in log_rows if r["feature"] == feature)
            delta_str = f"{row['delta']:+,.0f}"
            print(f"{row['step']:>4}  {row['status']:<6}  {row['mae_with']:>10,.0f}  "
                  f"{delta_str:>9}  {row['n_features']:>5}  {feature}  [resumed]")
            continue

        if feature not in selected:
            continue

        candidate = [f for f in selected if f != feature]
        t0 = time.time()
        mae_without, _ = _fit(train_df, test_df, candidate,
                               args.time_limit, args.preset, args.evaluator)
        elapsed = time.time() - t0
        delta   = mae_without - current_mae
        status  = "REMOVE" if delta < 0 else "KEEP"

        if status == "REMOVE":
            selected.remove(feature)
            current_mae = mae_without

        global_step = len(log_rows) + 1
        row = {
            "step":       global_step,
            "feature":    feature,
            "status":     status,
            "mae_with":   round(mae_without),
            "delta":      round(delta),
            "n_features": len(selected),
            "elapsed_s":  round(elapsed, 1),
        }
        log_rows.append(row)

        delta_str = f"{delta:+,.0f}"
        print(f"{global_step:>4}  {status:<6}  {mae_without:>10,.0f}  "
              f"{delta_str:>9}  {len(selected):>5}  {feature}")

        save_checkpoint(
            mode, selected,
            [f for f in remaining if f not in {r["feature"] for r in log_rows}],
            current_mae, log_rows, evaluator=args.evaluator,
        )

    write_output(selected, log_rows, mode)


# ==========================================================================
# Forward selection
# ==========================================================================

def run_forward(args, train_df, test_df, all_features):
    mode = "forward"

    if args.resume and CKPT_FILE.exists():
        ckpt = json.loads(CKPT_FILE.read_text())
        if ckpt.get("mode") not in (mode, None):
            print("  [warning] checkpoint is from a different mode — ignoring --resume")
        else:
            ckpt_evaluator = ckpt.get("evaluator")
            if ckpt_evaluator and ckpt_evaluator != args.evaluator:
                print(f"  [warning] checkpoint used evaluator '{ckpt_evaluator}' "
                      f"but --evaluator is '{args.evaluator}' — results will be mixed")
            selected    = ckpt["selected"]
            remaining   = ckpt["remaining"]
            current_mae = ckpt["current_mae"]
            log_rows    = ckpt["log"]
            seed_size   = ckpt.get("seed_size", len(selected))
            print(f"\n  Resumed: {len(selected)} selected | {len(remaining)} remaining "
                  f"| current MAE {current_mae:,.0f}")
            _run_forward_loop(args, train_df, test_df, selected, remaining,
                              current_mae, log_rows, mode, seed_size=seed_size)
            return

    # ── Seed-top-n mode: rank a pool of features, take top N as seed ──────
    if args.seed_top_n is not None:
        if CKPT_FILE.exists():
            ckpt = json.loads(CKPT_FILE.read_text())
            checkpoint_selected = ckpt.get("selected", [])
        else:
            checkpoint_selected = []

        # Fall back to BACKWARD_START if no checkpoint (or checkpoint is empty)
        if not checkpoint_selected:
            print(f"  No checkpoint found — using BACKWARD_START ({len(BACKWARD_START)} features) as ranking pool.")
            checkpoint_selected = BACKWARD_START

        # Only keep features that actually exist in master
        checkpoint_selected = [f for f in checkpoint_selected if f in set(all_features)]
        n = args.seed_top_n
        if n >= len(checkpoint_selected):
            print(f"  [warning] --seed-top-n {n} >= {len(checkpoint_selected)} checkpoint features; "
                  f"using all of them as seed")
            n = len(checkpoint_selected)

        print(f"\n{'─' * 72}")
        print(f"Step 1  Ranking {len(checkpoint_selected)} checkpoint features by importance …")
        t0 = time.time()
        _, model_ranked = _fit(train_df, test_df, checkpoint_selected,
                               args.time_limit, args.preset, args.evaluator)
        feature_order = rank_by_importance(model_ranked, test_df, checkpoint_selected)
        print(f"  Done  ({time.time()-t0:.0f}s)")

        seed_features = feature_order[:n]
        print(f"\n  Top {n} seed features (ranked by importance):")
        for i, f in enumerate(seed_features, 1):
            print(f"    {i:>2}. {f}")

        # Candidates = every master feature not already in the seed
        remaining = [f for f in all_features if f not in set(seed_features)]
        print(f"\n  Candidates: {len(remaining)} features "
              f"({len(checkpoint_selected) - n} from checkpoint pool + "
              f"{len(all_features) - len(checkpoint_selected)} others)")

        print(f"\n{'─' * 72}")
        print(f"Step 2  Evaluating seed ({n} features)…")
        t0 = time.time()
        current_mae, _ = _fit(train_df, test_df, seed_features,
                               args.time_limit, args.preset, args.evaluator)
        print(f"  Seed MAE : {current_mae:>10,.0f}  ({time.time()-t0:.0f}s)")

        selected = list(seed_features)
        log_rows = []
        save_checkpoint(mode, selected, remaining, current_mae, log_rows,
                        seed_size=n, evaluator=args.evaluator)

        print(f"\n{'─' * 72}")
        print(f"Step 3  Greedy forward selection  ({len(remaining)} candidates)")
        _run_forward_loop(args, train_df, test_df, selected, remaining,
                          current_mae, log_rows, mode, seed_size=n)
        return

    # ── Seed-file mode ────────────────────────────────────────────────────
    if args.seed_file:
        file_features = parse_feature_file(args.seed_file)
        missing = [f for f in file_features if f not in set(all_features)]
        if missing:
            print(f"  [warning] {len(missing)} features in seed file not in master — skipping:")
            for f in missing:
                print(f"    {f}")
        seed_features = [f for f in file_features if f in set(all_features)]
        remaining     = [f for f in all_features if f not in set(seed_features)]

        print(f"\n{'─' * 72}")
        print(f"Step 1  Evaluating seed from {args.seed_file}  ({len(seed_features)} features)…")
        t0 = time.time()
        current_mae, _ = _fit(train_df, test_df, seed_features,
                               args.time_limit, args.preset, args.evaluator)
        print(f"  Seed MAE : {current_mae:>10,.0f}  ({time.time()-t0:.0f}s)")

        selected = list(seed_features)
        log_rows = []
        save_checkpoint(mode, selected, remaining, current_mae, log_rows,
                        seed_size=len(seed_features), evaluator=args.evaluator)

        print(f"\n{'─' * 72}")
        print(f"Step 2  Greedy forward selection  ({len(remaining)} candidates)")
        _run_forward_loop(args, train_df, test_df, selected, remaining,
                          current_mae, log_rows, mode, seed_size=len(seed_features))
        return

    if not args.skip_importance:
        print(f"\n{'─' * 72}")
        print(f"Step 1  Training initial model with all {len(all_features)} features …")
        t0 = time.time()
        mae_all, model_all = _fit(
            train_df, test_df, all_features,
            time_limit=args.time_limit * 2,
            preset=args.preset,
            evaluator=args.evaluator,
        )
        print(f"  All-features MAE : {mae_all:>10,.0f}  ({time.time()-t0:.0f}s)")
        print(f"  Computing feature importances…")
        feature_order = rank_by_importance(model_all, test_df, all_features)
    else:
        feature_order = (
            [f for f in KEEP_FEATURES if f in all_features] +
            [f for f in all_features if f not in KEEP_FEATURES]
        )
        print(f"\n  Skipping importance step; using KEEP_FEATURES order.")

    seed_features = feature_order[:args.seed_size]
    remaining     = [f for f in feature_order if f not in seed_features]

    print(f"\n{'─' * 72}")
    print(f"Step 2  Evaluating seed set ({args.seed_size} features)…")
    t0 = time.time()
    current_mae, _ = _fit(train_df, test_df, seed_features,
                          args.time_limit, args.preset, args.evaluator)
    print(f"  Seed MAE : {current_mae:>10,.0f}  ({time.time()-t0:.0f}s)")

    if args.evaluate_only:
        print(f"\n  --evaluate-only: done.")
        return

    selected = list(seed_features)
    log_rows = []
    save_checkpoint(mode, selected, remaining, current_mae, log_rows,
                    seed_size=args.seed_size, evaluator=args.evaluator)

    print(f"\n{'─' * 72}")
    print(f"Step 3  Greedy forward selection  ({len(remaining)} candidates)")
    _run_forward_loop(args, train_df, test_df, selected, remaining,
                      current_mae, log_rows, mode, seed_size=args.seed_size)


def _run_forward_loop(args, train_df, test_df, selected, remaining,
                      current_mae, log_rows, mode, seed_size=None):
    print(f"{'─' * 72}")
    print(f"{'#':>4}  {'Status':<5}  {'MAE+feat':>10}  {'Delta':>9}  {'#Feat':>5}  Feature")
    print(f"{'─' * 72}")

    already_done = {row["feature"] for row in log_rows}

    for feature in remaining:
        if feature in already_done:
            row = next(r for r in log_rows if r["feature"] == feature)
            delta_str = f"{row['delta']:+,.0f}"
            print(f"{row['step']:>4}  {row['status']:<5}  {row['mae_with']:>10,.0f}  "
                  f"{delta_str:>9}  {row['n_features']:>5}  {feature}  [resumed]")
            continue

        candidate = selected + [feature]
        t0 = time.time()
        mae_with, _ = _fit(train_df, test_df, candidate,
                           args.time_limit, args.preset, args.evaluator)
        elapsed = time.time() - t0
        delta   = mae_with - current_mae
        status  = "KEEP" if delta < 0 else "DROP"

        if status == "KEEP":
            selected.append(feature)
            current_mae = mae_with

        global_step = len(log_rows) + 1
        row = {
            "step":       global_step,
            "feature":    feature,
            "status":     status,
            "mae_with":   round(mae_with),
            "delta":      round(delta),
            "n_features": len(selected),
            "elapsed_s":  round(elapsed, 1),
        }
        log_rows.append(row)

        delta_str = f"{delta:+,.0f}"
        print(f"{global_step:>4}  {status:<5}  {mae_with:>10,.0f}  "
              f"{delta_str:>9}  {len(selected):>5}  {feature}")

        save_checkpoint(
            mode, selected,
            [f for f in remaining if f not in {r["feature"] for r in log_rows}],
            current_mae, log_rows,
            seed_size=seed_size if seed_size is not None else args.seed_size,
            evaluator=args.evaluator,
        )

    write_output(selected, log_rows, mode,
                 seed_size=seed_size if seed_size is not None else args.seed_size)


# ==========================================================================
# Swap search helpers
# ==========================================================================

def _load_full_feature_df():
    """Run the full build_features pipeline without KEEP_FEATURES pruning."""
    from build_features import (
        add_calendar_features, add_google_trends_features, add_holiday_features,
        add_dow_month_interactions, add_volume_lag_features, add_momentum_features,
        add_dow_specific_lags, add_within_week_features, add_volume_regime_features,
        add_lag365_same_dow, add_log_features, add_top_feature_interactions,
        add_expanding_dow_month_avg, add_holiday_expected_volume, add_last_week_total,
        add_volume_context_features, add_regime_and_decay_features, add_new_features,
        add_group_features, add_anchor_family_features,
    )
    df = load_and_merge_data()
    df = df.sort_values("Date").reset_index(drop=True)
    df = add_calendar_features(df, "Date")
    df = add_google_trends_features(df)
    df, major_dates, presidents_dates = add_holiday_features(df)
    df = add_dow_month_interactions(df)
    df = add_volume_lag_features(df, TARGET_COL)
    df = add_momentum_features(df, TARGET_COL)
    df = add_dow_specific_lags(df, TARGET_COL)
    df = add_within_week_features(df, TARGET_COL)
    df = add_volume_regime_features(df, TARGET_COL)
    df = add_lag365_same_dow(df, TARGET_COL)
    df = add_log_features(df, TARGET_COL)
    df = add_top_feature_interactions(df, TARGET_COL)
    df = add_expanding_dow_month_avg(df, TARGET_COL)
    df = add_holiday_expected_volume(df, TARGET_COL, major_dates, presidents_dates)
    df = add_last_week_total(df, TARGET_COL)
    df = add_volume_context_features(df, TARGET_COL)
    df = add_regime_and_decay_features(df, TARGET_COL)
    df = add_new_features(df, TARGET_COL)
    df = add_group_features(df, TARGET_COL)
    df = add_anchor_family_features(df, TARGET_COL)

    internal_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=internal_cols)
    all_feat_cols = [c for c in df.columns if c not in ["Date", TARGET_COL]]
    for col in all_feat_cols:
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].ffill()

    df = df[df[TARGET_COL].notna()]
    essential = [f for f in ["Volume_lag1", "Volume_lag3", "Volume_lag7"] if f in df.columns]
    df = df.dropna(subset=essential).reset_index(drop=True)
    return df


def _fit_swap(train_df, test_df, features, evaluator):
    """Single-threaded model fit for safe use inside joblib Parallel."""
    valid = [f for f in features if f in train_df.columns]
    if not valid:
        return float("inf")

    X_tr = train_df[valid]
    y_tr = train_df[TARGET_COL]
    X_te = test_df[valid]
    y_te = test_df[TARGET_COL].values
    preds = []

    if evaluator in ("catboost", "ensemble"):
        m = CatBoostRegressor(
            iterations=1000, learning_rate=0.03, depth=6,
            l2_leaf_reg=3.0, subsample=0.85, colsample_bylevel=0.85,
            random_seed=RANDOM_STATE, verbose=0, thread_count=1,
        )
        m.fit(X_tr, y_tr)
        preds.append(m.predict(X_te))

    if evaluator in ("lgbm", "ensemble"):
        m = LGBMRegressor(
            n_estimators=1500, learning_rate=0.02, num_leaves=64,
            min_child_samples=40, subsample=0.85, colsample_bytree=0.85,
            random_state=RANDOM_STATE, n_jobs=1, verbosity=-1,
        )
        m.fit(X_tr, y_tr)
        preds.append(m.predict(X_te))

    if evaluator in ("xgb", "ensemble"):
        m = XGBRegressor(
            n_estimators=1000, learning_rate=0.03, max_depth=4,
            min_child_weight=4, subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=1.1,
            random_state=RANDOM_STATE, tree_method="hist", n_jobs=1, verbosity=0,
        )
        m.fit(X_tr, y_tr)
        preds.append(m.predict(X_te))

    return mean_absolute_error(y_te, np.mean(preds, axis=0)) if preds else float("inf")


def _eval_one_swap(cand, base_set, train_df, test_df, evaluator):
    return cand, _fit_swap(train_df, test_df, base_set + [cand], evaluator)


def _save_swap_checkpoint(selected, outer_done, current_mae, log_rows, evaluator):
    ckpt = {
        "mode":        "swap",
        "selected":    selected,
        "outer_done":  outer_done,
        "current_mae": current_mae,
        "log":         log_rows,
        "evaluator":   evaluator,
    }
    CKPT_FILE.write_text(json.dumps(ckpt, indent=2))


# ==========================================================================
# Swap search
# ==========================================================================

def run_swap(args, train_df, test_df, all_features):
    if args.swap_start:
        start_set_raw = parse_feature_file(args.swap_start)
        missing       = [f for f in start_set_raw if f not in set(all_features)]
        start_set     = [f for f in start_set_raw if f in set(all_features)]
        print(f"  Starting set : {args.swap_start} ({len(start_set)}/{len(start_set_raw)} features)")
        if missing:
            print(f"  [warning] {len(missing)} feature(s) in seed file not in pipeline — skipping:")
            for f in missing:
                print(f"    {f}")
    else:
        start_set = [f for f in KEEP_FEATURES if f in set(all_features)]
        print(f"  Starting set : KEEP_FEATURES ({len(start_set)} features)")

    if args.resume and CKPT_FILE.exists():
        ckpt = json.loads(CKPT_FILE.read_text())
        if ckpt.get("mode") != "swap":
            print("  [warning] checkpoint is from a different mode — ignoring --resume")
        else:
            selected    = ckpt["selected"]
            outer_done  = ckpt.get("outer_done", [])
            current_mae = ckpt["current_mae"]
            log_rows    = ckpt["log"]
            print(f"\n  Resumed: {len(selected)} in set | {len(outer_done)} outer done "
                  f"| current MAE {current_mae:,.0f}")
            _run_swap_loop(args, train_df, test_df, all_features,
                           selected, start_set, outer_done, current_mae, log_rows)
            return

    print(f"\n{'─' * 72}")
    print(f"Step 1  Baseline with {len(start_set)} features …")
    t0 = time.time()
    baseline_mae = _fit_swap(train_df, test_df, start_set, args.evaluator)
    print(f"  Baseline MAE : {baseline_mae:>10,.0f}  ({time.time()-t0:.0f}s)")

    selected   = list(start_set)
    outer_done = []
    log_rows   = []
    _save_swap_checkpoint(selected, outer_done, baseline_mae, log_rows, args.evaluator)

    n_cands = len([f for f in all_features if f not in set(selected)])
    print(f"\n{'─' * 72}")
    print(f"Step 2  Swap search  ({len(start_set)} outer features × ~{n_cands} candidates each)")
    if args.n_jobs != 1:
        print(f"  Parallel inner loop  n_jobs={args.n_jobs}")
    _run_swap_loop(args, train_df, test_df, all_features,
                   selected, start_set, outer_done, baseline_mae, log_rows)


def _run_swap_loop(args, train_df, test_df, all_features,
                   selected, start_set, outer_done, current_mae, log_rows):
    use_parallel = (
        _JOBLIB_AVAILABLE and args.n_jobs != 1
        and args.evaluator in FAST_EVALUATORS
    )
    n_outer = len(start_set)
    swaps_made = []

    print(f"{'─' * 72}")
    print(f"{'Outer':>5}  {'Removed':<30}  {'Best swap / result'}")
    print(f"{'─' * 72}")

    for step_i, xx in enumerate(start_set, 1):
        if xx in outer_done:
            continue

        if xx not in selected:
            print(f"{step_i:>5}  {xx:<30}  [already swapped out — skip]")
            outer_done.append(xx)
            _save_swap_checkpoint(selected, outer_done, current_mae, log_rows, args.evaluator)
            continue

        base_without = [f for f in selected if f != xx]
        candidates   = [f for f in all_features if f not in set(selected)]
        if args.candidates_keep_only:
            keep_set   = set(KEEP_FEATURES)
            candidates = [c for c in candidates if c in keep_set]

        print(f"{step_i:>5}  {xx:<30}  testing {len(candidates)} candidates…", flush=True)
        t_outer = time.time()

        if use_parallel:
            results = Parallel(n_jobs=args.n_jobs)(
                joblib_delayed(_eval_one_swap)(c, base_without, train_df, test_df, args.evaluator)
                for c in candidates
            )
        else:
            results = [
                _eval_one_swap(c, base_without, train_df, test_df, args.evaluator)
                for c in candidates
            ]

        results.sort(key=lambda x: x[1])
        elapsed = time.time() - t_outer

        for cand, mae in results:
            log_rows.append({
                "step":         step_i,
                "removed":      xx,
                "added":        cand,
                "mae":          round(mae),
                "baseline_mae": round(current_mae),
                "delta":        round(mae - current_mae),
                "improved":     mae < current_mae,
            })

        best_cand, best_mae = results[0] if results else (None, float("inf"))
        delta = best_mae - current_mae

        # Print top 5
        print(f"        {'─'*60}")
        print(f"        {'Candidate':<35}  {'MAE':>10}  {'Delta':>9}")
        for cand, mae in results[:5]:
            tag = "✓" if mae < current_mae else " "
            print(f"     {tag}  {cand:<35}  {mae:>10,.0f}  {mae-current_mae:>+9,.0f}")
        if len(results) > 5:
            print(f"        … {len(results)-5} more (see swap_log.csv)")
        print(f"        [{elapsed:.0f}s]")

        if best_cand is not None and best_mae < current_mae:
            print(f"  → SWAP: remove '{xx}' | add '{best_cand}'  Δ={delta:+,.0f}")
            selected.remove(xx)
            selected.append(best_cand)
            current_mae = best_mae
            swaps_made.append({"removed": xx, "added": best_cand, "delta": delta})
        else:
            print(f"  → no improvement  (best Δ={delta:+,.0f})")

        outer_done.append(xx)
        _save_swap_checkpoint(selected, outer_done, current_mae, log_rows, args.evaluator)
        pd.DataFrame(log_rows).to_csv(OUT_DIR / "swap_log.csv", index=False)

    _write_swap_output(selected, log_rows, swaps_made, current_mae)


def _write_swap_output(selected, log_rows, swaps_made, final_mae):
    pd.DataFrame(log_rows).to_csv(OUT_DIR / "swap_log.csv", index=False)

    lines = ["KEEP_FEATURES = ["]
    for f in selected:
        marker = "  # NEW" if f not in KEEP_FEATURES else ""
        lines.append(f'    "{f}",{marker}')
    lines.append("]")
    out_path = OUT_DIR / "swap_selected_features.txt"
    out_path.write_text("\n".join(lines))

    print(f"\n{'=' * 72}")
    print(f"SWAP SEARCH COMPLETE")
    print(f"  Final MAE  : {final_mae:,.0f}")
    print(f"  Swaps made : {len(swaps_made)}")
    if swaps_made:
        print(f"\n  Accepted swaps:")
        for s in swaps_made:
            print(f"    remove '{s['removed']}'  →  add '{s['added']}'  Δ={s['delta']:+,.0f}")
    print(f"\n  Final feature set ({len(selected)} features):")
    for f in selected:
        tag = "  ← NEW" if f not in KEEP_FEATURES else ""
        print(f"    {f}{tag}")
    print(f"\n  Outputs → {OUT_DIR}/")
    print(f"    swap_log.csv                — all {len(log_rows)} candidate evaluations")
    print(f"    swap_selected_features.txt  — KEEP_FEATURES snippet (copy-paste ready)")


# ==========================================================================
# Entry point
# ==========================================================================

def main():
    args = parse_args()

    if args.evaluator in FAST_EVALUATORS and not _FAST_MODELS_AVAILABLE:
        print(f"ERROR: --evaluator {args.evaluator} requires xgboost, lightgbm, and catboost.")
        print("  pip install xgboost lightgbm catboost")
        return

    OUT_DIR.mkdir(exist_ok=True)

    # ── Swap mode needs the full (unpruned) feature pipeline ──────────────
    if args.swap:
        if args.evaluator not in FAST_EVALUATORS:
            print("ERROR: --swap requires a fast evaluator (--evaluator catboost/lgbm/xgb/ensemble)")
            return
        if args.n_jobs != 1 and not _JOBLIB_AVAILABLE:
            print("WARNING: joblib not found — running sequentially (--n-jobs ignored)")
            args.n_jobs = 1

        print("=" * 72)
        print("Greedy Feature Selection  —  Swap Search")
        print("=" * 72)
        print("  Building full feature pipeline (no KEEP_FEATURES pruning)…")
        df_full = _load_full_feature_df()
        df_full["Date"] = pd.to_datetime(df_full["Date"])
        all_features_full = [c for c in df_full.columns if c not in ["Date", TARGET_COL]]

        test_end = min(TEST_END, df_full["Date"].max())
        train_df = df_full[df_full["Date"] <= TRAIN_END].copy()
        test_df  = df_full[(df_full["Date"] >= TEST_START) & (df_full["Date"] <= test_end)].copy()

        print(f"  Train      : {train_df['Date'].min().date()} → {train_df['Date'].max().date()}  ({len(train_df):,} rows)")
        print(f"  Test       : {test_df['Date'].min().date()} → {test_df['Date'].max().date()}  ({len(test_df):,} rows)")
        print(f"  Features   : {len(all_features_full)} total  ({len(KEEP_FEATURES)} current + "
              f"{len(all_features_full)-len(KEEP_FEATURES)} candidates)")
        print(f"  Evaluator  : {args.evaluator}  |  n_jobs={args.n_jobs}")
        run_swap(args, train_df, test_df, all_features_full)
        return

    # ── Forward / backward modes use pruned master_features.csv ──────────
    df = load_master()
    df["Date"] = pd.to_datetime(df["Date"])
    all_features = get_feature_columns(df)

    test_end = min(TEST_END, df["Date"].max())
    train_df = df[df["Date"] <= TRAIN_END].copy()
    test_df  = df[(df["Date"] >= TEST_START) & (df["Date"] <= test_end)].copy()

    mode_label = "Backward Elimination" if args.backward else "Forward Selection"
    print("=" * 72)
    print(f"Greedy Feature Selection  —  {mode_label}")
    print("=" * 72)
    print(f"  Train      : {train_df['Date'].min().date()} → {train_df['Date'].max().date()}  ({len(train_df):,} rows)")
    print(f"  Test       : {test_df['Date'].min().date()}  → {test_df['Date'].max().date()}  ({len(test_df):,} rows)")
    print(f"  Evaluator  : {args.evaluator}")
    if args.evaluator == "autogluon":
        print(f"  Time/model : {args.time_limit}s  |  Preset: {args.preset}")

    if args.backward:
        run_backward(args, train_df, test_df, all_features)
    else:
        print(f"  All features : {len(all_features)}")
        if args.seed_file:
            print(f"  Seed file    : {args.seed_file}")
        else:
            print(f"  Seed size    : {args.seed_size}")
        run_forward(args, train_df, test_df, all_features)


if __name__ == "__main__":
    main()
