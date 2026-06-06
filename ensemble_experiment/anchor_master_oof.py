"""
anchor_master_oof.py
====================
Extract the `anchor_master` value (already a function of past data, see
build_features.py:941) for the OOF date range. No training — anchor_master
is a deterministic, backward-looking feature, so reading it for the OOF
dates is itself the "forecast".

Outputs (under ensemble_experiment/output/):
    anchor_master_oof.csv      Date, Volume, pred_anchor_master
    anchor_master_summary.json
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from build_features import (
    TARGET_COL,
    build_features_from_df,
    load_and_merge_data,
)

OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)
TABULAR_OOF_PATH = OUT_DIR / "ag_tabular_oof.csv"


def parse_args():
    p = argparse.ArgumentParser()
    return p.parse_args()


def main():
    parse_args()
    if not TABULAR_OOF_PATH.exists():
        raise FileNotFoundError(
            f"{TABULAR_OOF_PATH} missing. Run ag_tabular.py first so we know "
            "the OOF date range to align against."
        )

    tabular_oof = pd.read_csv(TABULAR_OOF_PATH, parse_dates=["Date"])
    tabular_oof = tabular_oof[["Date", TARGET_COL]].rename(columns={TARGET_COL: "Volume"})

    print("Building feature dataframe (prune=False) to read anchor_master...")
    df = load_and_merge_data()
    df = build_features_from_df(df, verbose=False, prune=False)
    if "anchor_master" not in df.columns:
        raise RuntimeError("anchor_master column not produced by build_features — check pipeline.")

    df["Date"] = pd.to_datetime(df["Date"])
    sub = df[["Date", "anchor_master"]].rename(columns={"anchor_master": "pred_anchor_master"})

    out = tabular_oof.merge(sub, on="Date", how="left")
    n_missing = int(out["pred_anchor_master"].isna().sum())
    if n_missing:
        print(f"  WARNING: {n_missing} OOF dates have no anchor_master value")

    out_csv = OUT_DIR / "anchor_master_oof.csv"
    out.to_csv(out_csv, index=False)
    valid = out.dropna(subset=["pred_anchor_master"])
    overall_mae = float(mean_absolute_error(valid["Volume"], valid["pred_anchor_master"]))
    print(f"\nSaved anchor_master OOF → {out_csv}  (n={len(out)}, MAE = {overall_mae:,.0f})")

    summary = {
        "source_column": "anchor_master",
        "source_file": "build_features.build_features_from_df (prune=False)",
        "n_oof": int(len(out)),
        "n_missing": n_missing,
        "overall_oof_mae": overall_mae,
    }
    with open(OUT_DIR / "anchor_master_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
