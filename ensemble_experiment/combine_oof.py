"""
combine_oof.py
==============
Merge the OOF CSVs produced by the other scripts in this folder into
one combined CSV and print a comparison table (per-model MAE, residual
correlations).

Inputs (under ensemble_experiment/output/):
    ag_tabular_oof.csv         pred_ag_tabular
    ag_timeseries_oof.csv      pred_ag_timeseries
    anchor_master_oof.csv      pred_anchor_master
    seasonal_naive_oof.csv     pred_seasonal_naive
    yoy_delta_oof.csv          pred_yoy_delta  (replaces prophet in production)

Output:
    combined_oof.csv           Date, Volume, pred_ag_tabular, pred_ag_timeseries,
                               pred_anchor_master, pred_seasonal_naive, pred_yoy_delta
    combined_summary.json      per-model MAE + residual correlation matrix
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = HERE / "output"

INPUTS = [
    ("ag_tabular_oof.csv", "pred_ag_tabular"),
    ("ag_timeseries_oof.csv", "pred_ag_timeseries"),
    ("anchor_master_oof.csv", "pred_anchor_master"),
    ("seasonal_naive_oof.csv", "pred_seasonal_naive"),
    ("yoy_delta_oof.csv", "pred_yoy_delta"),
]


def load_one(path, col):
    if not path.exists():
        print(f"  ! {path.name} missing — skipping {col}")
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    needed = ["Date", "Volume", col]
    if not all(c in df.columns for c in needed):
        # ag_tabular uses TARGET_COL ("Volume") header already.
        for alt in ["volume"]:
            if alt in df.columns and "Volume" not in df.columns:
                df = df.rename(columns={alt: "Volume"})
        if not all(c in df.columns for c in needed):
            raise RuntimeError(f"{path.name} missing one of {needed}; has {df.columns.tolist()}")
    return df[needed]


def main():
    print("Loading per-model OOF files...")
    frames = []
    pred_cols = []
    for fname, col in INPUTS:
        f = load_one(OUT_DIR / fname, col)
        if f is None:
            continue
        frames.append(f)
        pred_cols.append(col)

    if not frames:
        raise RuntimeError("No OOF files found. Run the four model scripts first.")

    base = frames[0][["Date", "Volume"]].copy()
    for f in frames:
        col = [c for c in f.columns if c.startswith("pred_")][0]
        base = base.merge(f[["Date", col]], on="Date", how="left")

    out = base.dropna(subset=pred_cols).reset_index(drop=True)
    print(f"\nCombined OOF rows: {len(out)}  "
          f"({out['Date'].min().date()} → {out['Date'].max().date()})")
    print(f"Dropped {len(base) - len(out)} rows with missing predictions.")

    out_csv = OUT_DIR / "combined_oof.csv"
    out.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    y = out["Volume"].values
    print("\n" + "=" * 60)
    print("Per-model OOF MAE")
    print("=" * 60)
    maes = {}
    for c in pred_cols:
        m = float(mean_absolute_error(y, out[c].values))
        maes[c] = m
        print(f"  {c:<25} MAE {m:>12,.0f}")

    print("\nResidual correlation matrix (lower off-diagonal = more diverse):")
    resid = out[pred_cols].subtract(out["Volume"], axis=0)
    resid.columns = [c.replace("pred_", "") for c in pred_cols]
    corr = resid.corr().round(3)
    print(corr.to_string())

    simple_avg = out[pred_cols].mean(axis=1).values
    avg_mae = float(mean_absolute_error(y, simple_avg))
    print(f"\nSimple average across all {len(pred_cols)} models: MAE {avg_mae:,.0f}")

    summary = {
        "n_rows": int(len(out)),
        "date_start": str(out["Date"].min().date()),
        "date_end": str(out["Date"].max().date()),
        "per_model_mae": maes,
        "simple_avg_mae": avg_mae,
        "residual_correlation": corr.to_dict(),
    }
    with open(OUT_DIR / "combined_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {OUT_DIR / 'combined_summary.json'}")


if __name__ == "__main__":
    main()
