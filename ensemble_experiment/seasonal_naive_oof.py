"""
seasonal_naive_oof.py
=====================
Seasonal-naive baseline: pred(D) = Volume(D - 364), i.e. the same weekday
52 weeks earlier. No training — like anchor_master, this is a deterministic
backward-looking value, so reading it for the OOF dates is itself the
"forecast". Raw (no growth adjustment): the trailing-YoY growth scaling was
tested and only helps in the normal regime while hurting shoulders/peak.

Outputs (under ensemble_experiment/output/):
    seasonal_naive_oof.csv      Date, Volume, pred_seasonal_naive
    seasonal_naive_summary.json
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)
TABULAR_OOF_PATH = OUT_DIR / "ag_tabular_oof.csv"
VOLUME_PATH = ROOT / "data" / "tsa_volume.csv"

LAG_DAYS = 364  # 52 weeks: same day-of-week, one year back


def main():
    if not TABULAR_OOF_PATH.exists():
        raise FileNotFoundError(
            f"{TABULAR_OOF_PATH} missing. Run ag_tabular.py first so we know "
            "the OOF date range to align against."
        )
    dates = pd.read_csv(TABULAR_OOF_PATH, parse_dates=["Date"])["Date"]
    vol = (pd.read_csv(VOLUME_PATH, parse_dates=["Date"])
           .set_index("Date")["Volume"])

    out = pd.DataFrame({"Date": dates})
    out["Volume"] = out["Date"].map(vol)
    out["pred_seasonal_naive"] = (out["Date"] - pd.Timedelta(days=LAG_DAYS)).map(vol)

    missing = out["pred_seasonal_naive"].isna() | out["Volume"].isna()
    if missing.any():
        raise RuntimeError(
            f"{missing.sum()} OOF dates lack a lag-{LAG_DAYS} source value in "
            f"{VOLUME_PATH.name}: {out.loc[missing, 'Date'].dt.date.tolist()[:5]}..."
        )

    out_csv = OUT_DIR / "seasonal_naive_oof.csv"
    out.to_csv(out_csv, index=False)

    mae = float(mean_absolute_error(out["Volume"], out["pred_seasonal_naive"]))
    summary = {
        "n_rows": int(len(out)),
        "date_start": str(out["Date"].min().date()),
        "date_end": str(out["Date"].max().date()),
        "lag_days": LAG_DAYS,
        "mae": mae,
    }
    with open(OUT_DIR / "seasonal_naive_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved → {out_csv}  (n={len(out)}, MAE {mae:,.0f})")


if __name__ == "__main__":
    main()
