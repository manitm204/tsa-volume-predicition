import argparse
import os
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VARS   = ["snowfall_sum", "snow_depth_max"]

HUBS = [
    {"city": "ATL", "latitude": 33.6407, "longitude": -84.4277,  "weight": 0.110},
    {"city": "LAX", "latitude": 33.9416, "longitude": -118.4085, "weight": 0.085},
    {"city": "ORD", "latitude": 41.9742, "longitude": -87.9073,  "weight": 0.080},
    {"city": "DFW", "latitude": 32.8998, "longitude": -97.0403,  "weight": 0.075},
    {"city": "DEN", "latitude": 39.8561, "longitude": -104.6737, "weight": 0.065},
    {"city": "JFK", "latitude": 40.6413, "longitude": -73.7781,  "weight": 0.060},
    {"city": "SFO", "latitude": 37.6213, "longitude": -122.3790, "weight": 0.055},
    {"city": "SEA", "latitude": 47.4502, "longitude": -122.3088, "weight": 0.050},
    {"city": "LAS", "latitude": 36.0840, "longitude": -115.1537, "weight": 0.050},
    {"city": "MCO", "latitude": 28.4294, "longitude": -81.3089,  "weight": 0.050},
    {"city": "MIA", "latitude": 25.7959, "longitude": -80.2870,  "weight": 0.045},
    {"city": "CLT", "latitude": 35.2140, "longitude": -80.9431,  "weight": 0.045},
    {"city": "PHX", "latitude": 33.4343, "longitude": -112.0116, "weight": 0.045},
    {"city": "BOS", "latitude": 42.3656, "longitude": -71.0096,  "weight": 0.040},
    {"city": "MSP", "latitude": 44.8848, "longitude": -93.2223,  "weight": 0.035},
    {"city": "EWR", "latitude": 40.6895, "longitude": -74.1745,  "weight": 0.035},
]

SNOW_THRESHOLD_CM = 1.0


def _parse_response(json_data, hub):
    daily = json_data.get("daily", {})
    df = pd.DataFrame({
        "date":           pd.to_datetime(daily.get("time", [])),
        "snowfall_sum":   daily.get("snowfall_sum",   []),
        "snow_depth_max": daily.get("snow_depth_max", []),
    })
    df["city"]   = hub["city"]
    df["weight"] = hub["weight"]
    return df


def _evict_old_cache(cache_dir, city, start_date, end_date):
    """Delete older cache files for this city+start_date if a newer end_date is being saved."""
    prefix = f"archive_{city}_{start_date}_"
    for fname in os.listdir(cache_dir):
        if fname.startswith(prefix) and fname.endswith(".csv"):
            existing_end = fname[len(prefix):-4]
            if existing_end < end_date:
                os.remove(os.path.join(cache_dir, fname))


def fetch_hub_archive(hub, start_date, end_date, cache_dir, max_retries=6):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir, f"archive_{hub['city']}_{start_date}_{end_date}.csv"
    )
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, parse_dates=["date"])

    params = {
        "latitude":   hub["latitude"],
        "longitude":  hub["longitude"],
        "start_date": start_date,
        "end_date":   end_date,
        "daily":      ",".join(DAILY_VARS),
        "timezone":   "America/New_York",
    }
    for attempt in range(max_retries):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(min(2 ** attempt, 60))
                continue
            r.raise_for_status()
            df = _parse_response(r.json(), hub)
            _evict_old_cache(cache_dir, hub['city'], start_date, end_date)
            df.to_csv(cache_file, index=False)
            return df
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Archive fetch failed for {hub['city']}: {e}")
            time.sleep(min(2 ** attempt, 60))


def fetch_hub_forecast(hub, forecast_days, max_retries=6):
    params = {
        "latitude":      hub["latitude"],
        "longitude":     hub["longitude"],
        "daily":         ",".join(DAILY_VARS),
        "timezone":      "America/New_York",
        "forecast_days": forecast_days,
    }
    for attempt in range(max_retries):
        try:
            r = requests.get(FORECAST_URL, params=params, timeout=60)
            if r.status_code in (429, 502, 503, 504):
                wait = min(2 ** attempt, 60)
                if attempt < max_retries - 1:
                    time.sleep(wait)
                    continue
            r.raise_for_status()
            return _parse_response(r.json(), hub)
        except Exception as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Forecast fetch failed for {hub['city']}: {e}")
            time.sleep(min(2 ** attempt, 60))


def build_national_features(df_hubs):
    rows = []
    for dt, grp in df_hubs.groupby("date"):
        sf = grp["snowfall_sum"].fillna(0).to_numpy()
        sd = grp["snow_depth_max"].fillna(0).to_numpy()
        w  = grp["weight"].to_numpy()

        snowing_mask = sf >= SNOW_THRESHOLD_CM
        snowing_idx  = np.where(snowing_mask)[0]

        rows.append({
            "date":                    dt,
            "wt_avg_snow_depth_max":   float((sd * w).sum() / w.sum()),
            "p90_snowfall_sum":        float(np.percentile(sf, 90)),
            "wt_avg_snowfall_sum":     float((sf * w).sum() / w.sum()),
            "n_hubs_snowing":          int(snowing_mask.sum()),
            "top3_hubs_snowfall_mean": float(np.sort(sf)[::-1][:3].mean()),
            "vol_wtd_storm_impact":    float(
                (w[snowing_idx] * sf[snowing_idx]).sum()
            ) if snowing_idx.size > 0 else 0.0,
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def build_lag_features(df):
    df = df.copy()

    # wt_avg_snow_depth_max: lead1, lag1, lag2
    sd = df["wt_avg_snow_depth_max"]
    df["wt_avg_snow_depth_max_lead1"] = sd.shift(-1)
    df["wt_avg_snow_depth_max_lag1"]  = sd.shift(1)
    df["wt_avg_snow_depth_max_lag2"]  = sd.shift(2)

    #= p90_snowfall_sum: lead1 
    df["p90_snowfall_sum_lead1"] = df["p90_snowfall_sum"].shift(-1)

    # wt_avg_snowfall_sum: lead1
    df["wt_avg_snowfall_sum_lead1"] = df["wt_avg_snowfall_sum"].shift(-1)

    # n_hubs_snowing: lag1 only
    df["n_hubs_snowing_lag1"] = df["n_hubs_snowing"].shift(1)

    # top3_hubs_snowfall_mean: lag1 only
    df["top3_hubs_snowfall_mean_lag1"] = df["top3_hubs_snowfall_mean"].shift(1)

    # vol_wtd_storm_impact: lag1
    df["vol_wtd_storm_impact_lag1"] = df["vol_wtd_storm_impact"].shift(1)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_date",    default="2022-01-01")
    ap.add_argument("--forecast_days", type=int, default=10)
    ap.add_argument("--cache_dir",     default="cache")
    ap.add_argument("--out_csv",       default="data/weather_national_features.csv")
    args = ap.parse_args()

    yesterday = str(date.today() - timedelta(days=1))
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)

    print(f"[Archive] {args.start_date} -> {yesterday}")
    archive_frames = []
    for hub in HUBS:
        print(f"  {hub['city']}...", end=" ", flush=True)
        archive_frames.append(
            fetch_hub_archive(hub, args.start_date, yesterday, args.cache_dir)
        )
        time.sleep(0.5)
        print("ok")

    archive_hubs = pd.concat(archive_frames, ignore_index=True)
    archive_hubs["date"] = pd.to_datetime(archive_hubs["date"]).dt.normalize()

    print(f"\n[Forecast] Next {args.forecast_days} days")
    forecast_frames = []
    for hub in HUBS:
        print(f"  {hub['city']}...", end=" ", flush=True)
        forecast_frames.append(fetch_hub_forecast(hub, args.forecast_days))
        time.sleep(0.5)
        print("ok")

    forecast_hubs = pd.concat(forecast_frames, ignore_index=True)
    forecast_hubs["date"] = pd.to_datetime(forecast_hubs["date"]).dt.normalize()

    # Archive wins on any overlapping dates
    archive_dates = set(archive_hubs["date"].unique())
    forecast_hubs = forecast_hubs[~forecast_hubs["date"].isin(archive_dates)]

    all_hubs = pd.concat([archive_hubs, forecast_hubs], ignore_index=True)

    print("\nBuilding national features...")
    nat_df = build_national_features(all_hubs)

    nat_df.to_csv(args.out_csv, index=False)
    print(f"Saved raw    : {args.out_csv}  ({len(nat_df):,} rows, {len(nat_df.columns)} cols)")

    lag_df = build_lag_features(nat_df)
    lag_path = args.out_csv.replace(".csv", "_with_lags.csv")
    lag_df.to_csv(lag_path, index=False)
    lag_cols = [c for c in lag_df.columns if c not in nat_df.columns]
    print(f"\nDate range: {nat_df['date'].min().date()} -> {nat_df['date'].max().date()}")
    print(f"  Archive : {len(archive_hubs['date'].unique())} days")
    print(f"  Forecast: {len(forecast_hubs['date'].unique())} days "
          f"({forecast_hubs['date'].min().date()} -> {forecast_hubs['date'].max().date()})")


if __name__ == "__main__":
    main()