"""
build_features.py
=================
Shared config, metrics, data loading, and feature engineering.
Streamlined: only builds features that are actually used by the model.

Best ensemble OOF MAE: 63,815 (CatBoost solo, 77 features)

Usage:
    python build_features.py          # builds and saves master_features.csv

Other files import from here:
    from build_features import (
        load_master, TARGET_COL, N_SPLITS, TEST_SIZE, GAP,
        FINAL_HOLDOUT_DAYS, RANDOM_STATE, get_feature_columns,
        rmse, mase, smape, evaluate_fold,
        naive_last_value, seasonal_naive, lag365_naive,
    )
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================================================
# Config
# ==========================================================================
TSA_PATH = "./data/tsa_volume.csv"
WEATHER_PATH = "./data/weather_national_features_with_lags.csv"
MASTER_PATH = "./master_features.csv"
GOOGLE_TRENDS_PATH = "./data/google_trends.csv"

START_DATE = "2022-01-01"
N_SPLITS = 4
TEST_SIZE = 91
GAP = 0
FINAL_HOLDOUT_DAYS = 0
RANDOM_STATE = 42

TARGET_COL = "Volume"
USE_LOG_TARGET = False


# ==========================================================================
# Log-transform helpers
# ==========================================================================
def transform_target(y):
    if USE_LOG_TARGET:
        return np.log1p(np.asarray(y, dtype=float))
    return np.asarray(y, dtype=float)


def inverse_transform_target(y_pred):
    if USE_LOG_TARGET:
        return np.expm1(np.asarray(y_pred, dtype=float))
    return np.asarray(y_pred, dtype=float)


# ==========================================================================
# Metrics
# ==========================================================================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mase(y_true, y_pred, y_train, m=7):
    y_train = np.asarray(y_train)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_train) <= m:
        return np.nan
    naive_errors = np.abs(y_train[m:] - y_train[:-m])
    denom = np.mean(naive_errors)
    if denom == 0:
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / denom


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-8, denom)
    return np.mean(np.abs(y_true - y_pred) / denom) * 100


def pinball_loss(y_true, y_pred, q):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    err = y_true - y_pred
    return np.mean(np.maximum(q * err, (q - 1) * err))


def evaluate_fold(y_train, y_true, y_pred, label="fold"):
    return {
        "fold": label,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase_7": mase(y_true, y_pred, y_train, m=7),
    }


# ==========================================================================
# Baselines
# ==========================================================================
def naive_last_value(train_series, horizon):
    return np.repeat(train_series.iloc[-1], horizon)


def seasonal_naive(train_series, horizon, season=7):
    vals = train_series.iloc[-season:].values
    return np.tile(vals, int(np.ceil(horizon / season)))[:horizon]


def lag365_naive(train_series, horizon):
    if len(train_series) < 365:
        return np.repeat(train_series.iloc[-1], horizon)
    return train_series.iloc[-365:-365 + horizon].values


# ==========================================================================
# Helpers
# ==========================================================================
def get_feature_columns(df):
    return [c for c in df.columns if c not in ["Date", TARGET_COL]]


def load_master(path=MASTER_PATH):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df[df["Date"] >= pd.to_datetime(START_DATE)].reset_index(drop=True)
    return df


# ==========================================================================
# Data loading
# ==========================================================================
def load_and_merge_data(tsa_path=TSA_PATH, weather_path=WEATHER_PATH,
                        start_date=START_DATE):
    tsa = pd.read_csv(tsa_path)
    weather = pd.read_csv(weather_path)

    tsa.columns = [c.strip() for c in tsa.columns]
    weather.columns = [c.strip() for c in weather.columns]
    weather.rename(columns={"date": "Date"}, inplace=True)

    if "Date" not in weather.columns:
        if "ate" in weather.columns:
            weather = weather.rename(columns={"ate": "Date"})
        else:
            raise ValueError("Weather file must contain a Date column.")

    tsa["Date"] = pd.to_datetime(tsa["Date"])
    weather["Date"] = pd.to_datetime(weather["Date"])

    tsa = tsa.sort_values("Date").drop_duplicates("Date")
    weather = weather.sort_values("Date").drop_duplicates("Date")

    tsa = tsa[tsa["Date"] >= pd.to_datetime(start_date)].copy()
    df = tsa.merge(weather, on="Date", how="left")
    df = df[df["Date"] >= pd.to_datetime(start_date)].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # Merge Google Trends
    try:
        gt = load_google_trends()
        gt_cols = ["Date"] + [c for c in TRENDS_COLS if c in gt.columns]
        gt = gt[gt_cols]
        df = df.merge(gt, on="Date", how="left")
    except (FileNotFoundError, Exception) as e:
        print(f"  Google Trends not loaded: {e}")

    return df


# ==========================================================================
# Google Trends features
# ==========================================================================
TRENDS_COLS = ["airport_parking", "flights", "frontier_airlines",
               "southwest_airlines", "tsa_precheck",
               "cheap_flights", "boarding_pass", "vacation", "flight_tracker"]


def load_google_trends(path=GOOGLE_TRENDS_PATH):
    gt = pd.read_csv(path)
    gt.columns = [c.strip() for c in gt.columns]
    gt.rename(columns={"date": "Date"}, inplace=True)
    gt["Date"] = pd.to_datetime(gt["Date"])
    gt = gt.sort_values("Date").drop_duplicates("Date")
    return gt


def add_google_trends_features(df, lag=1):
    """Add Google Trends features lagged by `lag` days.

    Lag is configurable so we can test multiple horizons. Live inference uses
    yesterday's Google Trends value (lag=1), which is available by the time
    the daily pipeline runs.
    """
    df = df.copy()
    available_cols = [c for c in TRENDS_COLS if c in df.columns]
    if not available_cols:
        return df

    suffix = f"lag{lag}"
    for col in available_cols:
        lagged = df[col].shift(lag)
        df[f"{col}_{suffix}"] = lagged
        df[f"{col}_{suffix}_roll7"] = lagged.rolling(7, min_periods=4).mean()
        roll7 = df[f"{col}_{suffix}_roll7"]
        df[f"{col}_{suffix}_roll7_diff"] = roll7 - roll7.shift(7)

    lag_cols = [f"{c}_{suffix}" for c in available_cols if f"{c}_{suffix}" in df.columns]
    if lag_cols:
        normalized = pd.DataFrame()
        for col in lag_cols:
            exp_min = df[col].expanding().min()
            exp_max = df[col].expanding().max()
            exp_range = exp_max - exp_min
            normalized[col] = np.where(exp_range > 0, (df[col] - exp_min) / exp_range, 0.5)
        df[f"trends_composite_{suffix}"] = normalized.mean(axis=1)

    for col in available_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


# ==========================================================================
# Calendar features
# ==========================================================================
def add_calendar_features(df, date_col="Date"):
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["doy_cos"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.25)
    return df


# ==========================================================================
# Holiday helpers (internal)
# ==========================================================================
def _nth_weekday_of_month(year, month, weekday, n):
    first = pd.Timestamp(year=year, month=month, day=1)
    days_until = (weekday - first.weekday()) % 7
    return first + pd.Timedelta(days=days_until) + pd.Timedelta(weeks=n - 1)


def _get_major_holiday_dates(years):
    from dateutil.easter import easter as _easter_date
    holidays = {}
    thanksgiving, memorial, labor, july4, christmas, newyear, easter, halloween = \
        [], [], [], [], [], [], [], []
    for y in years:
        thanksgiving.append(_nth_weekday_of_month(y, 11, weekday=3, n=4))
        last_may = pd.Timestamp(year=y, month=5, day=31)
        memorial.append(last_may - pd.Timedelta(days=(last_may.weekday() - 0) % 7))
        labor.append(_nth_weekday_of_month(y, 9, weekday=0, n=1))
        july4.append(pd.Timestamp(year=y, month=7, day=4))
        christmas.append(pd.Timestamp(year=y, month=12, day=25))
        newyear.append(pd.Timestamp(year=y, month=1, day=1))
        easter.append(pd.Timestamp(_easter_date(y)))
        halloween.append(pd.Timestamp(year=y, month=10, day=31))
    holidays["thanksgiving"] = pd.to_datetime(thanksgiving)
    holidays["memorial"] = pd.to_datetime(memorial)
    holidays["labor"] = pd.to_datetime(labor)
    holidays["july4"] = pd.to_datetime(july4)
    holidays["christmas"] = pd.to_datetime(christmas)
    holidays["newyear"] = pd.to_datetime(newyear)
    holidays["easter"] = pd.to_datetime(easter)
    holidays["halloween"] = pd.to_datetime(halloween)
    return holidays


def _compute_rel_days(dates, holiday_dates):
    date_vals = dates.values.astype("datetime64[D]")
    hdays = holiday_dates.values.astype("datetime64[D]")
    rel = np.empty(len(dates), dtype=int)
    for i, day in enumerate(date_vals):
        closest_idx = np.argmin(np.abs(hdays - day))
        rel[i] = (day - hdays[closest_idx]).astype(int)
    return rel


# ==========================================================================
# Holiday features
# ==========================================================================
def add_holiday_features(df):
    df = df.copy()
    d = pd.to_datetime(df["Date"])
    df["Date"] = d
    years = sorted(d.dt.year.unique())
    cal = USFederalHolidayCalendar()
    fed_holidays = pd.to_datetime(
        cal.holidays(start=d.min(), end=d.max() + pd.Timedelta(days=400))
    )
    major_dates = _get_major_holiday_dates(years)

    easter_dates = major_dates["easter"]
    halloween_dates = major_dates["halloween"]
    all_holiday_dates = pd.DatetimeIndex(
        pd.concat([fed_holidays.to_series(), easter_dates.to_series(),
                   halloween_dates.to_series()])
    ).sort_values().drop_duplicates()
    df["is_holiday"] = d.isin(all_holiday_dates).astype(int)

    holiday_days = all_holiday_dates.values.astype("datetime64[D]")
    all_days = d.values.astype("datetime64[D]")
    nearest_dist = []
    for day in all_days:
        nearest_dist.append(np.min(np.abs(holiday_days - day)).astype(int))
    df["days_to_holiday"] = nearest_dist

    signed_dist = []
    for day in all_days:
        diffs = holiday_days - day
        abs_diffs = np.abs(diffs)
        closest_idx = np.argmin(abs_diffs)
        diff_val = diffs[closest_idx]
        if hasattr(diff_val, 'astype'):
            days_int = int(diff_val.astype('timedelta64[D]').astype(int))
        else:
            days_int = int(diff_val.days)
        signed_dist.append(days_int)
    df["days_to_holiday_signed"] = [-d for d in signed_dist]
    df["is_post_holiday"] = ((df["days_to_holiday_signed"] > 0) &
                              (df["days_to_holiday_signed"] <= 7)).astype(int)

    for hname, hdates in major_dates.items():
        df[f"_{hname}_rel_day"] = _compute_rel_days(d, hdates)
    presidents_dates = []
    for y in years:
        presidents_dates.append(_nth_weekday_of_month(y, 2, weekday=0, n=3))
    presidents_dates = pd.to_datetime(presidents_dates)
    df["_presidents_rel_day"] = _compute_rel_days(d, presidents_dates)

    return df, major_dates, presidents_dates


# ==========================================================================
# Volume lag features
# ==========================================================================
def add_volume_lag_features(df, target_col=TARGET_COL):
    df = df.copy()
    for lag in [1, 3, 7, 14, 28]:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    shifted = df[target_col].shift(1)
    df[f"{target_col}_roll_mean_28"] = shifted.rolling(28).mean()
    df[f"{target_col}_roll_std_28"] = shifted.rolling(28).std()
    df[f"{target_col}_roll_max_28"] = shifted.rolling(28).max()
    df["vol_diff_1"] = df[target_col].shift(1) - df[target_col].shift(2)
    df["vol_diff_7"] = df[target_col].shift(1) - df[target_col].shift(8)
    return df


# ==========================================================================
# Momentum features
# ==========================================================================
def add_momentum_features(df, target_col=TARGET_COL):
    df = df.copy()
    vol = df[target_col]
    shifted = vol.shift(1)
    rm7 = shifted.rolling(7).mean()
    above = (shifted > rm7).astype(int)
    below = (shifted < rm7).astype(int)
    streak_above = np.zeros(len(df))
    streak_below = np.zeros(len(df))
    for i in range(1, len(df)):
        if above.iloc[i] == 1: streak_above[i] = streak_above[i - 1] + 1
        if below.iloc[i] == 1: streak_below[i] = streak_below[i - 1] + 1
    df["streak_above_rm7"] = streak_above
    df["streak_below_rm7"] = streak_below
    return df


# ==========================================================================
# DOW-specific lags
# ==========================================================================
def add_dow_specific_lags(df, target_col=TARGET_COL):
    df = df.copy()
    vol = df[target_col].values
    same_dow_1w = np.full(len(vol), np.nan)
    same_dow_2w = np.full(len(vol), np.nan)
    same_dow_3w = np.full(len(vol), np.nan)
    same_dow_4w = np.full(len(vol), np.nan)
    for i in range(7, len(vol)):  same_dow_1w[i] = vol[i - 7]
    for i in range(14, len(vol)): same_dow_2w[i] = vol[i - 14]
    for i in range(21, len(vol)): same_dow_3w[i] = vol[i - 21]
    for i in range(28, len(vol)): same_dow_4w[i] = vol[i - 28]
    dow_4w_mean = np.full(len(vol), np.nan)
    for i in range(28, len(vol)):
        vals = np.array([same_dow_4w[i], same_dow_3w[i], same_dow_2w[i], same_dow_1w[i]])
        if not np.any(np.isnan(vals)): dow_4w_mean[i] = np.mean(vals)
    df["dow_4w_mean"] = dow_4w_mean
    return df


# ==========================================================================
# Within-week features
# ==========================================================================
def add_within_week_features(df, target_col=TARGET_COL):
    df = df.copy()
    dates = pd.to_datetime(df["Date"])
    dow = dates.dt.dayofweek.values
    vol = df[target_col].values

    # Week cumulative sum
    week_cumsum = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        current_dow = dow[i]
        if current_dow == 0:
            week_cumsum[i] = 0
        else:
            cumvol = 0.0
            for lookback in range(1, current_dow + 1):
                if i - lookback >= 0 and dow[i - lookback] == current_dow - lookback:
                    cumvol += vol[i - lookback]
                else: break
            week_cumsum[i] = cumvol
    df["week_vol_cumsum"] = week_cumsum

    # DOW weekly share
    dow_weekly_share = np.full(len(df), np.nan)
    for i in range(28, len(df)):
        current_dow_val = dow[i]; shares = []
        for w in range(1, 5):
            week_start = i - w * 7 - current_dow_val
            if week_start < 0 or week_start + 7 > len(vol): continue
            week_total = np.sum(vol[week_start:week_start + 7])
            day_val = vol[week_start + current_dow_val]
            if week_total > 0 and not np.isnan(day_val): shares.append(day_val / week_total)
        if shares: dow_weekly_share[i] = np.mean(shares)
    df["dow_weekly_share"] = dow_weekly_share
    return df


# ==========================================================================
# Volume regime features
# ==========================================================================
def add_volume_regime_features(df, target_col=TARGET_COL):
    df = df.copy()
    vol = df[target_col]
    same_dow_vals_8w = []
    for i in range(len(df)):
        if i < 8 * 7: same_dow_vals_8w.append(np.nan); continue
        indices = [i - 7 * w for w in range(1, 9) if i - 7 * w >= 0]
        vals = vol.iloc[indices].values
        valid = vals[~np.isnan(vals)]
        same_dow_vals_8w.append(np.mean(valid) if len(valid) > 0 else np.nan)
    df["vol_same_dow_mean_8w"] = same_dow_vals_8w
    return df


# ==========================================================================
# Lag365 same-DOW alignment
# ==========================================================================
def add_lag365_same_dow(df, target_col=TARGET_COL):
    df = df.copy()
    dates = pd.to_datetime(df["Date"])
    vol = df[target_col].values
    dow = dates.dt.dayofweek.values
    woy = dates.dt.isocalendar().week.values.astype(int)
    year = dates.dt.year.values
    lookup = {}
    for i in range(len(df)):
        lookup[(year[i], woy[i], dow[i])] = vol[i]

    same_dow_ly = np.full(len(df), np.nan)
    for i in range(len(df)):
        key = (year[i] - 1, woy[i], dow[i])
        if key in lookup: same_dow_ly[i] = lookup[key]
    df["lag365_same_dow"] = same_dow_ly

    shifted = df[target_col].shift(1).values
    df["vol_vs_lag365_same_dow"] = np.where(same_dow_ly > 0, shifted / same_dow_ly, np.nan)

    same_dow_ly_3w = np.full(len(df), np.nan)
    for i in range(len(df)):
        vals = []
        for week_offset in [-1, 0, 1]:
            target_woy = woy[i] + week_offset
            if target_woy < 1: target_woy = 52
            elif target_woy > 52: target_woy = 1
            key = (year[i] - 1, target_woy, dow[i])
            if key in lookup: vals.append(lookup[key])
        if vals: same_dow_ly_3w[i] = np.mean(vals)
    df["lag365_same_dow_3w_mean"] = same_dow_ly_3w
    return df


# ==========================================================================
# Log-scale features (only kept ones)
# ==========================================================================
def add_log_features(df, target_col=TARGET_COL):
    df = df.copy()
    for col_name, source in [
        ("log_Volume_lag7", f"{target_col}_lag7"),
        ("log_Volume_lag14", f"{target_col}_lag14"),
    ]:
        if source in df.columns:
            df[col_name] = np.log1p(df[source].clip(lower=0))
    return df


# ==========================================================================
# Top-feature interactions (only kept ones)
# ==========================================================================
def add_top_feature_interactions(df, target_col=TARGET_COL):
    df = df.copy()
    if "dow_4w_mean" in df.columns and "dayofweek" in df.columns:
        shifted = df[target_col].shift(1)
        lag1_ratio = np.where(df["dow_4w_mean"] > 0, shifted / df["dow_4w_mean"], np.nan)
        df["lag1_dow_ratio_x_dow"] = lag1_ratio * df["dayofweek"]
    return df


# ==========================================================================
# DOW × month interactions
# ==========================================================================
def add_dow_month_interactions(df):
    df = df.copy()
    df["dow_x_month"] = df["dayofweek"] * df["month"]
    return df


# ==========================================================================
# Expanding DOW × month averages
# ==========================================================================
def add_expanding_dow_month_avg(df, target_col=TARGET_COL):
    df = df.copy()
    dates = pd.to_datetime(df["Date"])
    vol = df[target_col].values
    dow = dates.dt.dayofweek.values
    month = dates.dt.month.values

    expanding_mean = np.full(len(df), np.nan)
    accum = {}
    for i in range(len(df)):
        key = (dow[i], month[i])
        if key in accum and len(accum[key]) >= 3: expanding_mean[i] = np.mean(accum[key])
        if key not in accum: accum[key] = []
        if not np.isnan(vol[i]): accum[key].append(vol[i])
    df["dow_month_expanding_mean"] = expanding_mean

    dow_expanding = np.full(len(df), np.nan)
    accum_dow = {}
    for i in range(len(df)):
        d = dow[i]
        if d in accum_dow and len(accum_dow[d]) >= 5: dow_expanding[i] = np.mean(accum_dow[d])
        if d not in accum_dow: accum_dow[d] = []
        if not np.isnan(vol[i]): accum_dow[d].append(vol[i])
    df["dow_expanding_mean"] = dow_expanding

    month_expanding = np.full(len(df), np.nan)
    accum_month = {}
    for i in range(len(df)):
        m = month[i]
        if m in accum_month and len(accum_month[m]) >= 10: month_expanding[i] = np.mean(accum_month[m])
        if m not in accum_month: accum_month[m] = []
        if not np.isnan(vol[i]): accum_month[m].append(vol[i])
    df["month_expanding_mean"] = month_expanding
    return df


# ==========================================================================
# Holiday expected volume + shape features
# ==========================================================================
def add_holiday_expected_volume(df, target_col, major_dates, presidents_dates):
    df = df.copy()
    dates = pd.to_datetime(df["Date"])
    vol = df[target_col].values
    date_years = dates.dt.year.values
    all_holidays = dict(major_dates)
    all_holidays["presidents"] = presidents_dates

    holiday_expected_lookups = {}
    holiday_aligned_lookups = {}
    for hname, hdates in all_holidays.items():
        rel_col = f"_{hname}_rel_day"
        if rel_col not in df.columns: continue
        rel_days = df[rel_col].values
        year_relday_vol = {}
        for i in range(len(df)):
            rd = rel_days[i]; y = date_years[i]
            if abs(rd) > 14: continue
            if y not in year_relday_vol: year_relday_vol[y] = {}
            year_relday_vol[y][rd] = vol[i]
        holiday_aligned_lookups[hname] = year_relday_vol
        all_years = sorted(year_relday_vol.keys())
        expected_lookup = {}
        for cur_year in all_years:
            expected_lookup[cur_year] = {}
            for rd in range(-14, 15):
                prior_vols = [year_relday_vol[y][rd] for y in all_years
                              if y < cur_year and rd in year_relday_vol[y]]
                if prior_vols: expected_lookup[cur_year][rd] = np.mean(prior_vols)
        holiday_expected_lookups[hname] = expected_lookup

    nearest_expected = np.full(len(df), np.nan)
    nearest_expected_tomorrow = np.full(len(df), np.nan)
    nearest_aligned_lag = np.full(len(df), np.nan)
    for i in range(len(df)):
        cur_year = date_years[i]
        best_hname = None; best_rel = 999
        for hname in all_holidays:
            rel_col = f"_{hname}_rel_day"
            if rel_col not in df.columns: continue
            rd = df[rel_col].values[i]
            if abs(rd) < abs(best_rel): best_rel = rd; best_hname = hname
        if best_hname is None or abs(best_rel) > 14: continue
        exp_lookup = holiday_expected_lookups.get(best_hname, {}).get(cur_year, {})
        if best_rel in exp_lookup: nearest_expected[i] = exp_lookup[best_rel]
        if (best_rel + 1) in exp_lookup: nearest_expected_tomorrow[i] = exp_lookup[best_rel + 1]
        aligned_lookup = holiday_aligned_lookups.get(best_hname, {})
        prev_year = cur_year - 1
        if prev_year in aligned_lookup and best_rel in aligned_lookup[prev_year]:
            nearest_aligned_lag[i] = aligned_lookup[prev_year][best_rel]

    df["nearest_holiday_expected_vol"] = nearest_expected
    df["holiday_expected_change"] = nearest_expected_tomorrow - nearest_expected
    df["holiday_expected_change_pct"] = np.where(
        nearest_expected > 0, (nearest_expected_tomorrow - nearest_expected) / nearest_expected, np.nan)
    if "dow_month_expanding_mean" in df.columns:
        df["holiday_expected_vs_normal"] = np.where(
            df["dow_month_expanding_mean"] > 0,
            nearest_expected / df["dow_month_expanding_mean"], np.nan)
    df["nearest_holiday_aligned_lag"] = nearest_aligned_lag
    shifted = df[target_col].shift(1).values
    df["nearest_holiday_aligned_ratio"] = np.where(
        nearest_aligned_lag > 0, shifted / nearest_aligned_lag, np.nan)
    return df


# ==========================================================================
# Holiday-DOW pooled features (general, not per-holiday)
# ==========================================================================
def add_holiday_pooled_dow_features(df, target_col=TARGET_COL):
    """
    Two general (holiday-agnostic) features that give the model an explicit
    DOW-normalized view of holiday effects:

    1) holiday_dow_shift_ratio:
         nearest_holiday_aligned_lag / dow_expanding_mean
       Multiplier interpretation. A "Sunday before Memorial Day" gets
       ~ (prior-year Sunday-before-Memorial-Day volume) / (typical Sunday)
       so the model can directly scale its DOW baseline.

    2) holiday_class_dow_ratio:
         leak-free expanding mean of (volume / dow_expanding_mean) keyed by
         (signed_days_to_nearest_holiday, holiday_DOW). Pools across ALL
         holidays that share the same DOW pattern, e.g. every Monday-holiday
         eve-Sunday (MLK, Presidents, Memorial, Labor, occasional July4)
         collapses into a single estimate with ~5x more observations than
         the single-holiday aligned lookup.
    """
    df = df.copy()

    if "nearest_holiday_aligned_lag" in df.columns and "dow_expanding_mean" in df.columns:
        aligned = df["nearest_holiday_aligned_lag"].values.astype(float)
        dow_exp = df["dow_expanding_mean"].values.astype(float)
        df["holiday_dow_shift_ratio"] = np.where(
            (dow_exp > 0) & ~np.isnan(aligned),
            aligned / np.where(dow_exp > 0, dow_exp, 1.0),
            1.0,
        )

    needed = ["days_to_holiday_signed", "days_to_holiday",
              "dayofweek", "dow_expanding_mean"]
    if all(c in df.columns for c in needed):
        signed_rd = df["days_to_holiday_signed"].values
        days_to = df["days_to_holiday"].values
        # holiday_DOW derived inline so this function doesn't depend on the
        # later add_regime_and_decay_features call order:
        h_dow = (df["dayofweek"].values - signed_rd) % 7
        dow_exp = df["dow_expanding_mean"].values
        vol = df[target_col].values

        cell_accum = {}
        ratios = np.full(len(df), 1.0)
        for i in range(len(df)):
            if np.isnan(signed_rd[i]) or np.isnan(h_dow[i]) or days_to[i] > 14:
                # still accumulate this row's contribution to no cell — skip
                continue
            key = (int(signed_rd[i]), int(h_dow[i]))
            # leak-free: read BEFORE writing this row
            if key in cell_accum and len(cell_accum[key]) >= 3:
                ratios[i] = float(np.mean(cell_accum[key]))
            if not np.isnan(vol[i]) and not np.isnan(dow_exp[i]) and dow_exp[i] > 0:
                cell_accum.setdefault(key, []).append(vol[i] / dow_exp[i])
        df["holiday_class_dow_ratio"] = ratios

    return df


# ==========================================================================
# Last full week's average
# ==========================================================================
def add_last_week_total(df, target_col=TARGET_COL):
    df = df.copy()
    dates = pd.to_datetime(df["Date"])
    dow = dates.dt.dayofweek.values
    vol = df[target_col].values
    last_week_avg = np.full(len(df), np.nan)
    for i in range(8, len(df)):
        current_dow = dow[i]
        days_since_sunday = (current_dow + 1) % 7
        if days_since_sunday == 0: days_since_sunday = 7
        sunday_idx = i - days_since_sunday; monday_idx = sunday_idx - 6
        if monday_idx >= 0 and sunday_idx < i:
            week_vals = vol[monday_idx:sunday_idx + 1]
            if len(week_vals) == 7 and not np.any(np.isnan(week_vals)):
                last_week_avg[i] = np.mean(week_vals)
    df["last_week_avg"] = last_week_avg
    return df


# ==========================================================================
# Volume context features
# ==========================================================================
def add_volume_context_features(df, target_col=TARGET_COL):
    df = df.copy()
    shifted = df[target_col].shift(1)

    df["vol_roll_median_14"] = shifted.rolling(14).median()

    lag7 = df[target_col].shift(7)
    if f"{target_col}_roll_mean_28" in df.columns and "lag365_same_dow_3w_mean" in df.columns:
        df["recent_trend_vs_ly"] = np.where(
            df["lag365_same_dow_3w_mean"] > 0,
            df[f"{target_col}_roll_mean_28"] / df["lag365_same_dow_3w_mean"], np.nan)

    if "holiday_expected_vs_normal" in df.columns and "lag365_same_dow" in df.columns:
        df["holiday_x_lag365"] = df["holiday_expected_vs_normal"] * df["lag365_same_dow"]

    if "lag365_same_dow" in df.columns and "dow_month_expanding_mean" in df.columns:
        df["lag365_vs_expanding"] = np.where(
            df["dow_month_expanding_mean"] > 0, df["lag365_same_dow"] / df["dow_month_expanding_mean"], np.nan)

    if "dow_expanding_mean" in df.columns:
        dow_exp = df["dow_expanding_mean"].values
        ratio_to_expected = np.where(dow_exp > 0, shifted.values / dow_exp, np.nan)
        df["recent_vol_vs_expected_7d"] = pd.Series(ratio_to_expected).rolling(7, min_periods=3).mean().values

    # Friday/Saturday strength (for Sundays)
    dates_ctx = pd.to_datetime(df["Date"]); dow_ctx = dates_ctx.dt.dayofweek.values
    vol_vals = df[target_col].values
    fri_sat_strength = np.full(len(df), np.nan)
    for i in range(2, len(df)):
        if dow_ctx[i] == 6:
            fri_vol = vol_vals[i - 2]; sat_vol = vol_vals[i - 1]
            fri_exp_v = [vol_vals[i - 2 - w * 7] for w in range(1, 5) if i - 2 - w * 7 >= 0]
            sat_exp_v = [vol_vals[i - 1 - w * 7] for w in range(1, 5) if i - 1 - w * 7 >= 0]
            if fri_exp_v and sat_exp_v:
                fri_exp = np.mean(fri_exp_v); sat_exp = np.mean(sat_exp_v)
                if fri_exp > 0 and sat_exp > 0:
                    fri_sat_strength[i] = (fri_vol / fri_exp + sat_vol / sat_exp) / 2
    df["friday_saturday_strength"] = fri_sat_strength
    return df


# ==========================================================================
# Regime, decay, and targeted features
# ==========================================================================
def add_regime_and_decay_features(df, target_col=TARGET_COL):
    df = df.copy()
    shifted = df[target_col].shift(1)
    vals = shifted.values

    # ── 1) Regime vs expected ─────────────────────────────────────────
    if "dow_expanding_mean" in df.columns:
        dow_exp = df["dow_expanding_mean"].values
        ratio_to_expected = np.where(dow_exp > 0, vals / dow_exp, np.nan)
        ratio_s = pd.Series(ratio_to_expected)
        df["recent_vol_vs_expected_3d"] = ratio_s.rolling(3, min_periods=2).mean().values
        df["recent_vol_vs_expected_14d"] = ratio_s.rolling(14, min_periods=5).mean().values

    # ── 2) Regime vs lag365 anchor ────────────────────────────────────
    if "lag365_same_dow" in df.columns:
        lag365 = df["lag365_same_dow"].values
        ratio_to_ly = np.where(lag365 > 0, vals / lag365, np.nan)
        ratio_ly_s = pd.Series(ratio_to_ly)
        df["recent_vol_vs_lag365_7d"] = ratio_ly_s.rolling(7, min_periods=3).mean().values
        df["recent_vol_vs_lag365_14d"] = ratio_ly_s.rolling(14, min_periods=5).mean().values

    # ── 3) Decay from recent local peak ───────────────────────────────
    peak14 = shifted.rolling(14, min_periods=5).max()
    df["current_vs_recent_peak_14d"] = np.where(peak14 > 0, shifted / peak14, np.nan)

    # ── 4) Post-holiday decay × regime ────────────────────────────────
    if "holiday_expected_change_pct" in df.columns and "current_vs_recent_peak_14d" in df.columns:
        df["holiday_decay_shape"] = df["holiday_expected_change_pct"] * df["current_vs_recent_peak_14d"]

    # ── 5) Friday/Saturday decomposition for Sunday ───────────────────
    dates = pd.to_datetime(df["Date"]); dow = dates.dt.dayofweek.values
    raw = df[target_col].values
    fri_vs_expected = np.full(len(df), np.nan)
    fri_sat_sum_vs_expected = np.full(len(df), np.nan)
    for i in range(2, len(df)):
        if dow[i] != 6: continue
        fri_val = raw[i - 2]; sat_val = raw[i - 1]
        fri_hist = [raw[i - 2 - 7 * w] for w in range(1, 5) if i - 2 - 7 * w >= 0]
        sat_hist = [raw[i - 1 - 7 * w] for w in range(1, 5) if i - 1 - 7 * w >= 0]
        if len(fri_hist) >= 2:
            fri_exp = np.mean(fri_hist)
            if fri_exp > 0: fri_vs_expected[i] = fri_val / fri_exp
        if len(fri_hist) >= 2 and len(sat_hist) >= 2:
            combo_exp = np.mean(fri_hist) + np.mean(sat_hist)
            if combo_exp > 0: fri_sat_sum_vs_expected[i] = (fri_val + sat_val) / combo_exp
    df["fri_vs_expected"] = fri_vs_expected
    df["fri_sat_sum_vs_expected"] = fri_sat_sum_vs_expected

    # ── 6) Regime-adjusted anchors ────────────────────────────────────
    if "lag365_same_dow" in df.columns and "recent_vol_vs_lag365_7d" in df.columns:
        df["lag365_regime_adjusted_7d"] = df["lag365_same_dow"] * df["recent_vol_vs_lag365_7d"]
    if "lag365_same_dow" in df.columns and "recent_vol_vs_lag365_14d" in df.columns:
        df["lag365_regime_adjusted_14d"] = df["lag365_same_dow"] * df["recent_vol_vs_lag365_14d"]
    if "lag365_same_dow" in df.columns and "recent_vol_vs_lag365_7d" in df.columns:
        df["lag365_final_anchor"] = df["lag365_same_dow"] * df["recent_vol_vs_lag365_7d"]
    if "lag365_same_dow_3w_mean" in df.columns and "lag365_final_anchor" in df.columns:
        df["lag365_blend_anchor"] = 0.7 * df["lag365_same_dow_3w_mean"] + 0.3 * df["lag365_final_anchor"]
    if "friday_saturday_strength" in df.columns and "recent_vol_vs_expected_7d" in df.columns:
        df["weekend_strength_x_regime"] = df["friday_saturday_strength"] * df["recent_vol_vs_expected_7d"]
    if "lag365_final_anchor" in df.columns and "friday_saturday_strength" in df.columns:
        df["sunday_anchor"] = df["lag365_final_anchor"] * df["friday_saturday_strength"]

        
    

    # ── Weather-penalized anchor ──────────────────────────────────────
    if "lag365_blend_anchor" in df.columns:
        weather_penalized = df["lag365_blend_anchor"].copy()
        if "vol_wtd_storm_impact" in df.columns:
            storm = df["vol_wtd_storm_impact"].clip(0, 2)
            weather_penalized = weather_penalized * (1 - 0.12 * storm - 0.08 * storm ** 2)
        if "p90_snowfall_sum" in df.columns:
            weather_penalized = weather_penalized * (1 - 0.03 * df["p90_snowfall_sum"].clip(0, 3))
        df["weather_penalized_anchor"] = weather_penalized

    # ── 7) Regime acceleration ────────────────────────────────────────
    if "recent_vol_vs_expected_7d" in df.columns:
        rvse7 = df["recent_vol_vs_expected_7d"]
        df["regime_accel_7d"] = rvse7 - rvse7.shift(7)
        df["regime_accel_3d"] = rvse7 - rvse7.shift(3)
    if "lag365_regime_adjusted_14d" in df.columns and "regime_accel_7d" in df.columns:
        df["anchor_accel_adjusted"] = (
            df["lag365_regime_adjusted_14d"] * (1 + df["regime_accel_7d"].clip(-0.15, 0.15)))

    # ── 8) Consecutive decline counter ────────────────────────────────
    vol_shifted = df[target_col].shift(1).values
    vol_shifted2 = df[target_col].shift(2).values
    declining = np.where(
        (~np.isnan(vol_shifted)) & (~np.isnan(vol_shifted2)),
        (vol_shifted < vol_shifted2).astype(int), 0)
    consec_decline = np.zeros(len(df))
    for i in range(1, len(df)):
        if declining[i] == 1: consec_decline[i] = consec_decline[i - 1] + 1
    df["consecutive_decline_days"] = consec_decline
    decline_mag = np.zeros(len(df))
    for i in range(1, len(df)):
        if declining[i] == 1 and not np.isnan(vol_shifted[i]) and not np.isnan(vol_shifted2[i]):
            decline_mag[i] = decline_mag[i - 1] + (vol_shifted2[i] - vol_shifted[i])
    df["decline_magnitude"] = decline_mag

    # ── 9) Holiday-DOW interaction ────────────────────────────────────
    if "nearest_holiday_expected_vol" in df.columns and "dayofweek" in df.columns:
        df["holiday_expected_x_dow"] = df["nearest_holiday_expected_vol"] * df["dayofweek"] / 1e6
    if "days_to_holiday" in df.columns and "dayofweek" in df.columns:
        df["days_to_holiday_x_dow"] = df["days_to_holiday"] * df["dayofweek"]
    if "days_to_holiday_signed" in df.columns and "dayofweek" in df.columns:
        holiday_dow = (df["dayofweek"].values - df["days_to_holiday_signed"].values) % 7
        df["nearest_holiday_dow"] = holiday_dow
        df["holiday_on_weekend"] = (holiday_dow >= 5).astype(int)

    # ── 10) Nonlinear weather severity ────────────────────────────────
    if "vol_wtd_storm_impact" in df.columns:
        storm = df["vol_wtd_storm_impact"]
        df["storm_impact_sq"] = storm ** 2
        df["storm_severe_flag"] = (storm > 0.5).astype(int)
    if "vol_wtd_storm_impact" in df.columns and "recent_vol_vs_expected_7d" in df.columns:
        df["storm_x_regime"] = df["vol_wtd_storm_impact"] * df["recent_vol_vs_expected_7d"]

    # ── 11) Faster regime-adjusted anchors ────────────────────────────
    if "lag365_same_dow" in df.columns and "recent_vol_vs_expected_3d" in df.columns:
        df["lag365_regime_adjusted_3d"] = df["lag365_same_dow"] * df["recent_vol_vs_expected_3d"]
    if "lag365_same_dow_3w_mean" in df.columns and "recent_vol_vs_lag365_7d" in df.columns:
        df["blend_anchor_tight"] = (
            0.5 * df["lag365_same_dow_3w_mean"] * df["recent_vol_vs_lag365_7d"] +
            0.5 * df.get("lag365_final_anchor", df["lag365_same_dow"] * df["recent_vol_vs_lag365_7d"]))
        


    # ── 12) Month × regime anchor ─────────────────────────────────────
    if "lag365_blend_anchor" in df.columns and "month" in df.columns:
        df["anchor_x_month"] = df["lag365_blend_anchor"] * df["month"] / 1e6

    # ── 13) Pre-holiday regime ────────────────────────────────────────
    if "days_to_holiday_signed" in df.columns and "recent_vol_vs_expected_7d" in df.columns:
        pre_holiday_zone = (
            (df["days_to_holiday_signed"] >= -7) & (df["days_to_holiday_signed"] <= -4)).astype(int)
        df["pre_holiday_regime"] = pre_holiday_zone * df["recent_vol_vs_expected_7d"]
    if "days_to_holiday" in df.columns and "lag365_blend_anchor" in df.columns:
        df["anchor_x_holiday_dist"] = np.where(
            df["days_to_holiday"] <= 14,
            df["lag365_blend_anchor"] * (1 / (1 + df["days_to_holiday"])), 0)

    # ── 14) Volume-level anchor adjustment ────────────────────────────
    if "lag365_blend_anchor" in df.columns and "Volume_roll_max_28" in df.columns:
        df["recent_max_vs_anchor"] = np.where(
            df["lag365_blend_anchor"] > 0,
            df["Volume_roll_max_28"] / df["lag365_blend_anchor"], np.nan)

    # ── 15) Sunday regime components ──────────────────────────────────
    if "fri_sat_sum_vs_expected" in df.columns and "recent_vol_vs_expected_7d" in df.columns:
        dates_tmp = pd.to_datetime(df["Date"])
        is_sunday = (dates_tmp.dt.dayofweek == 6)
        df["sunday_weekend_regime"] = np.where(
            is_sunday,
            df["fri_sat_sum_vs_expected"] * df["recent_vol_vs_expected_7d"], np.nan)

    # ── 16) Weather × anchor interaction ──────────────────────────────
    if "weather_penalized_anchor" in df.columns and "recent_vol_vs_expected_7d" in df.columns:
        df["weather_regime_anchor"] = df["weather_penalized_anchor"] * df["recent_vol_vs_expected_7d"]

    # ── Clip ratios ───────────────────────────────────────────────────
    for col in ["recent_vol_vs_expected_3d", "recent_vol_vs_expected_14d",
                 "recent_vol_vs_lag365_7d", "recent_vol_vs_lag365_14d",
                 "current_vs_recent_peak_14d",
                 "fri_vs_expected", "fri_sat_sum_vs_expected"]:
        if col in df.columns: df[col] = df[col].clip(lower=0.6, upper=1.4)

    df["anchor_master"] = (
        0.45 * df["lag365_final_anchor"] +
        0.35 * df["lag365_regime_adjusted_14d"] +
        0.20 * df["lag365_regime_adjusted_7d"]
    )

    df["anchor_gap_7d"] = (
        df["Volume"].shift(1) - df["lag365_final_anchor"]
    ).rolling(7).mean()

    df["anchor_super"] = (
        0.35 * df["anchor_master"] +
        0.30 * df["blend_anchor_tight"] +
        0.20 * df["lag365_blend_anchor"] +
        0.15 * df["weather_penalized_anchor"]
    )

    df["lag7_minus_anchor_master"] = df["Volume_lag7"] - df["anchor_master"]

    df["anchor_residual_7d"] = (
        (df["Volume"].shift(1) - df["anchor_master"])
    ).rolling(7).mean()

    df["week_progress_ratio"] = np.where(
        df["dow_weekly_share"] > 0,
        df["week_vol_cumsum"] / (df["dow_weekly_share"] * df["last_week_avg"] * 7),
        np.nan
    )

    df["volatility_7d"] = (
        df["Volume"].shift(1).rolling(7).std()
    )


    df["volatility_ratio_7d"] = np.where(
        df["dow_expanding_mean"] > 0,
        df["volatility_7d"] / df["dow_expanding_mean"],
        np.nan
    )

    df["log_vol_trend_3d"] = (
    np.log1p(df["Volume"].shift(1)) -
    np.log1p(df["Volume"].shift(4))
    )

    df["lag7_vs_anchor_master"] = np.where(
        df["anchor_master"] > 0,
        df["Volume_lag7"] / df["anchor_master"],
        np.nan
    )

    df["log_lag7_vs_anchor"] = (
        np.log1p(df["Volume_lag7"]) - np.log1p(df["anchor_master"])
    )

    df["anchor_trend_7d"] = (
        df["anchor_master"] - df["anchor_master"].shift(7)
    ) / df["anchor_master"].shift(7)

    df["lag7_x_dow_share"] = df["Volume_lag7"] * df["dow_weekly_share"]

    df["anchor_x_dow_share"] = df["anchor_master"] * df["dow_weekly_share"]

    df["same_dow_momentum_2w"] = df["Volume_lag7"] - df["Volume_lag14"]

    df["regime_gap_3d_7d"] = (
        df["recent_vol_vs_expected_3d"] - df["recent_vol_vs_expected_7d"]
    )

    return df


# ==========================================================================
# New advanced same-DOW and contextual features
# ==========================================================================
def add_new_features(df, target_col=TARGET_COL):
    df = df.copy()
    dates = pd.to_datetime(df["Date"])
    dow = dates.dt.dayofweek.values
    vol = df[target_col].values
    n = len(df)

    # ── 1) same_dow_residual_7d ───────────────────────────────────────
    # 7-day rolling mean of (lag1 - lag8): how much recent days have
    # systematically beaten/missed the same DOW the prior week
    dow_residual_daily = df[target_col].shift(1) - df[target_col].shift(8)
    df["same_dow_residual_7d"] = dow_residual_daily.rolling(7, min_periods=3).mean()

    # ── 2) same_dow_streak ────────────────────────────────────────────
    # Consecutive same-DOW week-over-week increases going backward
    # (how many weeks in a row this DOW has been rising vs the prior week)
    same_dow_streak = np.zeros(n)
    for i in range(14, n):
        count = 0
        k = 1
        while (i - 7 * k >= 7
               and not np.isnan(vol[i - 7 * k])
               and not np.isnan(vol[i - 7 * (k + 1)])):
            if vol[i - 7 * k] > vol[i - 7 * (k + 1)]:
                count += 1
                k += 1
            else:
                break
        same_dow_streak[i] = count
    df["same_dow_streak"] = same_dow_streak

    # ── 3) week_progress_vs_lastweek ─────────────────────────────────
    # Current week cumulative volume / same cumulative position last week
    if "week_vol_cumsum" in df.columns:
        prior = df["week_vol_cumsum"].shift(7)
        df["week_progress_vs_lastweek"] = np.where(prior > 0, df["week_vol_cumsum"] / prior, np.nan)

    # ── 4) same_dow_regime_ratio ──────────────────────────────────────
    # vol[i-7] / lag365_same_dow: how last week's same DOW compares directly
    # to the same DOW last year (more responsive than the 4w-mean version)
    if "lag365_same_dow" in df.columns:
        df["same_dow_regime_ratio"] = np.where(
            df["lag365_same_dow"] > 0,
            df[target_col].shift(7) / df["lag365_same_dow"],
            np.nan,
        )

    # ── 5) rolling_same_dow_std ───────────────────────────────────────
    # Std dev across the 8 most recent same-DOW values
    rolling_same_dow_std = np.full(n, np.nan)
    for i in range(56, n):
        vals = np.array([vol[i - 7 * w] for w in range(1, 9)])
        if not np.any(np.isnan(vals)):
            rolling_same_dow_std[i] = np.std(vals)
    df["rolling_same_dow_std"] = rolling_same_dow_std

    # ── 6) same_dow_percentile ────────────────────────────────────────
    # Where yesterday's volume ranks within the last 8 same-DOW values;
    # captures whether today is shaping up above/below the recent DOW baseline
    same_dow_percentile = np.full(n, np.nan)
    for i in range(56, n):
        hist = [vol[i - 7 * w] for w in range(1, 9)]
        if not np.any(np.isnan(hist)):
            current = vol[i - 1]
            if not np.isnan(current):
                same_dow_percentile[i] = sum(current > h for h in hist) / len(hist)
    df["same_dow_percentile"] = same_dow_percentile

    # ── 7) weekend_return_pressure ────────────────────────────────────
    # (yesterday + day-before) / (2 × 4-week same-DOW mean):
    # how strong recent momentum is relative to the DOW baseline
    if "dow_4w_mean" in df.columns:
        lag1 = df[target_col].shift(1)
        lag2 = df[target_col].shift(2)
        df["weekend_return_pressure"] = np.where(
            df["dow_4w_mean"] > 0,
            (lag1 + lag2) / (df["dow_4w_mean"] * 2),
            np.nan,
        )

    # ── 8) holiday_shape_similarity ───────────────────────────────────
    # (vol_lag7 - vol_lag14) - (lag365_same_dow - lag365_same_dow_3w_mean):
    # whether the current week-over-week change matches the typical holiday
    # approach/departure pattern seen at this date last year
    if "lag365_same_dow" in df.columns and "lag365_same_dow_3w_mean" in df.columns:
        df["holiday_shape_similarity"] = (
            (df[target_col].shift(7) - df[target_col].shift(14)) -
            (df["lag365_same_dow"] - df["lag365_same_dow_3w_mean"])
        )

    # ── 9) lag365_growth_accel ────────────────────────────────────────
    # How fast the YoY-growth ratio is accelerating: change in the 7-day
    # rolling YoY ratio vs the same ratio one week ago
    if "recent_vol_vs_lag365_7d" in df.columns:
        rvl = df["recent_vol_vs_lag365_7d"]
        df["lag365_growth_accel"] = rvl - rvl.shift(7)

    # ── 10) volume_accel_3d / volume_accel_pct_3d ────────────────────
    # Difference between the 3-day mean ending yesterday and the 3-day
    # mean ending 3 days before that; pct variant normalises by the base
    base_3d = df[target_col].shift(4).rolling(3, min_periods=2).mean()
    recent_3d = df[target_col].shift(1).rolling(3, min_periods=2).mean()
    df["volume_accel_3d"] = recent_3d - base_3d
    df["volume_accel_pct_3d"] = np.where(base_3d > 0, df["volume_accel_3d"] / base_3d, np.nan)

    # ── 11) adaptive_anchor ───────────────────────────────────────────
    # Blended near-term anchor weighting lag365 more than pure same-DOW
    # lags: useful when the YoY trend dominates short-term momentum
    if "lag365_same_dow" in df.columns:
        df["adaptive_anchor"] = (
            0.55 * df["lag365_same_dow"] +
            0.30 * df[target_col].shift(7) +
            0.15 * df[target_col].shift(14)
        )

    # ── lag365 / 2-year anchor family ────────────────────────────────
    year = dates.dt.year.values
    woy = dates.dt.isocalendar().week.values.astype(int)
    lookup = {}
    for i in range(n):
        if not np.isnan(vol[i]):
            lookup[(year[i], woy[i], dow[i])] = vol[i]

    # ── 12) lag730_same_dow ───────────────────────────────────────────
    # Volume for the exact same day-of-week 2 years ago (104 weeks);
    # second independent anchor point for secular trend estimation
    lag730 = np.full(n, np.nan)
    for i in range(n):
        key = (year[i] - 2, woy[i], dow[i])
        if key in lookup:
            lag730[i] = lookup[key]
    df["lag730_same_dow"] = lag730

    # ── 13) lag365_2yr_avg ────────────────────────────────────────────
    # Average of lag365 and lag730 same-DOW; more stable baseline that
    # smooths out any single year's anomaly (COVID recovery, weather, etc.)
    if "lag365_same_dow" in df.columns:
        lag365_arr = df["lag365_same_dow"].values
        df["lag365_2yr_avg"] = np.where(
            ~np.isnan(lag730) & ~np.isnan(lag365_arr),
            (lag365_arr + lag730) / 2,
            np.where(~np.isnan(lag365_arr), lag365_arr, np.nan),
        )

    # ── 14) lag365_yoy_growth_rate ────────────────────────────────────
    # YoY growth embedded in the anchors (lag365/lag730 - 1); captures
    # the secular travel-demand trend the model should extrapolate
    if "lag365_same_dow" in df.columns:
        df["lag365_yoy_growth_rate"] = np.where(
            lag730 > 0,
            df["lag365_same_dow"] / lag730 - 1,
            np.nan,
        )

    # ── 15) same_dow_3w_slope ─────────────────────────────────────────
    # OLS slope through vol[i-21], vol[i-14], vol[i-7]; for 3 equally
    # spaced points the slope = (last - first) / span
    same_dow_3w_slope = np.full(n, np.nan)
    for i in range(21, n):
        v21 = vol[i - 21]
        v7 = vol[i - 7]
        if not np.isnan(v21) and not np.isnan(v7):
            same_dow_3w_slope[i] = (v7 - v21) / 14.0
    df["same_dow_3w_slope"] = same_dow_3w_slope

    # ── 16) lag365_week_total ─────────────────────────────────────────
    # Total TSA volume for the same ISO-week last year; gives a whole-week
    # YoY reference even when a single day's lag365 is holiday-contaminated
    woy_totals: dict = {}
    woy_counts: dict = {}
    for i in range(n):
        key = (year[i], woy[i])
        if not np.isnan(vol[i]):
            woy_totals[key] = woy_totals.get(key, 0.0) + vol[i]
            woy_counts[key] = woy_counts.get(key, 0) + 1
    lag365_week_total = np.full(n, np.nan)
    for i in range(n):
        key = (year[i] - 1, woy[i])
        if key in woy_totals and woy_counts.get(key, 0) >= 5:
            lag365_week_total[i] = woy_totals[key]
    df["lag365_week_total"] = lag365_week_total

    # ── 17) recent_vs_lag365_4w ───────────────────────────────────────
    # 4-week rolling mean of vol[i-7k] / lag365_same_dow[i-7k] for k=1..4;
    # shows how consistently this DOW has been beating/missing last year
    if "lag365_same_dow" in df.columns:
        lag365_arr = df["lag365_same_dow"].values
        recent_vs_lag365_4w = np.full(n, np.nan)
        for i in range(28, n):
            ratios = []
            for k in range(1, 5):
                idx = i - 7 * k
                if not np.isnan(vol[idx]) and lag365_arr[idx] > 0:
                    ratios.append(vol[idx] / lag365_arr[idx])
            if len(ratios) >= 2:
                recent_vs_lag365_4w[i] = np.mean(ratios)
        df["recent_vs_lag365_4w"] = recent_vs_lag365_4w

    # ── 18) vol_same_dow_vs_8w ────────────────────────────────────────
    # vol[i-7] / 8-week same-DOW rolling mean; fast signal of whether
    # last week's same DOW was above or below its recent baseline
    if "vol_same_dow_mean_8w" in df.columns:
        df["vol_same_dow_vs_8w"] = np.where(
            df["vol_same_dow_mean_8w"] > 0,
            df[target_col].shift(7) / df["vol_same_dow_mean_8w"],
            np.nan,
        )

    # ── 19) dow_month_lag365_ratio ────────────────────────────────────
    # lag365 same-DOW / long-run DOW×month expanding mean; tells the model
    # whether last year was a "hot" or "cold" instance of this DOW in this month
    if "lag365_same_dow" in df.columns and "dow_month_expanding_mean" in df.columns:
        df["dow_month_lag365_ratio"] = np.where(
            df["dow_month_expanding_mean"] > 0,
            df["lag365_same_dow"] / df["dow_month_expanding_mean"],
            np.nan,
        )

    # ── 20) lag365_days_to_holiday ────────────────────────────────────
    # days_to_holiday at the approximate lag365 date; small = the lag365
    # anchor may be holiday-contaminated and should be down-weighted
    if "days_to_holiday" in df.columns:
        dth = df["days_to_holiday"].values
        lag365_dth = np.full(n, np.nan)
        for i in range(365, n):
            lag365_dth[i] = dth[i - 365]
        df["lag365_days_to_holiday"] = lag365_dth

    # ── 21) adaptive_anchor_residual_7d ──────────────────────────────
    # Rolling 7d mean of vol[i-1] / adaptive_anchor; tracks how well
    # the blended near-term anchor has been matching actual volume
    if "adaptive_anchor" in df.columns:
        shifted = df[target_col].shift(1).values
        ratio = np.where(df["adaptive_anchor"].values > 0,
                         shifted / df["adaptive_anchor"].values, np.nan)
        df["adaptive_anchor_residual_7d"] = (
            pd.Series(ratio).rolling(7, min_periods=3).mean().values
        )

    # ── 22) same_dow_residual_trend ───────────────────────────────────
    # Week-over-week change in the 7d same-DOW residual; positive = the
    # DOW is accelerating relative to the prior-week benchmark
    df["same_dow_residual_trend"] = (
        df["same_dow_residual_7d"] - df["same_dow_residual_7d"].shift(7)
    )

    # ── 23) weekly_ramp_slope / weekly_ramp_slope_pct ────────────────
    # Short-term ramp: difference between the 3d mean ending yesterday
    # and the 3d mean ending 4 days ago; pct variant vs vol[i-7]
    df["weekly_ramp_slope"] = df["volume_accel_3d"]
    df["weekly_ramp_slope_pct"] = np.where(
        df[target_col].shift(7) > 0,
        df["weekly_ramp_slope"] / df[target_col].shift(7),
        np.nan,
    )

    # ── KEEP_FEATURES_CHAT_NEW_20 ─────────────────────────────────────────

    # 1) Same-DOW residual momentum
    df["same_dow_residual_trend_2w"] = (
        (df["Volume_lag7"] - df["vol_same_dow_vs_8w"]) -
        (df["Volume_lag14"] - df["vol_same_dow_vs_8w"].shift(7))
    )

    # 2) Week acceleration ratio
    df["week_accel_ratio"] = np.where(
        df["week_vol_cumsum"].shift(7) > 0,
        df["week_vol_cumsum"] / df["week_vol_cumsum"].shift(7),
        np.nan,
    )

    # 3) Weekly shape deviation
    df["weekly_shape_deviation"] = (
        df["dow_weekly_share"] - df["dow_weekly_share"].shift(7)
    )

    # 4) Friday overshoot pressure
    df["friday_overshoot"] = np.where(
        df["dayofweek"] == 4,
        df["Volume_lag1"] / df["dow_expanding_mean"],
        np.nan,
    )

    # 5) Weekend exhaustion
    df["weekend_exhaustion"] = np.where(
        df["dayofweek"] <= 1,
        df["weekend_return_pressure"] - df["regime_gap_3d_7d"],
        np.nan,
    )

    # 6) Holiday ramp steepness
    df["holiday_ramp_steepness"] = (
        df["nearest_holiday_expected_vol"] -
        df["nearest_holiday_expected_vol"].shift(3)
    )

    # 7) same_dow_percentile already computed above (section 6)

    # 8) Weekly regime acceleration
    df["weekly_regime_accel"] = (
        df["regime_gap_3d_7d"] - df["regime_gap_3d_7d"].shift(7)
    )

    # 9) Holiday regime gap
    df["holiday_regime_gap"] = np.where(
        df["nearest_holiday_expected_vol"] > 0,
        df["Volume_lag7"] / df["nearest_holiday_expected_vol"],
        np.nan,
    )

    # 10) Volume curvature
    df["volume_curvature_3d"] = df["vol_diff_1"] - df["vol_diff_1"].shift(1)

    # 11) Weekly overperformance streak
    _overperf = (df["regime_gap_3d_7d"] > 0).astype(int)
    df["weekly_overperf_streak"] = _overperf.groupby(
        (_overperf == 0).cumsum()
    ).cumcount()

    # 12) Same-DOW z-score
    df["same_dow_zscore"] = np.where(
        df["rolling_same_dow_std"] > 0,
        (df["Volume_lag7"] - df["vol_same_dow_vs_8w"]) / df["rolling_same_dow_std"],
        np.nan,
    )

    # 13) Early-week carryover (Volume_lag2 computed inline)
    _vol_lag2 = df[target_col].shift(2)
    df["monday_tuesday_carry"] = np.where(
        df["dayofweek"] <= 1,
        (df["Volume_lag1"] + _vol_lag2) / df["last_week_avg"],
        np.nan,
    )

    # 14) Holiday asymmetry
    df["holiday_asymmetry"] = (
        abs(df["days_to_holiday"]) - abs(df["days_to_holiday"].shift(7))
    )

    # 15) Same-DOW slope (4-week)
    df["same_dow_slope_4w"] = (df["Volume_lag7"] - df["Volume_lag28"]) / 3

    # 16) Rolling max overshoot
    df["rolling_max_overshoot"] = np.where(
        df["Volume_roll_max_28"] > 0,
        df["Volume_lag1"] / df["Volume_roll_max_28"],
        np.nan,
    )

    # 17) Holiday proximity momentum
    df["holiday_proximity_momentum"] = (
        df["days_to_holiday"].shift(7) - df["days_to_holiday"]
    )

    # 18) Weekend amplification factor
    df["weekend_amplification"] = df["fri_vs_expected"] * df["dow_weekly_share"]

    # 19) Regime smoothness
    df["regime_smoothness"] = df["regime_gap_3d_7d"].rolling(7).std()

    # 20) Lag7 vs current week pace
    df["lag7_vs_current_weekpace"] = np.where(
        df["week_vol_cumsum"] > 0,
        df["Volume_lag7"] / df["week_vol_cumsum"],
        np.nan,
    )

    # ── KEEP_FEATURES_CLAUDE_20 ───────────────────────────────────────────────

    _dt = pd.to_datetime(df["Date"])
    _month = _dt.dt.month
    _day   = _dt.dt.day

    # 1) doy_sin — sine of day-of-year (complement to existing doy_cos)
    df["doy_sin"] = np.sin(2 * np.pi * _dt.dt.dayofyear / 365.25)

    # 2) quarter — Q1-Q4
    df["quarter"] = _dt.dt.quarter.astype(float)

    # 3) post_holiday_flag — was yesterday a holiday?
    df["post_holiday_flag"] = df["is_holiday"].shift(1).fillna(0)

    # 4) prior_7d_total — rolling 7-day volume sum (always calendar-week-agnostic)
    df["prior_7d_total"] = df[target_col].shift(1).rolling(7).sum()

    # 5) vol_vs_rm28 — lag1 relative to 28-day rolling mean
    df["vol_vs_rm28"] = np.where(
        df["Volume_roll_mean_28"] > 0,
        df["Volume_lag1"] / df["Volume_roll_mean_28"],
        np.nan,
    )

    # 6) rolling_vol_trend_14d — 14d mean minus 28d mean (trend direction)
    df["rolling_vol_trend_14d"] = (
        df[target_col].shift(1).rolling(14).mean() - df["Volume_roll_mean_28"]
    )

    # 7) vol_coef_variation — 28d std / 28d mean (recent volatility level)
    df["vol_coef_variation"] = np.where(
        df["Volume_roll_mean_28"] > 0,
        df["Volume_roll_std_28"] / df["Volume_roll_mean_28"],
        np.nan,
    )

    # 8) lag_ratio_3_7 — short vs medium lag ratio
    df["lag_ratio_3_7"] = np.where(
        df["Volume_lag7"] > 0,
        df["Volume_lag3"] / df["Volume_lag7"],
        np.nan,
    )

    # 9) lag7_vs_dow_expanding — lag7 vs DOW expanding mean
    df["lag7_vs_dow_expanding"] = np.where(
        df["dow_expanding_mean"] > 0,
        df["Volume_lag7"] / df["dow_expanding_mean"],
        np.nan,
    )

    # 10) summer_flag — June/July/August
    df["summer_flag"] = _month.isin([6, 7, 8]).astype(float)

    # 11) holiday_season_flag — Thanksgiving through New Year's (Nov 20 – Jan 5)
    df["holiday_season_flag"] = (
        ((_month == 11) & (_day >= 20)) | (_month == 12) | ((_month == 1) & (_day <= 5))
    ).astype(float)

    # 12) thu_pre_weekend — Thursday lag1 vs DOW expanding mean (pre-weekend ramp)
    df["thu_pre_weekend"] = np.where(
        (df["dayofweek"] == 3) & (df["dow_expanding_mean"] > 0),
        df["Volume_lag1"] / df["dow_expanding_mean"],
        np.nan,
    )

    # 13) is_long_weekend — Fri or Mon within 3 days of a holiday
    df["is_long_weekend"] = np.where(
        ((df["dayofweek"] == 4) | (df["dayofweek"] == 0)) & (df["days_to_holiday"] <= 3),
        1.0, 0.0,
    )

    # 14) yoy_ratio_7d — Volume_lag7 vs same week 52 weeks ago (364-day shift)
    _vol_lag7_yoy = df["Volume_lag7"].shift(364)
    df["yoy_ratio_7d"] = np.where(
        _vol_lag7_yoy > 0,
        df["Volume_lag7"] / _vol_lag7_yoy,
        np.nan,
    )

    # 15) week_of_year — ISO week number
    df["week_of_year"] = _dt.dt.isocalendar().week.astype(int)

    # 16) prior_week_total — sum of lag1 through lag7
    df["prior_week_total"] = sum(df[target_col].shift(i) for i in range(1, 8))

    # 17) vol_momentum_3d — 3-day rolling mean vs lag7
    _rm3 = df[target_col].shift(1).rolling(3).mean()
    df["vol_momentum_3d"] = np.where(
        df["Volume_lag7"] > 0,
        _rm3 / df["Volume_lag7"],
        np.nan,
    )

    # 18) pre_holiday_ramp — regime_gap weighted by closeness to holiday
    df["pre_holiday_ramp"] = np.where(
        df["days_to_holiday"] <= 7,
        df["regime_gap_3d_7d"] * (7 - df["days_to_holiday"]) / 7,
        0.0,
    )

    # 19) lag28_ratio — 4-week trend ratio
    df["lag28_ratio"] = np.where(
        df["Volume_lag28"] > 0,
        df["Volume_lag7"] / df["Volume_lag28"],
        np.nan,
    )

    # 20) dow_regime_interaction — regime strength weighted by DOW share
    df["dow_regime_interaction"] = df["regime_gap_3d_7d"] * df["dow_weekly_share"]

    return df


# ==========================================================================
# Group features: same-DOW trend, week pace, YoY regime,
#                 anchor reliability, week shape, holiday
# ==========================================================================
def add_group_features(df, target_col=TARGET_COL):
    df = df.copy()
    dates = pd.to_datetime(df["Date"])
    dow = dates.dt.dayofweek.values
    month_arr = dates.dt.month.values
    year_arr = dates.dt.year.values
    vol = df[target_col].values
    n = len(df)

    # Per-DOW expanding means, forward-filled to every row (no leakage:
    # each Monday's row reflects the mean of all prior Mondays only).
    _dow_exp: dict = {}
    for _d in range(7):
        _arr = np.full(n, np.nan)
        _accum: list = []
        for i in range(n):
            if dow[i] == _d:
                if len(_accum) >= 5:
                    _arr[i] = np.mean(_accum)
                if not np.isnan(vol[i]):
                    _accum.append(vol[i])
        _last = np.nan
        for i in range(n):
            if not np.isnan(_arr[i]):
                _last = _arr[i]
            elif not np.isnan(_last):
                _arr[i] = _last
        _dow_exp[_d] = _arr

    # ── Group 1: Same-DOW Trend ───────────────────────────────────────────
    lag7_v  = df[target_col].shift(7).values
    lag14_v = df[target_col].shift(14).values
    lag21_v = df[target_col].shift(21).values
    lag28_v = df[target_col].shift(28).values

    # 1. same_dow_trend_3w — average weekly change over 3 same-DOW windows
    df["same_dow_trend_3w"] = np.where(
        ~np.isnan(lag7_v) & ~np.isnan(lag21_v),
        (lag7_v - lag21_v) / 2.0,
        np.nan,
    )

    # 2. same_dow_trend_4w — 4-week same-DOW slope
    df["same_dow_trend_4w"] = np.where(
        ~np.isnan(lag7_v) & ~np.isnan(lag28_v),
        (lag7_v - lag28_v) / 3.0,
        np.nan,
    )

    # 3. same_dow_growth_rate_3w — relative growth lag7/lag21
    df["same_dow_growth_rate_3w"] = np.where(
        ~np.isnan(lag7_v) & (lag21_v > 0),
        lag7_v / lag21_v,
        np.nan,
    )

    # 4. same_dow_acceleration — second difference of same-DOW values
    df["same_dow_acceleration"] = np.where(
        ~np.isnan(lag7_v) & ~np.isnan(lag14_v) & ~np.isnan(lag21_v),
        (lag7_v - lag14_v) - (lag14_v - lag21_v),
        np.nan,
    )

    # 5. same_dow_consistency — std of last 4 same-DOW values
    _sdow = pd.concat([
        df[target_col].shift(7), df[target_col].shift(14),
        df[target_col].shift(21), df[target_col].shift(28),
    ], axis=1)
    df["same_dow_consistency"] = _sdow.std(axis=1, skipna=False)

    # ── Group 2: Current Week Pace ────────────────────────────────────────
    if "week_vol_cumsum" in df.columns:
        cumsum = df["week_vol_cumsum"]
        _1w = cumsum.shift(7)
        _2w = cumsum.shift(14)

        # 6. week_vs_lastweek_same_point
        df["week_vs_lastweek_same_point"] = np.where(
            _1w > 0, cumsum / _1w, np.nan
        )

        # 7. week_vs_2weeksago_same_point
        df["week_vs_2weeksago_same_point"] = np.where(
            _2w > 0, cumsum / _2w, np.nan
        )

        # 8. week_pace_acceleration
        df["week_pace_acceleration"] = (
            df["week_vs_lastweek_same_point"] - df["week_vs_2weeksago_same_point"]
        )

        # 9. week_progress_vs_4w_avg
        _hist4 = pd.concat([cumsum.shift(7 * k) for k in range(1, 5)], axis=1)
        _mean4 = _hist4.mean(axis=1)
        df["week_progress_vs_4w_avg"] = np.where(_mean4 > 0, cumsum / _mean4, np.nan)

        # 10. week_pace_zscore
        _hist8 = pd.concat([cumsum.shift(7 * k) for k in range(1, 9)], axis=1)
        _mean8 = _hist8.mean(axis=1)
        _std8  = _hist8.std(axis=1)
        df["week_pace_zscore"] = np.where(
            _std8 > 0, (cumsum - _mean8) / _std8, np.nan
        )

    # ── Group 3: YoY Regime ───────────────────────────────────────────────

    # 11. current_year_same_dow_avg — expanding within-year same-DOW mean
    cysda = np.full(n, np.nan)
    _acc_yd: dict = {}
    for i in range(n):
        key = (year_arr[i], dow[i])
        if key in _acc_yd and len(_acc_yd[key]) >= 2:
            cysda[i] = np.mean(_acc_yd[key])
        _acc_yd.setdefault(key, [])
        if not np.isnan(vol[i]):
            _acc_yd[key].append(vol[i])
    df["current_year_same_dow_avg"] = cysda

    # 12. current_year_same_dow_vs_last_year
    if "lag365_same_dow" in df.columns:
        _lag365_v = df["lag365_same_dow"].values
        df["current_year_same_dow_vs_last_year"] = np.where(
            (_lag365_v > 0) & ~np.isnan(cysda),
            cysda / _lag365_v,
            np.nan,
        )

    # 13. current_month_same_dow_avg — expanding within-month same-DOW mean
    cmsda = np.full(n, np.nan)
    _acc_md: dict = {}
    for i in range(n):
        key = (year_arr[i], month_arr[i], dow[i])
        if key in _acc_md and len(_acc_md[key]) >= 2:
            cmsda[i] = np.mean(_acc_md[key])
        _acc_md.setdefault(key, [])
        if not np.isnan(vol[i]):
            _acc_md[key].append(vol[i])
    df["current_month_same_dow_avg"] = cmsda

    # 14. current_month_vs_year_same_dow
    df["current_month_vs_year_same_dow"] = np.where(
        (~np.isnan(cysda)) & (cysda > 0) & (~np.isnan(cmsda)),
        cmsda / cysda,
        np.nan,
    )

    # 15. same_dow_ytd_growth — current-year expanding avg / prior full-year avg
    _full_yd: dict = {}
    for i in range(n):
        key = (year_arr[i], dow[i])
        _full_yd.setdefault(key, [])
        if not np.isnan(vol[i]):
            _full_yd[key].append(vol[i])
    prior_yd_avg = np.full(n, np.nan)
    for i in range(n):
        key = (year_arr[i] - 1, dow[i])
        if key in _full_yd and len(_full_yd[key]) >= 4:
            prior_yd_avg[i] = np.mean(_full_yd[key])
    df["same_dow_ytd_growth"] = np.where(
        (prior_yd_avg > 0) & ~np.isnan(cysda),
        cysda / prior_yd_avg,
        np.nan,
    )

    # ── Group 4: Anchor Reliability ───────────────────────────────────────
    if "lag365_same_dow" in df.columns:
        _sh = df[target_col].shift(1).values
        _l3 = df["lag365_same_dow"].values
        _valid_a = _l3 > 0

        _err = np.where(_valid_a, _sh - _l3, np.nan)
        _rat = np.where(_valid_a, _sh / _l3, np.nan)

        # 16. lag365_error_7d
        df["lag365_error_7d"] = pd.Series(_err).rolling(7, min_periods=3).mean().values

        # 17. lag365_error_pct_7d
        _epct = pd.Series(_rat).rolling(7, min_periods=3).mean()
        df["lag365_error_pct_7d"] = _epct.values

        # 18. anchor_hit_rate_28d
        _hit = np.where(~np.isnan(_rat), (np.abs(_rat - 1) < 0.05).astype(float), np.nan)
        df["anchor_hit_rate_28d"] = pd.Series(_hit).rolling(28, min_periods=10).mean().values

        # 19. anchor_bias_28d
        _bias = np.where(_valid_a, (_sh - _l3) / _l3, np.nan)
        df["anchor_bias_28d"] = pd.Series(_bias).rolling(28, min_periods=10).mean().values

        # 20. anchor_error_trend — is lag365 getting better or worse recently?
        df["anchor_error_trend"] = (_epct - _epct.shift(14)).values

    # ── Group 5: Week Shape ───────────────────────────────────────────────
    # For day i with DOW d, Monday of this week was d days ago (no leakage).
    mon_v = np.full(n, np.nan)
    tue_v = np.full(n, np.nan)
    wed_v = np.full(n, np.nan)
    for i in range(n):
        d = dow[i]
        if d >= 1 and i - d >= 0:
            mon_v[i] = vol[i - d]
        if d >= 2 and i - d + 1 >= 0:
            tue_v[i] = vol[i - d + 1]
        if d >= 3 and i - d + 2 >= 0:
            wed_v[i] = vol[i - d + 2]

    _me = _dow_exp[0]
    _te = _dow_exp[1]
    _we = _dow_exp[2]

    # 21. monday_strength
    df["monday_strength"] = np.where(
        ~np.isnan(mon_v) & (_me > 0), mon_v / _me, np.nan
    )

    # 22. tuesday_strength
    df["tuesday_strength"] = np.where(
        ~np.isnan(tue_v) & (_te > 0), tue_v / _te, np.nan
    )

    # 23. monday_tuesday_combo
    df["monday_tuesday_combo"] = np.where(
        ~np.isnan(mon_v) & ~np.isnan(tue_v) & ((_me + _te) > 0),
        (mon_v + tue_v) / (_me + _te),
        np.nan,
    )

    # 24. first_half_week_strength
    df["first_half_week_strength"] = np.where(
        ~np.isnan(mon_v) & ~np.isnan(tue_v) & ~np.isnan(wed_v) & ((_me + _te + _we) > 0),
        (mon_v + tue_v + wed_v) / (_me + _te + _we),
        np.nan,
    )

    # 25. early_week_surprise — actual cumsum minus expected cumsum so far
    if "week_vol_cumsum" in df.columns:
        _wcs = df["week_vol_cumsum"].values
        _exp_cs = np.full(n, np.nan)
        for i in range(n):
            d = dow[i]
            if d == 0:
                _exp_cs[i] = 0.0
            else:
                exps = [_dow_exp[dd][i] for dd in range(d)]
                if all(not np.isnan(e) for e in exps):
                    _exp_cs[i] = float(np.sum(exps))
        df["early_week_surprise"] = np.where(
            ~np.isnan(_wcs) & ~np.isnan(_exp_cs) & (_exp_cs > 0),
            _wcs - _exp_cs,
            np.nan,
        )

    # ── Group 6: Holiday ──────────────────────────────────────────────────

    # 26. holiday_ramp_speed
    if "nearest_holiday_expected_vol" in df.columns:
        df["holiday_ramp_speed"] = (
            df["nearest_holiday_expected_vol"]
            - df["nearest_holiday_expected_vol"].shift(3)
        )

    # 27. holiday_distance_velocity
    if "days_to_holiday" in df.columns:
        df["holiday_distance_velocity"] = (
            df["days_to_holiday"].shift(7) - df["days_to_holiday"]
        )

    # 28. holiday_regime_strength
    if "recent_vol_vs_expected_7d" in df.columns and "holiday_expected_vs_normal" in df.columns:
        df["holiday_regime_strength"] = (
            df["recent_vol_vs_expected_7d"] * df["holiday_expected_vs_normal"]
        )

    # 29. holiday_shape_error — how much vol.shift(1) deviates from holiday template
    if "nearest_holiday_expected_vol" in df.columns:
        _nhe = df["nearest_holiday_expected_vol"].values
        _sh2 = df[target_col].shift(1).values
        df["holiday_shape_error"] = np.where(
            _nhe > 0, (_sh2 - _nhe) / _nhe, np.nan
        )

    # 30. holiday_alignment_confidence — 1/(1+|days_to_holiday - same day last year|)
    if "days_to_holiday" in df.columns:
        _dth = df["days_to_holiday"].values.astype(float)
        _dth_ly = np.full(n, np.nan)
        for i in range(365, n):
            _dth_ly[i] = _dth[i - 365]
        df["holiday_alignment_confidence"] = np.where(
            ~np.isnan(_dth_ly),
            1.0 / (1.0 + np.abs(_dth - _dth_ly)),
            np.nan,
        )

    return df


# ==========================================================================
# Anchor family features: same-DOW anchor variants, better lag365s,
#                         week projection
# ==========================================================================
def add_anchor_family_features(df, target_col=TARGET_COL):
    df = df.copy()
    dates = pd.to_datetime(df["Date"])
    dow = dates.dt.dayofweek.values
    year_arr = dates.dt.year.values
    vol = df[target_col].values
    n = len(df)
    woy = dates.dt.isocalendar().week.values.astype(int)

    # ── Family 1: Same-DOW Anchor Variants ───────────────────────────────

    # 1. vol_same_dow_median_8w — robust to holiday outliers
    median_8w = np.full(n, np.nan)
    for i in range(56, n):
        vals = np.array([vol[i - 7 * w] for w in range(1, 9)])
        valid = vals[~np.isnan(vals)]
        if len(valid) >= 4:
            median_8w[i] = np.median(valid)
    df["vol_same_dow_median_8w"] = median_8w

    # 2. vol_same_dow_weighted_8w — recency-weighted: 0.4/0.3/0.2/0.1
    lag7_v  = df[target_col].shift(7).values
    lag14_v = df[target_col].shift(14).values
    lag21_v = df[target_col].shift(21).values
    lag28_v = df[target_col].shift(28).values
    df["vol_same_dow_weighted_8w"] = np.where(
        ~np.isnan(lag7_v) & ~np.isnan(lag14_v) & ~np.isnan(lag21_v) & ~np.isnan(lag28_v),
        0.4 * lag7_v + 0.3 * lag14_v + 0.2 * lag21_v + 0.1 * lag28_v,
        np.nan,
    )

    # 3. vol_same_dow_ema_8w — exponential moving average (alpha=2/9) over
    #    last 8 same-DOW observations, computed oldest→newest
    alpha = 2.0 / 9.0
    ema_8w = np.full(n, np.nan)
    for i in range(56, n):
        vals_desc = [vol[i - 7 * w] for w in range(1, 9)]  # lag7 first (newest)
        valid_desc = [v for v in vals_desc if not np.isnan(v)]
        if len(valid_desc) >= 4:
            ema = valid_desc[-1]  # oldest valid
            for v in reversed(valid_desc[:-1]):  # oldest→newest (skip last already used)
                ema = alpha * v + (1 - alpha) * ema
            ema_8w[i] = ema
    df["vol_same_dow_ema_8w"] = ema_8w

    # 4. vol_same_dow_trimmed_8w — drop highest and lowest, average rest
    trimmed_8w = np.full(n, np.nan)
    for i in range(56, n):
        vals = np.array([vol[i - 7 * w] for w in range(1, 9)])
        valid = np.sort(vals[~np.isnan(vals)])
        if len(valid) >= 4:
            trimmed_8w[i] = np.mean(valid[1:-1])
    df["vol_same_dow_trimmed_8w"] = trimmed_8w

    # 5. vol_same_dow_holiday_adj_8w — mean excluding holiday-adjacent weeks
    if "days_to_holiday" in df.columns:
        dth = df["days_to_holiday"].values
        hadj_8w = np.full(n, np.nan)
        for i in range(56, n):
            clean = [vol[i - 7 * w] for w in range(1, 9)
                     if i - 7 * w >= 0
                     and not np.isnan(vol[i - 7 * w])
                     and dth[i - 7 * w] > 7]
            if len(clean) >= 2:
                hadj_8w[i] = np.mean(clean)
        df["vol_same_dow_holiday_adj_8w"] = hadj_8w

    # ── Family 2: Better lag365 ───────────────────────────────────────────
    lookup: dict = {}
    for i in range(n):
        if not np.isnan(vol[i]):
            lookup[(year_arr[i], woy[i], dow[i])] = vol[i]

    def _woy_clamp(w):
        if w < 1:  return w + 52
        if w > 52: return w - 52
        return w

    # 6. lag365_same_dow_5w_mean — 5-week window (−2…+2) around same woy last year
    lag365_5w = np.full(n, np.nan)
    for i in range(n):
        vals = [lookup[(year_arr[i] - 1, _woy_clamp(woy[i] + off), dow[i])]
                for off in range(-2, 3)
                if (year_arr[i] - 1, _woy_clamp(woy[i] + off), dow[i]) in lookup]
        if vals:
            lag365_5w[i] = np.mean(vals)
    df["lag365_same_dow_5w_mean"] = lag365_5w

    # 7. lag365_same_dow_median — median of the 5-week window
    lag365_med = np.full(n, np.nan)
    for i in range(n):
        vals = [lookup[(year_arr[i] - 1, _woy_clamp(woy[i] + off), dow[i])]
                for off in range(-2, 3)
                if (year_arr[i] - 1, _woy_clamp(woy[i] + off), dow[i]) in lookup]
        if len(vals) >= 3:
            lag365_med[i] = np.median(vals)
    df["lag365_same_dow_median"] = lag365_med

    # 8. lag365_same_dow_weighted — 0.5×woy + 0.25×(woy−1) + 0.25×(woy+1)
    lag365_wtd = np.full(n, np.nan)
    for i in range(n):
        v0  = lookup.get((year_arr[i] - 1, woy[i],                   dow[i]))
        vm1 = lookup.get((year_arr[i] - 1, _woy_clamp(woy[i] - 1), dow[i]))
        vp1 = lookup.get((year_arr[i] - 1, _woy_clamp(woy[i] + 1), dow[i]))
        if v0 is not None:
            total_v = 0.5 * v0
            total_w = 0.5
            if vm1 is not None: total_v += 0.25 * vm1; total_w += 0.25
            if vp1 is not None: total_v += 0.25 * vp1; total_w += 0.25
            lag365_wtd[i] = total_v / total_w
    df["lag365_same_dow_weighted"] = lag365_wtd

    # 8.5/8.6 lag365_same_dow_clean and ..._5w_clean — same-DOW lookup
    # filtered by holiday-proximity on BOTH ends. The pair is masked to NaN
    # when this-year OR last-year date is within ±7d of a major holiday —
    # so the feature only fires when both ends are "normal" calendar days.
    # Captures the intuition: "the Nth Tuesday of the year is a strong
    # reference, as long as neither end is holiday-contaminated."
    if "days_to_holiday" in df.columns:
        dth_arr = df["days_to_holiday"].values
        dth_lookup: dict = {}
        for i in range(n):
            dth_lookup[(year_arr[i], woy[i], dow[i])] = dth_arr[i]

        lag365_clean = np.full(n, np.nan)
        for i in range(n):
            if dth_arr[i] <= 7:
                continue
            key = (year_arr[i] - 1, woy[i], dow[i])
            if key not in lookup:
                continue
            last_dth = dth_lookup.get(key)
            if last_dth is None or last_dth <= 7:
                continue
            lag365_clean[i] = lookup[key]
        df["lag365_same_dow_clean"] = lag365_clean

        lag365_5w_clean = np.full(n, np.nan)
        for i in range(n):
            if dth_arr[i] <= 7:
                continue
            vals = []
            for off in range(-2, 3):
                key = (year_arr[i] - 1, _woy_clamp(woy[i] + off), dow[i])
                if key not in lookup:
                    continue
                last_dth = dth_lookup.get(key)
                if last_dth is None or last_dth <= 7:
                    continue
                vals.append(lookup[key])
            if len(vals) >= 2:
                lag365_5w_clean[i] = np.mean(vals)
        df["lag365_same_dow_5w_clean"] = lag365_5w_clean

    # 9. lag365_x_recent_regime — lag365 × recent_vol_vs_expected_14d
    if "lag365_same_dow" in df.columns and "recent_vol_vs_expected_14d" in df.columns:
        df["lag365_x_recent_regime"] = (
            df["lag365_same_dow"] * df["recent_vol_vs_expected_14d"]
        )

    # 10. lag365_residual_anchor — lag365 + 14d rolling mean of recent residuals
    if "lag365_same_dow" in df.columns:
        _resid = df[target_col].shift(1) - df["lag365_same_dow"]
        df["lag365_residual_anchor"] = (
            df["lag365_same_dow"] + _resid.rolling(14, min_periods=5).mean()
        )

    # ── Family 3: Week Projection ─────────────────────────────────────────
    # Per-DOW expanding means (forward-filled) for expected-pace computation
    _dow_exp: dict = {}
    for _d in range(7):
        _arr = np.full(n, np.nan)
        _accum: list = []
        for i in range(n):
            if dow[i] == _d:
                if len(_accum) >= 5:
                    _arr[i] = np.mean(_accum)
                if not np.isnan(vol[i]):
                    _accum.append(vol[i])
        _last = np.nan
        for i in range(n):
            if not np.isnan(_arr[i]):
                _last = _arr[i]
            elif not np.isnan(_last):
                _arr[i] = _last
        _dow_exp[_d] = _arr

    # 11. expected_week_total — average of last 4 complete Mon–Sun week totals
    exp_week_total = np.full(n, np.nan)
    for i in range(n):
        mon_this = i - dow[i]
        totals = []
        for k in range(1, 5):
            mon_k = mon_this - 7 * k
            if mon_k >= 0 and mon_k + 7 <= n:
                wk = vol[mon_k:mon_k + 7]
                if not np.any(np.isnan(wk)):
                    totals.append(float(np.sum(wk)))
        if totals:
            exp_week_total[i] = np.mean(totals)
    df["expected_week_total"] = exp_week_total

    # Expected cumsum up to (but not including) today
    _exp_cs = np.full(n, np.nan)
    for i in range(n):
        d = dow[i]
        if d == 0:
            _exp_cs[i] = 0.0
        else:
            exps = [_dow_exp[dd][i] for dd in range(d)]
            if all(not np.isnan(e) for e in exps):
                _exp_cs[i] = float(np.sum(exps))

    if "week_vol_cumsum" in df.columns:
        _wcs = df["week_vol_cumsum"].values

        # 12. actual_vs_expected_pace — ratio of actual progress to expected progress
        df["actual_vs_expected_pace"] = np.where(
            ~np.isnan(_wcs) & ~np.isnan(_exp_cs) & (_exp_cs > 0),
            _wcs / _exp_cs,
            np.nan,
        )

        # 13. week_completion_ratio — what fraction of expected week total has elapsed
        df["week_completion_ratio"] = np.where(
            ~np.isnan(_exp_cs) & (exp_week_total > 0),
            _exp_cs / exp_week_total,
            np.nan,
        )

        # 14. week_remaining_expected — expected volume still to come this week
        df["week_remaining_expected"] = np.where(
            ~np.isnan(_exp_cs) & ~np.isnan(exp_week_total),
            exp_week_total - _exp_cs,
            np.nan,
        )

        # 15. week_projected_total — project current pace to end of week
        _compl = df["week_completion_ratio"].values
        df["week_projected_total"] = np.where(
            ~np.isnan(_wcs) & (_compl > 0),
            _wcs / _compl,
            np.nan,
        )

    return df


# ==========================================================================
# Feature pipeline

KEEP_FEATURES = [
    "regime_gap_3d_7d",
    "days_to_holiday_x_dow",
    "vol_diff_1",
    "dow_weekly_share",
    "Volume_lag7",
    "week_vol_cumsum",
    "last_week_avg",
    "lag7_x_dow_share",
    "nearest_holiday_aligned_lag",
    "log_Volume_lag7",
    "month",
    "doy_cos",
    "dow_sin",
    "dow_x_month",
    "days_to_holiday",
    "nearest_holiday_expected_vol",
    "holiday_decay_shape",
    "holiday_expected_x_dow",
    "anchor_accel_adjusted",
    "pre_holiday_regime",
    "holiday_expected_vs_normal",
    "same_dow_acceleration",
    "dayofweek",
    "prior_7d_total",
    "holiday_asymmetry",
    "week_accel_ratio",
    "week_progress_vs_4w_avg",
    "vol_same_dow_trimmed_8w",
    "holiday_regime_strength",
    "lag365_residual_anchor",
    "Volume_lag3",
    "lag365_error_7d",
    "first_half_week_strength",
    "vol_same_dow_median_8w",
]

# new_features = [
#     # ── Group 1: same-DOW trend ───────────────────────────────────────────
#     "same_dow_trend_3w",
#     "same_dow_trend_4w",
#     "same_dow_growth_rate_3w",
#     "same_dow_acceleration",
#     "same_dow_consistency",
#     # ── Group 2: current week pace ────────────────────────────────────────
#     "week_vs_lastweek_same_point",
#     "week_vs_2weeksago_same_point",
#     "week_pace_acceleration",
#     "week_progress_vs_4w_avg",
#     "week_pace_zscore",
#     # ── Group 3: YoY regime ───────────────────────────────────────────────
#     "current_year_same_dow_avg",
#     "current_year_same_dow_vs_last_year",
#     "current_month_same_dow_avg",
#     "current_month_vs_year_same_dow",
#     "same_dow_ytd_growth",
#     # ── Group 4: anchor reliability ───────────────────────────────────────
#     "lag365_error_7d",
#     "lag365_error_pct_7d",
#     "anchor_hit_rate_28d",
#     "anchor_bias_28d",
#     "anchor_error_trend",
#     # ── Group 5: week shape ───────────────────────────────────────────────
#     "monday_strength",
#     "tuesday_strength",
#     "monday_tuesday_combo",
#     "first_half_week_strength",
#     "early_week_surprise",
#     # ── Group 6: holiday ──────────────────────────────────────────────────
#     "holiday_ramp_speed",
#     "holiday_distance_velocity",
#     "holiday_regime_strength",
#     "holiday_shape_error",
#     "holiday_alignment_confidence",
#     # ── Family 1: same-DOW anchor variants ───────────────────────────────
#     "vol_same_dow_median_8w",
#     "vol_same_dow_weighted_8w",
#     "vol_same_dow_ema_8w",
#     "vol_same_dow_trimmed_8w",
#     "vol_same_dow_holiday_adj_8w",
#     # ── Family 2: better lag365 ───────────────────────────────────────────
#     "lag365_same_dow_5w_mean",
#     "lag365_same_dow_median",
#     "lag365_same_dow_weighted",
#     "lag365_x_recent_regime",
#     "lag365_residual_anchor",
#     # ── Family 3: week projection ─────────────────────────────────────────
#     "expected_week_total",
#     "actual_vs_expected_pace",
#     "week_completion_ratio",
#     "week_remaining_expected",
#     "week_projected_total",
# ]


# KEEP_FEATURES_CHAT_NEW_20 = [
#     "same_dow_residual_trend_2w",   # 1
#     "week_accel_ratio",             # 2
#     "weekly_shape_deviation",       # 3
#     "friday_overshoot",             # 4
#     "weekend_exhaustion",           # 5
#     "holiday_ramp_steepness",       # 6
#     "same_dow_percentile",          # 7 (already computed)
#     "weekly_regime_accel",          # 8
#     "holiday_regime_gap",           # 9
#     "volume_curvature_3d",          # 10
#     "weekly_overperf_streak",       # 11
#     "same_dow_zscore",              # 12
#     "monday_tuesday_carry",         # 13
#     "holiday_asymmetry",            # 14
#     "same_dow_slope_4w",            # 15
#     "rolling_max_overshoot",        # 16
#     "holiday_proximity_momentum",   # 17
#     "weekend_amplification",        # 18
#     "regime_smoothness",            # 19
#     "lag7_vs_current_weekpace",     # 20
# ]


# KEEP_FEATURES_CLAUDE_20 = [
#     "doy_sin",                  # 1  — complement to doy_cos
#     "quarter",                  # 2  — Q1-Q4
#     "post_holiday_flag",        # 3  — was yesterday a holiday
#     "prior_7d_total",           # 4  — rolling 7-day volume sum
#     "vol_vs_rm28",              # 5  — lag1 vs 28-day mean
#     "rolling_vol_trend_14d",    # 6  — 14d mean minus 28d mean
#     "vol_coef_variation",       # 7  — 28d std/mean
#     "lag_ratio_3_7",            # 8  — lag3 / lag7
#     "lag7_vs_dow_expanding",    # 9  — lag7 vs DOW expanding mean
#     "summer_flag",              # 10 — June/July/August
#     "holiday_season_flag",      # 11 — Nov 20 - Jan 5
#     "thu_pre_weekend",          # 12 — Thursday pre-weekend ramp
#     "is_long_weekend",          # 13 — Fri/Mon near holiday
#     "yoy_ratio_7d",             # 14 — lag7 vs same week last year
#     "week_of_year",             # 15 — ISO week number
#     "prior_week_total",         # 16 — sum of lag1..lag7
#     "vol_momentum_3d",          # 17 — 3-day mean / lag7
#     "pre_holiday_ramp",         # 18 — regime_gap × holiday proximity
#     "lag28_ratio",              # 19 — lag7 / lag28
#     "dow_regime_interaction",   # 20 — regime_gap × dow_weekly_share
# ]

# for f in new_features:
#     if f not in KEEP_FEATURES:
#         KEEP_FEATURES.append(f)

# for f in KEEP_FEATURES_CHAT_NEW_20:
#     if f not in KEEP_FEATURES:
#         KEEP_FEATURES.append(f)

# for f in KEEP_FEATURES_CLAUDE_20:
#     if f not in KEEP_FEATURES:
#         KEEP_FEATURES.append(f)

# KEEP_FEATURES = KEEP_FEATURES + KEEP_FEATURES_CLAUDE_20

# KEEP_FEATURES = [
#     # ── core lags ─────────────────────────────────────────────────────
#     "log_Volume_lag7", "log_Volume_lag14",
#     "Volume_lag3", "Volume_lag7", "Volume_lag14", "Volume_lag28",
#     "vol_diff_1", "vol_diff_7",
#     # ── calendar ──────────────────────────────────────────────────────
#     "dayofweek", "month", "doy_cos", "dow_sin", "dow_cos",
#     "dow_x_month", "dow_month_expanding_mean", "dow_expanding_mean", "month_expanding_mean",
#     # ── holiday ───────────────────────────────────────────────────────
#     "is_holiday", "days_to_holiday", "days_to_holiday_signed",
#     "nearest_holiday_expected_vol", "holiday_expected_change",
#     "holiday_expected_vs_normal", "holiday_x_lag365", "holiday_decay_shape",
#     "holiday_expected_x_dow", "days_to_holiday_x_dow", "nearest_holiday_dow",
#     "holiday_on_weekend", "nearest_holiday_aligned_lag", "nearest_holiday_aligned_ratio",
#     "holiday_shape_similarity",
#     # ── anchors ───────────────────────────────────────────────────────
#     "lag365_blend_anchor", "weather_penalized_anchor", "anchor_master",
#     "lag365_regime_adjusted_3d", "blend_anchor_tight",
#     "anchor_x_month", "anchor_x_holiday_dist", "anchor_x_dow_share",
#     "anchor_accel_adjusted",
#     "pre_holiday_regime", "recent_max_vs_anchor",
#     "adaptive_anchor", "adaptive_anchor_residual_7d",
#     # ── lag365 family ─────────────────────────────────────────────────
#     "lag365_vs_expanding", "lag365_growth_accel",
#     "lag730_same_dow", "lag365_2yr_avg", "lag365_yoy_growth_rate",
#     "lag365_week_total", "recent_vs_lag365_4w",
#     "dow_month_lag365_ratio", "lag365_days_to_holiday",
#     # ── regime ────────────────────────────────────────────────────────
#     "recent_vol_vs_expected_7d", "recent_vol_vs_expected_3d",
#     "recent_vol_vs_lag365_7d", "recent_trend_vs_ly",
#     "regime_accel_3d", "regime_gap_3d_7d",
#     "weekend_strength_x_regime", "storm_x_regime",
#     "sunday_weekend_regime", "weather_regime_anchor",
#     "sunday_anchor",
#     # ── DOW / within-week ─────────────────────────────────────────────
#     "dow_4w_mean", "dow_weekly_share", "lag7_x_dow_share",
#     "vol_same_dow_mean_8w", "vol_same_dow_vs_8w",
#     "same_dow_momentum_2w", "same_dow_3w_slope",
#     "same_dow_residual_7d", "same_dow_streak",
#     "same_dow_regime_ratio", "same_dow_percentile",
#     "week_vol_cumsum", "week_progress_vs_lastweek", "last_week_avg",
#     # ── momentum / volatility ─────────────────────────────────────────
#     "streak_above_rm7", "consecutive_decline_days", "decline_magnitude",
#     "current_vs_recent_peak_14d", "vol_roll_median_14", "Volume_roll_max_28",
#     "volume_accel_3d", "volume_accel_pct_3d", "rolling_same_dow_std",
#     # ── weather ───────────────────────────────────────────────────────
#     "storm_impact_sq", "storm_severe_flag",
#     # ── weekend ───────────────────────────────────────────────────────
#     "friday_saturday_strength", "fri_vs_expected", "fri_sat_sum_vs_expected",
#     "weekend_return_pressure",
#     # ── interactions ──────────────────────────────────────────────────
#     "lag1_dow_ratio_x_dow",
#     "same_dow_residual_trend", "weekly_ramp_slope", "weekly_ramp_slope_pct",
# ]

def build_features_from_df(df, verbose=True, prune=True):
    def log(msg):
        if verbose: print(msg)

    log("Adding calendar features...")
    df = add_calendar_features(df, "Date")
    log("Adding Google Trends features...")
    df = add_google_trends_features(df)
    log("Adding holiday features...")
    df, major_dates, presidents_dates = add_holiday_features(df)
    log("Adding DOW × month interactions...")
    df = add_dow_month_interactions(df)
    log("Adding volume lag features...")
    df = add_volume_lag_features(df, TARGET_COL)
    log("Adding momentum features...")
    df = add_momentum_features(df, TARGET_COL)
    log("Adding DOW-specific lags...")
    df = add_dow_specific_lags(df, TARGET_COL)
    log("Adding within-week features...")
    df = add_within_week_features(df, TARGET_COL)
    log("Adding volume regime features...")
    df = add_volume_regime_features(df, TARGET_COL)
    log("Adding lag365 same-DOW alignment...")
    df = add_lag365_same_dow(df, TARGET_COL)
    log("Adding log-scale features...")
    df = add_log_features(df, TARGET_COL)
    log("Adding top-feature interactions...")
    df = add_top_feature_interactions(df, TARGET_COL)
    log("Adding expanding DOW×month averages...")
    df = add_expanding_dow_month_avg(df, TARGET_COL)
    log("Adding holiday expected volume + shape features...")
    df = add_holiday_expected_volume(df, TARGET_COL, major_dates, presidents_dates)
    log("Adding holiday-DOW pooled features...")
    df = add_holiday_pooled_dow_features(df, TARGET_COL)
    log("Adding last week average...")
    df = add_last_week_total(df, TARGET_COL)
    log("Adding volume context features...")
    df = add_volume_context_features(df, TARGET_COL)
    log("Adding regime, decay, and targeted features...")
    df = add_regime_and_decay_features(df, TARGET_COL)
    log("Adding new same-DOW and contextual features...")
    df = add_new_features(df, TARGET_COL)
    log("Adding group features (same-DOW trend, week pace, YoY regime, anchor reliability, week shape, holiday)...")
    df = add_group_features(df, TARGET_COL)
    log("Adding anchor family features (same-DOW variants, better lag365, week projection)...")
    df = add_anchor_family_features(df, TARGET_COL)

    # ── Cleanup ───────────────────────────────────────────────────────
    internal_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=internal_cols)
    feature_cols = [c for c in df.columns if c not in ["Date", TARGET_COL]]
    for col in feature_cols:
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].ffill()

    # ── Prune to kept features ────────────────────────────────────────
    if prune:
        available = [f for f in KEEP_FEATURES if f in df.columns]
        dropped = [f for f in feature_cols if f not in KEEP_FEATURES and f in df.columns]
        df = df[["Date", TARGET_COL] + available]
        if verbose:
            print(f"\n  Feature pruning: kept {len(available)}, dropped {len(dropped)}")
            not_found = [f for f in KEEP_FEATURES if f not in available]
            if not_found:
                print(f"  WARNING: {len(not_found)} features in KEEP_FEATURES not found in data:")
                for f in not_found: print(f"    ✗ {f}")
    elif verbose:
        print(f"\n  Feature pruning skipped: {len(feature_cols)} features retained")

    return df


# ==========================================================================
# Master table builder
# ==========================================================================
def build_master_table():
    print("Loading raw data...")
    df = load_and_merge_data()
    df = build_features_from_df(df, verbose=True)
    df = df[df[TARGET_COL].notna()].copy()
    essential_cols = [f"{TARGET_COL}_lag{i}" for i in [1, 3, 7] if f"{TARGET_COL}_lag{i}" in df.columns]
    rows_before = len(df)
    df = df.dropna(subset=essential_cols).reset_index(drop=True)
    rows_after = len(df)
    feat_count = len(get_feature_columns(df))
    print(f"\nMaster table: {rows_after:,} rows, {feat_count} features  "
          f"(dropped {rows_before - rows_after} rows missing essential lags)")
    print(f"Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    return df


if __name__ == "__main__":
    df = build_master_table()
    df.to_csv(MASTER_PATH, index=False)
    print(f"\nSaved -> {MASTER_PATH}")
    feat_cols = get_feature_columns(df)
    print(f"Features: {len(feat_cols)}")
    print(f"Target: {TARGET_COL}")
    print(f"\nAll features:")
    for i, c in enumerate(feat_cols, 1):
        print(f"  {i:>3}. {c}")