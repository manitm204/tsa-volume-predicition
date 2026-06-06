"""
production_router.py
====================
The production router specification, as a pure importable module. No I/O,
no model training — just the deterministic logic that maps
    (date, tabular_pred, ts3_pred, prophet_pred, regime_features)
to a routed prediction + the regime label.

Used by:
    advanced_router_predict.py    daily shadow predictions
    dashboard/backend/main.py     re-classifying historical rows for the dashboard
    any future ablation script    keep one source of truth for the rules

Spec (locked in after regime_router_experiment + historical_alpha_calibration +
shoulder_width_sweep + extended_normal_test + router_weights_calibration):

    1. STORM           storm_severe_flag == 1 AND storm_impact_sq >= 2
                       pred = α(impact_sq) × tabular + (1−α) × weather_anchor
                         α = clip(1.0 − 0.07 × impact_sq, 0.3, 1.0)
                       (Smooth taper. Beats prior binary 0.9/0.5 schedule
                        by 22k MAE on cached OOF storm days. The schedule
                        was hand-picked rather than fit because with only
                        13 OOF storm days, any LOO parameter fit overfits
                        wildly — the in-sample optimal would have given
                        132k LOO MAE vs 79k for this fixed taper.)
    2. PEAK_HOLIDAY    |days_to_major_holiday_signed| <= 2
                       pred = tabular
    3. SHOULDER_PRE    3 <= days_to_major_holiday_signed <= 7  (close-in upcoming)
                       pred = 0.40 × tabular + 0.60 × anchor_master
                       (anchor_master beats tabular standalone on this slice
                        — 28k vs 34k OOF MAE — because close-in pre-holiday
                        demand is driven by YoY-stable advance bookings.
                        Window=7 was chosen by sweep over {7,10,14,21}; the
                        signal dilutes rapidly past day 7 because demand 8+
                        days out matches a normal calendar profile.)
    4. SHOULDER_POST  −7 <= days_to_major_holiday_signed <= −3 (close-in return)
                       pred = 0.70 × tabular + 0.10 × ts3 + 0.20 × prophet
                       (anchor doesn't help here — post-holiday return travel
                        is YoY-noisy; tabular's lag-7 still wins)
    5. NORMAL          default
                       pred = 0.72 × tabular + 0.15 × ts3 + 0.10 × prophet
                              + 0.03 × anchor_master
                       (Interpolation sweep current↔NNLS picked t=0.30 — the
                        minimum of a clean U on 110 OOF test rows. Improves
                        MAE 57,126 → 56,978 vs the older (0.85, .075, .075).
                        Full NNLS (0.42, 0.32, 0.17, 0.10) was +2.1k worse
                        on test — classic overfit.)
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────
# Locked-in spec — change here, every consumer picks it up.
# ──────────────────────────────────────────────────────────────────────────
STORM_TRIGGER_IMPACT = 2.0
# Smooth α taper. α = clip(STORM_ALPHA_INTERCEPT − STORM_ALPHA_SLOPE × impact_sq,
#                          STORM_ALPHA_FLOOR, STORM_ALPHA_CEIL)
# Hand-picked, not fit (13 OOF days is too few to fit reliably — LOO overfit).
STORM_ALPHA_INTERCEPT = 1.00
STORM_ALPHA_SLOPE     = 0.07
STORM_ALPHA_FLOOR     = 0.30
STORM_ALPHA_CEIL      = 1.00
PEAK_HOLIDAY_WINDOW = 2
SHOULDER_WINDOW = 7
# 4-tuples: (tab, ts3, prophet, anchor_master)
SHOULDER_PRE_WEIGHTS  = (0.40, 0.00, 0.00, 0.60)
SHOULDER_POST_WEIGHTS = (0.70, 0.10, 0.20, 0.00)
NORMAL_WEIGHTS        = (0.72, 0.15, 0.10, 0.03)

# Major US travel-surge holidays used for PEAK_HOLIDAY / SHOULDER triggers.
# Hand-curated: federal holidays where lag365_residual_anchor captures the
# surge magnitude correctly. Black Friday added separately because it's a
# travel-surge day even though not a federal holiday.
MAJOR_HOLIDAY_MONTHDAYS = [
    (1, 1),    # New Year's Day
    (7, 4),    # Independence Day
    (12, 24),  # Christmas Eve
    (12, 25),  # Christmas Day
    (12, 31),  # New Year's Eve
]


def _nth_weekday_of_month(year, month, weekday, n):
    """e.g. nth_weekday_of_month(2026, 5, 0, -1) = last Monday in May 2026."""
    if n > 0:
        first = pd.Timestamp(year, month, 1)
        offset = (weekday - first.weekday()) % 7
        return first + pd.Timedelta(days=offset + (n - 1) * 7)
    else:
        last = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
        offset = (last.weekday() - weekday) % 7
        return last - pd.Timedelta(days=offset + (abs(n) - 1) * 7)


def get_major_holiday_dates(years):
    """All major holiday dates falling in the given years."""
    dates = []
    for y in years:
        for m, d in MAJOR_HOLIDAY_MONTHDAYS:
            dates.append(pd.Timestamp(y, m, d))
        # Memorial Day = last Monday of May
        dates.append(_nth_weekday_of_month(y, 5, 0, -1))
        # Labor Day = first Monday of September
        dates.append(_nth_weekday_of_month(y, 9, 0, 1))
        # Thanksgiving = 4th Thursday of November
        thanksgiving = _nth_weekday_of_month(y, 11, 3, 4)
        dates.append(thanksgiving)
        # Black Friday = day after Thanksgiving
        dates.append(thanksgiving + pd.Timedelta(days=1))
    return pd.to_datetime(sorted(set(dates)))


def days_to_nearest_major_signed(date, holidays=None):
    """Days from `date` to nearest major holiday (positive = upcoming)."""
    if holidays is None:
        d = pd.Timestamp(date)
        holidays = get_major_holiday_dates([d.year - 1, d.year, d.year + 1])
    diffs = (holidays - pd.Timestamp(date)).days
    return int(diffs[np.argmin(np.abs(diffs))])


# ──────────────────────────────────────────────────────────────────────────
# The router itself
# ──────────────────────────────────────────────────────────────────────────
RegimeName = Literal[
    "STORM", "PEAK_HOLIDAY", "SHOULDER_PRE", "SHOULDER_POST", "NORMAL"
]


@dataclass
class RouterOutput:
    regime: RegimeName
    pred_router: float
    # Components saved for transparency / dashboard inspection.
    alpha_storm: float | None = None       # only set in STORM
    w_tab: float | None = None
    w_ts3: float | None = None
    w_prophet: float | None = None
    w_anchor: float | None = None          # anchor_master (lag365 momentum)


def storm_alpha_for(impact_sq: float) -> float:
    """Smooth taper: more anchor weight as the storm gets more severe."""
    raw = STORM_ALPHA_INTERCEPT - STORM_ALPHA_SLOPE * float(impact_sq)
    return float(np.clip(raw, STORM_ALPHA_FLOOR, STORM_ALPHA_CEIL))


def classify_regime(*, storm_severe_flag: int, storm_impact_sq: float,
                    days_to_major_signed: int) -> RegimeName:
    if int(storm_severe_flag) == 1 and storm_impact_sq >= STORM_TRIGGER_IMPACT:
        return "STORM"
    if abs(days_to_major_signed) <= PEAK_HOLIDAY_WINDOW:
        return "PEAK_HOLIDAY"
    if PEAK_HOLIDAY_WINDOW < abs(days_to_major_signed) <= SHOULDER_WINDOW:
        return "SHOULDER_PRE" if days_to_major_signed > 0 else "SHOULDER_POST"
    return "NORMAL"


def route(*, tabular_pred: float, ts3_pred: float, prophet_pred: float,
          anchor_master: float,
          weather_penalized_anchor: float,
          storm_severe_flag: int, storm_impact_sq: float,
          days_to_major_signed: int) -> RouterOutput:
    regime = classify_regime(
        storm_severe_flag=storm_severe_flag,
        storm_impact_sq=storm_impact_sq,
        days_to_major_signed=days_to_major_signed,
    )

    if regime == "STORM":
        a = storm_alpha_for(storm_impact_sq)
        pred = a * tabular_pred + (1 - a) * weather_penalized_anchor
        return RouterOutput(regime=regime, pred_router=float(pred), alpha_storm=a)

    if regime == "PEAK_HOLIDAY":
        return RouterOutput(regime=regime, pred_router=float(tabular_pred),
                            w_tab=1.0, w_ts3=0.0, w_prophet=0.0, w_anchor=0.0)

    if regime == "SHOULDER_PRE":
        wt, ws, wp, wa = SHOULDER_PRE_WEIGHTS
    elif regime == "SHOULDER_POST":
        wt, ws, wp, wa = SHOULDER_POST_WEIGHTS
    else:  # NORMAL
        wt, ws, wp, wa = NORMAL_WEIGHTS
    pred = (wt * tabular_pred + ws * ts3_pred +
            wp * prophet_pred + wa * anchor_master)
    return RouterOutput(regime=regime, pred_router=float(pred),
                        w_tab=wt, w_ts3=ws, w_prophet=wp, w_anchor=wa)


# ──────────────────────────────────────────────────────────────────────────
# Convenience: re-route a whole DataFrame (used by dashboard backend).
# ──────────────────────────────────────────────────────────────────────────
def route_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Expects columns: pred_tabular, pred_ts3, pred_prophet, anchor_master,
    storm_severe_flag, storm_impact_sq, weather_penalized_anchor,
    days_to_major_signed. Returns a copy with added regime + pred_router."""
    out = df.copy()
    regimes, preds, alphas, wts, wss, wps, was = [], [], [], [], [], [], []
    for row in out.itertuples():
        r = route(
            tabular_pred=row.pred_tabular,
            ts3_pred=row.pred_ts3,
            prophet_pred=row.pred_prophet,
            anchor_master=row.anchor_master,
            weather_penalized_anchor=row.weather_penalized_anchor,
            storm_severe_flag=row.storm_severe_flag,
            storm_impact_sq=row.storm_impact_sq,
            days_to_major_signed=row.days_to_major_signed,
        )
        regimes.append(r.regime)
        preds.append(r.pred_router)
        alphas.append(r.alpha_storm if r.alpha_storm is not None else np.nan)
        wts.append(r.w_tab if r.w_tab is not None else np.nan)
        wss.append(r.w_ts3 if r.w_ts3 is not None else np.nan)
        wps.append(r.w_prophet if r.w_prophet is not None else np.nan)
        was.append(r.w_anchor if r.w_anchor is not None else np.nan)
    out["regime"] = regimes
    out["pred_router"] = preds
    out["alpha_storm"] = alphas
    out["w_tab"] = wts
    out["w_ts3"] = wss
    out["w_prophet"] = wps
    out["w_anchor"] = was
    return out
