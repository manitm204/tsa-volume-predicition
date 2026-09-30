"""
production_router.py
====================
The production router specification, as a pure importable module. No I/O,
no model training — just the deterministic logic that maps
    (date, tabular_pred, ts3_pred, yoy_delta_pred, regime_features)
to a routed prediction + the regime label.

Used by:
    advanced_router_predict.py    daily shadow predictions
    dashboard/backend/main.py     re-classifying historical rows for the dashboard
    any future ablation script    keep one source of truth for the rules

Spec (locked in after regime_router_experiment + historical_alpha_calibration +
shoulder_width_sweep + extended_normal_test + router_weights_calibration;
re-weighted after ensemble_experiment/normal_weight_search.py dropped Prophet
in favor of yoy_delta — see ensemble_experiment/yoy_delta_oof.py):

    1. STORM           storm_severe_flag == 1 AND storm_impact_sq >= 2
                       pred = α(impact_sq) × tabular + (1−α) × weather_anchor
                         α = clip(1.0 − 0.07 × impact_sq, 0.3, 1.0)
                       (Smooth taper. Beats prior binary 0.9/0.5 schedule
                        by 22k MAE on cached OOF storm days. The schedule
                        was hand-picked rather than fit because with only
                        13 OOF storm days, any LOO parameter fit overfits
                        wildly — the in-sample optimal would have given
                        132k LOO MAE vs 79k for this fixed taper.)
    2. PEAK_HOLIDAY    min(days_since_prior, days_to_next) <= 2
                       pred = tabular
    3. SHOULDER_POST  3 <= days_since_prior_holiday <= 10  (post-holiday return)
                       — TAKES PRIORITY when both SHOULDER_PRE and SHOULDER_POST
                         could apply (e.g. Sat after a Fri holiday when Jul 4
                         is 7 days away).
                       pred = 0.333 × tabular + 0.333 × ts3 + 0.333 × anchor_master
                       (yoy_delta dropped 2026-09-21 — see below. NNLS on
                        combined_oof.csv, n=94, LOO MAE 65,876 vs the old
                        4-model production weights.)
    4. SHOULDER_PRE    3 <= days_to_next_holiday <= 7  (close-in upcoming)
                       pred = 0.300 × tabular + 0.700 × anchor_master  (ts3=0)
                       (yoy_delta dropped 2026-09-21 — see below. Grid search
                        on combined_oof.csv, n=52, LOO MAE 56,545. Historically
                        yoy_delta dominated this regime (0.700 weight, LOO MAE
                        36,117) — dropping it costs some accuracy specifically
                        here, but it's not safe to keep a component whose
                        trend term is unreliable across regimes just because
                        one regime benefited from it.)
    5. STORM_ECHO      not STORM/PEAK/SHOULDER, but storm_echo_flag == 1
                       (this day's yoy_delta t-364 anchor was itself a STORM
                        day — see build_features.add_yoy_delta_feature's
                        storm_echo_flag column)
                       pred = same weights as NORMAL_YOY_UNAVAILABLE for now
                        (tagged for analysis, not yet re-tuned — see
                        oof_regime_analysis.py)
    6. NORMAL_YOY_UNAVAILABLE   default, yoy_delta_trend_available == 0
                       (build_features.add_yoy_delta_feature's contaminated_1
                        flag: D or one of its delta_1 lookback anchors (D-7,
                        D-371) is holiday-adjacent, so the trend-correction
                        term has degraded to a naive last-year lookup)
                       pred = 0.600 × tabular + 0.400 × anchor_master  (ts3=yoy=0)
                       (yoy_delta dropped 2026-09-21: at the time NORMAL wasn't
                        yet split on trend availability — see NORMAL_YOY_AVAILABLE
                        below for where it earns its weight back. ts3 dropped
                        2026-09-28: a pooled LOO-grid search over tab/ts3/anchor
                        across all 225 NORMAL rows (both buckets, yoy excluded)
                        converged to (0.60, 0.00, 0.40) in 223/225 folds (99%),
                        independently matching the live daily_predict.py/
                        autogluon_predict.py NORMAL weights, and tied on LOO MAE
                        with the old 0.333-equal-split (47,715 vs 47,690) — see
                        ensemble_experiment/normal_yoy_split_search.py.)
    7. NORMAL_YOY_AVAILABLE    default, yoy_delta_trend_available == 1
                       (D and both delta_1 lookback anchors are holiday-clean
                        — trend-correction term carries real signal)
                       pred = 0.250 × tabular + 0.000 × ts3 + 0.250 × yoy_delta
                              + 0.500 × anchor_master
                       (Split from NORMAL after yoy_delta beat the 3-model
                        ensemble by 2-4x on 2026-09-25..27, all trend-available
                        days with no holiday nearby. Grid search on
                        ensemble_experiment/normal_yoy_split_search.py, n=120,
                        LOO MAE 52,841 vs 57,711 for the old 0.333/0.333/0/0.333
                        weights (-8.4%). ts3 drops out — its residuals
                        correlate 0.825 with tab's (near-redundant) vs
                        0.46-0.57 for yoy_delta, which is where the real
                        diversification comes from. tab's weight drop is
                        largely a bias correction: tab/ts3 run +12k/+19k
                        systematically high on this OOF window while
                        yoy_delta/anchor are ~unbiased (+839/+645) — worth
                        re-checking periodically in case that bias is
                        window-specific rather than stable.)

Holiday list (get_major_holiday_dates) covers, per year: New Year's Day,
MLK Day, Presidents Day, Memorial Day, Juneteenth, Independence Day, Labor
Day, Columbus Day, Veterans Day, Thanksgiving + Black Friday, Christmas Eve/
Day, New Year's Eve. MLK/Presidents/Columbus/Veterans added after diagnosing
2025-10-08→10-16 (Columbus Day week) as an omitted-holiday YoY growth spike
that misled yoy_delta despite being classified NORMAL.
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
SHOULDER_PRE_WINDOW  = 7
SHOULDER_POST_WINDOW = 10
# Legacy alias: any consumer that still imports SHOULDER_WINDOW gets the
# widest of the two — used only for OOF coverage masks in the older
# ensemble_experiment scripts.
SHOULDER_WINDOW = max(SHOULDER_PRE_WINDOW, SHOULDER_POST_WINDOW)
# 4-tuples: (tab, ts3, yoy_delta, anchor_master)
# yoy_delta dropped from production (2026-09-21): its holiday-contamination
# guard zeroes the trend-correction term for ~70% of all history, degrading
# it to a naive last-year lookup — this caused a ~134k MAE week (vs ~30-50k
# for the other 3 models) when 2026 volume diverged from 2025. Weight pinned
# to 0 below; re-derived tab/ts3/anchor via LOO grid/NNLS search over
# combined_oof.csv — see ensemble_experiment/normal_weight_search.py.
SHOULDER_PRE_WEIGHTS  = (0.300, 0.000, 0.000, 0.700)
SHOULDER_POST_WEIGHTS = (0.333, 0.333, 0.000, 0.333)

# NORMAL is split on build_features.add_yoy_delta_feature's
# yoy_delta_trend_available flag: whether the trend-correction term (delta_1/
# delta_2) carries real signal or has degraded to a naive last-year lookup
# because D (or its lookback anchors) is holiday-adjacent.
# UNAVAILABLE: matches the live daily_predict.py/autogluon_predict.py NORMAL
# weights (tab/anchor only, ts3=yoy=0). ensemble_experiment/
# normal_yoy_split_search.py confirmed this bucket doesn't benefit from
# yoy_delta (every fitted 4-model alternative did WORSE than an equal 3-model
# split — LOO MAE 47,690 vs 48,691-51,715). A follow-up pooled-LOO-grid
# search over just tab/ts3/anchor (225 rows = all of NORMAL, both buckets,
# yoy excluded) converged to (0.60, 0.00, 0.40) in 223/225 folds (99%) —
# independently reproducing the live pipeline's number — with LOO MAE 47,715
# on the unavailable subset, a statistical tie with the 0.333-equal-split's
# 47,690. Adopted 2026-09-28 to match production and drop the redundant ts3
# weight.
# AVAILABLE re-admits yoy_delta — grid search (LOO MAE 52,841 vs 57,711 for
# the old 0.333/0.333/0/0.333 weights, n=120, step=0.05) — see
# ensemble_experiment/normal_yoy_split_search.py, run 2026-09-28.
# ts3 drops to 0 here (as in SHOULDER_PRE): its residuals correlate 0.825
# with tab's (near-redundant), while yoy_delta correlates only 0.46-0.57
# with the others — the real diversification comes from yoy_delta, not ts3.
# tab's weight (0.60 in the legacy daily_predict.py/autogluon_predict.py
# 2-model NORMAL split, 0.333 in the old 4-model NORMAL_WEIGHTS) drops
# further to 0.25 here mainly because tab (and ts3) carry a systematic
# +12k/+19k overprediction bias on this OOF window while yoy_delta/anchor
# are ~unbiased (+839/+645) — a bias-correction effect, not just variance
# diversification. Worth re-running periodically in case that bias is
# window-specific (e.g. tab undercorrecting for a recent trend shift) rather
# than a stable property of tab on trend-available NORMAL days.
NORMAL_YOY_UNAVAILABLE_WEIGHTS = (0.600, 0.000, 0.000, 0.400)
# Re-fit 2026-09-30 after rebuilding yoy_delta as the 5-week drop-variant with
# a holiday-safe base (build_features.add_yoy_delta_feature). On all 225 normal
# OOF days the repaired yoy_delta is now ~tied with anchor on MAE, wins Sunday
# outright, and correlates only ~0.76 with anchor / ~0.4 with tab — so the LOO
# weight search shifts the bulk of the YoY-signal weight onto it.
NORMAL_YOY_AVAILABLE_WEIGHTS   = (0.350, 0.000, 0.450, 0.200)
NORMAL_WEIGHTS = NORMAL_YOY_UNAVAILABLE_WEIGHTS  # legacy alias, old callers

# Major US travel-surge holidays used for PEAK_HOLIDAY / SHOULDER triggers.
# Hand-curated: federal holidays where lag365_residual_anchor captures the
# surge magnitude correctly. Black Friday added separately because it's a
# travel-surge day even though not a federal holiday.
MAJOR_HOLIDAY_MONTHDAYS = [
    (1, 1),    # New Year's Day
    (6, 19),   # Juneteenth
    (7, 4),    # Independence Day
    (11, 11),  # Veterans Day
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
        # MLK Day = 3rd Monday of January
        dates.append(_nth_weekday_of_month(y, 1, 0, 3))
        # Presidents Day = 3rd Monday of February
        dates.append(_nth_weekday_of_month(y, 2, 0, 3))
        # Columbus Day = 2nd Monday of October
        dates.append(_nth_weekday_of_month(y, 10, 0, 2))
        # Thanksgiving = 4th Thursday of November
        thanksgiving = _nth_weekday_of_month(y, 11, 3, 4)
        dates.append(thanksgiving)
        # Black Friday = day after Thanksgiving
        dates.append(thanksgiving + pd.Timedelta(days=1))
    return pd.to_datetime(sorted(set(dates)))


def days_to_nearest_major_signed(date, holidays=None):
    """Days from `date` to nearest major holiday (positive = upcoming).

    Legacy helper — kept for callers that only need one signed scalar
    (older ensemble_experiment scripts). New code should use
    days_to_prior_and_next_major() so the asymmetric SHOULDER_PRE /
    SHOULDER_POST windows can be enforced independently.
    """
    if holidays is None:
        d = pd.Timestamp(date)
        holidays = get_major_holiday_dates([d.year - 1, d.year, d.year + 1])
    diffs = (holidays - pd.Timestamp(date)).days
    return int(diffs[np.argmin(np.abs(diffs))])


def days_to_prior_and_next_major(date, holidays=None):
    """Returns (days_since_prior_holiday, days_to_next_holiday) — both >= 0.

    If `date` itself is a major holiday, days_since_prior is 0 and
    days_to_next is the distance to the next holiday after it. If no
    prior/next holiday is in `holidays`, the corresponding side returns
    a large sentinel (10**6) so callers can ignore it.
    """
    if holidays is None:
        d = pd.Timestamp(date)
        holidays = get_major_holiday_dates([d.year - 1, d.year, d.year + 1, d.year + 2])
    diffs = (holidays - pd.Timestamp(date)).days
    past   = diffs[diffs <= 0]
    future = diffs[diffs >  0]
    days_prior = int(-past.max()) if len(past)   > 0 else 10**6
    days_next  = int(future.min()) if len(future) > 0 else 10**6
    return days_prior, days_next


# ──────────────────────────────────────────────────────────────────────────
# The router itself
# ──────────────────────────────────────────────────────────────────────────
RegimeName = Literal[
    "STORM", "PEAK_HOLIDAY", "SHOULDER_PRE", "SHOULDER_POST",
    "STORM_ECHO", "NORMAL_YOY_AVAILABLE", "NORMAL_YOY_UNAVAILABLE",
    "NORMAL",  # legacy, unsplit — only returned when yoy_delta_trend_available
               # isn't passed at all (old callers: autogluon_predict.py,
               # daily_predict.py). See classify_regime.
]


@dataclass
class RouterOutput:
    regime: RegimeName
    pred_router: float
    # Components saved for transparency / dashboard inspection.
    alpha_storm: float | None = None       # only set in STORM
    w_tab: float | None = None
    w_ts3: float | None = None
    w_yoy_delta: float | None = None
    w_anchor: float | None = None          # anchor_master (lag365 momentum)


def storm_alpha_for(impact_sq: float) -> float:
    """Smooth taper: more anchor weight as the storm gets more severe."""
    raw = STORM_ALPHA_INTERCEPT - STORM_ALPHA_SLOPE * float(impact_sq)
    return float(np.clip(raw, STORM_ALPHA_FLOOR, STORM_ALPHA_CEIL))


def classify_regime(*, storm_severe_flag: int, storm_impact_sq: float,
                    days_to_prior_major: int | None = None,
                    days_to_next_major: int | None = None,
                    days_to_major_signed: int | None = None,
                    storm_echo_flag: int = 0,
                    yoy_delta_trend_available: int | None = None) -> RegimeName:
    """Asymmetric POST-priority classifier.

    Pass `days_to_prior_major` AND `days_to_next_major` (both >= 0) for the
    full asymmetric spec. `days_to_major_signed` is accepted for backward
    compat with older callers — it gets unpacked into prior/next under
    the (less accurate) assumption that the nearest holiday is the only
    one in play.

    `storm_echo_flag` — pass 1 if this same calendar day one year ago
    (t-364) was itself a STORM day. yoy_delta's t-7/t-14 trend terms don't
    reach back that far, but its t-364 anchor does, so a day whose only
    disqualifier is a storm-contaminated YoY anchor still isn't a clean
    NORMAL day — it's tagged STORM_ECHO instead. Checked last, after the
    real-time storm/holiday windows, so it never overrides those.

    `yoy_delta_trend_available` — pass 0/1 (see build_features.
    add_yoy_delta_feature's yoy_delta_trend_available column) to split the
    default bucket into NORMAL_YOY_AVAILABLE / NORMAL_YOY_UNAVAILABLE.
    Left as None (default), the classifier returns the legacy unsplit
    "NORMAL" — required for backward compat with callers that predate the
    split (autogluon_predict.py, daily_predict.py call this function
    directly, not through route(), and key their own ENSEMBLE_WEIGHTS dicts
    on "NORMAL"; passing them a name they don't recognize would KeyError).
    route()/route_dataframe() in this module always pass an explicit 0/1.
    """
    if int(storm_severe_flag) == 1 and storm_impact_sq >= STORM_TRIGGER_IMPACT:
        return "STORM"

    if days_to_prior_major is None or days_to_next_major is None:
        if days_to_major_signed is None:
            raise ValueError("classify_regime needs prior+next or signed days")
        # Legacy path: signed days only tells us the nearest side. Set the
        # other side to a large sentinel so windows behave like the old
        # symmetric logic.
        if days_to_major_signed >= 0:
            days_to_next_major  = int(days_to_major_signed)
            days_to_prior_major = 10**6
        else:
            days_to_prior_major = int(-days_to_major_signed)
            days_to_next_major  = 10**6

    if min(days_to_prior_major, days_to_next_major) <= PEAK_HOLIDAY_WINDOW:
        return "PEAK_HOLIDAY"
    # POST priority: if both windows could apply, POST wins.
    if PEAK_HOLIDAY_WINDOW < days_to_prior_major <= SHOULDER_POST_WINDOW:
        return "SHOULDER_POST"
    if PEAK_HOLIDAY_WINDOW < days_to_next_major <= SHOULDER_PRE_WINDOW:
        return "SHOULDER_PRE"
    if int(storm_echo_flag) == 1:
        return "STORM_ECHO"
    if yoy_delta_trend_available is None:
        return "NORMAL"
    return "NORMAL_YOY_AVAILABLE" if int(yoy_delta_trend_available) else "NORMAL_YOY_UNAVAILABLE"


def route(*, tabular_pred: float, ts3_pred: float, yoy_delta_pred: float,
          anchor_master: float,
          weather_penalized_anchor: float,
          storm_severe_flag: int, storm_impact_sq: float,
          days_to_prior_major: int | None = None,
          days_to_next_major: int | None = None,
          days_to_major_signed: int | None = None,
          storm_echo_flag: int = 0,
          yoy_delta_trend_available: int = 1) -> RouterOutput:
    regime = classify_regime(
        storm_severe_flag=storm_severe_flag,
        storm_impact_sq=storm_impact_sq,
        days_to_prior_major=days_to_prior_major,
        days_to_next_major=days_to_next_major,
        days_to_major_signed=days_to_major_signed,
        storm_echo_flag=storm_echo_flag,
        yoy_delta_trend_available=yoy_delta_trend_available,
    )

    if regime == "STORM":
        a = storm_alpha_for(storm_impact_sq)
        pred = a * tabular_pred + (1 - a) * weather_penalized_anchor
        return RouterOutput(regime=regime, pred_router=float(pred), alpha_storm=a)

    if regime == "PEAK_HOLIDAY":
        return RouterOutput(regime=regime, pred_router=float(tabular_pred),
                            w_tab=1.0, w_ts3=0.0, w_yoy_delta=0.0, w_anchor=0.0)

    if regime == "SHOULDER_PRE":
        wt, ws, wy, wa = SHOULDER_PRE_WEIGHTS
    elif regime == "SHOULDER_POST":
        wt, ws, wy, wa = SHOULDER_POST_WEIGHTS
    elif regime == "NORMAL_YOY_AVAILABLE":
        wt, ws, wy, wa = NORMAL_YOY_AVAILABLE_WEIGHTS
    else:  # NORMAL_YOY_UNAVAILABLE or STORM_ECHO — same weights, not yet re-tuned
        wt, ws, wy, wa = NORMAL_YOY_UNAVAILABLE_WEIGHTS
    pred = (wt * tabular_pred + ws * ts3_pred +
            wy * yoy_delta_pred + wa * anchor_master)
    return RouterOutput(regime=regime, pred_router=float(pred),
                        w_tab=wt, w_ts3=ws, w_yoy_delta=wy, w_anchor=wa)


# ──────────────────────────────────────────────────────────────────────────
# Convenience: re-route a whole DataFrame (used by dashboard backend).
# ──────────────────────────────────────────────────────────────────────────
def route_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Expects columns: pred_tabular, pred_ts3, pred_yoy_delta, anchor_master,
    storm_severe_flag, storm_impact_sq, weather_penalized_anchor, and
    EITHER (days_to_prior_major + days_to_next_major) for the new asymmetric
    classifier, OR days_to_major_signed for the legacy path.
    Returns a copy with added regime + pred_router."""
    has_split = ("days_to_prior_major" in df.columns
                 and "days_to_next_major" in df.columns)
    out = df.copy()
    regimes, preds, alphas, wts, wss, wys, was = [], [], [], [], [], [], []
    for row in out.itertuples():
        if has_split:
            kwargs = dict(
                days_to_prior_major=int(row.days_to_prior_major),
                days_to_next_major=int(row.days_to_next_major),
            )
        else:
            kwargs = dict(days_to_major_signed=int(row.days_to_major_signed))
        r = route(
            tabular_pred=row.pred_tabular,
            ts3_pred=row.pred_ts3,
            yoy_delta_pred=row.pred_yoy_delta,
            anchor_master=row.anchor_master,
            weather_penalized_anchor=row.weather_penalized_anchor,
            storm_severe_flag=row.storm_severe_flag,
            storm_impact_sq=row.storm_impact_sq,
            storm_echo_flag=int(getattr(row, "storm_echo_flag", 0) or 0),
            yoy_delta_trend_available=int(getattr(row, "yoy_delta_trend_available", 1) or 0),
            **kwargs,
        )
        regimes.append(r.regime)
        preds.append(r.pred_router)
        alphas.append(r.alpha_storm if r.alpha_storm is not None else np.nan)
        wts.append(r.w_tab if r.w_tab is not None else np.nan)
        wss.append(r.w_ts3 if r.w_ts3 is not None else np.nan)
        wys.append(r.w_yoy_delta if r.w_yoy_delta is not None else np.nan)
        was.append(r.w_anchor if r.w_anchor is not None else np.nan)
    out["regime"] = regimes
    out["pred_router"] = preds
    out["alpha_storm"] = alphas
    out["w_tab"] = wts
    out["w_ts3"] = wss
    out["w_yoy_delta"] = wys
    out["w_anchor"] = was
    return out
