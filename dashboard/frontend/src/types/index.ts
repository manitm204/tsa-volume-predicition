export interface OverviewData {
  weekly_avg_forecast: number | null;
  weekly_avg_millions: number | null;
  weekly_avg_std_millions: number | null;
  n_actual: number | null;
  n_predicted: number | null;
  yesterday_actual: number | null;
  yesterday_predicted: number | null;
  yesterday_error: number | null;
  forecast_change_from_yesterday: number | null;
  bankroll: number;
  open_exposure: number;
  exposure_pct: number;
  orders_today: number;
  last_run_at: string | null;
  last_run_status: string;
  auto_trading: boolean;
  portfolio_source: string;
}

export interface DailyPoint {
  date: string;
  actual: number | null;
  predicted: number | null;
}

export interface WeekDay {
  date: string;
  day_name: string;
  status: "actual" | "predicted";
  volume: number | null;
  predicted: number | null;
  error: number | null;
  regime?: string | null;
  pred_tabular?: number | null;
  weights?: { tab: number; ts3: number; yoy_delta: number; anchor: number } | null;
  sigma_eff?: number | null;
}

export interface CurrentWeekData {
  days: WeekDay[];
  summary: Record<string, number | string | null>;
  mae: number | null;
  kalshi_avg: number | null;
}

export interface WeeklyAvgTrackerPoint {
  date: string;
  model_avg_millions: number | null;
  kalshi_avg_millions: number | null;
}

export interface WeeklyAvgTrackerData {
  week_monday: string | null;
  points: WeeklyAvgTrackerPoint[];
  settled_avg_millions: number | null;
}

export interface ForecastWeekOption {
  week_monday: string;
  settled_avg_millions: number | null;
  is_current: boolean;
}

export interface ProbRow {
  threshold_millions: number;
  model_p_over: number | null;
  model_p_under: number | null;
  kalshi_prob: number | null;
  over_edge: number | null;
  under_edge: number | null;
  yes_bid: number | null;
  yes_ask: number | null;
  yes_mid: number | null;
  no_mid: number | null;
  over_action: "buy_yes" | "skip";
  under_action: "buy_no" | "skip";
}

export interface Position {
  ticker: string;
  strike_millions: number | null;
  yes_shares: number;
  no_shares: number;
  avg_price: number | null;
  cost_dollars: number;
  yes_current_price: number | null;
  no_current_price: number | null;
  unrealized_pnl: number | null;
}

export interface OpenOrder {
  ticker: string;
  strike_millions: number | null;
  side: "yes" | "no";
  price: number;
  remaining: number;
  current_price: number | null;
  total_value: number;
}

export interface OrderHistoryRow {
  run_at: string;
  ticker: string;
  strike_millions: number | null;
  type: string;
  side: string;
  price: number | null;
  shares: number | null;
  pct: number | null;
  model_prob: number | null;
  market_prob: number | null;
  dry_run: boolean;
  cost: number | null;
}

export interface PnLData {
  total_invested: number;
  current_value: number;
  unrealized_pnl: number;
  realized_pnl: number | null;
  total_pnl: number;
  bankroll: number;
  by_position: {
    ticker: string;
    strike_millions: number | null;
    side: string;
    shares: number;
    avg_price: number | null;
    current_price: number | null;
    cost_dollars: number;
    unrealized_pnl: number | null;
  }[];
}

export interface BankrollData {
  bankroll: number;
  cash: number;
  invested: number;
  reserved_for_orders: number;
  cash_pct: number;
  invested_pct: number;
  reserved_pct: number;
}

export interface YoYPoint {
  date: string;
  y2024: number | null;
  y2025: number | null;
  y2026_actual: number | null;
  y2026_predicted: number | null;
}

export interface DayChange {
  date: string;
  day_name: string;
  kind: "new_actual" | "forecast_revision";
  prev_predicted: number;
  now_actual?: number;
  now_predicted?: number;
  surprise?: number;
  delta?: number;
}

export interface ProbabilityShift {
  strike_millions: number;
  prev_model_p_over: number;
  now_model_p_over: number;
  delta: number;
}

export interface ChangesData {
  has_data: boolean;
  reason?: string;
  today_run_date?: string;
  forecast_delta_passengers?: number | null;
  forecast_delta_pct?: number | null;
  confidence_delta_std_millions?: number | null;
  kalshi_delta_millions?: number | null;
  per_day_changes?: DayChange[];
  probability_shifts?: ProbabilityShift[];
  attribution?: {
    new_actual_contribution_k: number;
    forecast_revision_contribution_k: number;
    kalshi_delta_pct: number | null;
  };
}

export interface WeatherDay {
  date: string;
  wt_avg_snow_depth_max?: number | null;
  wt_avg_snowfall_sum?: number | null;
  n_hubs_snowing?: number | null;
  top3_hubs_snowfall_mean?: number | null;
  vol_wtd_storm_impact?: number | null;
}

export interface WeatherImpactData {
  has_data: boolean;
  reason?: string;
  days?: WeatherDay[];
  current_week_mean?: Record<string, number | null>;
  trailing_4w_mean?: Record<string, number | null>;
  composite_penalty?: number | null;
  trailing_penalty?: number | null;
  penalty_delta?: number | null;
  qualitative_label?: string;
  approx_volume_impact_k?: number | null;
}

export interface DriftStats {
  mae: number | null;
  bias: number | null;
  residual_std: number | null;
  n: number;
}

export interface FeatureAttribution {
  feature: string;
  label: string;
  category: "history" | "momentum" | "calendar" | "trend" | "weather" | "other";
  reason: string;
  contribution_passengers: number;
  contribution_millions: number;
  current_value: number;
  baseline_value: number;
  importance_rank: number;
  importance_passengers: number;
}

export interface FeatureAttributionsData {
  has_data: boolean;
  reason?: string;
  computed_at?: string | null;
  baseline_prediction_weekly_avg_millions?: number | null;
  method?: string;
  drivers: FeatureAttribution[];
}

export interface DriftData {
  has_data: boolean;
  recent_14d?: DriftStats;
  trailing_90d?: DriftStats;
  mae_ratio?: number | null;
  bias_flip?: boolean | null;
  drift_flag?: boolean;
  messages?: string[];
}

export interface OrderBookRow {
  ticker: string;
  strike_millions: number | null;
  yes_bid: number | null;
  yes_ask: number | null;
  no_bid: number | null;
  no_ask: number | null;
  yes_mid: number | null;
  market_prob: number | null;
  model_prob: number | null;
  spread: number | null;
  volume: number | null;
  open_interest: number | null;
}

// ── Shadow router (kept for backward compat) ─────────────────────────────────
export interface ShadowPrediction {
  target_date: string;
  made_on_date: string | null;
  actual_volume: number | null;
  pred_tabular: number | null;
  pred_ts3: number | null;
  pred_yoy_delta: number | null;
  pred_router: number | null;
  regime: "STORM" | "PEAK_HOLIDAY" | "SHOULDER_PRE" | "SHOULDER_POST" | "NORMAL" | null;
  alpha_storm: number | null;
  w_tab: number | null;
  w_ts3: number | null;
  w_yoy_delta: number | null;
  w_anchor: number | null;
  anchor_master: number | null;
  storm_severe_flag: number;
  storm_impact_sq: number | null;
  days_to_major_signed: number | null;
  err_tabular: number | null;
  err_router: number | null;
}

export interface ShadowRegimeRow {
  regime: string;
  n: number;
  tabular_mae: number;
  router_mae: number;
  delta: number;
}

export interface ShadowSummary {
  n_predictions: number;
  n_with_actuals: number;
  last_trained: { trained_at: string; n_rows: number; latest_date: string } | null;
  tabular_mae: number | null;
  router_mae: number | null;
  router_vs_tabular_delta: number | null;
  router_vs_tabular_pct: number | null;
  by_regime: ShadowRegimeRow[];
  first_target_date: string | null;
  last_target_date: string | null;
}

// ── Ensemble router ───────────────────────────────────────────────────────────
export interface EnsembleWeightRow {
  regime: string;
  tab: number | null;
  ts3: number | null;
  yoy_delta: number | null;
  anchor: number | null;
  sigma_raw: number;
  sigma_eff: number;
  note: string;
}

export interface EnsembleHistoryRow {
  target_date: string;
  made_on_date: string | null;
  predicted_volume: number | null;
  pred_tabular: number | null;
  regime: string | null;
  weights: { tab: number; ts3: number; yoy_delta: number; anchor: number } | null;
  sigma_eff: number | null;
}

export interface EnsembleSummary {
  n_predictions: number;
  regime_distribution: Record<string, number>;
  models_last_trained: { trained_at: string } | null;
  first_target_date: string | null;
  last_target_date: string | null;
}

// ── Dynamic NORMAL-regime weight refresh (refresh_normal_weights.py) ────────
export interface ModelWeights {
  tab: number;
  ts3: number;
  yoy_delta: number;
  anchor: number;
}

export interface DynamicWeightsData {
  has_data: boolean;
  regime?: string;
  model_order?: string[];
  full_history_weights?: ModelWeights;
  last30_weights?: ModelWeights;
  blend_weights?: ModelWeights;
  method?: string;
  generated_at?: string | null;
  n_full_history?: number | null;
  n_last30?: number | null;
  date_range?: [string, string] | null;
  quick_tabular_oof_mae?: number | null;
}

// ── Day explorer — cycle through each day of the current week ────────────────
export interface DayDetailRow {
  date: string;
  day_name: string;
  status: "actual" | "predicted";
  regime: string | null;
  ensemble: number | null;
  predicted: number | null;
  actual: number | null;
  error: number | null;
  pred_tabular: number | null;
  pred_ts3: number | null;
  pred_yoy_delta: number | null;
  pred_anchor: number | null;
  weights: ModelWeights | null;
  sigma_eff: number | null;
  sigma_raw: number | null;
  bell_curve: { x: number; y: number }[];
}

export interface DayDetailData {
  days: DayDetailRow[];
  week_start: string | null;
  has_older_week: boolean;
}

// ── Tomorrow forecast ─────────────────────────────────────────────────────────
export interface TomorrowData {
  has_data: boolean;
  reason?: string;
  target_date?: string;
  day_name?: string;
  regime?: string;
  pred_tabular?: number | null;
  pred_ts3?: number | null;
  pred_yoy_delta?: number | null;
  pred_anchor?: number | null;
  pred_ensemble?: number | null;
  weights?: { tab: number | null; ts3: number | null; yoy_delta: number | null; anchor: number | null } | null;
  sigma_raw?: number | null;
  sigma_eff?: number | null;
  bell_curve?: { x: number; y: number }[];
  thresholds?: TomorrowThreshold[];
  thresholds_source?: "kalshi" | "fallback";
  kalshi_error?: string | null;
  daily_positions?: Position[];
  daily_open_orders?: OpenOrder[];
  daily_orderbook?: OrderBookRow[];
}

export interface TomorrowThreshold {
  threshold_millions: number;
  ticker: string | null;
  p_over: number | null;
  p_under: number | null;
  market_p_over: number | null;
  market_p_under: number | null;
  edge_over: number | null;
  edge_under: number | null;
  yes_bid_cents: number | null;
  yes_ask_cents: number | null;
  volume: number | null;
  open_interest: number | null;
}
