import type {
  OverviewData, DailyPoint, CurrentWeekData, ProbRow,
  Position, OpenOrder, OrderHistoryRow, PnLData, BankrollData, OrderBookRow,
  WeeklyAvgTrackerData, ForecastWeekOption, YoYPoint,
  ChangesData, WeatherImpactData, DriftData,
  FeatureAttributionsData,
  ShadowPrediction, ShadowSummary,
  EnsembleWeightRow, EnsembleHistoryRow, EnsembleSummary, DynamicWeightsData,
  TomorrowData, DayDetailData,
} from "../types";

const BASE = "/api";

async function get<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(BASE + path, window.location.origin);
  if (params) Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function post<T>(path: string): Promise<T> {
  const res = await fetch(BASE + path, { method: "POST" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  health:          ()                             => get<{ status: string; last_run_at: string | null }>("/health"),
  overview:        ()                             => get<OverviewData>("/overview"),
  forecastDaily:   (start?: string, end?: string) => get<DailyPoint[]>("/forecast/daily", { ...(start && { start }), ...(end && { end }) }),
  forecastWeek:    ()                             => get<CurrentWeekData>("/forecast/current-week"),
  probabilities:   ()                             => get<ProbRow[]>("/probabilities"),
  positions:       ()                             => get<Position[]>("/positions"),
  openOrders:      ()                             => get<OpenOrder[]>("/orders/open"),
  orderHistory:    (limit = 200)                  => get<OrderHistoryRow[]>("/orders/history", { limit: String(limit) }),
  pnl:             ()                             => get<PnLData>("/pnl"),
  bankroll:        ()                             => get<BankrollData>("/bankroll"),
  orderbook:       ()                             => get<OrderBookRow[]>("/orderbook"),
  weeklyAvgTracker:(weekMonday?: string)          => get<WeeklyAvgTrackerData>("/forecast/weekly-avg-tracker", weekMonday ? { week_monday: weekMonday } : undefined),
  forecastWeeks:   ()                             => get<ForecastWeekOption[]>("/forecast/weeks"),
  yoyComparison:   ()                             => get<YoYPoint[]>("/yoy-comparison"),
  changes:         ()                             => get<ChangesData>("/changes"),
  weatherImpact:   ()                             => get<WeatherImpactData>("/weather-impact"),
  drift:           ()                             => get<DriftData>("/drift"),
  featureAttributions: ()                         => get<FeatureAttributionsData>("/feature-attributions"),
  shadowPredictions:  ()                           => get<ShadowPrediction[]>("/shadow/predictions"),
  shadowSummary:      ()                           => get<ShadowSummary>("/shadow/summary"),
  ensembleWeights:    ()                           => get<EnsembleWeightRow[]>("/ensemble/weights"),
  ensembleHistory:    ()                           => get<EnsembleHistoryRow[]>("/ensemble/history"),
  ensembleSummary:    ()                           => get<EnsembleSummary>("/ensemble/summary"),
  dynamicWeights:     ()                           => get<DynamicWeightsData>("/ensemble/dynamic-weights"),
  dayDetail:          (weeksBack = 0)              => get<DayDetailData>("/forecast/day-detail", { weeks_back: String(weeksBack) }),
  tomorrow:           ()                           => get<TomorrowData>("/tomorrow"),
  refreshPositions:   ()                           => post<{ ok: boolean; source: string; n_positions: number; n_open_orders: number; refreshed_at: string }>("/refresh-positions"),
};
