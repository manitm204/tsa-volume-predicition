// Trading-math helpers. All client-side derivations so the dashboard does not
// depend on backend changes for signal-grading, Kelly sizing, opportunity ranking.

import type { ProbRow } from "../types";

export const KELLY_FRACTION = 0.5;   // half-Kelly safety scale
export const KELLY_MIN_PCT  = 0.05;  // floor used by kalshi.py
export const FEE_RATE       = 0.07;  // Kalshi taker fee approx

export interface OpportunitySide {
  side: "OVER" | "UNDER";
  ticker: "yes" | "no";
  strike: number;
  modelProb: number;
  marketProb: number;
  edge: number;            // model − market
  expectedValue: number;   // EV per $1 staked
  kellyFraction: number;   // 0..1 of bankroll suggested by full Kelly
  kellyStake: number;      // $ size at half-Kelly scaled to bankroll
  contracts: number;       // # of contracts at price
  price: number | null;
  bid: number | null;
  ask: number | null;
  liquidity: "thin" | "ok" | "deep";
  signal: "strong" | "good" | "watch" | "skip";
}

/** Full Kelly criterion for a yes/no contract priced at `price`, with true
 * probability `p`. Returns the fraction of bankroll to stake (0..1). */
export function kellyFraction(p: number, price: number): number {
  if (price <= 0 || price >= 1) return 0;
  const b = (1 - price) / price;  // payoff per $1 staked if win
  const q = 1 - p;
  const f = (b * p - q) / b;
  return Math.max(0, f);
}

export function expectedValue(p: number, price: number): number {
  if (price <= 0 || price >= 1) return 0;
  return (p / price - 1);  // EV per $1 staked, ignoring fees
}

export function liquidityFromVolume(volume: number | null | undefined): "thin" | "ok" | "deep" {
  if (volume == null) return "thin";
  if (volume >= 200) return "deep";
  if (volume >= 50)  return "ok";
  return "thin";
}

export function buildOpportunities(rows: ProbRow[] | undefined, bankroll: number): OpportunitySide[] {
  if (!rows?.length) return [];
  const out: OpportunitySide[] = [];

  for (const r of rows) {
    // OVER side
    if (r.model_p_over != null && r.kalshi_prob != null && r.yes_ask != null) {
      const edge = r.over_edge ?? r.model_p_over - r.kalshi_prob;
      const price = r.yes_ask;
      const f = kellyFraction(r.model_p_over, price);
      const stake = Math.round(f * KELLY_FRACTION * bankroll * 100) / 100;
      const contracts = price > 0 ? Math.floor(stake / price) : 0;
      out.push({
        side: "OVER",
        ticker: "yes",
        strike: r.threshold_millions,
        modelProb: r.model_p_over,
        marketProb: r.kalshi_prob,
        edge,
        expectedValue: expectedValue(r.model_p_over, price),
        kellyFraction: f,
        kellyStake: stake,
        contracts,
        price,
        bid: r.yes_bid,
        ask: r.yes_ask,
        liquidity: "ok",
        signal: signalFromEdge(edge, f),
      });
    }
    // UNDER side
    if (r.model_p_under != null && r.kalshi_prob != null && r.yes_mid != null) {
      const noAsk = +(1 - (r.yes_bid ?? r.yes_mid)).toFixed(4); // approx no_ask
      const edge  = r.under_edge ?? r.model_p_under - (1 - r.kalshi_prob);
      const f     = kellyFraction(r.model_p_under, noAsk);
      const stake = Math.round(f * KELLY_FRACTION * bankroll * 100) / 100;
      const contracts = noAsk > 0 ? Math.floor(stake / noAsk) : 0;
      out.push({
        side: "UNDER",
        ticker: "no",
        strike: r.threshold_millions,
        modelProb: r.model_p_under,
        marketProb: 1 - r.kalshi_prob,
        edge,
        expectedValue: expectedValue(r.model_p_under, noAsk),
        kellyFraction: f,
        kellyStake: stake,
        contracts,
        price: noAsk,
        bid: r.no_mid != null ? +(r.no_mid - 0.01).toFixed(2) : null,
        ask: noAsk,
        liquidity: "ok",
        signal: signalFromEdge(edge, f),
      });
    }
  }
  return out.sort((a, b) => b.edge - a.edge);
}

export function signalFromEdge(edge: number, kellyF: number): "strong" | "good" | "watch" | "skip" {
  if (edge >= 0.08 && kellyF >= 0.10) return "strong";
  if (edge >= 0.04 && kellyF >= 0.05) return "good";
  if (edge >= 0.02)                   return "watch";
  return "skip";
}

/** Grade the overall weekly signal based on the strongest opportunity. */
export type Grade = "A+" | "A" | "B" | "C" | "D" | "F";

export function gradeSignal(bestEdge: number, bestKelly: number, confidence: number): Grade {
  // confidence is 0..1 (e.g. 1 - normalised std)
  const score = bestEdge * 100 * 0.6 + bestKelly * 100 * 0.25 + confidence * 100 * 0.15;
  if (score >= 22) return "A+";
  if (score >= 16) return "A";
  if (score >= 10) return "B";
  if (score >= 6)  return "C";
  if (score >= 3)  return "D";
  return "F";
}

export function gradeClass(g: Grade): string {
  switch (g) {
    case "A+": return "grade grade-aplus";
    case "A":  return "grade grade-a";
    case "B":  return "grade grade-b";
    case "C":  return "grade grade-c";
    case "D":  return "grade grade-d";
    case "F":  return "grade grade-f";
  }
}

/** Derive a 0..1 confidence from the model's weekly_avg_std (in millions). */
export function confidenceFromStd(stdM: number | null | undefined): number {
  if (stdM == null || stdM <= 0) return 0.5;
  // empirical: model std of 0.02M ≈ very tight (0.95); 0.08M ≈ wide (0.55)
  const x = Math.min(0.10, Math.max(0.015, stdM));
  return Math.max(0.3, Math.min(0.99, 1 - (x - 0.015) / 0.20));
}

/** Approximate normal PDF — used for forecast distribution chart. */
export function normalPdf(x: number, mu: number, sigma: number): number {
  if (sigma <= 0) return 0;
  const z = (x - mu) / sigma;
  return Math.exp(-0.5 * z * z) / (sigma * Math.sqrt(2 * Math.PI));
}

export function normalCdf(x: number, mu: number, sigma: number): number {
  if (sigma <= 0) return x >= mu ? 1 : 0;
  const z = (x - mu) / (sigma * Math.SQRT2);
  return 0.5 * (1 + erf(z));
}

function erf(x: number): number {
  // Abramowitz & Stegun approximation
  const a1 =  0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 =  1.061405429, p  = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  x = Math.abs(x);
  const t = 1.0 / (1.0 + p * x);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

/** Generate distribution points for a forecast charted area. */
export function distributionPoints(
  mu: number, sigma: number, n = 80,
): { x: number; pdf: number }[] {
  const lo = mu - 4 * sigma;
  const hi = mu + 4 * sigma;
  const step = (hi - lo) / (n - 1);
  return Array.from({ length: n }, (_, i) => {
    const x = lo + i * step;
    return { x, pdf: normalPdf(x, mu, sigma) };
  });
}
