import { useQuery } from "@tanstack/react-query";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine,
  BarChart, Bar, Legend, Cell,
} from "recharts";
import { api } from "../api/client";
import { PageHeader, type KpiItem } from "../components/PageHeader";
import { GlassSection } from "../components/ui/GlassCard";
import { MarketsPanel } from "../components/MarketsPanel";
import type { TomorrowThreshold } from "../types";

const REGIME_COLOR: Record<string, string> = {
  STORM: "#ef5466",
  PEAK_HOLIDAY: "#f59e0b",
  SHOULDER_PRE: "#60a5fa",
  SHOULDER_POST: "#a78bfa",
  NORMAL: "#22d3a4",
};

const MODEL_META = [
  { key: "pred_tabular", label: "Tabular",      color: "#60a5fa", desc: "AutoGluon tabular (autoregressive chain)" },
  { key: "pred_ts3",     label: "TS3",           color: "#a78bfa", desc: "AutoGluon TimeSeries (calendar + lag365)" },
  { key: "pred_prophet", label: "Prophet",       color: "#fb923c", desc: "Prophet with 5 regressors" },
  { key: "pred_anchor",  label: "Anchor",        color: "#facc15", desc: "anchor_master feature (lag-365 same-DOW)" },
];

function fmt(v: number | null | undefined, decimals = 0): string {
  if (v == null || isNaN(v)) return "—";
  return v.toLocaleString(undefined, { maximumFractionDigits: decimals });
}
function fmtM(v: number | null | undefined): string {
  if (v == null || isNaN(v)) return "—";
  return `${(v / 1e6).toFixed(4)}M`;
}
function fmtPct(v: number | null | undefined): string {
  if (v == null || isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}
function fmtPctSigned(v: number | null | undefined): string {
  if (v == null || isNaN(v)) return "—";
  return `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;
}

const BellTT = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass !p-2 !rounded-lg text-[11px]" style={{ background: "rgba(5,8,15,0.96)" }}>
      <span className="mono text-slate-300">{Number(payload[0].payload.x).toFixed(3)}M</span>
    </div>
  );
};

const EdgeBarTT = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass !p-2.5 !rounded-lg text-[11px] space-y-1" style={{ background: "rgba(5,8,15,0.96)" }}>
      <div className="mono text-slate-300 text-[12px] font-bold">{label}M</div>
      {payload.map((p: any) => (
        <div key={p.dataKey} className="mono" style={{ color: p.fill }}>
          {p.name}: {p.dataKey === "edge"
            ? `${p.value >= 0 ? "+" : ""}${(p.value * 100).toFixed(1)}%`
            : `${(p.value * 100).toFixed(1)}%`}
        </div>
      ))}
    </div>
  );
};

function closestThresholds(thresholds: TomorrowThreshold[], mu: number, n: number): TomorrowThreshold[] {
  if (!thresholds.length || mu == null) return [];
  return [...thresholds]
    .sort((a, b) => Math.abs(a.threshold_millions * 1e6 - mu) - Math.abs(b.threshold_millions * 1e6 - mu))
    .slice(0, n)
    .sort((a, b) => a.threshold_millions - b.threshold_millions);
}

export default function Tomorrow() {
  const { data, isLoading } = useQuery({
    queryKey: ["tomorrow"],
    queryFn: api.tomorrow,
    refetchInterval: 60_000,
  });

  if (isLoading) {
    return (
      <GlassSection title="Loading…">
        <p className="text-slate-500 text-[13px]">Fetching tomorrow's forecast…</p>
      </GlassSection>
    );
  }

  if (!data?.has_data) {
    return (
      <>
        <PageHeader title="Tomorrow" description="Next day's per-model forecast" kpis={[]} />
        <GlassSection title="No data">
          <p className="text-slate-500 text-[13px]">{data?.reason ?? "No predicted days found."}</p>
        </GlassSection>
      </>
    );
  }

  const regime     = data.regime ?? "NORMAL";
  const regColor   = REGIME_COLOR[regime] ?? "#94a3b8";
  const ensemble   = data.pred_ensemble;
  const sigmaEff   = data.sigma_eff;
  const weights    = data.weights as Record<string, number | null> | null;
  const bellCurve  = (data.bell_curve ?? []) as { x: number; y: number }[];
  const thresholds = (data.thresholds ?? []) as TomorrowThreshold[];
  const isKalshi   = data.thresholds_source === "kalshi";

  // Reference lines at ±1σ and ±2σ
  const s1lo = ensemble && sigmaEff ? (ensemble - sigmaEff) / 1e6 : null;
  const s1hi = ensemble && sigmaEff ? (ensemble + sigmaEff) / 1e6 : null;
  const s2lo = ensemble && sigmaEff ? (ensemble - 2 * sigmaEff) / 1e6 : null;
  const s2hi = ensemble && sigmaEff ? (ensemble + 2 * sigmaEff) / 1e6 : null;

  const nearestForBell = ensemble != null ? closestThresholds(thresholds, ensemble, 3) : [];

  const overChartData = thresholds.map((t) => ({
    strike: t.threshold_millions.toFixed(2),
    model:  t.p_over,
    market: t.market_p_over,
    edge:   t.edge_over,
  }));
  const underChartData = thresholds.map((t) => ({
    strike: t.threshold_millions.toFixed(2),
    model:  t.p_under,
    market: t.market_p_under,
    edge:   t.edge_under,
  }));

  const kpis: KpiItem[] = [
    {
      label: "Ensemble Prediction",
      value: fmtM(ensemble),
      sub: sigmaEff ? `σ = ${fmt(sigmaEff)} (Platt-adj)` : undefined,
      accent: "#22d3a4",
    },
    {
      label: "Regime",
      value: regime.replace("_", " "),
      accent: regColor,
    },
    {
      label: "σ_eff",
      value: sigmaEff ? `${(sigmaEff / 1000).toFixed(1)}k` : "—",
      sub: data.sigma_raw ? `raw: ${(data.sigma_raw / 1000).toFixed(0)}k` : undefined,
    },
    {
      label: "Target Date",
      value: data.target_date ?? "—",
      sub: data.day_name,
    },
  ];

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Tomorrow"
        description={`Per-model breakdown for ${data.target_date} · ${data.day_name}`}
        kpis={kpis}
      />

      {/* ── Per-model prediction cards ─────────────────────────── */}
      <GlassSection title="Per-model predictions" sub="Each model's raw prediction before ensemble blending">
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-4 mb-4">
          {MODEL_META.map(({ key, label, color, desc }) => {
            const val = data[key as keyof typeof data] as number | null;
            const wKey = key === "pred_tabular" ? "tab"
                       : key === "pred_ts3"     ? "ts3"
                       : key === "pred_prophet" ? "prophet"
                       : "anchor";
            const w = weights?.[wKey] ?? null;
            const diff = val != null && ensemble != null ? val - ensemble : null;
            return (
              <div
                key={key}
                className="glass rounded-xl p-4 flex flex-col gap-2"
                style={{ borderLeft: `3px solid ${color}` }}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[11px] font-bold uppercase tracking-widest" style={{ color }}>
                    {label}
                  </span>
                  {w != null && (
                    <span className="text-[10px] mono px-1.5 py-0.5 rounded"
                          style={{ background: `${color}22`, color }}>
                      {(w * 100).toFixed(1)}%
                    </span>
                  )}
                </div>
                <div className="mono text-xl font-bold text-slate-100">
                  {val != null ? fmtM(val) : <span className="text-slate-600 text-base">not available</span>}
                </div>
                {diff != null && (
                  <div className="mono text-[11px]" style={{ color: diff >= 0 ? "#22d3a4" : "#ef5466" }}>
                    {diff >= 0 ? "+" : ""}{fmt(diff)} vs ensemble
                  </div>
                )}
                <div className="text-[10px] text-slate-600 leading-tight">{desc}</div>
              </div>
            );
          })}
        </div>

        {/* Ensemble result row */}
        <div className="glass rounded-xl p-4 flex items-center justify-between"
             style={{ borderLeft: "3px solid #22d3a4" }}>
          <div>
            <div className="text-[11px] font-bold uppercase tracking-widest text-edge-up mb-1">
              Ensemble (blended)
            </div>
            <div className="text-[10px] text-slate-500">
              {weights
                ? Object.entries({ tab: weights.tab, ts3: weights.ts3, prophet: weights.prophet, anchor: weights.anchor })
                    .filter(([, v]) => v != null && (v as number) > 0)
                    .map(([k, v]) => `${k} ${((v as number) * 100).toFixed(1)}%`)
                    .join(" · ")
                : "α-blend (storm)"}
            </div>
          </div>
          <div className="mono text-2xl font-bold text-edge-up">
            {fmtM(ensemble)}
          </div>
        </div>
      </GlassSection>

      {/* ── Bell distribution ─────────────────────────────────── */}
      {bellCurve.length > 0 && (
        <GlassSection
          title="Forecast distribution"
          sub={`N(μ = ${fmtM(ensemble)}, σ = ${sigmaEff ? (sigmaEff / 1000).toFixed(1) + 'k' : '—'}) · 3 nearest ${isKalshi ? "Kalshi" : "fallback"} thresholds marked`}
        >
          <div className="h-[220px] sm:h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={bellCurve} margin={{ top: 28, right: 20, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="bellGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor="#22d3a4" stopOpacity={0.35} />
                  <stop offset="95%" stopColor="#22d3a4" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(99,140,255,0.07)" strokeDasharray="3 6" />
              <XAxis
                dataKey="x"
                type="number"
                domain={["dataMin", "dataMax"]}
                tickFormatter={(v) => `${Number(v).toFixed(2)}M`}
                tick={{ fontSize: 10, fill: "#475569" }}
                tickLine={false}
                axisLine={false}
              />
              <YAxis hide />
              <Tooltip content={<BellTT />} />

              {/* ±1σ shading */}
              {s1lo != null && <ReferenceLine x={s1lo} stroke="#22d3a4" strokeDasharray="3 3" strokeWidth={1.2} strokeOpacity={0.5} />}
              {s1hi != null && <ReferenceLine x={s1hi} stroke="#22d3a4" strokeDasharray="3 3" strokeWidth={1.2} strokeOpacity={0.5} />}
              {/* ±2σ shading */}
              {s2lo != null && <ReferenceLine x={s2lo} stroke="#22d3a4" strokeDasharray="2 4" strokeWidth={1} strokeOpacity={0.3} />}
              {s2hi != null && <ReferenceLine x={s2hi} stroke="#22d3a4" strokeDasharray="2 4" strokeWidth={1} strokeOpacity={0.3} />}

              {/* Threshold markers — 3 nearest to μ */}
              {nearestForBell.map((t) => (
                <ReferenceLine
                  key={`thr-${t.threshold_millions}`}
                  x={t.threshold_millions}
                  stroke="#fb923c"
                  strokeDasharray="4 3"
                  strokeWidth={1.4}
                  strokeOpacity={0.85}
                  label={{
                    value: `${t.threshold_millions.toFixed(2)}M · ${fmtPct(t.p_over)}↑`,
                    position: "top",
                    fill: "#fb923c",
                    fontSize: 10,
                    offset: 6,
                  }}
                />
              ))}

              {/* Ensemble mean */}
              {ensemble != null && (
                <ReferenceLine
                  x={ensemble / 1e6}
                  stroke="#22d3a4"
                  strokeWidth={2}
                  label={{ value: "μ", position: "top", fill: "#22d3a4", fontSize: 11 }}
                />
              )}

              <Area
                type="monotone"
                dataKey="y"
                stroke="#22d3a4"
                strokeWidth={2}
                fill="url(#bellGrad)"
                dot={false}
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
          </div>

          {/* σ legend */}
          <div className="flex flex-wrap gap-5 mt-2 text-[10px] text-slate-500">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-5 border-t border-dashed border-edge-up opacity-60" />
              ±1σ ({sigmaEff ? (sigmaEff / 1000).toFixed(1) + 'k' : '—'})
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-5 border-t border-dashed border-edge-up opacity-30" />
              ±2σ ({sigmaEff ? (sigmaEff * 2 / 1000).toFixed(1) + 'k' : '—'})
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-5 border-t border-dashed" style={{ borderColor: "#fb923c" }} />
              nearest threshold · P(over)
            </span>
          </div>
        </GlassSection>
      )}

      {/* ── Edge bar charts (OVER + UNDER) ─────────────────────── */}
      {isKalshi && thresholds.length > 0 && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
          <EdgeChart
            title="OVER side — model vs market"
            sub="Edge = model P(over) − market P(over). Positive means BUY YES has edge."
            data={overChartData}
          />
          <EdgeChart
            title="UNDER side — model vs market"
            sub="Edge = model P(under) − market P(under). Positive means BUY NO has edge."
            data={underChartData}
          />
        </div>
      )}

      {/* ── Daily Kalshi positions / orders / orderbook ───────── */}
      {isKalshi && (
        <MarketsPanel
          scopeLabel={`KXTRUFTSA daily markets for ${data.target_date}`}
          positions={data.daily_positions}
          openOrders={data.daily_open_orders}
          orderbook={data.daily_orderbook}
        />
      )}

      {/* ── P(over / under) table ──────────────────────────────── */}
      {thresholds.length > 0 && (
        <GlassSection
          title="Threshold probabilities"
          sub={
            isKalshi
              ? `Live Kalshi daily markets for ${data.target_date} (${thresholds.length} strikes, 10-min cache)`
              : `No Kalshi market for ${data.target_date}${data.kalshi_error ? ` — ${data.kalshi_error}` : ""} · using μ ± {0.5,1,1.5}σ fallback`
          }
          right={
            <span
              className="chip"
              style={{
                background: isKalshi ? "rgba(34,211,164,0.12)" : "rgba(148,163,184,0.12)",
                color:      isKalshi ? "#22d3a4" : "#94a3b8",
              }}
            >
              {isKalshi ? "KALSHI" : "FALLBACK"}
            </span>
          }
        >
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-left text-slate-500 border-b border-slate-800/60">
                <th className="py-2">Threshold</th>
                <th className="py-2 text-right">Model P(Over)</th>
                <th className="py-2 text-right">Model P(Under)</th>
                {isKalshi && <th className="py-2 text-right">Mkt P(Over)</th>}
                {isKalshi && <th className="py-2 text-right">Edge OVER</th>}
                {isKalshi && <th className="py-2 text-right">Edge UNDER</th>}
                <th className="py-2 text-right">Lean</th>
              </tr>
            </thead>
            <tbody>
              {thresholds.map((t) => {
                const lean = t.p_over == null ? "—"
                  : t.p_over > 0.55 ? "OVER"
                  : t.p_over < 0.45 ? "UNDER"
                  : "TOSS-UP";
                const leanColor = lean === "OVER" ? "#22d3a4"
                  : lean === "UNDER" ? "#ef5466"
                  : "#94a3b8";
                return (
                  <tr key={t.threshold_millions} className="border-b border-slate-800/30">
                    <td className="py-2 mono text-slate-300">{t.threshold_millions.toFixed(2)}M</td>
                    <td className="py-2 text-right mono font-semibold"
                        style={{ color: (t.p_over ?? 0) > 0.55 ? "#22d3a4" : (t.p_over ?? 1) < 0.45 ? "#ef5466" : "#94a3b8" }}>
                      {fmtPct(t.p_over)}
                    </td>
                    <td className="py-2 text-right mono font-semibold"
                        style={{ color: (t.p_under ?? 0) > 0.55 ? "#22d3a4" : (t.p_under ?? 1) < 0.45 ? "#ef5466" : "#94a3b8" }}>
                      {fmtPct(t.p_under)}
                    </td>
                    {isKalshi && (
                      <td className="py-2 text-right mono text-slate-300">{fmtPct(t.market_p_over)}</td>
                    )}
                    {isKalshi && (
                      <td className="py-2 text-right mono font-semibold"
                          style={{ color: (t.edge_over ?? 0) > 0.02 ? "#22d3a4" : (t.edge_over ?? 0) < -0.02 ? "#ef5466" : "#94a3b8" }}>
                        {fmtPctSigned(t.edge_over)}
                      </td>
                    )}
                    {isKalshi && (
                      <td className="py-2 text-right mono font-semibold"
                          style={{ color: (t.edge_under ?? 0) > 0.02 ? "#22d3a4" : (t.edge_under ?? 0) < -0.02 ? "#ef5466" : "#94a3b8" }}>
                        {fmtPctSigned(t.edge_under)}
                      </td>
                    )}
                    <td className="py-2 text-right">
                      <span className="text-[10px] font-semibold" style={{ color: leanColor }}>
                        {lean}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </GlassSection>
      )}
    </div>
  );
}

function EdgeChart({
  title, sub, data,
}: {
  title: string;
  sub: string;
  data: { strike: string; model: number | null; market: number | null; edge: number | null }[];
}) {
  return (
    <GlassSection title={title} sub={sub}>
      <div className="h-[200px] sm:h-[260px]">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 6" stroke="rgba(99,140,255,0.08)" />
          <XAxis
            dataKey="strike"
            tick={{ fontSize: 10, fill: "#475569" }}
            tickLine={false} axisLine={false}
            tickFormatter={(v) => `${v}M`}
          />
          <YAxis
            tick={{ fontSize: 10, fill: "#475569" }}
            tickLine={false} axisLine={false}
            domain={[-1, 1]}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            width={42}
          />
          <ReferenceLine y={0} stroke="rgba(148,163,184,0.4)" />
          <Tooltip content={<EdgeBarTT />} cursor={{ fill: "rgba(96,165,250,0.06)" }} />
          <Legend wrapperStyle={{ fontSize: 11, paddingTop: 4 }} iconType="circle" iconSize={8} />
          <Bar dataKey="model"  name="Model"  fill="#60a5fa" radius={[3, 3, 0, 0]} />
          <Bar dataKey="market" name="Market" fill="#94a3b8" radius={[3, 3, 0, 0]} />
          <Bar dataKey="edge"   name="Edge"   radius={[3, 3, 0, 0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={(d.edge ?? 0) >= 0 ? "#22d3a4" : "#ef5466"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      </div>
    </GlassSection>
  );
}
