import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine,
} from "recharts";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { api } from "../api/client";
import { PageHeader, type KpiItem } from "../components/PageHeader";
import { GlassSection } from "../components/ui/GlassCard";
import type { DayDetailRow } from "../types";

const REGIME_COLOR: Record<string, string> = {
  STORM: "#ef5466",
  PEAK_HOLIDAY: "#f59e0b",
  SHOULDER_PRE: "#60a5fa",
  SHOULDER_POST: "#a78bfa",
  NORMAL: "#22d3a4",
};

const MODEL_META = [
  { key: "pred_tabular",   wKey: "tab",       label: "Tabular",   color: "#60a5fa", desc: "AutoGluon tabular (autoregressive chain)" },
  { key: "pred_ts3",       wKey: "ts3",       label: "TS3",       color: "#a78bfa", desc: "AutoGluon TimeSeries (calendar + lag365)" },
  { key: "pred_yoy_delta", wKey: "yoy_delta", label: "YoY Delta", color: "#fb923c", desc: "Last year + recent same-weekday YoY delta" },
  { key: "pred_anchor",    wKey: "anchor",    label: "Anchor",    color: "#facc15", desc: "anchor_master feature (lag-365 same-DOW)" },
] as const;

function fmt(v: number | null | undefined): string {
  if (v == null || isNaN(v)) return "—";
  return v.toLocaleString(undefined, { maximumFractionDigits: 0 });
}
function fmtM(v: number | null | undefined): string {
  if (v == null || isNaN(v)) return "—";
  return `${(v / 1e6).toFixed(4)}M`;
}
function fmtPct(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

const BellTT = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass !p-2 !rounded-lg text-[11px]" style={{ background: "rgba(5,8,15,0.96)" }}>
      <span className="mono text-slate-300">{Number(payload[0].payload.x).toFixed(3)}M</span>
    </div>
  );
};

export default function DayExplorer() {
  const [weeksBack, setWeeksBack] = useState(0);
  const [idx, setIdx] = useState<number | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["day-detail", weeksBack],
    queryFn: () => api.dayDetail(weeksBack),
    refetchInterval: 60_000,
    placeholderData: (prev) => prev,
  });

  const days = data?.days ?? [];
  const hasOlderWeek = data?.has_older_week ?? false;

  if (isLoading && days.length === 0) {
    return (
      <GlassSection title="Loading…">
        <p className="text-slate-500 text-[13px]">Loading day-by-day breakdown…</p>
      </GlassSection>
    );
  }

  if (days.length === 0) {
    return (
      <>
        <PageHeader title="Day Explorer" description="Cycle through each day of the week" kpis={[]} />
        <GlassSection title="No data">
          <p className="text-slate-500 text-[13px]">No weekly forecast data available.</p>
        </GlassSection>
      </>
    );
  }

  // Default to today, or the first predicted day if today isn't in the list.
  const todayStr = new Date().toISOString().slice(0, 10);
  const defaultIdx = Math.max(0, days.findIndex(d => d.date === todayStr));
  const activeIdx = Math.min(idx ?? defaultIdx, days.length - 1);
  const d: DayDetailRow = days[activeIdx];
  const canGoPrev = activeIdx > 0 || hasOlderWeek;
  const canGoNext = activeIdx < days.length - 1 || weeksBack > 0;

  // Crossing a week boundary: go to the last/first day of the adjacent week.
  const goPrev = () => {
    if (activeIdx > 0) { setIdx(activeIdx - 1); return; }
    if (hasOlderWeek) { setWeeksBack(w => w + 1); setIdx(6); }
  };
  const goNext = () => {
    if (activeIdx < days.length - 1) { setIdx(activeIdx + 1); return; }
    if (weeksBack > 0) { setWeeksBack(w => w - 1); setIdx(0); }
  };

  const isActual = d.status === "actual";
  const regColor = REGIME_COLOR[d.regime ?? ""] ?? "#94a3b8";
  const bellCurve = d.bell_curve ?? [];

  const s1lo = d.predicted && d.sigma_eff ? (d.predicted - d.sigma_eff) / 1e6 : null;
  const s1hi = d.predicted && d.sigma_eff ? (d.predicted + d.sigma_eff) / 1e6 : null;

  const kpis: KpiItem[] = [
    {
      label: "Ensemble",
      value: (
        <span className="flex items-baseline gap-1.5">
          <span>{fmtM(d.predicted)}</span>
          {isActual && (
            <>
              <span className="text-slate-600 text-[13px] font-normal">/</span>
              <span style={{ color: "#f59e0b" }}>{fmtM(d.actual)}</span>
            </>
          )}
        </span>
      ),
      sub: isActual ? "predicted / actual" : "predicted",
      accent: "#22d3a4",
    },
    { label: "Regime", value: (d.regime ?? "—").replace("_", " "), accent: regColor },
    {
      label: isActual ? "Error" : "σ_eff",
      value: isActual ? (d.error != null ? `${d.error >= 0 ? "+" : ""}${fmt(d.error)}` : "—")
                       : (d.sigma_eff ? `${(d.sigma_eff / 1000).toFixed(1)}k` : "—"),
      accent: isActual ? (d.error != null && d.error >= 0 ? "#22d3a4" : "#ef5466") : "#a78bfa",
    },
    { label: "Date", value: d.date, sub: d.day_name },
  ];

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Day Explorer"
        description={weeksBack === 0
          ? "Ensemble + per-model prediction for each day of the current week"
          : `Week of ${data?.week_start ?? "—"} (${weeksBack} week${weeksBack === 1 ? "" : "s"} back)`}
        kpis={kpis}
      />

      {/* ── Day cycler ─────────────────────────────────────────── */}
      <div className="flex items-center gap-2">
        <button
          className="p-2 rounded-lg glass glass-hover disabled:opacity-30"
          disabled={!canGoPrev}
          onClick={goPrev}
          aria-label="Previous day"
        >
          <ChevronLeft size={16} />
        </button>

        <div className="flex-1 grid grid-cols-7 gap-1.5">
          {days.map((day, i) => {
            const active = i === activeIdx;
            const dActual = day.status === "actual";
            return (
              <button
                key={day.date}
                onClick={() => setIdx(i)}
                className="rounded-lg py-2 px-1 text-center transition-colors"
                style={{
                  background: active ? "rgba(34,211,164,0.14)" : "rgba(148,163,184,0.06)",
                  border: active ? "1px solid rgba(34,211,164,0.5)" : "1px solid transparent",
                }}
              >
                <div className="text-[10px] font-bold uppercase tracking-wide"
                     style={{ color: active ? "#22d3a4" : "#64748b" }}>
                  {day.day_name.slice(0, 3)}
                </div>
                <div className="text-[9px] mono mt-0.5" style={{ color: active ? "#94a3b8" : "#475569" }}>
                  {day.date.slice(5)}
                </div>
                <span
                  className={`inline-block mt-1 px-1 py-0.5 rounded text-[8px] font-semibold ${
                    dActual ? "badge-actual" : "badge-predicted"
                  }`}
                >
                  {dActual ? "actual" : "pred"}
                </span>
              </button>
            );
          })}
        </div>

        <button
          className="p-2 rounded-lg glass glass-hover disabled:opacity-30"
          disabled={!canGoNext}
          onClick={goNext}
          aria-label="Next day"
        >
          <ChevronRight size={16} />
        </button>
      </div>
      {weeksBack > 0 && (
        <button
          className="text-[11px] text-slate-500 hover:text-slate-300 transition-colors -mt-2"
          onClick={() => { setWeeksBack(0); setIdx(null); }}
        >
          ← back to current week
        </button>
      )}

      {/* ── Per-model prediction cards ─────────────────────────── */}
      <GlassSection
        title="Per-model predictions"
        sub={isActual
          ? "Each model's prediction made ahead of time for this now-actual day"
          : "Each model's raw prediction before ensemble blending"}
      >
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-4 mb-4">
          {MODEL_META.map(({ key, wKey, label, color, desc }) => {
            const val = d[key] as number | null;
            const w = d.weights?.[wKey as keyof typeof d.weights] ?? null;
            const diffEnsemble = val != null && d.predicted != null ? val - d.predicted : null;
            const diffActual = isActual && val != null && d.actual != null ? val - d.actual : null;
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
                  {val != null ? fmtM(val) : <span className="text-slate-600 text-base">not recorded</span>}
                </div>
                {diffEnsemble != null && (
                  <div className="mono text-[11px]" style={{ color: diffEnsemble >= 0 ? "#22d3a4" : "#ef5466" }}>
                    {diffEnsemble >= 0 ? "+" : ""}{fmt(diffEnsemble)} vs ensemble
                  </div>
                )}
                {diffActual != null && (
                  <div className="mono text-[11px]" style={{ color: diffActual >= 0 ? "#22d3a4" : "#ef5466" }}>
                    {diffActual >= 0 ? "+" : ""}{fmt(diffActual)} vs actual
                  </div>
                )}
                <div className="text-[10px] text-slate-600 leading-tight">{desc}</div>
              </div>
            );
          })}
        </div>

        <div className="glass rounded-xl p-4 flex items-center justify-between"
             style={{ borderLeft: "3px solid #22d3a4" }}>
          <div>
            <div className="text-[11px] font-bold uppercase tracking-widest text-edge-up mb-1">
              Ensemble (blended)
            </div>
            <div className="text-[10px] text-slate-500">
              {d.weights
                ? Object.entries(d.weights)
                    .filter(([, v]) => v != null && (v as number) > 0)
                    .map(([k, v]) => `${k} ${((v as number) * 100).toFixed(1)}%`)
                    .join(" · ")
                : "α-blend (storm) or not available"}
            </div>
          </div>
          <div className="flex items-baseline gap-3">
            {isActual && (
              <span className="text-[11px] text-slate-500">
                predicted {fmtM(d.predicted)} →
              </span>
            )}
            <span className="mono text-2xl font-bold text-edge-up">
              {isActual ? fmtM(d.actual) : fmtM(d.ensemble)}
            </span>
            {isActual && <span className="text-[10px] text-slate-500">actual</span>}
          </div>
        </div>
      </GlassSection>

      {/* ── Forecast distribution (past days only) ─────────────── */}
      {isActual && bellCurve.length > 0 && (
        <GlassSection
          title="Forecast distribution vs. actual"
          sub={`N(μ = ${fmtM(d.predicted)}, σ = ${d.sigma_eff ? (d.sigma_eff / 1000).toFixed(1) + 'k' : '—'}) · actual value marked`}
        >
          <div className="h-[220px] sm:h-[300px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={bellCurve} margin={{ top: 28, right: 20, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="bellGradExplorer" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22d3a4" stopOpacity={0.35} />
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

                {s1lo != null && <ReferenceLine x={s1lo} stroke="#22d3a4" strokeDasharray="3 3" strokeWidth={1.2} strokeOpacity={0.5} />}
                {s1hi != null && <ReferenceLine x={s1hi} stroke="#22d3a4" strokeDasharray="3 3" strokeWidth={1.2} strokeOpacity={0.5} />}

                {d.predicted != null && (
                  <ReferenceLine
                    x={d.predicted / 1e6}
                    stroke="#22d3a4"
                    strokeWidth={2}
                    label={{ value: "predicted (μ)", position: "top", fill: "#22d3a4", fontSize: 11 }}
                  />
                )}
                {d.actual != null && (
                  <ReferenceLine
                    x={d.actual / 1e6}
                    stroke="#f59e0b"
                    strokeWidth={2}
                    label={{ value: "actual", position: "insideTopRight", fill: "#f59e0b", fontSize: 11 }}
                  />
                )}

                <Area
                  type="monotone"
                  dataKey="y"
                  stroke="#22d3a4"
                  strokeWidth={2}
                  fill="url(#bellGradExplorer)"
                  dot={false}
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="flex flex-wrap gap-5 mt-2 text-[10px] text-slate-500">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-5 border-t-2" style={{ borderColor: "#22d3a4" }} />
              predicted (μ)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-5 border-t-2" style={{ borderColor: "#f59e0b" }} />
              actual
            </span>
            <span className="flex items-center gap-1.5">
              error: <span className="mono font-semibold" style={{ color: (d.error ?? 0) >= 0 ? "#22d3a4" : "#ef5466" }}>
                {d.error != null ? `${d.error >= 0 ? "+" : ""}${fmt(d.error)}` : "—"}
              </span>
              {d.predicted && d.error != null && (
                <span className="mono">({fmtPct(d.error / d.predicted)})</span>
              )}
            </span>
          </div>
        </GlassSection>
      )}
    </div>
  );
}
