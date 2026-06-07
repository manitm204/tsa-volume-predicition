import { useState, useMemo } from "react"
import { useQuery } from "@tanstack/react-query"
import {
  Bar, BarChart, LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine, Cell,
} from "recharts"
import { subDays, subMonths, parseISO, format } from "date-fns"
import { api } from "../api/client"
import { PageHeader } from "../components/PageHeader"
import { GlassSection } from "../components/ui/GlassCard"
import { ModelScorecard } from "../components/ModelScorecard"
import { ModelExplainability } from "../components/ModelExplainability"
import { DriftMonitor } from "../components/DriftMonitor"

type Range = "1w" | "1m" | "3m" | "6m" | "1y" | "all"
const ranges: { key: Range; label: string }[] = [
  { key: "1w",  label: "1W"  },
  { key: "1m",  label: "1M"  },
  { key: "3m",  label: "3M"  },
  { key: "6m",  label: "6M"  },
  { key: "1y",  label: "1Y"  },
  { key: "all", label: "All" },
]
function startFor(r: Range) {
  const now = new Date()
  if (r === "1w")  return subDays(now,    7).toISOString().slice(0, 10)
  if (r === "1m")  return subMonths(now,  1).toISOString().slice(0, 10)
  if (r === "3m")  return subMonths(now,  3).toISOString().slice(0, 10)
  if (r === "6m")  return subMonths(now,  6).toISOString().slice(0, 10)
  if (r === "1y")  return subMonths(now, 12).toISOString().slice(0, 10)
  return undefined
}

function RangeSelector({ value, onChange }: { value: Range; onChange: (r: Range) => void }) {
  return (
    <div className="flex items-center gap-0.5 p-0.5 rounded-lg bg-white/[0.04] ring-1 ring-white/[0.06]">
      {ranges.map((r) => (
        <button
          key={r.key}
          onClick={() => onChange(r.key)}
          className={`px-2 py-1 text-[10.5px] font-bold rounded-md transition-colors mono ${
            value === r.key
              ? "bg-edge-info/20 text-edge-info"
              : "text-slate-500 hover:text-slate-300"
          }`}
        >
          {r.label}
        </button>
      ))}
    </div>
  )
}

const TT = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="glass !p-3 !rounded-lg" style={{ background: "rgba(5,8,15,0.96)" }}>
      <p className="text-[11px] text-slate-500 mb-1">{label}</p>
      {payload.map((p: any) => p.value != null && (
        <p key={p.name} className="mono text-[12px]" style={{ color: p.color }}>
          {p.name}: {p.name === "Error" ? `${p.value > 0 ? "+" : ""}${Math.round(p.value)}k` : `${(p.value / 1e6).toFixed(3)}M`}
        </p>
      ))}
    </div>
  )
}

function useDailyChart(range: Range) {
  const { data, isLoading } = useQuery({
    queryKey:  ["forecast-daily", range],
    queryFn:   () => api.forecastDaily(startFor(range)),
    staleTime: 5 * 60_000,
  })
  const labelFmt = useMemo(() => {
    if (range === "1w" || range === "1m") return "MMM d"
    if (range === "3m" || range === "6m") return "MMM d"
    return "MMM d, ''yy"
  }, [range])
  const chartData = useMemo(() => (data ?? []).map(d => ({
    date:      format(parseISO(d.date), labelFmt),
    Actual:    d.actual    ?? undefined,
    Predicted: d.predicted ?? undefined,
    Error:     d.actual != null && d.predicted != null
      ? Math.round((d.actual - d.predicted) / 1000)
      : undefined,
  })), [data, labelFmt])
  return { chartData, isLoading }
}

export default function Forecast() {
  const [combinedRange, setCombinedRange] = useState<Range>("1m")

  const { chartData: combinedData, isLoading: combinedLoading } = useDailyChart(combinedRange)

  const { data: week } = useQuery({ queryKey: ["forecast-week"], queryFn: api.forecastWeek })

  // KPIs follow the combined chart's range so they stay in sync with what's on screen.
  const withBoth = combinedData.filter(d => d.Actual != null && d.Predicted != null)
  const mae = withBoth.length
    ? Math.round(withBoth.reduce((s, d) => s + Math.abs(d.Actual! - d.Predicted!), 0) / withBoth.length / 1000)
    : null
  const avgErr = withBoth.length
    ? Math.round(withBoth.reduce((s, d) => s + (d.Actual! - d.Predicted!), 0) / withBoth.length / 1000)
    : null
  const hitRate = withBoth.length
    ? Math.round(withBoth.filter(d => Math.abs(d.Actual! - d.Predicted!) < 50_000).length / withBoth.length * 100)
    : null

  // For the combined chart: pick a tick gap that scales with point count
  const combinedTickGap = combinedData.length > 200 ? 56 : combinedData.length > 90 ? 36 : 24

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Model Performance"
        description="Forecast accuracy, calibration, and feature attribution"
        kpis={[
          { label: "Mean Abs Error",  value: mae    != null ? `${mae}k`    : "—", accent: "#60a5fa",
            sub: `over ${combinedRange.toUpperCase()}` },
          { label: "Average Bias",
            value: avgErr != null ? `${avgErr >= 0 ? "+" : ""}${avgErr}k` : "—",
            accent: avgErr != null ? (Math.abs(avgErr) < 10 ? "#22d3a4" : avgErr > 0 ? "#f59e0b" : "#ef5466") : undefined,
            sub: avgErr != null ? (avgErr > 0 ? "Model under-predicts" : avgErr < 0 ? "Model over-predicts" : "Well-calibrated") : undefined },
          { label: "Within 50k", value: hitRate != null ? `${hitRate}%` : "—", accent: "#a78bfa" },
          { label: "Days Compared", value: `${withBoth.length}`, sub: `of ${combinedData.length} in range` },
        ]}
      />

      <ModelScorecard />

      <DriftMonitor />

      <ModelExplainability />

      {/* ─── Volume on top, error bars below (synced) ─── */}
      <GlassSection
        title="Daily Volume + Forecast Error"
        sub={`Top: actual vs predicted · Bottom: signed error (under/over-predicted) · ${combinedRange.toUpperCase()} window`}
        right={<RangeSelector value={combinedRange} onChange={setCombinedRange} />}
      >
        {combinedLoading ? <div className="skeleton h-[380px]" /> : (
          <div className="space-y-0">
            {/* Top: lines */}
            <div className="h-[180px] sm:h-[260px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={combinedData} syncId="model-combined" margin={{ top: 8, right: 14, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 6" stroke="rgba(99,140,255,0.07)" />
                <XAxis
                  dataKey="date"
                  tick={false}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(99,140,255,0.10)" }}
                  height={0}
                />
                <YAxis
                  tick={{ fill: "#475569", fontSize: 11 }}
                  tickLine={false} axisLine={false}
                  tickFormatter={v => `${(v / 1e6).toFixed(2)}M`}
                  width={48}
                  domain={["auto", "auto"]}
                />
                <Tooltip content={<TT />} cursor={{ stroke: "rgba(96,165,250,0.20)" }} />
                <Line type="monotone" dataKey="Actual"    name="Actual"    stroke="#60a5fa" strokeWidth={2}   dot={false} connectNulls={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="Predicted" name="Predicted" stroke="#f59e0b" strokeWidth={1.6} strokeDasharray="5 3" dot={false} connectNulls={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
            </div>

            {/* Bottom: error bars sharing same x */}
            <div className="h-[100px] sm:h-[130px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={combinedData} syncId="model-combined" margin={{ top: 0, right: 14, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 6" stroke="rgba(99,140,255,0.07)" />
                <XAxis
                  dataKey="date"
                  tick={{ fill: "#475569", fontSize: 10.5 }}
                  tickLine={false}
                  axisLine={{ stroke: "rgba(99,140,255,0.10)" }}
                  interval="preserveStartEnd"
                  minTickGap={combinedTickGap}
                />
                <YAxis
                  tick={{ fill: "#475569", fontSize: 10.5 }}
                  tickLine={false} axisLine={false}
                  tickFormatter={v => `${v}k`}
                  width={48}
                  domain={["auto","auto"]}
                />
                <Tooltip content={<TT />} cursor={{ fill: "rgba(96,165,250,0.05)" }} />
                <ReferenceLine y={0} stroke="rgba(99,140,255,0.18)" strokeWidth={1.5} />
                <Bar dataKey="Error" name="Error" radius={[3, 3, 0, 0]}>
                  {combinedData.map((d, i) => (
                    <Cell key={i} fill={(d.Error ?? 0) >= 0 ? "#22d3a4" : "#ef5466"} opacity={0.85} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            </div>

            <div className="flex flex-wrap items-center gap-5 mt-2 px-1 text-[10px] text-slate-500">
              <span className="flex items-center gap-1.5">
                <span className="inline-block w-3 h-[2.5px] bg-edge-info rounded" /> Actual
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block w-3 h-[0px]" style={{ borderTop: "2px dashed #f59e0b" }} /> Predicted
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: "#22d3a4", opacity: 0.85 }} /> Error ≥ 0 (under-predicted)
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: "#ef5466", opacity: 0.85 }} /> Error &lt; 0 (over-predicted)
              </span>
            </div>
          </div>
        )}
      </GlassSection>

      {week?.days && (
        <GlassSection
          title="Current Week — Day by Day"
          sub="Comparing today's model forecast against confirmed TSA actuals"
        >
          <table className="data-table">
            <thead>
              <tr>{["Day", "Date", "Status", "Actual", "Predicted", "Error"].map(h => <th key={h} className="label">{h}</th>)}</tr>
            </thead>
            <tbody>
              {week.days.map(d => {
                const vol  = d.volume    != null ? `${(d.volume    / 1e6).toFixed(3)}M` : "—"
                const pred = d.predicted != null ? `${(d.predicted / 1e6).toFixed(3)}M` : "—"
                const errK = d.error     != null ? Math.round(d.error / 1000)           : null
                return (
                  <tr key={d.date}>
                    <td className="font-semibold text-slate-100">{d.day_name}</td>
                    <td className="text-[12px] text-slate-500">{d.date}</td>
                    <td><span className={d.status === "actual" ? "badge badge-actual" : "badge badge-predicted"}>{d.status}</span></td>
                    <td className="mono text-slate-100">{vol}</td>
                    <td className="mono text-slate-500">{pred}</td>
                    <td>
                      {errK != null
                        ? <span className="mono font-semibold" style={{ color: errK >= 0 ? "#22d3a4" : "#ef5466" }}>{errK >= 0 ? "+" : ""}{errK}k</span>
                        : <span className="text-slate-700">—</span>}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </GlassSection>
      )}
    </div>
  )
}
