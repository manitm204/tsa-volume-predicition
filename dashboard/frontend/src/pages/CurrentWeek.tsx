import { useQuery } from "@tanstack/react-query"
import {
  ComposedChart, Bar, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Cell, LineChart, ReferenceLine,
} from "recharts"
import { api } from "../api/client"
import { format, parseISO } from "date-fns"
import { PageHeader } from "../components/PageHeader"
import { GlassSection } from "../components/ui/GlassCard"
import { ProgressBar } from "../components/ui/ProgressBar"
import { ProbabilityLadder } from "../components/ProbabilityLadder"
import { ForecastDistribution } from "../components/ForecastDistribution"
import { WeatherImpactCard } from "../components/WeatherImpactCard"
import { MarketsPanel } from "../components/MarketsPanel"
import { fmt } from "../lib/format"

const TT = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="glass !p-3 !rounded-lg" style={{ background: "rgba(5,8,15,0.96)" }}>
      <p className="text-[11px] text-slate-500 mb-1">{label}</p>
      {payload.map((p: any) => p.value != null && (
        <p key={p.name} className="mono text-[12px]" style={{ color: p.color }}>
          {p.name}: {(p.value / 1e6).toFixed(3)}M
        </p>
      ))}
    </div>
  )
}
const TrackerTT = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="glass !p-3 !rounded-lg" style={{ background: "rgba(5,8,15,0.96)" }}>
      <p className="text-[11px] text-slate-500 mb-1">{label}</p>
      {payload.map((p: any) => p.value != null && (
        <p key={p.name} className="mono text-[12px]" style={{ color: p.color }}>
          {p.name}: {p.value.toFixed(4)}M
        </p>
      ))}
    </div>
  )
}

export default function CurrentWeek() {
  const { data: week }       = useQuery({ queryKey: ["forecast-week"],      queryFn: api.forecastWeek      })
  const { data: tracker }    = useQuery({ queryKey: ["weekly-avg-tracker"], queryFn: api.weeklyAvgTracker })
  const { data: positions }  = useQuery({ queryKey: ["positions"],          queryFn: api.positions,    refetchInterval: 60_000 })
  const { data: openOrders } = useQuery({ queryKey: ["open-orders"],        queryFn: api.openOrders,   refetchInterval: 60_000 })
  const { data: orderbook }  = useQuery({ queryKey: ["orderbook"],          queryFn: api.orderbook,    refetchInterval: 60_000 })

  const summary   = week?.summary ?? {}
  const weekAvg   = typeof summary.weekly_avg_millions     === "number" ? summary.weekly_avg_millions     : null
  const weekStd   = typeof summary.weekly_avg_std_millions === "number" ? summary.weekly_avg_std_millions : null
  const nActual   = (summary.n_actual as number | null) ?? null
  const kalshiAvg = week?.kalshi_avg ?? null
  const diffK     = weekAvg != null && kalshiAvg != null ? (weekAvg - kalshiAvg) * 1000 : null

  const chartData   = (week?.days ?? []).map(d => ({
    day:    d.day_name.slice(0, 3),
    volume: d.volume ?? undefined,
    fill:   d.status === "actual" ? "#60a5fa" : "#f59e0b",
  }))
  const trackerData = (tracker ?? []).map(t => ({
    date:   format(parseISO(t.date), "MMM d"),
    Model:  t.model_avg_millions  ?? undefined,
    Kalshi: t.kalshi_avg_millions ?? undefined,
  }))

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="This Week"
        description="Current week's volume forecast compared against confirmed TSA actuals"
        kpis={[
          { label: "Model Weekly Avg", value: weekAvg != null ? `${weekAvg.toFixed(4)}M` : "—",
            sub: weekStd != null ? `± ${weekStd.toFixed(4)}M (1σ)` : undefined, accent: "#60a5fa" },
          { label: "Kalshi Implied", value: kalshiAvg != null ? `${kalshiAvg.toFixed(4)}M` : "—",
            sub: "Market consensus", accent: "#fbbf24" },
          { label: "Model vs Market",
            value: diffK != null
              ? <span style={{ color: diffK >= 0 ? "#22d3a4" : "#ef5466" }}>{diffK >= 0 ? "+" : ""}{diffK.toFixed(1)}k</span>
              : "—",
            sub: diffK != null
              ? (Math.abs(diffK) < 10 ? "In agreement" : diffK > 0 ? "Model higher" : "Model lower") : undefined },
          { label: "Days Confirmed", value: nActual != null ? `${nActual} / 7` : "—",
            sub: week?.mae != null ? `In-week MAE: ${Math.round(week.mae / 1000)}k` : undefined },
        ]}
        right={
          nActual != null ? (
            <div className="w-32">
              <ProgressBar value={nActual} max={7} tone="gradient" height={6} rightLabel={`${nActual}/7`} />
            </div>
          ) : undefined
        }
      />

      {/* Weather impact */}
      <WeatherImpactCard />

      {/* Distribution + Ladder */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <ForecastDistribution />
        <ProbabilityLadder />
      </div>

      {/* Daily breakdown + volume bars */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <GlassSection
          title="Daily Breakdown"
          sub="Volume and error for each day of the current week"
        >
          <table className="data-table">
            <thead>
              <tr>{["Day", "Regime", "Date", "Volume", "Status", "Error"].map(h => <th key={h} className="label">{h}</th>)}</tr>
            </thead>
            <tbody>
              {(week?.days ?? []).map(d => {
                const errK = d.error != null ? Math.round(d.error / 1000) : null
                const REGIME_COLOR: Record<string, string> = {
                  STORM: "#ef5466", PEAK_HOLIDAY: "#f59e0b",
                  SHOULDER_PRE: "#60a5fa", SHOULDER_POST: "#a78bfa", NORMAL: "#22d3a4",
                }
                return (
                  <tr key={d.date}>
                    <td className="font-semibold text-slate-100">{d.day_name}</td>
                    <td>
                      {d.regime ? (
                        <span
                          className="inline-block px-2 py-0.5 rounded text-[10px] font-semibold"
                          style={{
                            background: `${REGIME_COLOR[d.regime] ?? "#64748b"}22`,
                            color: REGIME_COLOR[d.regime] ?? "#94a3b8",
                          }}
                        >
                          {d.regime}
                        </span>
                      ) : (
                        <span className="text-slate-700">—</span>
                      )}
                    </td>
                    <td className="text-[12px] text-slate-500">{d.date}</td>
                    <td className="mono text-slate-100">{fmt(d.volume)}</td>
                    <td>
                      <span className={d.status === "actual" ? "badge badge-actual" : "badge badge-predicted"}>
                        {d.status}
                      </span>
                    </td>
                    <td>
                      {errK != null
                        ? <span className="mono font-semibold" style={{ color: errK >= 0 ? "#22d3a4" : "#ef5466" }}>
                            {errK >= 0 ? "+" : ""}{errK}k
                          </span>
                        : <span className="text-slate-700">—</span>}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </GlassSection>

        <GlassSection
          title="Volume by Day"
          sub="Blue = confirmed · Amber = forecasted"
          right={weekAvg != null && (
            <span className="chip">avg {weekAvg.toFixed(3)}M</span>
          )}
        >
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart data={chartData} margin={{ top: 6, right: 12, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 6" stroke="rgba(99,140,255,0.07)" />
              <XAxis dataKey="day" tick={{ fill: "#475569", fontSize: 12 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: "#475569", fontSize: 11 }} tickLine={false} axisLine={false}
                tickFormatter={v => `${(v/1e6).toFixed(2)}M`} width={44} domain={["auto","auto"]} />
              <Tooltip content={<TT />} cursor={{ fill: "rgba(96,165,250,0.05)" }} />
              {weekAvg != null && (
                <ReferenceLine y={weekAvg * 1e6} stroke="#22d3a4" strokeDasharray="3 4" strokeWidth={1.5} />
              )}
              <Bar dataKey="volume" name="Volume" radius={[6, 6, 0, 0]} maxBarSize={52}>
                {chartData.map((e, i) => <Cell key={i} fill={e.fill} />)}
              </Bar>
              {weekAvg != null && (
                <Line type="monotone" dataKey={() => weekAvg * 1e6} name="Avg Forecast"
                  stroke="#f59e0b" strokeWidth={1.5} strokeDasharray="5 3" dot={false} />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </GlassSection>
      </div>

      {/* Weekly Kalshi positions, orders, orderbook */}
      <MarketsPanel
        scopeLabel="KXTSAW (weekly) markets"
        positions={positions}
        openOrders={openOrders}
        orderbook={orderbook}
      />

      {/* Tracker */}
      {trackerData.length > 0 && (
        <GlassSection
          title="Weekly Avg Tracker"
          sub="How model (emerald) and Kalshi (amber) consensus evolved this week"
          right={
            <div className="flex items-center gap-4 text-[11px] text-slate-500">
              <span className="flex items-center gap-1.5">
                <span className="inline-block w-3 h-[2.5px] bg-edge-up rounded" /> Model
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block w-3 h-[0px]" style={{ borderTop: "2px dashed #f59e0b" }} /> Kalshi
              </span>
            </div>
          }
        >
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={trackerData} margin={{ top: 6, right: 14, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 6" stroke="rgba(99,140,255,0.07)" />
              <XAxis dataKey="date" tick={{ fill: "#475569", fontSize: 12 }} tickLine={false} axisLine={false} />
              <YAxis tick={{ fill: "#475569", fontSize: 11 }} tickLine={false} axisLine={false} tickFormatter={v => `${v.toFixed(2)}M`} width={52} domain={["auto","auto"]} />
              <Tooltip content={<TrackerTT />} cursor={{ stroke: "rgba(96,165,250,0.20)" }} />
              <Line type="monotone" dataKey="Model"  stroke="#22d3a4" strokeWidth={2.5} dot={{ r: 4, fill: "#22d3a4", strokeWidth: 0 }} connectNulls={false} />
              <Line type="monotone" dataKey="Kalshi" stroke="#f59e0b" strokeWidth={2}   strokeDasharray="5 3" dot={{ r: 4, fill: "#f59e0b", strokeWidth: 0 }} connectNulls={false} />
            </LineChart>
          </ResponsiveContainer>
        </GlassSection>
      )}
    </div>
  )
}
