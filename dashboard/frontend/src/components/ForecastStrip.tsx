import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";
import { subDays, parseISO, format } from "date-fns";
import { api } from "../api/client";
import { GlassSection } from "./ui/GlassCard";

const ChartTT = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="glass !p-3 !rounded-lg"
      style={{ background: "rgba(5,8,15,0.96)" }}
    >
      <p className="text-[11px] text-slate-500 mb-1">{label}</p>
      {payload.map((p: any) =>
        p.value != null && (
          <p key={p.name} className="mono text-[12px]" style={{ color: p.color }}>
            {p.name}: {(p.value / 1e6).toFixed(3)}M
          </p>
        ),
      )}
    </div>
  );
};

const RANGES = [
  { label: "7D",  days:   7 },
  { label: "30D", days:  30 },
  { label: "60D", days:  60 },
  { label: "90D", days:  90 },
  { label: "6M",  days: 180 },
  { label: "1Y",  days: 365 },
] as const;
type RangeKey = (typeof RANGES)[number]["label"];

export function ForecastStrip() {
  const [range, setRange] = useState<RangeKey>("30D");

  // Always fetch the widest window once, then slice client-side on range change
  const fullCutoff = subDays(new Date(), 365).toISOString().slice(0, 10);
  const { data: daily } = useQuery({
    queryKey: ["forecast-daily-strip", fullCutoff],
    queryFn: () => api.forecastDaily(fullCutoff),
    staleTime: 5 * 60_000,
  });

  const chartData = useMemo(() => {
    const days = RANGES.find((r) => r.label === range)!.days;
    const cutoff = subDays(new Date(), days);
    return (daily ?? [])
      .filter((d) => d.actual != null || d.predicted != null)
      .filter((d) => parseISO(d.date) >= cutoff)
      .map((d) => ({
        date:      format(parseISO(d.date), days <= 30 ? "MMM d" : "MMM d, yy"),
        Actual:    d.actual    ?? undefined,
        Predicted: d.predicted ?? undefined,
      }));
  }, [daily, range]);

  const all = chartData.flatMap((d) => [d.Actual, d.Predicted]).filter((v): v is number => v != null);
  const yMin = all.length ? Math.floor((Math.min(...all) * 0.96) / 1e5) * 1e5 : "auto";
  const yMax = all.length ? Math.ceil((Math.max(...all) * 1.02) / 1e5) * 1e5 : "auto";

  return (
    <GlassSection
      title="TSA Volume — actual vs. forecast"
      sub={`Last ${range.toLowerCase()} of confirmed actuals vs. model predictions`}
      className="w-full h-full"
      right={
        <div className="flex items-center gap-3">
          <div className="hidden md:flex items-center gap-3 text-[11px] text-slate-500">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-[2.5px] bg-edge-info rounded" /> Actual
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-[0px]" style={{ borderTop: "2px dashed #f59e0b" }} /> Predicted
            </span>
          </div>
          <div className="flex items-center gap-0.5 p-0.5 rounded-lg bg-white/[0.04] ring-1 ring-white/[0.06]">
            {RANGES.map((r) => (
              <button
                key={r.label}
                onClick={() => setRange(r.label)}
                className={`px-2 py-1 text-[10.5px] font-bold rounded-md transition-colors mono ${
                  range === r.label
                    ? "bg-edge-info/20 text-edge-info"
                    : "text-slate-500 hover:text-slate-300"
                }`}
              >
                {r.label}
              </button>
            ))}
          </div>
        </div>
      }
    >
      <div className="h-[180px] sm:h-[260px]">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 6, right: 14, bottom: 4, left: 0 }}>
          <defs>
            <linearGradient id="stripActual" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#60a5fa" stopOpacity={0.32} />
              <stop offset="100%" stopColor="#60a5fa" stopOpacity={0.0} />
            </linearGradient>
            <linearGradient id="stripPred" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#f59e0b" stopOpacity={0.22} />
              <stop offset="100%" stopColor="#f59e0b" stopOpacity={0.0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 6" stroke="rgba(99,140,255,0.08)" />
          <XAxis
            dataKey="date"
            tick={{ fill: "#475569", fontSize: 11 }}
            tickLine={false} axisLine={false}
            interval="preserveStartEnd" minTickGap={28}
          />
          <YAxis
            tick={{ fill: "#475569", fontSize: 11 }}
            tickLine={false} axisLine={false}
            tickFormatter={(v) => `${(v / 1e6).toFixed(1)}M`}
            width={44} domain={[yMin, yMax]}
          />
          <Tooltip content={<ChartTT />} cursor={{ stroke: "rgba(96,165,250,0.20)" }} />
          <Area
            type="monotone" dataKey="Actual" name="Actual"
            stroke="#60a5fa" strokeWidth={2}
            fill="url(#stripActual)"
            dot={false} connectNulls={false}
            isAnimationActive={false}
          />
          <Area
            type="monotone" dataKey="Predicted" name="Predicted"
            stroke="#f59e0b" strokeWidth={1.5}
            strokeDasharray="5 3"
            fill="url(#stripPred)"
            dot={false} connectNulls={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
      </div>
    </GlassSection>
  );
}
