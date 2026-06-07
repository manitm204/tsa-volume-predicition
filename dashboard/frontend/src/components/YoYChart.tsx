import { useQuery } from "@tanstack/react-query";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, ReferenceLine,
} from "recharts";
import { parseISO, format } from "date-fns";
import { api } from "../api/client";

const COLORS = {
  y2024:           "#374151",
  y2025:           "#60a5fa",
  y2026_actual:    "#34d399",
  y2026_predicted: "#f59e0b",
};

const CustomXTick = ({ x, y, index, chartData }: any) => {
  const item     = chartData?.[index];
  const isMonday = item?.isMonday;
  const dow      = item?.dow ?? "";
  const weekLabel = item?.weekLabel ?? "";

  return (
    <g transform={`translate(${x},${y})`}>
      {isMonday && weekLabel && (
        <text x={0} y={-6} fill="#374151" fontSize={9} textAnchor="middle">
          {weekLabel}
        </text>
      )}
      <text
        x={0} dy={14}
        fill={isMonday ? "#94a3b8" : "#374151"}
        fontSize={10}
        fontWeight={isMonday ? 600 : 400}
        textAnchor="middle"
      >
        {dow}
      </text>
    </g>
  );
};

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const item = payload[0]?.payload;
  return (
    <div
      style={{
        background: "rgba(8, 13, 24, 0.95)",
        border: "1px solid rgba(59, 130, 246, 0.15)",
        borderRadius: "10px",
        padding: "10px 14px",
        boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
        backdropFilter: "blur(12px)",
      }}
    >
      <p style={{ color: "#6b7280", fontSize: "11px", marginBottom: "6px", fontWeight: 600 }}>
        {item?.weekLabel} · {item?.dow}
      </p>
      {payload.map((p: any) =>
        p.value != null ? (
          <p key={p.name} style={{ color: p.color, fontSize: "12px", fontFamily: "monospace" }}>
            {p.name}: {p.value.toFixed(3)}M
          </p>
        ) : null
      )}
    </div>
  );
};

export default function YoYChart() {
  const { data, isLoading } = useQuery({
    queryKey: ["yoy-comparison"],
    queryFn: api.yoyComparison,
    staleTime: 5 * 60_000,
  });

  if (isLoading || !data?.length) {
    return (
      <div className="card">
        <p className="label mb-4">Year-over-Year Comparison</p>
        <div className="skeleton h-72" />
      </div>
    );
  }

  let lastActualIdx = -1;
  for (let i = data.length - 1; i >= 0; i--) {
    if (data[i].y2026_actual != null) { lastActualIdx = i; break; }
  }

  const chartData = data.map((d, i) => {
    const predicted =
      d.y2026_predicted != null ? d.y2026_predicted / 1e6
      : i === lastActualIdx      ? d.y2026_actual! / 1e6
      : null;

    const dt        = parseISO(d.date);
    const weekIdx   = Math.floor(i / 7);
    const weekOffset = weekIdx - 3;
    const weekLabel = weekOffset === 0 ? "This week" : `${weekOffset}w`;

    return {
      date:            d.date,
      dow:             format(dt, "EEE"),
      weekLabel,
      isMonday:        dt.getDay() === 1,
      "2024":          d.y2024        != null ? d.y2024        / 1e6 : null,
      "2025":          d.y2025        != null ? d.y2025        / 1e6 : null,
      "2026 Actual":   d.y2026_actual != null ? d.y2026_actual / 1e6 : null,
      "2026 Forecast": predicted,
    };
  });

  const mondayDates = chartData
    .filter((d, i) => d.isMonday && i > 0)
    .map(d => d.date);

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-5">
        <p className="label">Year-over-Year — Last 3 Weeks + Current</p>
        <div className="flex items-center gap-5 text-xs" style={{ color: "#374151" }}>
          <span className="flex items-center gap-1.5">
            <span style={{ width: 12, height: 2, background: COLORS.y2024, borderRadius: 2, display: "inline-block" }} />
            2024
          </span>
          <span className="flex items-center gap-1.5">
            <span style={{ width: 12, height: 2, background: COLORS.y2025, borderRadius: 2, display: "inline-block" }} />
            2025
          </span>
          <span className="flex items-center gap-1.5">
            <span style={{ width: 12, height: 2, background: COLORS.y2026_actual, borderRadius: 2, display: "inline-block" }} />
            2026 Actual
          </span>
          <span className="flex items-center gap-1.5">
            <span style={{ width: 14, height: 0, borderTop: `2px dashed ${COLORS.y2026_predicted}`, display: "inline-block" }} />
            2026 Forecast
          </span>
        </div>
      </div>
      <div className="h-[200px] sm:h-[300px]">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={chartData} margin={{ top: 16, right: 12, bottom: 4, left: 0 }}>
          <defs>
            <linearGradient id="gradYoY2026" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#34d399" stopOpacity={0.18} />
              <stop offset="100%" stopColor="#34d399" stopOpacity={0.01} />
            </linearGradient>
            <linearGradient id="gradYoY2025" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#60a5fa" stopOpacity={0.12} />
              <stop offset="100%" stopColor="#60a5fa" stopOpacity={0.01} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 6" stroke="rgba(19, 30, 51, 0.8)" />

          {mondayDates.map(date => (
            <ReferenceLine key={date} x={date} stroke="#1a2640" strokeDasharray="3 3" />
          ))}

          <XAxis
            dataKey="date"
            tick={(props) => <CustomXTick {...props} chartData={chartData} />}
            tickLine={false}
            axisLine={false}
            interval={0}
            height={32}
          />
          <YAxis
            tick={{ fill: "#374151", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={v => `${v.toFixed(1)}M`}
            width={44}
            domain={[
              (min: number) => Math.floor(min * 10) / 10,
              (max: number) => Math.ceil(max * 10)  / 10,
            ]}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone" dataKey="2024"
            stroke={COLORS.y2024} strokeWidth={1.5}
            fill="none"
            dot={false} connectNulls={false}
          />
          <Area
            type="monotone" dataKey="2025"
            stroke={COLORS.y2025} strokeWidth={1.5}
            fill="url(#gradYoY2025)"
            dot={false} connectNulls={false}
          />
          <Area
            type="monotone" dataKey="2026 Actual"
            stroke={COLORS.y2026_actual} strokeWidth={2.5}
            fill="url(#gradYoY2026)"
            dot={false} connectNulls={false}
          />
          <Area
            type="monotone" dataKey="2026 Forecast"
            stroke={COLORS.y2026_predicted} strokeWidth={2}
            strokeDasharray="5 3"
            fill="none"
            dot={false} connectNulls={false}
          />
        </AreaChart>
      </ResponsiveContainer>
      </div>
    </div>
  );
}
