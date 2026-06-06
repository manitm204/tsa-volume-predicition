import { Area, AreaChart, ResponsiveContainer, YAxis } from "recharts";

interface Props {
  data: (number | null | undefined)[];
  stroke?: string;
  fill?: string;
  height?: number;
  width?: number | string;
}

export function Sparkline({
  data, stroke = "#22d3a4", fill = "rgba(34,211,164,0.18)", height = 32, width = "100%",
}: Props) {
  const pts = data
    .map((v, i) => ({ i, v: v == null ? null : Number(v) }))
    .filter((p) => p.v != null) as { i: number; v: number }[];

  if (pts.length < 2) {
    return <div style={{ height, width }} className="opacity-30 text-[10px] text-slate-500">no data</div>;
  }

  const gradId = `spark-${Math.random().toString(36).slice(2, 8)}`;

  return (
    <div style={{ width, height }}>
      <ResponsiveContainer>
        <AreaChart data={pts} margin={{ top: 2, right: 0, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor={fill}   stopOpacity={1} />
              <stop offset="100%" stopColor={fill}   stopOpacity={0} />
            </linearGradient>
          </defs>
          <YAxis hide domain={["dataMin", "dataMax"]} />
          <Area
            type="monotone"
            dataKey="v"
            stroke={stroke}
            strokeWidth={1.6}
            fill={`url(#${gradId})`}
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
