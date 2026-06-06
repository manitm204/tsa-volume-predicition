import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, CartesianGrid,
} from "recharts";
import { api } from "../api/client";
import { GlassSection } from "./ui/GlassCard";
import { distributionPoints, normalCdf } from "../lib/signal";
import { fmtPct } from "../lib/format";

/** Continuous normal-approx distribution of the model's weekly-avg forecast,
 *  with Kalshi strikes overlaid and the kalshi implied mean. */
export function ForecastDistribution() {
  const { data: week  } = useQuery({ queryKey: ["forecast-week"],  queryFn: api.forecastWeek,  refetchInterval: 60_000 });
  const { data: probs } = useQuery({ queryKey: ["probabilities"], queryFn: api.probabilities, refetchInterval: 60_000 });

  const summary = week?.summary ?? {};
  const mu      = (summary.weekly_avg_millions     as number | null | undefined) ?? null;
  const sigma   = (summary.weekly_avg_std_millions as number | null | undefined) ?? null;
  const kAvg    = week?.kalshi_avg ?? null;

  const data = useMemo(() => {
    if (mu == null || sigma == null || sigma <= 0) return [];
    return distributionPoints(mu, sigma, 100).map((d) => ({ x: d.x, p: d.pdf }));
  }, [mu, sigma]);

  // 80% confidence band: mu ± 1.282 sigma
  const ci80Lo = mu != null && sigma != null ? mu - 1.282 * sigma : null;
  const ci80Hi = mu != null && sigma != null ? mu + 1.282 * sigma : null;

  if (mu == null || sigma == null) {
    return (
      <GlassSection title="Forecast Distribution" sub="Modeled probability density of the weekly average">
        <p className="text-center py-12 text-sm text-slate-500">No forecast available yet</p>
      </GlassSection>
    );
  }

  return (
    <GlassSection
      title="Forecast Distribution"
      sub={`Normal-approx around model μ = ${mu.toFixed(3)}M · σ = ${sigma.toFixed(4)}M · 80% CI ${ci80Lo?.toFixed(3)}—${ci80Hi?.toFixed(3)}M`}
    >
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={data} margin={{ top: 10, right: 18, bottom: 4, left: 0 }}>
          <defs>
            <linearGradient id="distFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%"   stopColor="#22d3a4" stopOpacity={0.35} />
              <stop offset="100%" stopColor="#22d3a4" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 6" stroke="rgba(99,140,255,0.07)" />
          <XAxis
            dataKey="x"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={(v) => `${v.toFixed(2)}M`}
            tick={{ fill: "#475569", fontSize: 11 }}
            tickLine={false} axisLine={false}
          />
          <YAxis hide domain={[0, "dataMax"]} />
          <Tooltip
            content={({ active, payload }) =>
              active && payload?.length ? (
                <div className="glass !p-2.5 !rounded-lg text-[12px]">
                  <div className="mono text-slate-300">{(payload[0].payload.x as number).toFixed(4)}M</div>
                  <div className="text-slate-500 text-[11px]">
                    P(avg ≥) {fmtPct(1 - normalCdf(payload[0].payload.x as number, mu, sigma))}
                  </div>
                </div>
              ) : null
            }
          />

          {/* Confidence band shading via two ReferenceLines + shaded area is approximated by
              extra Area drawn between ci80Lo/Hi — recharts doesn't natively support area bands,
              so we use ReferenceLine markers for now. */}
          {ci80Lo != null && (
            <ReferenceLine x={ci80Lo} stroke="rgba(96,165,250,0.40)" strokeDasharray="3 4" />
          )}
          {ci80Hi != null && (
            <ReferenceLine x={ci80Hi} stroke="rgba(96,165,250,0.40)" strokeDasharray="3 4" />
          )}

          {/* μ line */}
          <ReferenceLine
            x={mu}
            stroke="#22d3a4"
            strokeWidth={2}
            label={{ value: `μ ${mu.toFixed(3)}M`, fill: "#22d3a4", fontSize: 11, position: "top", offset: 6 }}
          />

          {/* Kalshi implied */}
          {kAvg != null && (
            <ReferenceLine
              x={kAvg}
              stroke="#f59e0b"
              strokeWidth={2}
              strokeDasharray="4 4"
              label={{ value: `Kalshi ${kAvg.toFixed(3)}M`, fill: "#f59e0b", fontSize: 11, position: "top", offset: 6 }}
            />
          )}

          {/* Strikes */}
          {(probs ?? []).map((p) => (
            <ReferenceLine
              key={`strike-${p.threshold_millions}`}
              x={p.threshold_millions}
              stroke="rgba(167,139,250,0.30)"
              strokeDasharray="2 4"
              label={{
                value: p.threshold_millions.toFixed(2),
                fill: "rgba(167,139,250,0.70)",
                fontSize: 9.5, position: "insideBottomLeft", offset: 4,
              }}
            />
          ))}

          <Area
            type="monotone"
            dataKey="p"
            stroke="#22d3a4"
            strokeWidth={2}
            fill="url(#distFill)"
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>

      <div className="grid grid-cols-3 gap-3 mt-3">
        <DistStat
          label="Most likely outcome"
          value={`${mu.toFixed(3)}M`}
          tone="up"
        />
        <DistStat
          label="80% range width"
          value={ci80Lo != null && ci80Hi != null ? `${((ci80Hi - ci80Lo) * 1000).toFixed(0)}k` : "—"}
          tone="info"
        />
        <DistStat
          label="P(beats Kalshi)"
          value={
            kAvg != null
              ? fmtPct(1 - normalCdf(kAvg, mu, sigma), 1)
              : "—"
          }
          tone="warn"
        />
      </div>
    </GlassSection>
  );
}

function DistStat({ label, value, tone }: { label: string; value: string; tone: "up" | "info" | "warn" }) {
  const color =
    tone === "up" ? "text-edge-up" : tone === "info" ? "text-edge-info" : "text-edge-warn";
  return (
    <div className="glass !p-3 !rounded-xl">
      <div className="label !text-[10px]">{label}</div>
      <div className={`display font-bold text-lg mt-1 mono ${color}`}>{value}</div>
    </div>
  );
}
