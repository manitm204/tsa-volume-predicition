import { useQuery } from "@tanstack/react-query";
import { Cloud, CloudRain, CloudSnow, Sun, Wind } from "lucide-react";
import { api } from "../api/client";
import { GlassSection } from "./ui/GlassCard";

const iconFor = (penalty: number | null | undefined) => {
  if (penalty == null) return Sun;
  if (penalty < 0.10) return Sun;
  if (penalty < 0.30) return Cloud;
  if (penalty < 0.60) return CloudRain;
  return CloudSnow;
};

const colorFor = (penalty: number | null | undefined) => {
  if (penalty == null) return { text: "text-slate-400", glow: "" };
  if (penalty < 0.10) return { text: "text-edge-up",     glow: "shadow-glow-emerald" };
  if (penalty < 0.30) return { text: "text-edge-info",   glow: "shadow-glow-blue" };
  if (penalty < 0.60) return { text: "text-edge-warn",   glow: "shadow-glow-amber" };
  return { text: "text-edge-down", glow: "shadow-glow-red" };
};

export function WeatherImpactCard() {
  const { data } = useQuery({ queryKey: ["weather-impact"], queryFn: api.weatherImpact, refetchInterval: 5 * 60_000 });

  if (!data) {
    return (
      <GlassSection title="Weather Impact" sub="Hub-weighted weather drag on this week's forecast">
        <div className="skeleton h-40" />
      </GlassSection>
    );
  }
  if (!data.has_data) {
    return (
      <GlassSection title="Weather Impact" sub={data.reason ?? "Weather feed not available"}>
        <p className="text-center py-6 text-sm text-slate-500">No weather data</p>
      </GlassSection>
    );
  }

  const penalty = data.composite_penalty ?? null;
  const trailing = data.trailing_penalty ?? null;
  const delta = data.penalty_delta ?? null;
  const impactK = data.approx_volume_impact_k ?? null;

  const Icon = iconFor(penalty);
  const { text: color, glow } = colorFor(penalty);

  return (
    <GlassSection
      title="Weather Impact"
      sub="16-hub composite storm + snowfall load · vs. trailing 4-week mean"
    >
      <div className="grid grid-cols-[auto_1fr] gap-5 items-center">
        <div className={`w-20 h-20 rounded-2xl flex items-center justify-center glass !p-0 ${glow}`}>
          <Icon size={36} className={color} />
        </div>

        <div className="min-w-0">
          <div className={`display text-xl font-bold ${color}`}>{data.qualitative_label}</div>
          <div className="text-[12px] text-slate-400 mt-1">
            Composite penalty <span className="mono text-slate-200">{penalty?.toFixed(3) ?? "—"}</span>
            {trailing != null && (
              <> · trailing 4w avg <span className="mono text-slate-500">{trailing.toFixed(3)}</span></>
            )}
            {delta != null && (
              <> · <span className="mono" style={{ color: delta <= 0 ? "#22d3a4" : "#ef5466" }}>
                {delta >= 0 ? "+" : ""}{delta.toFixed(3)} vs trailing
              </span></>
            )}
          </div>
        </div>
      </div>

      {/* Metric row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mt-5">
        <WStat
          Icon={CloudSnow}
          label="Hubs Snowing"
          now={data.current_week_mean?.n_hubs_snowing}
          prior={data.trailing_4w_mean?.n_hubs_snowing}
          digits={1}
        />
        <WStat
          Icon={CloudSnow}
          label="Snowfall (wt avg)"
          now={data.current_week_mean?.wt_avg_snowfall_sum}
          prior={data.trailing_4w_mean?.wt_avg_snowfall_sum}
          digits={2}
        />
        <WStat
          Icon={Wind}
          label="Storm Impact"
          now={data.current_week_mean?.vol_wtd_storm_impact}
          prior={data.trailing_4w_mean?.vol_wtd_storm_impact}
          digits={3}
        />
        <WStat
          Icon={CloudRain}
          label="Top-3 Hubs Snowfall"
          now={data.current_week_mean?.top3_hubs_snowfall_mean}
          prior={data.trailing_4w_mean?.top3_hubs_snowfall_mean}
          digits={2}
        />
      </div>

      {/* Estimated volume effect */}
      {impactK != null && (
        <div className="mt-4 p-3 rounded-xl flex items-center gap-3"
          style={{
            background: "rgba(96,140,255,0.04)",
            border: "1px solid rgba(99,140,255,0.10)",
          }}
        >
          <Cloud size={16} className={color} />
          <div className="flex-1">
            <div className="text-[12px] text-slate-300">
              Estimated per-day volume effect ·{" "}
              <span className="mono font-semibold" style={{ color: impactK < 0 ? "#ef5466" : "#22d3a4" }}>
                {impactK >= 0 ? "+" : ""}{impactK.toFixed(1)}k
              </span>{" "}
              <span className="text-slate-500">(heuristic — 6k per unit of composite penalty)</span>
            </div>
          </div>
        </div>
      )}
    </GlassSection>
  );
}

function WStat({
  Icon, label, now, prior, digits,
}: {
  Icon: typeof Sun;
  label: string;
  now: number | null | undefined;
  prior: number | null | undefined;
  digits: number;
}) {
  const nowVal = now == null ? "—" : now.toFixed(digits);
  const priorVal = prior == null ? null : prior.toFixed(digits);
  return (
    <div className="glass !p-3 !rounded-xl">
      <div className="flex items-center justify-between mb-1">
        <span className="label !text-[10px]">{label}</span>
        <Icon size={12} className="text-slate-500" />
      </div>
      <div className="mono display text-[16px] font-bold text-slate-100">{nowVal}</div>
      {priorVal != null && (
        <div className="text-[10.5px] text-slate-500 mt-0.5">
          vs <span className="mono">{priorVal}</span> trailing
        </div>
      )}
    </div>
  );
}
