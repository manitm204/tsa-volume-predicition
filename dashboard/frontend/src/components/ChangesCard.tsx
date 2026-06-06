import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { ArrowUpRight, ArrowDownRight, Sparkles, type LucideIcon } from "lucide-react";
import { api } from "../api/client";
import { GlassSection } from "./ui/GlassCard";
import { fmt, fmtPct } from "../lib/format";

interface Line {
  label: string;
  value: string;
  tone: "up" | "down" | "neutral" | "info";
  sub?: string;
  Icon?: LucideIcon;
}

export function ChangesCard() {
  const { data } = useQuery({ queryKey: ["changes"], queryFn: api.changes, refetchInterval: 60_000 });

  if (!data) {
    return (
      <GlassSection title="What Changed Since Yesterday" sub="Loading run diff">
        <div className="skeleton h-32" />
      </GlassSection>
    );
  }

  if (!data.has_data) {
    return (
      <GlassSection
        title="What Changed Since Yesterday"
        sub={data.reason ?? "Awaiting two consecutive pipeline runs to compare"}
      >
        <p className="text-center py-6 text-sm text-slate-500">No comparison snapshot yet</p>
      </GlassSection>
    );
  }

  const lines: Line[] = [];

  // Total weekly-avg forecast change (our model)
  if (data.forecast_delta_passengers != null) {
    const dK = data.forecast_delta_passengers / 1000;
    lines.push({
      label: "Our weekly-avg forecast",
      value: `${dK >= 0 ? "+" : ""}${Math.round(dK).toLocaleString()}k`,
      tone: dK >= 0 ? "up" : "down",
      sub: `Net change vs. yesterday's run${data.forecast_delta_pct != null ? ` · ${fmtPct(data.forecast_delta_pct, 2, true)}` : ""}`,
    });
  }

  // New actual contribution — "TSA actual vs. predicted" surprise
  if (data.attribution?.new_actual_contribution_k != null && Math.abs(data.attribution.new_actual_contribution_k) > 0.05) {
    const v = data.attribution.new_actual_contribution_k;
    lines.push({
      label: "TSA actual surprise",
      value: `${v >= 0 ? "+" : ""}${v.toFixed(1)}k`,
      tone: v >= 0 ? "up" : "down",
      sub: "Yesterday's actual came in this much above/below what we predicted",
    });
  }

  // Forward forecast revision — model's update for remaining days
  if (data.attribution?.forecast_revision_contribution_k != null && Math.abs(data.attribution.forecast_revision_contribution_k) > 0.05) {
    const v = data.attribution.forecast_revision_contribution_k;
    lines.push({
      label: "Forward revision",
      value: `${v >= 0 ? "+" : ""}${v.toFixed(1)}k`,
      tone: v >= 0 ? "up" : "down",
      sub: "Model revised the remaining days of the week by this much",
    });
  }

  // Kalshi market move
  if (data.kalshi_delta_millions != null && Math.abs(data.kalshi_delta_millions) > 0.0005) {
    const dK = data.kalshi_delta_millions * 1000;
    lines.push({
      label: "Kalshi market move",
      value: `${dK >= 0 ? "+" : ""}${dK.toFixed(1)}k`,
      tone: "info",
      sub: `Implied weekly-avg from market${data.attribution?.kalshi_delta_pct != null ? ` (${data.attribution.kalshi_delta_pct >= 0 ? "+" : ""}${data.attribution.kalshi_delta_pct.toFixed(2)}%)` : ""}`,
    });
  }

  const bigShifts = (data.probability_shifts ?? []).filter((s) => Math.abs(s.delta) >= 0.05);

  if (lines.length === 0 && bigShifts.length === 0) {
    return (
      <GlassSection
        title="What Changed Since Yesterday"
        sub="Quiet day — model is steady"
      >
        <p className="text-center py-4 text-sm text-slate-500">No material changes vs. yesterday's run</p>
      </GlassSection>
    );
  }

  return (
    <GlassSection
      title="What Changed Since Yesterday"
      sub="Real diff between today's run and the previous run · attribution where derivable"
      right={
        <span className="chip">
          <Sparkles size={11} /> {lines.length} change{lines.length === 1 ? "" : "s"}
        </span>
      }
    >
      <div
        className={`grid gap-3 ${
          lines.length >= 3
            ? "grid-cols-2 lg:grid-cols-4"
            : "grid-cols-1 sm:grid-cols-2"
        }`}
      >
        {lines.map((l, i) => {
          const color =
            l.tone === "up" ? "text-edge-up"
            : l.tone === "down" ? "text-edge-down"
            : l.tone === "info" ? "text-edge-info"
            : "text-slate-200";
          const Arrow = l.tone === "up" ? ArrowUpRight : l.tone === "down" ? ArrowDownRight : null;
          return (
            <motion.div
              key={`${l.label}-${i}`}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.2, delay: i * 0.04 }}
              className="glass !p-3 !rounded-xl"
            >
              <div className="flex items-center justify-between mb-1">
                <span className="label !text-[10px]">{l.label}</span>
                {Arrow && <Arrow size={13} className={color} />}
              </div>
              <div className={`display text-[20px] font-bold leading-none mono ${color}`}>
                {l.value}
              </div>
              {l.sub && <div className="text-[10.5px] text-slate-500 mt-1">{l.sub}</div>}
            </motion.div>
          );
        })}
      </div>

      {bigShifts.length > 0 && (
        <div className="mt-4 pt-4 border-t border-white/[0.06]">
          <div className="label !text-[10px] mb-2">Probability shifts ≥ 5pts</div>
          <div className="flex flex-wrap gap-2">
            {bigShifts.map((s) => (
              <div key={s.strike_millions} className="glass !p-2.5 !rounded-lg flex items-center gap-2">
                <span className="text-[10px] font-bold uppercase tracking-[0.1em] text-slate-500">
                  P(over {s.strike_millions.toFixed(2)}M)
                </span>
                <span className="mono font-semibold text-[12px]" style={{ color: s.delta >= 0 ? "#22d3a4" : "#ef5466" }}>
                  {fmtPct(s.delta, 1, true)}
                </span>
                <span className="text-[10px] text-slate-500">
                  {fmtPct(s.prev_model_p_over, 0)} → {fmtPct(s.now_model_p_over, 0)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {(data.per_day_changes?.length ?? 0) > 0 && (
        <div className="mt-4 pt-4 border-t border-white/[0.06]">
          <div className="label !text-[10px] mb-2">Per-day delta</div>
          <div className="flex flex-wrap gap-2">
            {data.per_day_changes!.map((d) => {
              const delta = d.kind === "new_actual"
                ? (d.surprise ?? 0)
                : (d.delta ?? 0);
              return (
                <div
                  key={d.date}
                  className="glass !p-2.5 !rounded-lg flex items-center gap-2"
                  title={`${d.day_name} · ${d.kind === "new_actual" ? "actual arrived" : "forecast revision"}`}
                >
                  <span className="text-[10px] font-bold uppercase tracking-[0.1em] text-slate-500">
                    {d.day_name.slice(0, 3)}
                  </span>
                  <span className={`mono font-semibold text-[12px]`} style={{ color: delta >= 0 ? "#22d3a4" : "#ef5466" }}>
                    {delta >= 0 ? "+" : ""}{Math.round(delta / 1000).toLocaleString()}k
                  </span>
                  {d.kind === "new_actual" && (
                    <span className="badge badge-actual !text-[9px] !py-0">new</span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </GlassSection>
  );
}
