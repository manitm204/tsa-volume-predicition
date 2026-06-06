import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import {
  CalendarHeart, TrendingUp, BarChart3, Activity, History, CloudSun, HelpCircle,
  type LucideIcon,
} from "lucide-react";
import { formatDistanceToNow, parseISO } from "date-fns";
import { api } from "../api/client";
import { GlassSection } from "./ui/GlassCard";
import type { FeatureAttribution } from "../types";

/**
 * "Why the Model Predicts This" — real leave-one-out attributions computed by
 * autogluon_predict.py against the production AutoGluon ensemble. Each feature's
 * value is replaced by the training-set median; the resulting prediction delta
 * across the week's predicted days, divided by 7, is its signed contribution
 * to the weekly average.
 */

const CATEGORY_ICON: Record<FeatureAttribution["category"], LucideIcon> = {
  history:  History,
  momentum: TrendingUp,
  calendar: CalendarHeart,
  trend:    Activity,
  weather:  CloudSun,
  other:    HelpCircle,
};

export function ModelExplainability() {
  const { data, isLoading } = useQuery({
    queryKey: ["feature-attributions"],
    queryFn: api.featureAttributions,
    refetchInterval: 5 * 60_000,
    staleTime:       60_000,
  });

  if (isLoading || !data) {
    return (
      <GlassSection title="Why the Model Predicts This" sub="Loading leave-one-out attributions">
        <div className="skeleton h-48" />
      </GlassSection>
    );
  }

  if (!data.has_data || !data.drivers.length) {
    return (
      <GlassSection
        title="Why the Model Predicts This"
        sub={data.reason ?? "Attributions not available"}
      >
        <p className="text-center py-6 text-sm text-slate-500">
          {data.reason ?? "No attributions yet — run the daily pipeline."}
        </p>
      </GlassSection>
    );
  }

  const drivers = data.drivers;
  const maxAbs  = Math.max(...drivers.map((d) => Math.abs(d.contribution_passengers)), 1);
  const lastRun = data.computed_at ? parseISO(data.computed_at) : null;

  return (
    <GlassSection
      title="Why the Model Predicts This"
      sub="Real LOO ablation against the trained AutoGluon model · bars sized by impact on weekly avg"
      right={<span className="chip"><BarChart3 size={11} /> {drivers.length} drivers</span>}
    >
      <div className="space-y-3">
        {drivers.map((d, i) => {
          const pct  = (Math.abs(d.contribution_passengers) / maxAbs) * 100;
          const isUp = d.contribution_passengers >= 0;
          const fillClass = isUp
            ? "bg-gradient-to-r from-emerald-500/80 to-emerald-300/70"
            : "bg-gradient-to-r from-rose-500/80 to-rose-300/70";
          const Icon = CATEGORY_ICON[d.category] ?? HelpCircle;
          const k    = Math.round(Math.abs(d.contribution_passengers) / 1000 * 10) / 10;
          return (
            <motion.div
              key={d.feature}
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.22, delay: i * 0.05 }}
              className="grid grid-cols-[170px_1fr_64px] gap-3 items-center"
            >
              <div className="flex items-center gap-2 min-w-0">
                <div className="w-7 h-7 rounded-lg flex items-center justify-center bg-white/[0.04] ring-1 ring-white/[0.06] flex-shrink-0">
                  <Icon size={13} className="text-edge-info" />
                </div>
                <div className="min-w-0">
                  <div
                    className="text-[12.5px] font-semibold text-slate-200 truncate"
                    title={d.feature}
                  >
                    {d.label}
                  </div>
                  <div className="text-[10.5px] text-slate-500 truncate" title={d.reason}>
                    {d.reason}
                  </div>
                </div>
              </div>

              <div className="relative h-2.5 rounded-full bg-surface-700/60 overflow-hidden">
                <div
                  className={`absolute inset-y-0 left-0 rounded-full ${fillClass}`}
                  style={{ width: `${pct}%` }}
                />
              </div>

              <div
                className="text-right mono text-[12px] font-semibold"
                style={{ color: isUp ? "#22d3a4" : "#ef5466" }}
                title={`Rank #${d.importance_rank} · global importance ${Math.round(d.importance_passengers / 1000)}k`}
              >
                {isUp ? "+" : "−"}{k.toFixed(1)}k
              </div>
            </motion.div>
          );
        })}
      </div>

      <p className="text-[10.5px] text-slate-500 mt-4">
        <span className="text-slate-400 font-semibold">Method:</span>{" "}
        {data.method ?? "Leave-one-out ablation"}
        {lastRun && (
          <>
            {" · "}
            <span title={lastRun.toISOString()}>computed {formatDistanceToNow(lastRun, { addSuffix: true })}</span>
          </>
        )}
      </p>
    </GlassSection>
  );
}
