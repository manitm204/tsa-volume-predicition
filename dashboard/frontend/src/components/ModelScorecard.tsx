import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Trophy, Target, Crosshair, Gauge, TrendingUp, Wallet } from "lucide-react";
import { subDays, subMonths, parseISO, isAfter, startOfYear } from "date-fns";
import { api } from "../api/client";
import { GlassSection } from "./ui/GlassCard";
import { fmtMoney, fmtPct } from "../lib/format";
import { Sparkline } from "./ui/Sparkline";

interface Stat {
  label: string;
  value: string;
  sub?: string;
  tone: "up" | "down" | "info" | "warn" | "violet" | "neutral";
  Icon: typeof Trophy;
  sparkline?: (number | null)[];
}

export function ModelScorecard() {
  const { data: daily } = useQuery({
    queryKey: ["forecast-daily-scorecard"],
    queryFn: () => api.forecastDaily(subMonths(new Date(), 12).toISOString().slice(0, 10)),
    staleTime: 5 * 60_000,
  });
  const { data: history } = useQuery({ queryKey: ["order-history"], queryFn: () => api.orderHistory(500) });

  const stats: Stat[] = useMemo(() => {
    const rows = (daily ?? []).filter((d) => d.actual != null && d.predicted != null);

    const window = (days: number) => {
      const cutoff = subDays(new Date(), days);
      return rows.filter((d) => isAfter(parseISO(d.date), cutoff));
    };
    const mae = (rs: typeof rows) =>
      rs.length ? rs.reduce((s, d) => s + Math.abs(d.actual! - d.predicted!), 0) / rs.length : null;
    const bias = (rs: typeof rows) =>
      rs.length ? rs.reduce((s, d) => s + (d.actual! - d.predicted!), 0) / rs.length : null;
    const hit = (rs: typeof rows, t = 50_000) =>
      rs.length ? rs.filter((d) => Math.abs(d.actual! - d.predicted!) < t).length / rs.length : null;

    const w30 = window(30);
    const w90 = window(90);
    const wYTD = rows.filter((d) => isAfter(parseISO(d.date), startOfYear(new Date())));

    const mae30  = mae(w30);
    const mae90  = mae(w90);
    const maeYTD = mae(wYTD);
    const bias90 = bias(w90);
    const hit90  = hit(w90);

    // 30-day rolling MAE sparkline (per-day window of 7)
    const spark = rows.slice(-60).map((_, i, arr) => {
      const s = arr.slice(Math.max(0, i - 6), i + 1);
      if (s.length < 3) return null;
      return s.reduce((acc, d) => acc + Math.abs(d.actual! - d.predicted!), 0) / s.length / 1000;
    });

    // Live trading metrics from order history
    const live = (history ?? []).filter((h) => !h.dry_run && h.price != null && h.shares != null);
    const totalSpent = live.reduce((s, h) => s + (h.cost ?? 0), 0);
    const avgEdge = live.length
      ? live.reduce((s, h) => {
          if (h.model_prob != null && h.market_prob != null) {
            const e = h.side === "yes"
              ? h.model_prob - h.market_prob
              : (1 - h.model_prob) - (1 - h.market_prob);
            return s + e;
          }
          return s;
        }, 0) / live.length
      : null;
    const winRate = live.length
      ? live.filter((h) => h.model_prob != null && h.market_prob != null &&
          ((h.side === "yes" && h.model_prob > h.market_prob) ||
           (h.side === "no"  && h.model_prob < h.market_prob))).length / live.length
      : null;

    return [
      {
        label: "30D MAE",
        value: mae30 != null ? `${Math.round(mae30 / 1000)}k` : "—",
        sub: `${w30.length} days compared`,
        tone: mae30 != null && mae30 < 60_000 ? "up" : mae30 != null && mae30 < 100_000 ? "info" : "warn",
        Icon: Target,
        sparkline: spark.slice(-30),
      },
      {
        label: "90D MAE",
        value: mae90 != null ? `${Math.round(mae90 / 1000)}k` : "—",
        sub: `${w90.length} days compared`,
        tone: "info",
        Icon: Crosshair,
        sparkline: spark,
      },
      {
        label: "YTD MAE",
        value: maeYTD != null ? `${Math.round(maeYTD / 1000)}k` : "—",
        sub: `${wYTD.length} days year-to-date`,
        tone: "violet",
        Icon: Gauge,
      },
      {
        label: "Bias (90D)",
        value: bias90 != null
          ? `${bias90 >= 0 ? "+" : ""}${Math.round(bias90 / 1000)}k`
          : "—",
        sub: bias90 == null ? undefined
          : bias90 > 5_000 ? "model under-predicts"
          : bias90 < -5_000 ? "model over-predicts"
          : "well-calibrated",
        tone: bias90 == null ? "neutral"
          : Math.abs(bias90) < 5_000 ? "up"
          : bias90 > 0 ? "warn" : "down",
        Icon: Trophy,
      },
      {
        label: "Hit Rate ±50k",
        value: hit90 != null ? fmtPct(hit90, 0) : "—",
        sub: "share of days within 50k",
        tone: hit90 != null && hit90 > 0.55 ? "up" : "info",
        Icon: TrendingUp,
      },
      {
        label: "Avg Edge Captured",
        value: avgEdge != null ? fmtPct(avgEdge, 1, true) : "—",
        sub: live.length ? `${live.length} live orders` : "no live orders yet",
        tone: avgEdge != null && avgEdge > 0 ? "up" : "neutral",
        Icon: Wallet,
      },
      {
        label: "Win Rate (entry edge)",
        value: winRate != null ? fmtPct(winRate, 0) : "—",
        sub: "share of live entries with positive edge",
        tone: winRate != null && winRate > 0.6 ? "up" : "info",
        Icon: Trophy,
      },
      {
        label: "Capital Deployed",
        value: fmtMoney(totalSpent, 0),
        sub: `${live.length} live orders, all time`,
        tone: "neutral",
        Icon: Wallet,
      },
    ];
  }, [daily, history]);

  return (
    <GlassSection
      title="Model Scorecard"
      sub="Forecast accuracy + live trading performance"
    >
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {stats.map((s) => {
          const color =
            s.tone === "up"     ? "text-edge-up"
            : s.tone === "down"  ? "text-edge-down"
            : s.tone === "info"  ? "text-edge-info"
            : s.tone === "warn"  ? "text-edge-warn"
            : s.tone === "violet"? "text-edge-violet"
            : "text-slate-200";
          const spark =
            s.tone === "up"     ? "#22d3a4"
            : s.tone === "down" ? "#ef5466"
            : s.tone === "warn" ? "#f59e0b"
            : s.tone === "violet"? "#a78bfa"
            : "#60a5fa";
          return (
            <div key={s.label} className="glass glass-hover !p-4 flex flex-col gap-2 min-h-[120px]">
              <div className="flex items-center justify-between">
                <span className="label">{s.label}</span>
                <s.Icon size={13} className={color} />
              </div>
              <div className={`display text-[22px] font-bold mono leading-none ${color}`}>{s.value}</div>
              {s.sub && <div className="text-[10.5px] text-slate-500">{s.sub}</div>}
              {s.sparkline && s.sparkline.length > 2 && (
                <Sparkline data={s.sparkline} stroke={spark} fill={`${spark}33`} height={24} />
              )}
            </div>
          );
        })}
      </div>
    </GlassSection>
  );
}
