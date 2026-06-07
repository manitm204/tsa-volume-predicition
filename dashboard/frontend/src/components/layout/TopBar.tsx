import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import { formatDistanceToNow, format } from "date-fns";
import { RefreshCw, Clock, Wifi, WifiOff, Menu } from "lucide-react";
import { StatusDot } from "../ui/StatusDot";
import { fmt, fmtPct } from "../../lib/format";

interface TopBarProps {
  title: string;
  onMenuClick?: () => void;
}

export default function TopBar({ title, onMenuClick }: TopBarProps) {
  const { data: ov, isError, refetch, isFetching } = useQuery({
    queryKey: ["overview"],
    queryFn: api.overview,
    refetchInterval: 60_000,
  });
  const { data: week } = useQuery({ queryKey: ["forecast-week"], queryFn: api.forecastWeek, refetchInterval: 60_000 });
  const { data: bank } = useQuery({ queryKey: ["bankroll"], queryFn: api.bankroll, refetchInterval: 60_000 });

  const lastRun = ov?.last_run_at ? new Date(ov.last_run_at) : null;
  const isLive  = !isError && !!ov;
  const kAvg    = week?.kalshi_avg ?? null;
  const mAvg    = (week?.summary?.weekly_avg_millions as number | null | undefined) ?? null;
  const edge    = mAvg != null && kAvg != null ? (mAvg - kAvg) * 1e6 : null;

  const tickerItems: { label: string; value: string; tone?: "up" | "down" | "warn" | "info" }[] = [
    { label: "MODEL", value: fmt(mAvg != null ? mAvg * 1e6 : null), tone: "info" },
    { label: "KALSHI", value: fmt(kAvg != null ? kAvg * 1e6 : null), tone: "warn" },
    { label: "EDGE", value: edge != null ? `${edge >= 0 ? "+" : ""}${Math.round(edge / 1000)}k` : "—", tone: (edge ?? 0) >= 0 ? "up" : "down" },
    { label: "BANKROLL", value: bank ? `$${bank.bankroll.toFixed(0)}` : "—" },
    { label: "DEPLOYED", value: bank ? fmtPct(bank.invested_pct / 100) : "—", tone: "info" },
    { label: "CASH", value: bank ? `$${bank.cash.toFixed(0)}` : "—", tone: "up" },
    { label: "ORDERS TODAY", value: ov?.orders_today?.toString() ?? "—" },
    { label: "STATUS", value: ov?.last_run_status?.toUpperCase() ?? "—", tone: ov?.last_run_status === "ok" ? "up" : "down" },
  ];
  const looped = [...tickerItems, ...tickerItems];

  const toneColor = (t?: string) => {
    if (t === "up") return "text-edge-up";
    if (t === "down") return "text-edge-down";
    if (t === "warn") return "text-edge-warn";
    if (t === "info") return "text-edge-info";
    return "text-slate-200";
  };

  return (
    <header
      className="fixed top-0 right-0 left-0 lg:left-60 z-20"
      style={{
        background: "rgba(5, 8, 15, 0.85)",
        backdropFilter: "blur(14px)",
        WebkitBackdropFilter: "blur(14px)",
        borderBottom: "1px solid rgba(99, 140, 255, 0.10)",
      }}
    >
      {/* Row 1 — title + status pills + refresh */}
      <div className="flex items-center justify-between gap-2 px-3 sm:px-4 lg:px-6 h-12">
        <div className="flex items-center gap-2 sm:gap-3 min-w-0">
          {/* Hamburger — mobile only */}
          <button
            onClick={onMenuClick}
            className="lg:hidden p-1.5 -ml-1 rounded-lg text-slate-300 hover:text-slate-100 hover:bg-white/5 transition-colors flex-shrink-0"
            aria-label="Open menu"
          >
            <Menu size={18} />
          </button>
          <h1 className="display text-[14px] sm:text-[15px] font-bold text-slate-100 tracking-tight truncate">{title}</h1>
          <span className="text-slate-700 hidden sm:inline">/</span>
          <span className="text-[11px] font-semibold uppercase tracking-[0.20em] text-slate-500 hidden sm:inline">TSA · KXTSAW</span>
        </div>

        <div className="flex items-center gap-2 sm:gap-3 flex-shrink-0">
          {ov && (
            <span
              className="flex items-center gap-1.5 text-[10.5px] sm:text-[11px] font-semibold px-2 sm:px-2.5 py-1 rounded-full"
              style={
                ov.auto_trading
                  ? { background: "rgba(34,211,164,0.12)", color: "#22d3a4", border: "1px solid rgba(34,211,164,0.30)", boxShadow: "0 0 12px rgba(34,211,164,0.15)" }
                  : { background: "rgba(71,85,105,0.30)", color: "#94a3b8", border: "1px solid rgba(71,85,105,0.40)" }
              }
            >
              <StatusDot tone={ov.auto_trading ? "live" : "off"} size={6} />
              {ov.auto_trading ? "AUTO" : "MANUAL"}
            </span>
          )}

          {lastRun && (
            <span className="hidden md:flex items-center gap-1.5 text-[11px] text-slate-400">
              <Clock size={11} />
              <span className="text-slate-300">{format(lastRun, "HH:mm 'UTC'")}</span>
              <span className="text-slate-500">· {formatDistanceToNow(lastRun, { addSuffix: true })}</span>
            </span>
          )}

          <div className="flex items-center gap-1.5">
            {isLive ? <Wifi size={12} className="text-edge-up" /> : <WifiOff size={12} className="text-edge-down" />}
            <span className="hidden sm:inline text-[11px] font-semibold" style={{ color: isLive ? "#22d3a4" : "#ef5466" }}>
              {isLive ? "LIVE" : "OFFLINE"}
            </span>
          </div>

          <button
            onClick={() => refetch()}
            className="p-1.5 rounded-lg transition-colors cursor-pointer text-slate-500 hover:text-edge-info hover:bg-white/5"
            title="Refresh"
          >
            <RefreshCw size={13} className={isFetching ? "animate-spin" : ""} />
          </button>
        </div>
      </div>

      {/* Row 2 — ticker strip (shorter on mobile) */}
      <div className="border-t border-white/[0.04] h-7 sm:h-8 flex items-center">
        <div className="ticker w-full">
          <div className="ticker-track text-[10px] sm:text-[11px]">
            {looped.map((it, i) => (
              <span key={i} className="inline-flex items-center gap-1.5">
                <span className="text-[10px] font-bold tracking-[0.16em] text-slate-600">{it.label}</span>
                <span className={`mono font-semibold ${toneColor(it.tone)}`}>{it.value}</span>
              </span>
            ))}
          </div>
        </div>
      </div>
    </header>
  );
}
