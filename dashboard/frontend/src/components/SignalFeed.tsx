import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { format, isToday, isYesterday, formatDistanceToNow, parseISO } from "date-fns";
import {
  ArrowUpRight, ArrowDownRight, RefreshCw, Clock, Crosshair, Pause,
} from "lucide-react";
import { api } from "../api/client";
import { GlassSection } from "./ui/GlassCard";
import { fmtMoney } from "../lib/format";
import type { OrderHistoryRow, OpenOrder } from "../types";

type TradeKind = "live_buy" | "live_sell" | "open_limit";

interface FeedRow {
  ts: Date;
  kind: TradeKind;
  ticker: string;
  strikeM: number | null;
  side: string; // YES / NO
  price: number | null;
  shares: number | null;
  cost: number | null;
  detail: string;
}

const kindMeta: Record<TradeKind, { label: string; tone: "up" | "down" | "info"; Icon: typeof ArrowUpRight }> = {
  live_buy:   { label: "BUY",   tone: "up",   Icon: ArrowUpRight },
  live_sell:  { label: "SELL",  tone: "down", Icon: ArrowDownRight },
  open_limit: { label: "LIMIT", tone: "info", Icon: Clock },
};

const toneStyles: Record<"up" | "down" | "info", { ring: string; text: string }> = {
  up:   { ring: "ring-edge-up/30  bg-emerald-500/10", text: "text-edge-up" },
  down: { ring: "ring-edge-down/30 bg-rose-500/10",    text: "text-edge-down" },
  info: { ring: "ring-edge-info/30 bg-sky-500/10",     text: "text-edge-info" },
};

function buildRows(history: OrderHistoryRow[] | undefined, openOrders: OpenOrder[] | undefined): FeedRow[] {
  const rows: FeedRow[] = [];

  for (const h of history ?? []) {
    if (h.dry_run || !h.run_at) continue;
    const isBuy = (h.type || "").toLowerCase().includes("buy");
    rows.push({
      ts:       parseISO(h.run_at),
      kind:     isBuy ? "live_buy" : "live_sell",
      ticker:   h.ticker,
      strikeM:  h.strike_millions,
      side:     (h.side || "").toUpperCase(),
      price:    h.price,
      shares:   h.shares,
      cost:     h.cost,
      detail:   `${h.type || ""}${h.pct != null ? ` · ${(h.pct * 100).toFixed(1)}%` : ""}`,
    });
  }

  for (const o of openOrders ?? []) {
    rows.push({
      ts:       new Date(),
      kind:     "open_limit",
      ticker:   o.ticker,
      strikeM:  o.strike_millions,
      side:     (o.side || "").toUpperCase(),
      price:    o.price,
      shares:   o.remaining,
      cost:     o.total_value,
      detail:   "resting limit",
    });
  }

  return rows.sort((a, b) => b.ts.getTime() - a.ts.getTime());
}

function bucketLabel(ts: Date, kind: TradeKind): "Open orders" | "Today" | "Yesterday" | "This week" | "Earlier" {
  if (kind === "open_limit") return "Open orders";
  if (isToday(ts)) return "Today";
  if (isYesterday(ts)) return "Yesterday";
  const ageDays = (Date.now() - ts.getTime()) / 86400000;
  if (ageDays < 7) return "This week";
  return "Earlier";
}

type TabKey = "all" | "today" | "week";

export function SignalFeed() {
  const qc = useQueryClient();
  const [tab, setTab] = useState<TabKey>("all");
  const [refreshing, setRefreshing] = useState(false);
  const [refreshedAt, setRefreshedAt] = useState<Date | null>(null);

  const { data: history } = useQuery({
    queryKey: ["order-history"],
    queryFn:  () => api.orderHistory(80),
    refetchInterval: 60_000,
  });
  const { data: openOrders } = useQuery({
    queryKey: ["open-orders"],
    queryFn:  api.openOrders,
    refetchInterval: 60_000,
  });

  const allRows = buildRows(history, openOrders);
  const rows = allRows.filter((r) => {
    if (tab === "all") return true;
    if (tab === "today") return r.kind === "open_limit" || isToday(r.ts);
    // week
    const ageDays = (Date.now() - r.ts.getTime()) / 86400000;
    return r.kind === "open_limit" || ageDays < 7;
  });

  // Group rows by bucket for visual separators
  const grouped: { bucket: string; rows: FeedRow[] }[] = [];
  for (const r of rows) {
    const b = bucketLabel(r.ts, r.kind);
    const last = grouped[grouped.length - 1];
    if (last && last.bucket === b) last.rows.push(r);
    else grouped.push({ bucket: b, rows: [r] });
  }

  const onRefresh = async () => {
    setRefreshing(true);
    try {
      await api.refreshPositions();
    } catch (e) {
      console.warn("refresh-positions failed", e);
    }
    await Promise.all([
      qc.invalidateQueries({ queryKey: ["order-history"] }),
      qc.invalidateQueries({ queryKey: ["open-orders"] }),
      qc.invalidateQueries({ queryKey: ["positions"] }),
      qc.invalidateQueries({ queryKey: ["overview"] }),
      qc.invalidateQueries({ queryKey: ["bankroll"] }),
    ]);
    setRefreshedAt(new Date());
    setRefreshing(false);
  };

  const liveCount = allRows.filter((r) => r.kind !== "open_limit").length;
  const limitCount = allRows.filter((r) => r.kind === "open_limit").length;

  return (
    <GlassSection
      className="w-full h-full flex flex-col"
      title="Signal Feed"
      sub={`${liveCount} live trade${liveCount === 1 ? "" : "s"} · ${limitCount} open limit${limitCount === 1 ? "" : "s"}`}
      right={
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-0.5 p-0.5 rounded-lg bg-white/[0.04] ring-1 ring-white/[0.06]">
            {(["all", "week", "today"] as TabKey[]).map((k) => (
              <button
                key={k}
                onClick={() => setTab(k)}
                className={`px-2 py-1 text-[10.5px] font-bold rounded-md transition-colors uppercase tracking-wide ${
                  tab === k
                    ? "bg-edge-up/20 text-edge-up"
                    : "text-slate-500 hover:text-slate-300"
                }`}
              >
                {k === "all" ? "All" : k === "week" ? "Week" : "Today"}
              </button>
            ))}
          </div>
          <button
            onClick={onRefresh}
            disabled={refreshing}
            title={refreshedAt ? `Refreshed ${formatDistanceToNow(refreshedAt)} ago` : "Refresh positions from Kalshi"}
            className="px-2 py-1 rounded-md text-[10.5px] font-bold ring-1 ring-white/[0.08] bg-white/[0.04] text-slate-300 hover:text-edge-up hover:ring-edge-up/40 disabled:opacity-50 flex items-center gap-1"
          >
            <RefreshCw size={11} className={refreshing ? "animate-spin" : ""} />
            {refreshing ? "Syncing" : "Refresh"}
          </button>
        </div>
      }
    >
      {rows.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center text-slate-500 py-8">
          <Pause size={18} className="mb-2 opacity-60" />
          <p className="text-sm">No trades in this window</p>
          <p className="text-[11px] text-slate-600 mt-1">Live orders & resting limits will appear here</p>
        </div>
      ) : (
        <div className="flex-1 min-h-0 overflow-y-auto pr-1 space-y-3">
          {grouped.map((g) => (
            <div key={g.bucket}>
              <div className="text-[10px] font-bold uppercase tracking-[0.15em] text-slate-500 mb-1.5 px-1">
                {g.bucket}
              </div>
              <div className="space-y-1">
                {g.rows.map((r, idx) => {
                  const meta = kindMeta[r.kind];
                  const style = toneStyles[meta.tone];
                  const sideTone = r.side === "YES" ? "text-edge-up" : r.side === "NO" ? "text-edge-down" : "text-slate-300";
                  return (
                    <motion.div
                      key={`${r.ticker}-${r.ts.toISOString()}-${idx}`}
                      initial={{ opacity: 0, x: 6 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ duration: 0.18, delay: Math.min(idx * 0.02, 0.1) }}
                      className="flex items-center gap-3 py-1.5 px-2 rounded-lg hover:bg-white/[0.03]"
                    >
                      <div className={`w-7 h-7 rounded-lg flex items-center justify-center ring-1 ${style.ring} flex-shrink-0`}>
                        {r.kind === "open_limit" ? (
                          <Clock size={13} className={style.text} />
                        ) : r.kind === "live_buy" ? (
                          <Crosshair size={13} className={style.text} />
                        ) : (
                          <ArrowDownRight size={13} className={style.text} />
                        )}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-baseline gap-1.5 text-[13px] font-semibold">
                          <span className={style.text}>{meta.label}</span>
                          <span className={sideTone}>{r.side}</span>
                          {r.strikeM != null && (
                            <span className="text-slate-200 mono">{r.strikeM.toFixed(2)}M</span>
                          )}
                        </div>
                        <div className="text-[11px] text-slate-500 truncate mono">
                          {r.shares ?? "—"}c @ {r.price != null ? `${(r.price * 100).toFixed(0)}¢` : "—"} · {fmtMoney(r.cost)}
                          {r.kind !== "open_limit" && (
                            <span className="text-slate-600 ml-1">· {format(r.ts, "HH:mm")}</span>
                          )}
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </GlassSection>
  );
}
