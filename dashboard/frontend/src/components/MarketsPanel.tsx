import { motion } from "framer-motion";
import { GlassSection } from "./ui/GlassCard";
import { fmtPct } from "../lib/format";
import type { Position, OpenOrder, OrderBookRow } from "../types";

function PriceCell({ v }: { v: number | null | undefined }) {
  return v != null
    ? <span className="mono text-slate-300">{v.toFixed(2)}</span>
    : <span className="text-slate-700">—</span>;
}

function PnlVal({ v }: { v: number | null | undefined }) {
  if (v == null) return <span className="text-slate-700">—</span>;
  return (
    <span className="mono font-semibold" style={{ color: v >= 0 ? "#22d3a4" : "#ef5466" }}>
      {v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}
    </span>
  );
}

function SideBadge({ side }: { side?: string }) {
  return <span className={side === "yes" ? "badge badge-yes" : "badge badge-no"}>{side === "yes" ? "YES" : "NO"}</span>;
}

function PosBadge({ isYes }: { isYes: boolean }) {
  return <span className={isYes ? "badge badge-over" : "badge badge-under"}>{isYes ? "OVER" : "UNDER"}</span>;
}

interface Props {
  scopeLabel: string; // "KXTSAW" | "KXTRUFTSA-26JUN05" — used in copy
  positions: Position[] | undefined;
  openOrders: OpenOrder[] | undefined;
  orderbook: OrderBookRow[] | undefined;
}

export function MarketsPanel({ scopeLabel, positions, openOrders, orderbook }: Props) {
  // Group open orders by (strike, side); show the highest-priced as next-to-fill,
  // aggregate the rest as "behind" + cumulative reserved.
  const groups = new Map<string, { top: OpenOrder; behind: number; resBehind: number }>();
  for (const o of openOrders ?? []) {
    const key = `${o.strike_millions?.toFixed(2)}-${o.side}`;
    const prev = groups.get(key);
    if (!prev) {
      groups.set(key, { top: o, behind: 0, resBehind: 0 });
    } else if (o.price > prev.top.price) {
      groups.set(key, {
        top: o,
        behind: prev.behind + 1,
        resBehind: prev.resBehind + prev.top.total_value,
      });
    } else {
      groups.set(key, { ...prev, behind: prev.behind + 1, resBehind: prev.resBehind + o.total_value });
    }
  }
  const grouped = [...groups.values()].sort(
    (a, b) => (b.top.strike_millions ?? 0) - (a.top.strike_millions ?? 0),
  );
  const totalOrders = openOrders?.length ?? 0;

  return (
    <>
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-5">
        <GlassSection
          title="Open Positions"
          sub={`Contracts held in ${scopeLabel} · mark-to-market`}
          right={<span className="chip">{positions?.length ?? 0} active</span>}
        >
          {!positions?.length ? (
            <p className="text-center py-8 text-sm text-slate-600">No open positions</p>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  {["Strike", "Position", "Shares", "Avg", "Mark", "Invested", "P&L"].map((h) => (
                    <th key={h} className="label">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {positions.map((p, i) => {
                  const isYes = p.yes_shares > 0;
                  return (
                    <motion.tr
                      key={i}
                      initial={{ opacity: 0, y: 4 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.16, delay: i * 0.03 }}
                    >
                      <td className="mono font-semibold text-slate-100">{p.strike_millions?.toFixed(2)}M</td>
                      <td><PosBadge isYes={isYes} /></td>
                      <td className="mono text-slate-400">{isYes ? p.yes_shares : p.no_shares}</td>
                      <td><PriceCell v={p.avg_price} /></td>
                      <td><PriceCell v={isYes ? p.yes_current_price : p.no_current_price} /></td>
                      <td className="mono text-slate-500">${p.cost_dollars.toFixed(2)}</td>
                      <td><PnlVal v={p.unrealized_pnl} /></td>
                    </motion.tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </GlassSection>

        <GlassSection
          title="Open Limit Orders"
          sub={`Next-to-fill per strike × side · ${totalOrders} order${totalOrders === 1 ? "" : "s"} resting`}
          right={<span className="chip">{grouped.length} markets</span>}
        >
          {!grouped.length ? (
            <p className="text-center py-8 text-sm text-slate-600">No open orders</p>
          ) : (
            <table className="data-table">
              <thead>
                <tr>
                  {["Strike", "Side", "Remain", "Next Limit", "Current", "Reserved"].map((h) => (
                    <th key={h} className="label">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {grouped.map((g, i) => (
                  <tr key={i}>
                    <td className="mono font-semibold text-slate-100">{g.top.strike_millions?.toFixed(2)}M</td>
                    <td><SideBadge side={g.top.side} /></td>
                    <td className="mono text-slate-400">{g.top.remaining}</td>
                    <td>
                      <span className="mono text-slate-300">{g.top.price.toFixed(2)}</span>
                      {g.behind > 0 && (
                        <span className="ml-2 text-[10px] text-slate-600">+{g.behind} behind</span>
                      )}
                    </td>
                    <td><PriceCell v={g.top.current_price} /></td>
                    <td className="mono text-slate-500">
                      ${(g.top.total_value + g.resBehind).toFixed(2)}
                      {g.behind > 0 && (
                        <span className="ml-1 text-[10px] text-slate-600">total</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </GlassSection>
      </div>

      {!!orderbook?.length && (
        <GlassSection
          title="Market Order Book"
          sub="Live bid/ask for each strike · highlighted rows = ≥3% edge"
          right={<span className="chip">{orderbook.length} markets</span>}
        >
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  {["Strike", "YES Bid", "YES Ask", "Mid", "Market %", "Model %", "Edge", "Spread", "Vol"].map((h) => (
                    <th key={h} className="label">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {orderbook.map((row, i) => {
                  const edge    = row.model_prob != null && row.market_prob != null ? (row.model_prob - row.market_prob) : null;
                  const hasEdge = edge != null && Math.abs(edge) >= 0.03;
                  return (
                    <tr key={i} style={hasEdge ? { background: "rgba(34,211,164,0.04)" } : {}}>
                      <td className="mono font-semibold text-slate-100">{row.strike_millions?.toFixed(2)}M</td>
                      <td className="mono text-edge-up">{row.yes_bid != null ? row.yes_bid.toFixed(2) : "—"}</td>
                      <td className="mono text-edge-down">{row.yes_ask != null ? row.yes_ask.toFixed(2) : "—"}</td>
                      <td className="mono text-slate-400">{row.yes_mid != null ? row.yes_mid.toFixed(2) : "—"}</td>
                      <td className="mono text-slate-500">{fmtPct(row.market_prob)}</td>
                      <td className="mono font-semibold" style={{ color: hasEdge ? "#22d3a4" : "#94a3b8" }}>
                        {fmtPct(row.model_prob)}
                      </td>
                      <td>
                        {edge != null
                          ? <span className={hasEdge ? "edge-strong mono" : edge < 0 ? "edge-neg mono" : "edge-neutral mono"}>
                              {fmtPct(edge, 1, true)}
                            </span>
                          : <span className="text-slate-700">—</span>}
                      </td>
                      <td className="mono text-slate-600">{row.spread != null ? row.spread.toFixed(2) : "—"}</td>
                      <td className="mono text-slate-600">{row.volume != null ? Math.round(row.volume).toLocaleString() : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </GlassSection>
      )}
    </>
  );
}
