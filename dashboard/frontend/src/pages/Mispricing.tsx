import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ComposedChart, Line, Scatter, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine, CartesianGrid, ZAxis,
} from "recharts";
import { api } from "../api/client";
import { PageHeader } from "../components/PageHeader";
import { GlassSection } from "../components/ui/GlassCard";
import { fmtPct } from "../lib/format";
import type { MispricingRow } from "../types";

type View = "survival" | "bell";

export default function Mispricing() {
  const { data } = useQuery({
    queryKey: ["mispricing"],
    queryFn: api.mispricing,
    refetchInterval: 60_000,
  });
  const [view, setView] = useState<View>("survival");
  const [showModel, setShowModel] = useState(false);

  const mu = data?.mu ?? null;
  const sigma = data?.sigma ?? null;
  const hasModel = data?.model_mu != null && data?.model_sigma != null;

  // Scatter points — the actual market mids.
  const survivalPts = useMemo(
    () => (data?.points ?? []).map((p) => ({
      x: p.strike, y: p.market_prob, fitted: p.fitted_prob,
      ticker: p.ticker, residual: p.residual,
    })),
    [data],
  );
  const densityPts = useMemo(
    () => (data?.density_points ?? []).map((d) => ({ x: d.x, y: d.density })),
    [data],
  );

  if (data && !data.has_data) {
    return (
      <>
        <PageHeader title="Kalshi Mispricing" description="Normal fit to the market's own implied probabilities" />
        <GlassSection title="No fit available" sub={data.reason ?? "Not enough quoted strikes to fit a distribution"}>
          <p className="text-center py-12 text-sm text-slate-500">
            Waiting for a weekly event with at least 3 quoted strikes.
          </p>
        </GlassSection>
      </>
    );
  }

  const curve = view === "survival" ? (data?.survival_curve ?? []) : (data?.pdf_curve ?? []);
  const scatter = view === "survival" ? survivalPts : densityPts;
  const modelCurve = view === "survival" ? (data?.model_survival_curve ?? []) : (data?.model_pdf_curve ?? []);
  const modelPts = view === "survival" ? (data?.model_survival_points ?? []) : [];
  const worst = (data?.mispricings ?? []).reduce<MispricingRow | undefined>(
    (a, b) => (b.abs_residual > (a?.abs_residual ?? -1) ? b : a),
    undefined,
  );

  return (
    <>
      <PageHeader
        title="Kalshi Mispricing"
        description={`Normal distribution fit (least-squares) to ${data?.event_ticker ?? "the active weekly event"}'s implied survival probabilities, with the actual mid prices overlaid.`}
        kpis={[
          { label: "Fitted μ (mean vol)", value: mu != null ? `${mu.toFixed(3)}M` : "—", accent: "#22d3a4" },
          { label: "Fitted σ", value: sigma != null ? `${(sigma * 1000).toFixed(0)}k` : "—", accent: "#60a5fa" },
          { label: "Strikes fit", value: data?.n_points ?? "—" },
          {
            label: "Largest gap",
            value: worst ? `${(worst.abs_residual * 100).toFixed(1)}pp` : "—",
            sub: worst ? `${worst.strike.toFixed(2)}M · ${worst.direction}` : undefined,
            accent: worst ? (worst.direction === "rich" ? "#f59e0b" : "#60a5fa") : undefined,
          },
        ]}
      />

      <GlassSection
        title={view === "survival" ? "Survival curve — P(volume > strike)" : "Implied density (bell)"}
        sub={
          view === "survival"
            ? "Dots are Kalshi mid prices; the line is the normal fitted to them. Dots off the line are mispriced."
            : "Dots are per-strike implied density (differenced mids); the line is the fit put through the same differencing — so dots off the line are genuinely mispriced."
        }
        right={
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowModel((v) => !v)}
              disabled={!hasModel}
              title={hasModel ? "Overlay my model's prediction" : "No model forecast available"}
              className={`px-3 py-1.5 rounded-lg border text-[11px] font-semibold transition-colors disabled:opacity-40 disabled:cursor-not-allowed ${
                showModel && hasModel
                  ? "border-sky-400/40 bg-sky-500/20 text-sky-300"
                  : "border-white/10 text-slate-400 hover:text-slate-200"
              }`}
            >
              My model
            </button>
            <div className="flex rounded-lg overflow-hidden border border-white/10 text-[11px] font-semibold">
              <button
                onClick={() => setView("survival")}
                className={`px-3 py-1.5 transition-colors ${view === "survival" ? "bg-emerald-500/20 text-edge-up" : "text-slate-400 hover:text-slate-200"}`}
              >
                Survival
              </button>
              <button
                onClick={() => setView("bell")}
                className={`px-3 py-1.5 transition-colors ${view === "bell" ? "bg-emerald-500/20 text-edge-up" : "text-slate-400 hover:text-slate-200"}`}
              >
                Bell
              </button>
            </div>
          </div>
        }
      >
        <div className="h-[280px] sm:h-[360px]">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart margin={{ top: 10, right: 18, bottom: 4, left: 0 }}>
              <CartesianGrid strokeDasharray="3 6" stroke="rgba(99,140,255,0.07)" />
              <XAxis
                dataKey="x"
                type="number"
                domain={["dataMin", "dataMax"]}
                tickFormatter={(v) => `${Number(v).toFixed(2)}M`}
                tick={{ fill: "#475569", fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                allowDuplicatedCategory={false}
              />
              <YAxis
                type="number"
                domain={view === "survival" ? [0, 1] : [0, "dataMax"]}
                tickFormatter={(v) => (view === "survival" ? `${Math.round(Number(v) * 100)}%` : Number(v).toFixed(0))}
                tick={{ fill: "#475569", fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                width={38}
              />
              <ZAxis range={[60, 60]} />
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const pt = payload.find((p) => p.name === "market");
                  const d = (pt ?? payload[0]).payload as Record<string, number | string>;
                  return (
                    <div className="glass !p-2.5 !rounded-lg text-[12px]">
                      <div className="mono text-slate-300">{Number(d.x).toFixed(3)}M</div>
                      {view === "survival" && d.fitted != null ? (
                        <>
                          <div className="text-slate-400 text-[11px]">market {fmtPct(Number(d.y))}</div>
                          <div className="text-slate-500 text-[11px]">fitted {fmtPct(Number(d.fitted))}</div>
                          <div className={`text-[11px] ${Number(d.residual) >= 0 ? "text-edge-warn" : "text-edge-info"}`}>
                            resid {(Number(d.residual) * 100 >= 0 ? "+" : "")}{(Number(d.residual) * 100).toFixed(1)}pp
                          </div>
                        </>
                      ) : (
                        <div className="text-slate-400 text-[11px]">density {Number(d.y).toFixed(2)}</div>
                      )}
                    </div>
                  );
                }}
              />

              {mu != null && (
                <ReferenceLine
                  x={mu}
                  stroke="#22d3a4"
                  strokeWidth={1.5}
                  strokeDasharray="4 4"
                  label={{ value: `μ ${mu.toFixed(3)}M`, fill: "#22d3a4", fontSize: 10.5, position: "top" }}
                />
              )}

              {/* Fitted normal curve */}
              <Line
                data={curve}
                dataKey="y"
                name="fitted"
                stroke="#22d3a4"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />

              {/* Actual market mid points */}
              <Scatter
                data={scatter}
                dataKey="y"
                name="market"
                fill="#f59e0b"
                isAnimationActive={false}
              />

              {/* My model overlay */}
              {showModel && hasModel && (
                <Line
                  data={modelCurve}
                  dataKey="y"
                  name="model"
                  stroke="#60a5fa"
                  strokeWidth={2}
                  strokeDasharray="5 4"
                  dot={false}
                  isAnimationActive={false}
                />
              )}
              {showModel && hasModel && view === "survival" && (
                <Scatter
                  data={modelPts}
                  dataKey="y"
                  name="model-pts"
                  fill="#60a5fa"
                  isAnimationActive={false}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        <div className="flex flex-wrap items-center gap-x-5 gap-y-1.5 mt-3 text-[11px] text-slate-400">
          <span className="flex items-center gap-1.5"><span className="inline-block w-4 h-0.5 bg-[#22d3a4]" />Kalshi fit</span>
          <span className="flex items-center gap-1.5"><span className="inline-block w-2 h-2 rounded-full bg-[#f59e0b]" />Kalshi mid</span>
          {showModel && hasModel && (
            <>
              <span className="flex items-center gap-1.5"><span className="inline-block w-4 h-0.5 border-t-2 border-dashed border-[#60a5fa]" />My model</span>
              <span className="text-slate-500">
                μ {data?.model_mu?.toFixed(3)}M · σ {data?.model_sigma != null ? `${(data.model_sigma * 1000).toFixed(0)}k` : "—"}
              </span>
            </>
          )}
        </div>
      </GlassSection>

      <GlassSection
        title="Mispricings"
        sub="Strikes ranked by how far the Kalshi mid sits from the fitted normal. 'Rich' = market prices 'above' higher than the fit; 'cheap' = lower."
        className="mt-4"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-[12.5px]">
            <thead>
              <tr className="text-left text-slate-500 border-b border-white/10">
                <th className="py-2 pr-3 font-semibold">Strike</th>
                <th className="py-2 pr-3 font-semibold text-right">Market</th>
                <th className="py-2 pr-3 font-semibold text-right">Fitted</th>
                <th className="py-2 pr-3 font-semibold text-right">Gap</th>
                <th className="py-2 pr-3 font-semibold text-right">Dir</th>
                <th className="py-2 pr-3 font-semibold text-right">Vol</th>
              </tr>
            </thead>
            <tbody className="mono">
              {(data?.mispricings ?? []).map((m) => (
                <tr key={m.ticker} className="border-b border-white/[0.04]">
                  <td className="py-1.5 pr-3 text-slate-300">{m.strike.toFixed(2)}M</td>
                  <td className="py-1.5 pr-3 text-right text-slate-200">{fmtPct(m.market_prob)}</td>
                  <td className="py-1.5 pr-3 text-right text-slate-400">{fmtPct(m.fitted_prob)}</td>
                  <td className={`py-1.5 pr-3 text-right font-semibold ${m.residual >= 0 ? "text-edge-warn" : "text-edge-info"}`}>
                    {m.residual >= 0 ? "+" : ""}{(m.residual * 100).toFixed(1)}pp
                  </td>
                  <td className="py-1.5 pr-3 text-right">
                    <span className={m.direction === "rich" ? "text-edge-warn" : "text-edge-info"}>
                      {m.direction}
                    </span>
                  </td>
                  <td className="py-1.5 pr-3 text-right text-slate-500">{m.volume != null ? m.volume.toFixed(0) : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </GlassSection>
    </>
  );
}
