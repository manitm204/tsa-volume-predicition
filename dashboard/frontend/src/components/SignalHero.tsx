import { motion } from "framer-motion";
import { ArrowUpRight, ArrowDownRight, Zap, Target, Coins, ShieldCheck } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { GlassCard } from "./ui/GlassCard";
import { GradePill } from "./ui/GradePill";
import { ProgressBar } from "./ui/ProgressBar";
import { StatusDot } from "./ui/StatusDot";
import { buildOpportunities, confidenceFromStd, gradeSignal } from "../lib/signal";
import { fmt, fmtMoney, fmtPct, signed } from "../lib/format";

export function SignalHero() {
  const { data: ov }    = useQuery({ queryKey: ["overview"],      queryFn: api.overview,      refetchInterval: 60_000 });
  const { data: week }  = useQuery({ queryKey: ["forecast-week"], queryFn: api.forecastWeek,  refetchInterval: 60_000 });
  const { data: probs } = useQuery({ queryKey: ["probabilities"], queryFn: api.probabilities, refetchInterval: 60_000 });

  const summary = week?.summary ?? {};
  const modelAvg  = (summary.weekly_avg_millions     as number | null | undefined) ?? null;
  const stdM      = (summary.weekly_avg_std_millions as number | null | undefined) ?? null;
  const kalshiAvg = week?.kalshi_avg ?? null;

  const edgeM   = modelAvg != null && kalshiAvg != null ? modelAvg - kalshiAvg : null;
  const edgeK   = edgeM != null ? edgeM * 1000 : null;
  const conf    = confidenceFromStd(stdM);

  const opps    = buildOpportunities(probs, ov?.bankroll ?? 250);
  const best    = opps[0] ?? null;
  const grade   = gradeSignal(best?.edge ?? 0, best?.kellyFraction ?? 0, conf);

  const recAction = best
    ? `BUY ${best.side} ${best.strike.toFixed(2)}M`
    : "NO ACTION";
  const recDir = best?.side === "OVER";

  const expectedROI = best?.expectedValue ?? null;
  const kellySize   = best?.kellyStake ?? null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
    >
      <GlassCard strong bordered glow="emerald" padded={false} className="relative overflow-hidden animate-glow">
        {/* subtle aurora background */}
        <div
          className="absolute inset-0 pointer-events-none"
          aria-hidden
          style={{
            background:
              "radial-gradient(600px 240px at 0% 0%, rgba(34,211,164,0.10), transparent 60%)," +
              "radial-gradient(700px 260px at 100% 100%, rgba(96,165,250,0.10), transparent 60%)",
          }}
        />

        <div className="relative p-7">
          {/* Eyebrow */}
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-2">
              <Zap size={14} className="text-edge-up" />
              <span className="label !text-edge-up">THIS WEEK&apos;S SIGNAL</span>
              <StatusDot tone="live" size={6} className="ml-1" />
            </div>
            <span className="text-[11px] text-slate-500">
              {best
                ? `Best of ${opps.filter((o) => o.signal !== "skip").length} active markets`
                : "No edge above threshold"}
            </span>
          </div>

          {/* MAIN GRID: Recommendation | Stats | Grade */}
          <div className="grid grid-cols-1 xl:grid-cols-[1.4fr_1fr_auto] gap-8 items-start">
            {/* Recommendation */}
            <div>
              <div className="text-[11px] uppercase tracking-[0.20em] text-slate-500 font-bold mb-2">
                Recommendation
              </div>
              <div
                className={`display text-[44px] leading-[1.05] font-bold ${
                  best ? (recDir ? "gradient-text-emerald" : "gradient-text-red") : "text-slate-300"
                }`}
              >
                {recAction}
              </div>
              <div className="flex items-center gap-2 mt-3">
                {best && (
                  <>
                    <span className={`badge ${recDir ? "badge-buy" : "badge-sell"}`}>
                      {recDir ? <ArrowUpRight size={11} /> : <ArrowDownRight size={11} />}
                      {recDir ? "BUY YES" : "BUY NO"}
                    </span>
                    <span className="chip">
                      <Target size={10} /> Edge {fmtPct(best.edge, 1, true)}
                    </span>
                    <span className="chip">
                      <Coins size={10} /> EV {fmtPct(best.expectedValue, 1, true)}
                    </span>
                  </>
                )}
                {!best && (
                  <span className="text-[12px] text-slate-500">
                    Sit tight — model and market are too close to act.
                  </span>
                )}
              </div>

              {/* Confidence bar */}
              <div className="mt-6 max-w-md">
                <ProgressBar
                  value={conf}
                  tone={conf >= 0.85 ? "emerald" : conf >= 0.65 ? "blue" : "amber"}
                  label="Model confidence"
                  rightLabel={fmtPct(conf, 0)}
                />
              </div>
            </div>

            {/* Inline stats column */}
            <div className="grid grid-cols-2 gap-4">
              <HeroStat
                label="Model Weekly Avg"
                value={fmt(modelAvg != null ? modelAvg * 1e6 : null, 3)}
                sub={stdM != null ? `± ${stdM.toFixed(4)}M` : undefined}
                tone="info"
              />
              <HeroStat
                label="Kalshi Implied"
                value={fmt(kalshiAvg != null ? kalshiAvg * 1e6 : null, 3)}
                sub="Market consensus"
                tone="warn"
              />
              <HeroStat
                label="Edge"
                value={edgeK != null ? `${edgeK >= 0 ? "+" : ""}${Math.round(edgeK).toLocaleString()}k` : "—"}
                sub={edgeK != null ? (Math.abs(edgeK) < 10 ? "In agreement" : edgeK > 0 ? "Model higher" : "Model lower") : undefined}
                tone={(edgeK ?? 0) >= 0 ? "up" : "down"}
              />
              <HeroStat
                label="Expected ROI"
                value={expectedROI != null ? fmtPct(expectedROI, 1, true) : "—"}
                sub={best ? `at ${best.price?.toFixed(2)}¢ ask` : "no live edge"}
                tone={(expectedROI ?? 0) > 0 ? "up" : "neutral"}
              />
            </div>

            {/* Grade column */}
            <div className="flex flex-col items-center justify-between gap-4 min-w-[160px]">
              <div className="text-center">
                <div className="text-[11px] uppercase tracking-[0.20em] text-slate-500 font-bold mb-3">
                  Signal Strength
                </div>
                <GradePill grade={grade} size="lg" className="!h-20 !min-w-[88px] !text-[40px] shadow-glow-emerald" />
              </div>
              <div className="w-full glass !p-3 !rounded-xl text-center">
                <div className="label !text-[10px] mb-1">Kelly Size (½K)</div>
                <div className="display text-2xl font-bold text-edge-up">
                  {kellySize != null ? fmtMoney(kellySize, 0) : "$0"}
                </div>
                <div className="text-[10px] text-slate-500 mt-1">
                  {best && best.contracts > 0 ? `${best.contracts} contracts @ ${best.price?.toFixed(2)}¢` : "no position rec"}
                </div>
              </div>
              <div className="w-full flex items-center gap-2 px-2">
                <ShieldCheck size={11} className="text-edge-up" />
                <span className="text-[10.5px] text-slate-500">
                  ½-Kelly capped at {fmtMoney((ov?.bankroll ?? 250) * 0.25, 0)} per side
                </span>
              </div>
            </div>
          </div>
        </div>
      </GlassCard>
    </motion.div>
  );
}

function HeroStat({
  label, value, sub, tone,
}: { label: string; value: string; sub?: string; tone: "info" | "warn" | "up" | "down" | "neutral" }) {
  const toneColor =
    tone === "up" ? "text-edge-up"
    : tone === "down" ? "text-edge-down"
    : tone === "info" ? "text-edge-info"
    : tone === "warn" ? "text-edge-warn"
    : "text-slate-200";

  return (
    <div className="glass !p-3.5 !rounded-xl">
      <div className="label !text-[10px]">{label}</div>
      <div className={`display text-xl font-bold mt-1 ${toneColor} mono`}>{value}</div>
      {sub && <div className="text-[10.5px] text-slate-500 mt-0.5">{sub}</div>}
    </div>
  );
}
