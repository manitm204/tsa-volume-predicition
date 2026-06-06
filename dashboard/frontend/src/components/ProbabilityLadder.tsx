import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { api } from "../api/client";
import { GlassSection } from "./ui/GlassCard";
import { fmtPct } from "../lib/format";

/** Shows P(weekly avg > strike) for every strike, model vs. market overlay. */
export function ProbabilityLadder() {
  const { data: probs } = useQuery({ queryKey: ["probabilities"], queryFn: api.probabilities, refetchInterval: 60_000 });

  if (!probs?.length) {
    return (
      <GlassSection title="Probability Ladder" sub="Model P(weekly avg ≥ strike) for every market">
        <div className="skeleton h-64" />
      </GlassSection>
    );
  }

  const rows = [...probs].sort((a, b) => a.threshold_millions - b.threshold_millions);

  return (
    <GlassSection
      title="Probability Ladder"
      sub="Model P(weekly avg ≥ strike) vs. Kalshi implied · bar shows model prob"
      right={
        <div className="flex items-center gap-3 text-[11px] text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-1.5 rounded-sm" style={{ background: "linear-gradient(90deg,#22d3a4,#60a5fa)" }} />
            Model
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-3 h-0" style={{ borderTop: "2px dashed #f59e0b" }} />
            Kalshi
          </span>
        </div>
      }
    >
      <div className="space-y-2">
        {rows.map((r, i) => {
          const m = r.model_p_over;
          const k = r.kalshi_prob;
          if (m == null) return null;
          const isLow = m < 0.50;
          const edge = m != null && k != null ? m - k : null;
          return (
            <motion.div
              key={r.threshold_millions}
              initial={{ opacity: 0, x: -6 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.22, delay: i * 0.04 }}
              className="ladder-row px-1"
            >
              {/* Strike */}
              <div className="mono font-semibold text-slate-200 text-[13px]">
                {r.threshold_millions.toFixed(2)}M
              </div>

              {/* Bar */}
              <div className="ladder-track relative">
                <div
                  className={`ladder-fill ${isLow ? "ladder-fill-low" : ""}`}
                  style={{ width: `${Math.max(0, Math.min(1, m)) * 100}%` }}
                />
                {/* Kalshi marker */}
                {k != null && (
                  <div
                    className="absolute top-[-3px] bottom-[-3px] w-[2px] bg-edge-warn shadow-[0_0_8px_rgba(245,158,11,0.65)]"
                    style={{ left: `${Math.max(0, Math.min(1, k)) * 100}%` }}
                    title={`Kalshi ${fmtPct(k)}`}
                  />
                )}
              </div>

              {/* Model % */}
              <div className="mono text-right text-[12.5px] font-semibold text-slate-200">
                {fmtPct(m, 0)}
              </div>

              {/* Edge */}
              <div className="text-right">
                {edge != null && (
                  <span className={
                    edge >= 0.06 ? "edge-strong mono text-[11px]"
                    : edge >= 0.02 ? "edge-pos mono text-[11px]"
                    : edge <= -0.02 ? "edge-neg mono text-[11px]"
                    : "edge-neutral mono text-[11px]"
                  }>
                    {fmtPct(edge, 0, true)}
                  </span>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>
    </GlassSection>
  );
}
