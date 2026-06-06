import { useQuery } from "@tanstack/react-query";
import { ShieldAlert, ShieldCheck, ActivitySquare, AlertTriangle } from "lucide-react";
import { api } from "../api/client";
import { GlassSection } from "./ui/GlassCard";
import { ProgressBar } from "./ui/ProgressBar";

export function DriftMonitor() {
  const { data } = useQuery({ queryKey: ["drift"], queryFn: api.drift, refetchInterval: 5 * 60_000 });

  if (!data) {
    return (
      <GlassSection title="Drift Monitor" sub="Loading drift signal">
        <div className="skeleton h-40" />
      </GlassSection>
    );
  }
  if (!data.has_data) {
    return (
      <GlassSection title="Drift Monitor" sub="Insufficient prediction history">
        <p className="text-center py-6 text-sm text-slate-500">Not enough data to compute drift yet</p>
      </GlassSection>
    );
  }

  const drift = !!data.drift_flag;
  const StatusIcon = drift ? ShieldAlert : ShieldCheck;
  const statusColor = drift ? "text-edge-down" : "text-edge-up";
  const recent   = data.recent_14d!;
  const trailing = data.trailing_90d!;

  const ratio = data.mae_ratio ?? null;
  const fmtErr = (v: number | null | undefined) => v == null ? "—" : `${Math.round(v / 1000)}k`;
  const fmtBias = (v: number | null | undefined) => v == null ? "—" : `${v >= 0 ? "+" : ""}${Math.round(v / 1000)}k`;

  return (
    <GlassSection
      title="Drift Monitor"
      sub="Recent (14d) accuracy vs. trailing (90d) baseline"
      right={
        <span className={`badge ${drift ? "badge-no" : "badge-live"} flex items-center gap-1.5`}>
          <StatusIcon size={12} />
          {drift ? "DRIFT DETECTED" : "STABLE"}
        </span>
      }
    >
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_1px_1fr] gap-5">
        {/* Recent column */}
        <div>
          <div className="label !text-[10px] mb-3">Last 14 days · {recent.n ?? 0} samples</div>
          <div className="space-y-3">
            <MetricRow label="MAE"        value={fmtErr(recent.mae)}        tone="info"   Icon={ActivitySquare} />
            <MetricRow label="Bias"       value={fmtBias(recent.bias)}      tone={Math.abs(recent.bias ?? 0) < 5000 ? "up" : Math.abs(recent.bias ?? 0) > 30_000 ? "down" : "warn"} Icon={ActivitySquare} />
            <MetricRow label="Residual σ" value={fmtErr(recent.residual_std)} tone="violet" Icon={ActivitySquare} />
          </div>
        </div>

        {/* Divider */}
        <div className="hidden lg:block bg-white/[0.06]" />

        {/* Trailing column */}
        <div>
          <div className="label !text-[10px] mb-3">Trailing 90 days · {trailing.n ?? 0} samples</div>
          <div className="space-y-3">
            <MetricRow label="MAE"        value={fmtErr(trailing.mae)}        tone="neutral" Icon={ActivitySquare} />
            <MetricRow label="Bias"       value={fmtBias(trailing.bias)}      tone="neutral" Icon={ActivitySquare} />
            <MetricRow label="Residual σ" value={fmtErr(trailing.residual_std)} tone="neutral" Icon={ActivitySquare} />
          </div>
        </div>
      </div>

      {/* Drift ratio bar */}
      {ratio != null && (
        <div className="mt-5 pt-4 border-t border-white/[0.06]">
          <ProgressBar
            value={Math.min(2, ratio)}
            max={2}
            tone={ratio >= 1.40 ? "red" : ratio >= 1.10 ? "amber" : "emerald"}
            label="Recent / trailing MAE ratio"
            rightLabel={`${ratio.toFixed(2)}× ${ratio < 1 ? "(improving)" : ratio > 1 ? "(degrading)" : ""}`}
          />
          <div className="flex justify-between text-[10px] text-slate-600 mt-1">
            <span>0.5×</span><span>1.0× baseline</span><span>2.0×</span>
          </div>
        </div>
      )}

      {/* Messages */}
      {(data.messages?.length ?? 0) > 0 && (
        <div className="mt-4 space-y-1.5">
          {data.messages!.map((m, i) => (
            <div key={i} className="flex items-start gap-2 text-[12px] text-slate-300">
              {drift
                ? <AlertTriangle size={12} className="text-edge-warn mt-0.5 flex-shrink-0" />
                : <ShieldCheck size={12} className={statusColor + " mt-0.5 flex-shrink-0"} />}
              <span>{m}</span>
            </div>
          ))}
        </div>
      )}
    </GlassSection>
  );
}

function MetricRow({
  label, value, tone, Icon,
}: { label: string; value: string; tone: "up" | "down" | "info" | "warn" | "violet" | "neutral"; Icon: typeof ActivitySquare }) {
  const color =
    tone === "up" ? "text-edge-up"
    : tone === "down" ? "text-edge-down"
    : tone === "warn" ? "text-edge-warn"
    : tone === "info" ? "text-edge-info"
    : tone === "violet" ? "text-edge-violet"
    : "text-slate-300";
  return (
    <div className="flex items-center justify-between glass !p-3 !rounded-xl">
      <div className="flex items-center gap-2">
        <Icon size={12} className="text-slate-600" />
        <span className="text-[12px] text-slate-400">{label}</span>
      </div>
      <span className={`mono font-bold text-[14px] ${color}`}>{value}</span>
    </div>
  );
}
