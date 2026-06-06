import { cn } from "@/lib/utils";

interface Props {
  value: number;        // 0..1 or 0..100 depending on `max`
  max?: number;
  tone?: "emerald" | "blue" | "amber" | "red" | "violet" | "gradient";
  height?: number;
  label?: string;
  rightLabel?: string;
  className?: string;
}

const toneStyle: Record<NonNullable<Props["tone"]>, string> = {
  emerald:  "bg-gradient-to-r from-emerald-500 to-emerald-300 shadow-[0_0_12px_rgba(34,211,164,0.35)]",
  blue:     "bg-gradient-to-r from-blue-600 to-sky-400 shadow-[0_0_12px_rgba(59,130,246,0.30)]",
  amber:    "bg-gradient-to-r from-amber-600 to-amber-300 shadow-[0_0_10px_rgba(245,158,11,0.25)]",
  red:      "bg-gradient-to-r from-rose-600 to-rose-400 shadow-[0_0_10px_rgba(239,68,68,0.28)]",
  violet:   "bg-gradient-to-r from-violet-600 to-fuchsia-400 shadow-[0_0_10px_rgba(167,139,250,0.28)]",
  gradient: "bg-gradient-to-r from-emerald-400 via-sky-400 to-violet-400 shadow-[0_0_14px_rgba(96,165,250,0.30)]",
};

export function ProgressBar({
  value, max = 1, tone = "emerald", height = 8, label, rightLabel, className,
}: Props) {
  const pct = Math.max(0, Math.min(1, value / max)) * 100;
  return (
    <div className={cn("w-full", className)}>
      {(label || rightLabel) && (
        <div className="flex items-center justify-between mb-1.5">
          {label && <span className="text-[11px] text-slate-500 font-medium tracking-wide">{label}</span>}
          {rightLabel && <span className="text-[11px] mono text-slate-300 font-semibold">{rightLabel}</span>}
        </div>
      )}
      <div
        className="rounded-full overflow-hidden bg-surface-700/60"
        style={{ height }}
      >
        <div
          className={cn("h-full rounded-full transition-[width] duration-500 ease-out", toneStyle[tone])}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
