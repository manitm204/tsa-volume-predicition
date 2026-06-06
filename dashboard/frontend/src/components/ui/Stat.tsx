import * as React from "react";
import { cn } from "@/lib/utils";
import { ArrowUpRight, ArrowDownRight, Minus, type LucideIcon } from "lucide-react";
import { Sparkline } from "./Sparkline";

type Tone = "neutral" | "up" | "down" | "info" | "warn" | "violet";

const toneText: Record<Tone, string> = {
  neutral: "text-slate-200",
  up:      "text-edge-up",
  down:    "text-edge-down",
  info:    "text-edge-info",
  warn:    "text-edge-warn",
  violet:  "text-edge-violet",
};

const toneIcon: Record<Tone, string> = {
  neutral: "text-slate-500",
  up:      "text-edge-up",
  down:    "text-edge-down",
  info:    "text-edge-info",
  warn:    "text-edge-warn",
  violet:  "text-edge-violet",
};

interface Props {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  tone?: Tone;
  Icon?: LucideIcon;
  delta?: { value: number; suffix?: string };  // small change indicator
  sparkline?: (number | null | undefined)[];
  sparkColor?: string;
  className?: string;
  mono?: boolean;
}

export function Stat({
  label, value, sub, tone = "neutral", Icon, delta, sparkline, sparkColor, className, mono,
}: Props) {
  const arrow = delta == null ? null
    : delta.value > 0 ? <ArrowUpRight size={13} className="text-edge-up" />
    : delta.value < 0 ? <ArrowDownRight size={13} className="text-edge-down" />
    : <Minus size={13} className="text-slate-500" />;

  const deltaColor = delta == null ? ""
    : delta.value > 0 ? "text-edge-up"
    : delta.value < 0 ? "text-edge-down"
    : "text-slate-500";

  return (
    <div className={cn("glass glass-hover p-4 flex flex-col gap-2 min-h-[112px]", className)}>
      <div className="flex items-center justify-between">
        <span className="label">{label}</span>
        {Icon && <Icon size={14} className={toneIcon[tone]} />}
      </div>

      <div className="flex items-end justify-between gap-2 mt-auto">
        <div className="flex flex-col">
          <span className={cn("display text-2xl font-bold leading-none", toneText[tone], mono && "mono")}>
            {value}
          </span>
          {sub && <span className="text-[11px] text-slate-500 mt-1.5">{sub}</span>}
        </div>

        {delta && (
          <span className={cn("inline-flex items-center gap-0.5 text-xs font-semibold mono mb-0.5", deltaColor)}>
            {arrow}
            {Math.abs(delta.value).toFixed(1)}{delta.suffix ?? "%"}
          </span>
        )}
      </div>

      {sparkline && sparkline.length > 1 && (
        <div className="mt-1 -mx-1">
          <Sparkline
            data={sparkline}
            stroke={sparkColor ?? (tone === "down" ? "#ef5466" : tone === "up" ? "#22d3a4" : "#60a5fa")}
            fill={sparkColor ?? (tone === "down" ? "rgba(239,84,102,0.18)" : tone === "up" ? "rgba(34,211,164,0.18)" : "rgba(96,165,250,0.18)")}
            height={28}
          />
        </div>
      )}
    </div>
  );
}
