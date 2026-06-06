import { cn } from "@/lib/utils";

type Tone = "live" | "warn" | "err" | "off";

const dot: Record<Tone, string> = {
  live: "bg-edge-up shadow-[0_0_8px_rgba(34,211,164,0.75)] animate-pulse-dot",
  warn: "bg-edge-warn shadow-[0_0_8px_rgba(245,158,11,0.65)]",
  err:  "bg-edge-down shadow-[0_0_8px_rgba(239,84,102,0.75)]",
  off:  "bg-slate-600",
};

export function StatusDot({ tone = "live", size = 8, className }: { tone?: Tone; size?: number; className?: string }) {
  return (
    <span
      className={cn("inline-block rounded-full", dot[tone], className)}
      style={{ width: size, height: size }}
    />
  );
}
