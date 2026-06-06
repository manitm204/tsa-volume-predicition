import * as React from "react";
import { cn } from "@/lib/utils";

type Glow = "none" | "emerald" | "blue" | "red" | "amber" | "violet";

interface Props extends React.HTMLAttributes<HTMLDivElement> {
  glow?: Glow;
  strong?: boolean;
  bordered?: boolean;
  padded?: boolean;
}

const glowMap: Record<Glow, string> = {
  none:    "",
  emerald: "shadow-glow-emerald",
  blue:    "shadow-glow-blue",
  red:     "shadow-glow-red",
  amber:   "shadow-glow-amber",
  violet:  "shadow-glow-violet",
};

export function GlassCard({
  className, glow = "none", strong, bordered, padded = true, children, ...props
}: Props) {
  return (
    <div
      className={cn(
        strong ? "glass-strong" : "glass",
        "glass-hover",
        padded && "p-5",
        bordered && "neon-border",
        glowMap[glow],
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export function GlassSection({
  title, sub, right, children, glow, className,
}: {
  title: string;
  sub?: React.ReactNode;
  right?: React.ReactNode;
  children: React.ReactNode;
  glow?: Glow;
  className?: string;
}) {
  return (
    <GlassCard glow={glow} className={className}>
      <div className="flex items-start justify-between mb-4 gap-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-200 tracking-tight">{title}</h3>
          {sub && <p className="text-xs mt-0.5 text-slate-500">{sub}</p>}
        </div>
        {right}
      </div>
      {children}
    </GlassCard>
  );
}
