import * as React from "react"
import { motion } from "framer-motion"

export interface KpiItem {
  label: string
  value: React.ReactNode
  sub?: string | React.ReactNode
  accent?: string          // css color string
  wide?: boolean
}

interface PageHeaderProps {
  title: string
  description?: string
  kpis?: KpiItem[]
  right?: React.ReactNode
}

function KpiCell({ label, value, sub, accent }: KpiItem) {
  return (
    <div className="flex flex-col gap-1 min-w-0">
      <span className="label">{label}</span>
      <span
        className="display font-bold text-[18px] sm:text-[22px] leading-none mono tracking-tight truncate"
        style={{ color: accent ?? "#e2e8f0" }}
      >
        {value}
      </span>
      {sub && <span className="text-[11px] text-slate-500 truncate">{sub}</span>}
    </div>
  )
}

export function PageHeader({ title, description, kpis, right }: PageHeaderProps) {
  const n = Math.min(kpis?.length ?? 0, 4)
  // Mobile: 2 columns when 2+ kpis. sm+: full count.
  const mobileCols = n <= 1 ? 1 : 2
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, ease: "easeOut" }}
      className="glass !p-4 sm:!p-5 mb-4 sm:mb-5"
    >
      <div className="flex items-start justify-between gap-3 flex-wrap">
        <div className="min-w-0">
          <h2 className="display text-[16px] sm:text-[18px] font-bold tracking-tight text-slate-100">{title}</h2>
          {description && <p className="text-[12px] sm:text-[12.5px] mt-0.5 text-slate-500">{description}</p>}
        </div>
        {right && <div className="shrink-0">{right}</div>}
      </div>

      {kpis && kpis.length > 0 && (
        <>
          <style>{`@media (min-width: 640px) { .kpi-grid-${n} { grid-template-columns: repeat(${n}, minmax(0,1fr)) !important; } }`}</style>
          <div
            className={`kpi-grid-${n} mt-4 sm:mt-5 pt-4 sm:pt-5 grid gap-x-4 sm:gap-x-8 gap-y-3`}
            style={{
              borderTop: "1px solid rgba(99,140,255,0.10)",
              gridTemplateColumns: `repeat(${mobileCols}, minmax(0,1fr))`,
            }}
          >
            {kpis.map((k, i) => <KpiCell key={i} {...k} />)}
          </div>
        </>
      )}
    </motion.div>
  )
}
