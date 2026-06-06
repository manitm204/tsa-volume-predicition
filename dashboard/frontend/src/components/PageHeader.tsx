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
        className="display font-bold text-[22px] leading-none mono tracking-tight truncate"
        style={{ color: accent ?? "#e2e8f0" }}
      >
        {value}
      </span>
      {sub && <span className="text-[11px] text-slate-500 truncate">{sub}</span>}
    </div>
  )
}

export function PageHeader({ title, description, kpis, right }: PageHeaderProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.28, ease: "easeOut" }}
      className="glass !p-5 mb-5"
    >
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="display text-[18px] font-bold tracking-tight text-slate-100">{title}</h2>
          {description && <p className="text-[12.5px] mt-0.5 text-slate-500">{description}</p>}
        </div>
        {right && <div className="shrink-0">{right}</div>}
      </div>

      {kpis && kpis.length > 0 && (
        <div
          className="mt-5 pt-5 grid gap-x-8 gap-y-3"
          style={{
            borderTop: "1px solid rgba(99,140,255,0.10)",
            gridTemplateColumns: `repeat(${Math.min(kpis.length, 4)}, minmax(0,1fr))`,
          }}
        >
          {kpis.map((k, i) => <KpiCell key={i} {...k} />)}
        </div>
      )}
    </motion.div>
  )
}
