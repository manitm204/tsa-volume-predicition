import { NavLink } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import {
  Zap, Crosshair, LineChart, CalendarRange, TrendingUp, GitBranch, Scale, CalendarDays, X, type LucideIcon,
} from "lucide-react";
import { api } from "../../api/client";
import { StatusDot } from "../ui/StatusDot";

interface Link {
  to: string;
  label: string;
  Icon: LucideIcon;
  hint?: string;
}

const links: Link[] = [
  { to: "/",         label: "Command",   Icon: Crosshair,     hint: "Today's signal"          },
  { to: "/tomorrow", label: "Tomorrow",  Icon: TrendingUp,    hint: "Next day forecast"       },
  { to: "/day-explorer", label: "Day Explorer", Icon: CalendarDays, hint: "Cycle per-model breakdown"},
  { to: "/week",     label: "This Week", Icon: CalendarRange, hint: "Forecast ladder"         },
  { to: "/forecast", label: "Model",     Icon: LineChart,     hint: "Accuracy / scorecard"    },
  { to: "/shadow",   label: "Ensemble",  Icon: GitBranch,     hint: "Regime routing & weights"},
  { to: "/dynamic-weights", label: "Weight Refresh", Icon: Scale, hint: "Full-history vs last-30d blend"},
];

interface SidebarProps {
  open?: boolean;
  onClose?: () => void;
}

export default function Sidebar({ open = false, onClose }: SidebarProps) {
  const { data } = useQuery({
    queryKey: ["overview-sidebar"],
    queryFn: api.overview,
    refetchInterval: 60_000,
  });

  return (
    <>
      {/* Backdrop — mobile only, when drawer is open */}
      <div
        className={`fixed inset-0 z-30 bg-black/60 backdrop-blur-sm lg:hidden transition-opacity ${
          open ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
        onClick={onClose}
        aria-hidden
      />

      <aside
        className={`fixed inset-y-0 left-0 w-60 flex flex-col z-40 transition-transform duration-200 ease-out
          ${open ? "translate-x-0" : "-translate-x-full"} lg:translate-x-0`}
        style={{
          background: "linear-gradient(180deg, #06090f 0%, #03060e 100%)",
          borderRight: "1px solid rgba(99, 140, 255, 0.10)",
        }}
      >
        {/* Brand */}
        <div className="px-5 pt-5 pb-4 flex items-center justify-between" style={{ borderBottom: "1px solid rgba(99, 140, 255, 0.08)" }}>
          <div className="flex items-center gap-2.5">
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0 shadow-glow-emerald"
              style={{ background: "linear-gradient(135deg, #22d3a4 0%, #11785e 100%)" }}
            >
              <Zap size={16} className="text-emerald-50" strokeWidth={2.5} />
            </div>
            <div className="leading-tight">
              <div className="display text-[15px] font-bold text-slate-100">TSA · EDGE</div>
              <div className="text-[10px] font-semibold tracking-[0.22em] text-edge-up uppercase">Quant Terminal</div>
            </div>
          </div>
          {/* Close button — mobile only */}
          <button
            onClick={onClose}
            className="lg:hidden p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-white/5 transition-colors"
            aria-label="Close menu"
          >
            <X size={18} />
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 py-4 px-3 space-y-1 no-scrollbar overflow-y-auto">
          {links.map(({ to, label, Icon, hint }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) => isActive ? "nav-active" : "nav-inactive"}
              style={{
                display: "block",
                padding: "10px 12px",
                borderRadius: 10,
                textDecoration: "none",
              }}
            >
              <div className="flex items-center gap-2.5">
                <Icon size={15} strokeWidth={2} />
                <span className="text-[13.5px] font-semibold">{label}</span>
              </div>
              {hint && <div className="text-[10.5px] mt-0.5 ml-[26px] opacity-70">{hint}</div>}
            </NavLink>
          ))}
        </nav>

        {/* Footer status */}
        <div
          className="px-4 py-4 space-y-3"
          style={{ borderTop: "1px solid rgba(99, 140, 255, 0.08)" }}
        >
          <div className="glass !p-3 !rounded-xl">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] uppercase tracking-[0.18em] font-bold text-slate-500">Bankroll</span>
              <StatusDot tone={data?.last_run_status === "ok" ? "live" : "warn"} size={7} />
            </div>
            <div className="flex items-end justify-between gap-1">
              <span className="mono text-lg font-bold text-slate-100">${data?.bankroll?.toFixed(0) ?? "—"}</span>
              <span className="mono text-[11px] text-edge-warn font-semibold">
                {data?.exposure_pct != null ? `${data.exposure_pct.toFixed(0)}% used` : "—"}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <StatusDot tone={data?.auto_trading ? "live" : "off"} size={7} />
            <span className="text-[11px] text-slate-500 font-medium">
              {data?.auto_trading ? "Auto-trading armed" : "Manual mode"}
            </span>
          </div>
        </div>
      </aside>
    </>
  );
}
