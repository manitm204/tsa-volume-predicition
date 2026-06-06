import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { PageHeader } from "../components/PageHeader";
import { GlassSection } from "../components/ui/GlassCard";
import type { EnsembleWeightRow, EnsembleHistoryRow, EnsembleSummary } from "../types";

// ── Regime colours + labels ───────────────────────────────────────────────────
const REGIME_COLOR: Record<string, string> = {
  STORM:         "#ef5466",
  PEAK_HOLIDAY:  "#f59e0b",
  SHOULDER_PRE:  "#60a5fa",
  SHOULDER_POST: "#a78bfa",
  NORMAL:        "#22d3a4",
};

const REGIME_LABEL: Record<string, string> = {
  NORMAL:        "Normal",
  SHOULDER_PRE:  "Shoulder Pre",
  SHOULDER_POST: "Shoulder Post",
  PEAK_HOLIDAY:  "Peak Holiday",
  STORM:         "Storm",
};

// ── Model segment colours for the stacked weight bar ─────────────────────────
const MODEL_COLOR = {
  tab:     "#60a5fa",
  ts3:     "#22d3a4",
  prophet: "#f59e0b",
  anchor:  "#a78bfa",
};

// ── Formatters ────────────────────────────────────────────────────────────────
function fmtPct(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function fmtK(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${(v / 1000).toFixed(1)}k`;
}

function fmtVol(v: number | null | undefined): string {
  if (v == null) return "—";
  return (v / 1e6).toFixed(3) + "M";
}

// ── Regime badge ──────────────────────────────────────────────────────────────
function RegimeBadge({ regime }: { regime: string | null | undefined }) {
  if (!regime) return <span className="text-slate-700">—</span>;
  const color = REGIME_COLOR[regime] ?? "#64748b";
  return (
    <span
      className="inline-block px-2 py-0.5 rounded text-[10px] font-semibold"
      style={{ background: `${color}22`, color }}
    >
      {REGIME_LABEL[regime] ?? regime}
    </span>
  );
}

// ── Mini stacked weight bar ───────────────────────────────────────────────────
function WeightBar({ row }: { row: EnsembleWeightRow }) {
  if (row.tab == null) {
    return <span className="text-[10px] text-slate-500 italic">α-blend</span>;
  }
  const segments: { key: keyof typeof MODEL_COLOR; val: number }[] = [
    { key: "tab",     val: row.tab     ?? 0 },
    { key: "ts3",     val: row.ts3     ?? 0 },
    { key: "prophet", val: row.prophet ?? 0 },
    { key: "anchor",  val: row.anchor  ?? 0 },
  ];
  return (
    <div className="flex h-3 w-28 rounded overflow-hidden gap-px">
      {segments.map(({ key, val }) =>
        val > 0 ? (
          <div
            key={key}
            style={{ width: `${val * 100}%`, background: MODEL_COLOR[key] }}
            title={`${key}: ${(val * 100).toFixed(1)}%`}
          />
        ) : null
      )}
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function EnsembleRouter() {
  const { data: weekData } = useQuery({
    queryKey: ["forecast-week"],
    queryFn: api.forecastWeek,
    refetchInterval: 60_000,
  });

  const { data: weights = [], isLoading: weightsLoading } = useQuery({
    queryKey: ["ensemble-weights"],
    queryFn: api.ensembleWeights,
    staleTime: 5 * 60_000,
  });

  const { data: history = [], isLoading: histLoading } = useQuery({
    queryKey: ["ensemble-history"],
    queryFn: api.ensembleHistory,
    refetchInterval: 60_000,
  });

  const { data: summary } = useQuery<EnsembleSummary>({
    queryKey: ["ensemble-summary"],
    queryFn: api.ensembleSummary,
    refetchInterval: 60_000,
  });

  // ── KPI derivations ─────────────────────────────────────────────────────────
  const days = weekData?.days ?? [];

  const predictedDays = days.filter(d => d.status === "predicted" && d.regime);
  const predictedCount = predictedDays.length;

  // Build "3 Normal · 1 Shoulder Pre" string
  const regimeCounts: Record<string, number> = {};
  for (const d of predictedDays) {
    const r = d.regime ?? "UNKNOWN";
    regimeCounts[r] = (regimeCounts[r] ?? 0) + 1;
  }
  const regimeThisWeek = Object.entries(regimeCounts)
    .map(([r, n]) => `${n} ${REGIME_LABEL[r] ?? r}`)
    .join(" · ") || "—";

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Ensemble Router"
        description="Per-regime ensemble of 4 models (tabular, TS3, prophet, anchor_master) with Platt-calibrated σ for probabilities."
        kpis={[
          {
            label: "Predicted Days",
            value: String(predictedCount),
            sub: "this week with regime",
            accent: "#22d3a4",
          },
          {
            label: "Regimes This Week",
            value: regimeThisWeek,
            sub: "from weekly_forecast.csv",
            accent: "#60a5fa",
          },
          {
            label: "Platt σ Range",
            value: "32k – 82k",
            sub: "σ_eff across all regimes",
            accent: "#a78bfa",
          },
          {
            label: "History Rows",
            value: summary?.n_predictions != null ? String(summary.n_predictions) : "—",
            sub: summary?.last_target_date ? `last: ${summary.last_target_date}` : "prediction_history.csv",
          },
        ]}
      />

      {/* ── Section 1: This Week's Routing ─────────────────────────────────── */}
      <GlassSection
        title="This Week's Routing"
        sub="Ensemble weights and calibrated σ applied to each day's prediction"
      >
        {days.length === 0 ? (
          <p className="text-[13px] text-slate-500">No weekly forecast data available.</p>
        ) : (
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-left text-slate-500 border-b border-slate-800/60">
                {["Day", "Date", "Status", "Regime", "Tab%", "TS3%", "Prophet%", "Anchor%", "σ_eff", "Volume"].map(h => (
                  <th key={h} className="py-2 pr-3 font-semibold">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {days.map(d => {
                const isActual = d.status === "actual";
                return (
                  <tr key={d.date} className="border-b border-slate-800/30">
                    <td className="py-2 pr-3 font-semibold text-slate-100">{d.day_name}</td>
                    <td className="py-2 pr-3 mono text-slate-500 text-[11px]">{d.date}</td>
                    <td className="py-2 pr-3">
                      <span className={d.status === "actual" ? "badge badge-actual" : "badge badge-predicted"}>
                        {d.status}
                      </span>
                    </td>
                    <td className="py-2 pr-3">
                      {isActual
                        ? <span className="text-slate-700">—</span>
                        : <RegimeBadge regime={d.regime} />}
                    </td>
                    <td className="py-2 pr-3 mono text-slate-400">
                      {isActual || !d.weights ? "—" : fmtPct(d.weights.tab)}
                    </td>
                    <td className="py-2 pr-3 mono text-slate-400">
                      {isActual || !d.weights ? "—" : fmtPct(d.weights.ts3)}
                    </td>
                    <td className="py-2 pr-3 mono text-slate-400">
                      {isActual || !d.weights ? "—" : fmtPct(d.weights.prophet)}
                    </td>
                    <td className="py-2 pr-3 mono text-slate-400">
                      {isActual || !d.weights ? "—" : fmtPct(d.weights.anchor)}
                    </td>
                    <td className="py-2 pr-3 mono text-slate-300">
                      {isActual || d.sigma_eff == null ? "—" : fmtK(d.sigma_eff)}
                    </td>
                    <td className="py-2 mono text-slate-100 font-semibold">
                      {fmtVol(d.volume ?? d.predicted)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </GlassSection>

      {/* ── Section 2: Ensemble Weights by Regime ──────────────────────────── */}
      <GlassSection
        title="Ensemble Weights by Regime"
        sub="Static reference table — hardcoded from cross-validated weight optimization"
      >
        {weightsLoading ? (
          <p className="text-[13px] text-slate-500">Loading…</p>
        ) : (
          <>
            {/* Legend */}
            <div className="flex gap-4 mb-3 text-[11px] text-slate-500">
              {(["tab", "ts3", "prophet", "anchor"] as const).map(k => (
                <span key={k} className="flex items-center gap-1.5">
                  <span className="inline-block w-3 h-2.5 rounded-sm" style={{ background: MODEL_COLOR[k] }} />
                  {k === "tab" ? "Tabular" : k === "ts3" ? "TS3" : k === "prophet" ? "Prophet" : "Anchor"}
                </span>
              ))}
            </div>

            <table className="w-full text-[12px]">
              <thead>
                <tr className="text-left text-slate-500 border-b border-slate-800/60">
                  {["Regime", "Tabular", "TS3", "Prophet", "Anchor", "σ_raw", "σ_eff", "Weight Split", "Note"].map(h => (
                    <th key={h} className="py-2 pr-3 font-semibold">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(weights as EnsembleWeightRow[]).map(row => (
                  <tr key={row.regime} className="border-b border-slate-800/30">
                    <td className="py-2.5 pr-3">
                      <RegimeBadge regime={row.regime} />
                    </td>
                    <td className="py-2.5 pr-3 mono text-slate-300">{fmtPct(row.tab)}</td>
                    <td className="py-2.5 pr-3 mono text-slate-300">{fmtPct(row.ts3)}</td>
                    <td className="py-2.5 pr-3 mono text-slate-300">{fmtPct(row.prophet)}</td>
                    <td className="py-2.5 pr-3 mono text-slate-300">{fmtPct(row.anchor)}</td>
                    <td className="py-2.5 pr-3 mono text-slate-500">{fmtK(row.sigma_raw)}</td>
                    <td className="py-2.5 pr-3 mono text-slate-200 font-semibold">{fmtK(row.sigma_eff)}</td>
                    <td className="py-2.5 pr-3">
                      <WeightBar row={row} />
                    </td>
                    <td className="py-2.5 text-[11px] text-slate-500 italic">{row.note || ""}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </GlassSection>

      {/* ── Section 3: Prediction History ──────────────────────────────────── */}
      <GlassSection
        title="Prediction History"
        sub="Most recent 30 rows from prediction_history.csv — one row per day per run"
      >
        {histLoading ? (
          <p className="text-[13px] text-slate-500">Loading…</p>
        ) : (history as EnsembleHistoryRow[]).length === 0 ? (
          <p className="text-[13px] text-slate-500">
            No prediction history yet. Runs via <code className="mono text-slate-400">autogluon_predict.py</code> will
            populate <code className="mono text-slate-400">output_autogluon_predict/prediction_history.csv</code>.
          </p>
        ) : (
          <table className="w-full text-[12px]">
            <thead>
              <tr className="text-left text-slate-500 border-b border-slate-800/60">
                {["Date", "Made On", "Regime", "Ensemble Pred", "Tabular Pred", "Δ (ens − tab)"].map(h => (
                  <th key={h} className="py-2 pr-3 font-semibold">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(history as EnsembleHistoryRow[]).map((row, i) => {
                const delta =
                  row.predicted_volume != null && row.pred_tabular != null
                    ? row.predicted_volume - row.pred_tabular
                    : null;
                const deltaK = delta != null ? Math.round(delta / 1000) : null;
                return (
                  <tr key={`${row.target_date}-${i}`} className="border-b border-slate-800/30">
                    <td className="py-2 pr-3 mono text-slate-300">{row.target_date}</td>
                    <td className="py-2 pr-3 mono text-slate-500 text-[11px]">{row.made_on_date ?? "—"}</td>
                    <td className="py-2 pr-3">
                      <RegimeBadge regime={row.regime} />
                    </td>
                    <td className="py-2 pr-3 mono text-slate-100 font-semibold">
                      {fmtVol(row.predicted_volume)}
                    </td>
                    <td className="py-2 pr-3 mono text-slate-400">
                      {fmtVol(row.pred_tabular)}
                    </td>
                    <td className="py-2 mono font-semibold">
                      {deltaK != null ? (
                        <span style={{ color: deltaK >= 0 ? "#22d3a4" : "#ef5466" }}>
                          {deltaK >= 0 ? "+" : ""}{deltaK}k
                        </span>
                      ) : (
                        <span className="text-slate-700">—</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </GlassSection>
    </div>
  );
}
