import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { PageHeader } from "../components/PageHeader";
import { GlassSection } from "../components/ui/GlassCard";
import type { ModelWeights } from "../types";

const MODEL_COLOR: Record<keyof ModelWeights, string> = {
  tab:       "#60a5fa",
  ts3:       "#22d3a4",
  yoy_delta: "#f59e0b",
  anchor:    "#a78bfa",
};

const MODEL_LABEL: Record<keyof ModelWeights, string> = {
  tab:       "Tabular",
  ts3:       "TS3",
  yoy_delta: "YoY Delta",
  anchor:    "Anchor",
};

const MODEL_KEYS = ["tab", "ts3", "yoy_delta", "anchor"] as const;

function fmtPct(v: number | null | undefined): string {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function WeightBars({ weights }: { weights: ModelWeights | undefined }) {
  return (
    <div className="space-y-2.5">
      {MODEL_KEYS.map(k => {
        const v = weights?.[k] ?? 0;
        return (
          <div key={k} className="flex items-center gap-3">
            <span className="w-20 text-[11px] text-slate-400 shrink-0">{MODEL_LABEL[k]}</span>
            <div className="flex-1 h-3.5 rounded-full bg-slate-800/60 overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{ width: `${v * 100}%`, background: MODEL_COLOR[k] }}
              />
            </div>
            <span className="w-14 text-right text-[11px] mono text-slate-200 font-semibold shrink-0">
              {fmtPct(v)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function WeightPanel({ title, sub, weights, accent }: {
  title: string; sub: string; weights: ModelWeights | undefined; accent: string;
}) {
  return (
    <div className="glass !p-4 !rounded-xl">
      <div className="mb-3">
        <h4 className="text-[13px] font-semibold" style={{ color: accent }}>{title}</h4>
        <p className="text-[11px] text-slate-500 mt-0.5">{sub}</p>
      </div>
      <WeightBars weights={weights} />
    </div>
  );
}

export default function DynamicWeights() {
  const { data, isLoading } = useQuery({
    queryKey: ["ensemble-dynamic-weights"],
    queryFn: api.dynamicWeights,
    staleTime: 60_000,
  });

  const generatedAt = data?.generated_at
    ? new Date(data.generated_at).toLocaleString()
    : "—";

  return (
    <div className="space-y-5 animate-fade-in">
      <PageHeader
        title="Dynamic NORMAL Weights"
        description="Full-history vs. last-30-day ensemble fits, and the 50/50 blend actually in production — from refresh_normal_weights.py."
        kpis={[
          { label: "Last Refreshed", value: generatedAt, sub: "generated_at", accent: "#22d3a4" },
          { label: "Full History N", value: String(data?.n_full_history ?? "—"), sub: "NORMAL days", accent: "#60a5fa" },
          { label: "Last-30 N", value: String(data?.n_last30 ?? "—"), sub: "NORMAL days", accent: "#f59e0b" },
          { label: "Quick Tab OOF MAE", value: data?.quick_tabular_oof_mae != null ? `${(data.quick_tabular_oof_mae / 1000).toFixed(1)}k` : "—", accent: "#a78bfa" },
        ]}
      />

      {isLoading ? (
        <GlassSection title="Loading…" sub="">
          <p className="text-[13px] text-slate-500">Loading dynamic weights…</p>
        </GlassSection>
      ) : !data?.has_data ? (
        <GlassSection title="No dynamic weights found" sub="">
          <p className="text-[13px] text-slate-500">
            <code className="mono">ensemble_experiment/output/dynamic_normal_weights.json</code> does not exist yet.
            Run <code className="mono">python ensemble_experiment/refresh_normal_weights.py</code> to generate it.
          </p>
        </GlassSection>
      ) : (
        <>
          <GlassSection
            title="Weight Comparison"
            sub={`${data.date_range?.[0]} → ${data.date_range?.[1]} · ${data.method ?? ""}`}
          >
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <WeightPanel
                title="Full History"
                sub={`Direct-MAE fit on all ${data.n_full_history} NORMAL days`}
                weights={data.full_history_weights}
                accent="#60a5fa"
              />
              <WeightPanel
                title="Last 30 Days"
                sub={`Direct-MAE fit on the most recent ${data.n_last30} NORMAL days`}
                weights={data.last30_weights}
                accent="#f59e0b"
              />
              <WeightPanel
                title="Blend (In Use)"
                sub="50/50 average — this is what NORMAL regime predictions use"
                weights={data.blend_weights}
                accent="#22d3a4"
              />
            </div>
          </GlassSection>

          <GlassSection title="Legend" sub="">
            <div className="flex gap-4 text-[11px] text-slate-500 flex-wrap">
              {MODEL_KEYS.map(k => (
                <span key={k} className="flex items-center gap-1.5">
                  <span className="inline-block w-3 h-2.5 rounded-sm" style={{ background: MODEL_COLOR[k] }} />
                  {MODEL_LABEL[k]}
                </span>
              ))}
            </div>
          </GlassSection>
        </>
      )}
    </div>
  );
}
