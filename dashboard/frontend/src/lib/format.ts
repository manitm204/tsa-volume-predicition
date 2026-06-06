// Numeric formatters used everywhere in the dashboard.
// All accept null/undefined and return em-dash for missing values.

export function fmt(n: number | null | undefined, digits = 3): string {
  if (n == null || Number.isNaN(n)) return "—";
  return `${(n / 1e6).toFixed(digits)}M`;
}

export function fmtK(n: number | null | undefined, signed = true): string {
  if (n == null || Number.isNaN(n)) return "—";
  const s = signed && n >= 0 ? "+" : "";
  return `${s}${Math.round(n / 1000).toLocaleString()}k`;
}

export function fmtPct(v: number | null | undefined, digits = 1, signed = false): string {
  if (v == null || Number.isNaN(v)) return "—";
  const pct = v * 100;
  const s = signed && pct >= 0 ? "+" : "";
  return `${s}${pct.toFixed(digits)}%`;
}

export function fmtMoney(v: number | null | undefined, digits = 2, signed = false): string {
  if (v == null || Number.isNaN(v)) return "—";
  const s = signed && v >= 0 ? "+" : v < 0 ? "−" : "";
  return `${s}$${Math.abs(v).toFixed(digits)}`;
}

export function fmtPrice(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function signed(v: number | null | undefined, formatter: (n: number) => string): string {
  if (v == null || Number.isNaN(v)) return "—";
  const s = v >= 0 ? "+" : "−";
  return `${s}${formatter(Math.abs(v))}`;
}

export function fmtCompact(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  if (Math.abs(v) >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (Math.abs(v) >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return v.toFixed(0);
}
