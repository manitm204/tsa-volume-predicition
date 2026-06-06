/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        display: ['"Space Grotesk"', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'ui-monospace', 'monospace'],
      },
      colors: {
        ink:    { 950: "#03060e", 900: "#05080f", 850: "#070c18", 800: "#0a1020", 700: "#0e1530", 600: "#141d3d" },
        surface: {
          950: "#020617",
          900: "#080d18",
          800: "#0d1526",
          700: "#131e33",
          600: "#1a2640",
        },
        edge: {
          up:    "#22d3a4",
          down:  "#ef5466",
          warn:  "#f59e0b",
          info:  "#60a5fa",
          violet:"#a78bfa",
        },
        grade: {
          aplus: "#22d3a4",
          a:     "#34d399",
          b:     "#60a5fa",
          c:     "#a78bfa",
          d:     "#f59e0b",
          f:     "#ef5466",
        },
      },
      backgroundImage: {
        "grid-faint":
          "linear-gradient(rgba(99,140,255,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(99,140,255,0.04) 1px, transparent 1px)",
        "radial-spot":
          "radial-gradient(800px 320px at 20% -20%, rgba(34,211,164,0.10), transparent 60%), radial-gradient(700px 300px at 90% -10%, rgba(96,165,250,0.10), transparent 60%)",
      },
      backgroundSize: {
        "grid-32": "32px 32px",
      },
      keyframes: {
        "border-beam": { "100%": { "offset-distance": "100%" } },
        pulseDot: {
          "0%, 100%": { opacity: "1", transform: "scale(1)" },
          "50%":      { opacity: "0.4", transform: "scale(0.85)" },
        },
        fadeInUp: {
          "0%":   { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        slideInRight: {
          "0%":   { opacity: "0", transform: "translateX(8px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
        shimmer: {
          "0%":   { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        tickerSlide: {
          "0%":   { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
        glowPulse: {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(34,211,164,0.0), 0 0 24px rgba(34,211,164,0.18)" },
          "50%":      { boxShadow: "0 0 0 4px rgba(34,211,164,0.04), 0 0 32px rgba(34,211,164,0.28)" },
        },
      },
      animation: {
        "pulse-dot":   "pulseDot 2s ease-in-out infinite",
        "fade-in":     "fadeInUp 0.25s ease-out",
        "slide-in":    "slideInRight 0.30s ease-out",
        shimmer:       "shimmer 1.8s linear infinite",
        "border-beam": "border-beam calc(var(--duration)*1s) infinite linear",
        "ticker":      "tickerSlide 60s linear infinite",
        "glow":        "glowPulse 3.2s ease-in-out infinite",
      },
      boxShadow: {
        "glow-blue":    "0 0 24px rgba(59, 130, 246, 0.24), 0 0 56px rgba(59, 130, 246, 0.08)",
        "glow-emerald": "0 0 24px rgba(34, 211, 164, 0.28), 0 0 56px rgba(34, 211, 164, 0.10)",
        "glow-red":     "0 0 24px rgba(239, 68, 68, 0.24), 0 0 56px rgba(239, 68, 68, 0.08)",
        "glow-amber":   "0 0 24px rgba(245, 158, 11, 0.22), 0 0 56px rgba(245, 158, 11, 0.06)",
        "glow-violet":  "0 0 24px rgba(167, 139, 250, 0.22), 0 0 56px rgba(167, 139, 250, 0.06)",
        card:           "0 4px 32px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255,255,255,0.03)",
        "card-hover":   "0 6px 40px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255,255,255,0.05)",
      },
    },
  },
  plugins: [],
};
