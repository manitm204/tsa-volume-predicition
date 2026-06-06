import { useLocation } from "react-router-dom";
import Sidebar from "./Sidebar";
import TopBar from "./TopBar";

const titles: Record<string, string> = {
  "/":         "Command Center",
  "/tomorrow": "Tomorrow",
  "/forecast": "Model Performance",
  "/week":     "This Week",
  "/shadow":   "Ensemble Router",
  "/ensemble": "Ensemble Router",
};

export default function Layout({ children }: { children: React.ReactNode }) {
  const { pathname } = useLocation();
  const title = titles[pathname] ?? "Dashboard";

  return (
    <div className="min-h-screen bg-ink-950 text-slate-200">
      <Sidebar />
      <TopBar title={title} />
      <main
        className="ml-60 pt-[80px] min-h-screen relative"
        style={{
          background:
            "radial-gradient(900px 320px at 12% -10%, rgba(34,211,164,0.06), transparent 60%)," +
            "radial-gradient(900px 320px at 88% -8%, rgba(96,165,250,0.06), transparent 60%)," +
            "#03060e",
        }}
      >
        {/* faint grid overlay */}
        <div
          className="absolute inset-0 pointer-events-none opacity-[0.6] grid-bg"
          aria-hidden
        />
        <div className="relative p-6 max-w-[1640px] mx-auto">{children}</div>
      </main>
    </div>
  );
}
