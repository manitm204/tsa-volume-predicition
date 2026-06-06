import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { SignalHero } from "../components/SignalHero";
import { SignalFeed } from "../components/SignalFeed";
import { ForecastStrip } from "../components/ForecastStrip";
import { ChangesCard } from "../components/ChangesCard";
import YoYChart from "../components/YoYChart";

export default function Overview() {
  const { isLoading } = useQuery({
    queryKey: ["overview"],
    queryFn: api.overview,
    refetchInterval: 60_000,
  });

  if (isLoading) {
    return (
      <div className="space-y-5 animate-fade-in">
        <div className="skeleton h-[260px]" />
        <div className="skeleton h-32" />
      </div>
    );
  }

  return (
    <div className="space-y-5 animate-fade-in">
      {/* HERO */}
      <SignalHero />

      {/* WHAT CHANGED */}
      <ChangesCard />

      {/* TWO-COLUMN: forecast + feed (feed matches chart height) */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-5 items-stretch xl:h-[372px]">
        <div className="xl:col-span-2 flex min-h-0">
          <ForecastStrip />
        </div>
        <div className="flex min-h-0">
          <SignalFeed />
        </div>
      </div>

      {/* YoY */}
      <YoYChart />
    </div>
  );
}
