import { PortfolioAllocator } from "@/components/portfolio/allocator";
import { getLeaderboard } from "@/lib/api";

// Route segment config must be a literal — Next cannot statically analyse an
// imported constant here. Keep in step with REVALIDATE_SECONDS in lib/api.ts.
export const revalidate = 3600;
export const maxDuration = 60;

export const metadata = {
  title: "Portfolio",
  description:
    "Equal-weight view of the top-ranked stocks. A view of the ranking, not an optimised portfolio and not a backtest.",
};

export default async function PortfolioPage() {
  const data = await getLeaderboard();

  return (
    <div className="space-y-6">
      <header className="max-w-3xl">
        <h1 className="font-display text-[1.15rem] font-bold tracking-tight text-bright">
          Portfolio
        </h1>
        <p className="mt-2 font-prose text-[0.86rem] leading-relaxed text-mid">
          An equal-weight allocation across the top-ranked stocks — a view of
          the ranking, not an optimised portfolio and not a backtest. Position
          sizing, a cost model and risk metrics are Phase 4 of the roadmap.
        </p>
      </header>

      <PortfolioAllocator entries={data.entries} />
    </div>
  );
}
