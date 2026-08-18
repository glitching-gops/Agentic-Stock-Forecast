"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { Badge, Callout, Card, Stat, evidenceTone } from "@/components/ui";
import { cx, money, probability, signed, signedPct } from "@/lib/format";
import type { LeaderboardEntry } from "@/lib/types";

/**
 * Equal-weight view of the top-ranked names.
 *
 * One deliberate departure from the Streamlit page this replaces: the
 * candidate pool defaults to stocks the system actually ranks, not to the top
 * N of a list that is 93/95 zeros. "Equal weight across the top 10" over a
 * board where only two names clear the evidence gate silently allocates 80% of
 * a notional book to stocks the ranking explicitly declines to rank. The
 * unranked names are one checkbox away, and the checkbox says what it does.
 */
export function PortfolioAllocator({ entries }: { entries: LeaderboardEntry[] }) {
  const [size, setSize] = useState(10);
  const [includeUnranked, setIncludeUnranked] = useState(false);

  const reviewed = useMemo(
    () =>
      entries.filter(
        (e) => e.critic_verdict === "APPROVED" || e.critic_verdict === "FLAGGED",
      ),
    [entries],
  );

  const rankedPool = useMemo(
    () => reviewed.filter((e) => e.score_basis === "RANKED"),
    [reviewed],
  );

  const pool = includeUnranked ? reviewed : rankedPool;
  const holdings = pool.slice(0, size);
  const weight = holdings.length > 0 ? 100 / holdings.length : 0;

  const weightedExcess =
    holdings.reduce((sum, e) => sum + (e.pred_excess_return ?? 0), 0) /
    Math.max(holdings.length, 1);

  const ics = holdings
    .map((e) => e.eval_rank_ic)
    .filter((v): v is number => v !== null);
  const meanIc = ics.length
    ? ics.reduce((a, b) => a + b, 0) / ics.length
    : Number.NaN;
  const strong = holdings.filter(
    (e) => e.forecast_confidence === "STRONG",
  ).length;

  const sectors = useMemo(() => {
    const map = new Map<string, number>();
    for (const holding of holdings) {
      const key = holding.sector ?? "Unclassified";
      map.set(key, (map.get(key) ?? 0) + weight);
    }
    return [...map.entries()].sort((a, b) => b[1] - a[1]);
  }, [holdings, weight]);

  return (
    <div className="space-y-6">
      <Callout tone="warning" title="This is a ranking, not a backtested strategy.">
        Nothing here is a portfolio simulation. No transaction costs (an Indian
        delivery round trip runs roughly 30–60 bps before market impact), no
        slippage, no liquidity limits, no T+1 settlement and no benchmark
        comparison are modelled anywhere. A cost-aware simulator reporting
        Sharpe, Sortino, max drawdown and Calmar against NIFTY 50 TR is Phase 4
        of the roadmap. Until it exists,{" "}
        <strong>nothing on this page has been shown to make money.</strong>{" "}
        <Link href="/methodology">What is actually measured →</Link>
      </Callout>

      <Card className="flex flex-wrap items-end gap-6 p-4">
        <label className="flex-1 min-w-64">
          <span className="text-[0.65rem] font-semibold uppercase tracking-[0.09em] text-mist-500">
            Positions
          </span>
          <div className="mt-2 flex items-center gap-3">
            <input
              type="range"
              min={2}
              max={20}
              step={1}
              value={size}
              onChange={(e) => setSize(Number(e.target.value))}
              className="h-1 flex-1 cursor-pointer appearance-none rounded-full bg-ink-500 accent-[var(--color-brand-500)]"
            />
            <span className="nums w-6 text-sm font-semibold text-mist-100">
              {size}
            </span>
          </div>
        </label>

        <label className="flex cursor-pointer items-start gap-2.5 pb-1">
          <input
            type="checkbox"
            checked={includeUnranked}
            onChange={(e) => setIncludeUnranked(e.target.checked)}
            className="mt-0.5 h-4 w-4 accent-[var(--color-brand-500)]"
          />
          <span className="max-w-xs text-xs leading-relaxed text-mist-400">
            Include reviewed stocks that score{" "}
            <span className="nums">0.0</span> — they failed the evidence gate or
            are not long candidates.
          </span>
        </label>
      </Card>

      {holdings.length === 0 ? (
        <Card className="px-6 py-10 text-center">
          <div className="text-sm font-semibold text-mist-300">
            No stock currently qualifies.
          </div>
          <p className="mx-auto mt-2 max-w-md text-sm text-mist-500">
            Nothing clears the evidence gate with a positive forecast and a
            passing critic verdict, so there is no allocation to make. Tick the
            box above to see what an unfiltered top-{size} would have contained.
          </p>
        </Card>
      ) : (
        <>
          {holdings.length < size ? (
            <Callout tone="muted">
              Only {holdings.length} of the requested {size} positions could be
              filled — the candidate pool is that small.
            </Callout>
          ) : null}

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <Stat
              label="Positions"
              value={holdings.length}
              sub={`Equal weight, ${weight.toFixed(1)}% each.`}
            />
            <Stat
              label="Weighted excess signal"
              value={signedPct(weightedExcess)}
              tone={weightedExcess >= 0 ? "positive" : "negative"}
              sub="Average predicted 30-session return relative to each stock's own benchmark. NOT an expected portfolio return."
              help="Excludes market direction entirely, along with costs and slippage."
            />
            <Stat
              label="Mean rank IC"
              value={signed(meanIc)}
              tone={meanIc > 0 ? "positive" : "negative"}
              sub="Out-of-sample. 0 means no skill."
            />
            <Stat
              label="Strong evidence"
              value={`${strong} of ${holdings.length}`}
              tone={strong === 0 ? "muted" : "positive"}
              sub="Passed all three held-out checks."
            />
          </div>

          <div className="overflow-x-auto rounded-xl border border-ink-500/70">
            <table className="w-full min-w-[840px] border-collapse text-sm">
              <thead>
                <tr className="border-b border-ink-500/70 bg-ink-800/70 text-left">
                  <th className="px-3 py-2.5 text-[0.68rem] font-semibold uppercase tracking-[0.07em] text-mist-500">
                    Stock
                  </th>
                  <th className="px-3 py-2.5 text-right text-[0.68rem] font-semibold uppercase tracking-[0.07em] text-mist-500">
                    Price
                  </th>
                  <th className="px-3 py-2.5 text-right text-[0.68rem] font-semibold uppercase tracking-[0.07em] text-mist-500">
                    Implied target
                  </th>
                  <th className="px-3 py-2.5 text-right text-[0.68rem] font-semibold uppercase tracking-[0.07em] text-mist-500">
                    Excess
                  </th>
                  <th className="px-3 py-2.5 text-right text-[0.68rem] font-semibold uppercase tracking-[0.07em] text-mist-500">
                    P(out)
                  </th>
                  <th className="px-3 py-2.5 text-[0.68rem] font-semibold uppercase tracking-[0.07em] text-mist-500">
                    Evidence
                  </th>
                  <th className="px-3 py-2.5 text-right text-[0.68rem] font-semibold uppercase tracking-[0.07em] text-mist-500">
                    Score
                  </th>
                  <th className="px-3 py-2.5 text-right text-[0.68rem] font-semibold uppercase tracking-[0.07em] text-mist-500">
                    Weight
                  </th>
                </tr>
              </thead>
              <tbody>
                {holdings.map((holding) => (
                  <tr
                    key={holding.ticker}
                    className="border-b border-ink-500/40 last:border-b-0 hover:bg-ink-600/40"
                  >
                    <td className="px-3 py-2.5">
                      <Link
                        href={`/stocks/${holding.ticker}`}
                        className="font-medium text-mist-100 hover:text-brand-300"
                      >
                        {holding.company ?? holding.ticker}
                      </Link>
                      <div className="text-xs text-mist-500">
                        {holding.sector ?? "—"}
                      </div>
                    </td>
                    <td className="nums px-3 py-2.5 text-right text-mist-200">
                      {money(holding.current_price)}
                    </td>
                    <td className="nums px-3 py-2.5 text-right text-mist-300">
                      {money(holding.forecast_price)}
                    </td>
                    <td
                      className={cx(
                        "nums px-3 py-2.5 text-right font-semibold",
                        (holding.pred_excess_return ?? 0) >= 0
                          ? "text-pos-500"
                          : "text-neg-500",
                      )}
                    >
                      {signedPct(holding.pred_excess_return)}
                    </td>
                    <td className="nums px-3 py-2.5 text-right text-mist-200">
                      {probability(holding.prob_outperform)}
                    </td>
                    <td className="px-3 py-2.5">
                      <Badge tone={evidenceTone(holding.forecast_confidence)}>
                        {holding.forecast_confidence === "INSUFFICIENT"
                          ? "None"
                          : (holding.forecast_confidence ?? "—")}
                      </Badge>
                    </td>
                    <td className="nums px-3 py-2.5 text-right text-mist-300">
                      {(holding.composite_score ?? 0).toFixed(1)}
                    </td>
                    <td className="nums px-3 py-2.5 text-right font-semibold text-mist-100">
                      {weight.toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <Card className="p-4">
            <h2 className="mb-3 text-sm font-semibold text-mist-100">
              Sector allocation
            </h2>
            <ul className="space-y-2">
              {sectors.map(([sector, pct]) => (
                <li key={sector} className="flex items-center gap-3">
                  <span className="w-52 shrink-0 truncate text-xs text-mist-400">
                    {sector}
                  </span>
                  <div className="h-2 flex-1 overflow-hidden rounded-full bg-ink-600">
                    <div
                      className="h-full rounded-full bg-brand-500"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="nums w-12 shrink-0 text-right text-xs font-semibold text-mist-200">
                    {pct.toFixed(1)}%
                  </span>
                </li>
              ))}
            </ul>
          </Card>
        </>
      )}
    </div>
  );
}
