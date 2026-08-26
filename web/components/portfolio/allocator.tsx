"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import {
  Badge,
  Empty,
  Eyebrow,
  Note,
  Panel,
  Readout,
  SectionHead,
  evidenceTone,
} from "@/components/ui";
import { cx, decimal, money, probability, signed, signedPct } from "@/lib/format";
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
      <Note tone="bar" title="This is a ranking, not a backtested strategy">
        Nothing here is a portfolio simulation. No transaction costs (an Indian
        delivery round trip runs roughly 30–60 bps before market impact), no
        slippage, no liquidity limits, no T+1 settlement and no benchmark
        comparison are modelled anywhere. A cost-aware simulator reporting
        Sharpe, Sortino, max drawdown and Calmar against NIFTY 50 TR is Phase 4
        of the roadmap. Until it exists,{" "}
        <strong>nothing on this page has been shown to make money.</strong>{" "}
        <Link href="/methodology">What is actually measured →</Link>
      </Note>

      <Panel className="flex flex-wrap items-end gap-x-8 gap-y-4 px-4 py-3">
        <label className="min-w-64 flex-1">
          <Eyebrow as="span">Positions</Eyebrow>
          <div className="mt-2 flex items-center gap-3">
            <input
              type="range"
              min={2}
              max={20}
              step={1}
              value={size}
              onChange={(e) => setSize(Number(e.target.value))}
              className="h-px flex-1 cursor-pointer appearance-none bg-rule-hi accent-[var(--color-bright)]"
            />
            <span className="w-6 text-[0.8rem] text-bright">{size}</span>
          </div>
        </label>

        <label className="flex cursor-pointer items-start gap-2.5 pb-1">
          <input
            type="checkbox"
            checked={includeUnranked}
            onChange={(e) => setIncludeUnranked(e.target.checked)}
            className="mt-0.5 h-3.5 w-3.5 accent-[var(--color-bright)]"
          />
          <span className="max-w-xs font-prose text-[0.78rem] leading-relaxed text-dim">
            Include reviewed stocks that score <span>0.0</span> — they failed the
            evidence gate or are not long candidates.
          </span>
        </label>
      </Panel>

      {holdings.length === 0 ? (
        <Empty title="No stock currently qualifies">
          Nothing clears the evidence gate with a positive forecast and a
          passing critic verdict, so there is no allocation to make. Tick the
          box above to see what an unfiltered top-{size} would have contained.
        </Empty>
      ) : (
        <>
          {holdings.length < size ? (
            <Note tone="dim">
              Only {holdings.length} of the requested {size} positions could be
              filled — the candidate pool is that small.
            </Note>
          ) : null}

          <div className="grid grid-cols-1 gap-x-6 gap-y-5 sm:grid-cols-2 xl:grid-cols-4">
            <Readout
              label="Positions"
              value={holdings.length}
              sub={`Equal weight, ${weight.toFixed(1)}% each.`}
            />
            <Readout
              label="Weighted excess signal"
              value={signedPct(weightedExcess)}
              tone={weightedExcess >= 0 ? "pos" : "neg"}
              help="Excludes market direction entirely, along with costs and slippage."
              sub="Average predicted 30-session return relative to each stock's own benchmark. NOT an expected portfolio return."
            />
            <Readout
              label="Mean rank IC"
              value={signed(meanIc)}
              tone={meanIc > 0 ? "pos" : "neg"}
              sub="Out-of-sample. 0 means no skill."
            />
            <Readout
              label="Strong evidence"
              value={`${strong} / ${holdings.length}`}
              tone={strong === 0 ? "dim" : "pos"}
              sub="Passed all three held-out checks."
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full min-w-[840px] border-collapse text-[0.76rem]">
              <thead>
                <tr className="bg-inset text-left">
                  <Th>Stock</Th>
                  <Th align="right">Price</Th>
                  <Th align="right">Implied target</Th>
                  <Th align="right">Excess</Th>
                  <Th align="right">P(out)</Th>
                  <Th>Evidence</Th>
                  <Th align="right">Score</Th>
                  <Th align="right">Weight</Th>
                </tr>
              </thead>
              <tbody>
                {holdings.map((holding) => (
                  <tr
                    key={holding.ticker}
                    className="border-b border-rule/70 hover:bg-raise"
                  >
                    <td className="px-2 py-1.5">
                      <Link
                        href={`/stocks/${holding.ticker}`}
                        className="text-bright hover:underline"
                      >
                        {holding.company ?? holding.ticker}
                      </Link>
                      <div className="text-[0.68rem] text-dim">
                        {holding.sector ?? "—"}
                      </div>
                    </td>
                    <td className="px-2 py-1.5 text-right text-text">
                      {money(holding.current_price)}
                    </td>
                    <td className="px-2 py-1.5 text-right text-mid">
                      {money(holding.forecast_price)}
                    </td>
                    <td
                      className={cx(
                        "px-2 py-1.5 text-right",
                        (holding.pred_excess_return ?? 0) >= 0
                          ? "text-pos"
                          : "text-neg",
                      )}
                    >
                      {signedPct(holding.pred_excess_return)}
                    </td>
                    <td className="px-2 py-1.5 text-right text-text">
                      {probability(holding.prob_outperform)}
                    </td>
                    <td className="px-2 py-1.5">
                      <Badge tone={evidenceTone(holding.forecast_confidence)}>
                        {holding.forecast_confidence === "INSUFFICIENT"
                          ? "None"
                          : (holding.forecast_confidence ?? "—")}
                      </Badge>
                    </td>
                    <td className="px-2 py-1.5 text-right text-mid">
                      {decimal(holding.composite_score ?? 0, 1)}
                    </td>
                    <td className="px-2 py-1.5 text-right text-bright">
                      {weight.toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <section>
            <SectionHead as="h2" title="Sector allocation" />
            <ul className="space-y-1.5">
              {sectors.map(([sector, pct]) => (
                <li key={sector} className="flex items-center gap-3">
                  <span className="w-52 shrink-0 truncate text-[0.72rem] text-dim">
                    {sector}
                  </span>
                  <div className="h-[5px] flex-1 bg-inset">
                    <div
                      className="h-full bg-bright"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="w-12 shrink-0 text-right text-[0.72rem] text-text">
                    {pct.toFixed(1)}%
                  </span>
                </li>
              ))}
            </ul>
          </section>
        </>
      )}
    </div>
  );
}

function Th({
  children,
  align = "left",
}: {
  children: React.ReactNode;
  align?: "left" | "right";
}) {
  return (
    <th
      className={cx(
        "border-y border-rule px-2 py-1",
        align === "right" ? "text-right" : "text-left",
      )}
    >
      <Eyebrow as="span">{children}</Eyebrow>
    </th>
  );
}
