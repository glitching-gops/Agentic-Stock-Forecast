import Link from "next/link";

import { Eyebrow } from "@/components/ui";
import { cx, scoreBasisExplainer, scoreBasisLabel } from "@/lib/format";
import type { LeaderboardEntry } from "@/lib/types";

/**
 * The whole universe as one matrix — a row per sector, a cell per name, shaded
 * by why that name scored what it scored.
 *
 * This exists because the honest headline of this project is a count, and a
 * count is the one thing a ranked table is bad at showing. Ninety-three of
 * ninety-five rows score exactly 0.0; a table sorted by score puts the two
 * survivors on top and lets a reader scroll away believing they are the point.
 * The matrix puts the ratio first and cannot be scrolled past.
 *
 * Rows are sectors rather than an arbitrary run, so the second reading is free:
 * whether the surviving names cluster anywhere. They do not.
 */

/**
 * Shading runs from lit to unlit in the order a reader should care about.
 * These are greys, not hues — see the palette note in globals.css. Only the
 * critic's own veto earns the amber, because that is the one category where a
 * threshold was applied rather than missing.
 */
const BASIS_CELL: Record<string, string> = {
  RANKED: "bg-bright",
  NOT_LONG: "bg-mid",
  FLAGGED_OUT: "bg-bar-dim",
  NO_EVIDENCE: "bg-rule-hi",
  NO_FORECAST: "bg-inset ring-1 ring-inset ring-rule",
};

const LEGEND_ORDER = [
  "RANKED",
  "NOT_LONG",
  "FLAGGED_OUT",
  "NO_EVIDENCE",
  "NO_FORECAST",
] as const;

function cellClass(basis: string | null | undefined): string {
  return BASIS_CELL[basis ?? ""] ?? "bg-rule";
}

export function UniverseStrip({ entries }: { entries: LeaderboardEntry[] }) {
  const sectors = groupBySector(entries);
  const counts = countByBasis(entries);
  const ranked = entries.filter((e) => e.score_basis === "RANKED");

  return (
    <section aria-labelledby="universe-heading" className="border border-rule bg-shell">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-b border-rule px-4 py-2.5">
        <h2
          id="universe-heading"
          className="font-display text-[0.9rem] font-bold tracking-tight text-bright"
        >
          {ranked.length} of {entries.length} rated
        </h2>
        <p className="font-prose text-[0.78rem] text-dim">
          One cell per name, grouped by sector. A lit cell cleared the evidence
          gate with a forecast pointing up.
        </p>
      </div>

      <div className="space-y-px px-4 py-3">
        {sectors.map(({ sector, rows }) => (
          <div key={sector} className="flex items-center gap-3">
            <Eyebrow className="w-[9.5rem] shrink-0 truncate text-right leading-4">
              {sector}
            </Eyebrow>
            <div className="flex flex-wrap gap-[3px]">
              {rows.map((entry) => (
                <Link
                  key={entry.ticker}
                  href={`/stocks/${entry.ticker}`}
                  title={`${entry.company ?? entry.ticker} (${entry.ticker.replace(/\.NS$/, "")}) — ${scoreBasisLabel(entry.score_basis)}`}
                  className={cx(
                    "h-3.5 w-3.5 transition-transform hover:scale-125",
                    cellClass(entry.score_basis),
                  )}
                >
                  <span className="sr-only">
                    {entry.company ?? entry.ticker}: {scoreBasisLabel(entry.score_basis)}
                  </span>
                </Link>
              ))}
            </div>
            <span className="ml-auto shrink-0 text-[0.66rem] text-dim">
              {rows.length}
            </span>
          </div>
        ))}
      </div>

      <dl className="flex flex-wrap gap-x-5 gap-y-1.5 border-t border-rule px-4 py-2.5">
        {LEGEND_ORDER.filter((basis) => counts[basis] > 0).map((basis) => (
          <div
            key={basis}
            className="flex items-center gap-1.5"
            title={scoreBasisExplainer(basis)}
          >
            <span
              aria-hidden
              className={cx("h-2.5 w-2.5 shrink-0", cellClass(basis))}
            />
            <dt className="text-[0.66rem] uppercase tracking-[0.1em] text-dim">
              {scoreBasisLabel(basis)}
            </dt>
            <dd className="text-[0.7rem] text-text">{counts[basis]}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function groupBySector(entries: LeaderboardEntry[]) {
  const map = new Map<string, LeaderboardEntry[]>();
  for (const entry of entries) {
    const sector = entry.sector ?? "Unclassified";
    const bucket = map.get(sector);
    if (bucket) bucket.push(entry);
    else map.set(sector, [entry]);
  }

  return Array.from(map, ([sector, rows]) => ({
    sector,
    // Lit cells first inside a sector, so a rated name is never buried at the
    // end of a 22-cell row where it reads as noise.
    rows: rows.sort(
      (a, b) =>
        LEGEND_ORDER.indexOf(a.score_basis as (typeof LEGEND_ORDER)[number]) -
          LEGEND_ORDER.indexOf(b.score_basis as (typeof LEGEND_ORDER)[number]) ||
        a.ticker.localeCompare(b.ticker),
    ),
  })).sort((a, b) => b.rows.length - a.rows.length || a.sector.localeCompare(b.sector));
}

function countByBasis(entries: LeaderboardEntry[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const entry of entries) {
    const key = entry.score_basis ?? "NO_FORECAST";
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return counts;
}
