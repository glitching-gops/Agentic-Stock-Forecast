import Link from "next/link";

import { Eyebrow } from "@/components/ui";
import {
  cx,
  evidenceState,
  evidenceStateExplainer,
  evidenceStateLabel,
} from "@/lib/format";
import type { CurrentForecast, EvidenceState } from "@/lib/types";

/**
 * The whole universe as one matrix — a row per sector, a cell per name, shaded
 * by what the held-out evaluation supports for that name.
 *
 * This exists because the honest headline of this project is a count, and a
 * count is the one thing a table is bad at showing. Three of ninety-six tickers
 * clear the evidence gate, which is exactly what chance produces; a table
 * sorted by anything puts the survivors on top and lets a reader scroll away
 * believing they are the point. The matrix puts the ratio first and cannot be
 * scrolled past.
 *
 * Rows are sectors rather than an arbitrary run, so the second reading is free:
 * whether the graded names cluster anywhere. They do not.
 */

/**
 * Shading runs from lit to unlit in the order a reader should care about.
 * These are greys, not hues — see the palette note in globals.css. Only the
 * "no forecast at all" case is drawn as an outline rather than a fill, because
 * it is the one category that is an absence rather than a measurement.
 */
const STATE_CELL: Record<EvidenceState, string> = {
  STRONG: "bg-bright",
  WEAK: "bg-mid",
  INSUFFICIENT: "bg-rule-hi",
  NO_FORECAST: "bg-inset ring-1 ring-inset ring-rule",
};

const LEGEND_ORDER: EvidenceState[] = [
  "STRONG",
  "WEAK",
  "INSUFFICIENT",
  "NO_FORECAST",
];

export function UniverseStrip({
  forecasts,
}: {
  forecasts: CurrentForecast[];
}) {
  const sectors = groupBySector(forecasts);
  const counts = countByState(forecasts);
  const graded = forecasts.filter((f) => {
    const state = evidenceState(f);
    return state === "STRONG" || state === "WEAK";
  });

  return (
    <section
      aria-labelledby="universe-heading"
      className="border border-rule bg-shell"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1 border-b border-rule px-4 py-2.5">
        <h2
          id="universe-heading"
          className="font-display text-[0.9rem] font-bold tracking-tight text-bright"
        >
          {graded.length} of {forecasts.length} carry held-out evidence
        </h2>
        <p className="font-prose text-[0.78rem] text-dim">
          One cell per name, grouped by sector. A lit cell passed at least two
          of its three checks on data the model never trained on.
        </p>
      </div>

      <div className="space-y-px px-4 py-3">
        {sectors.map(({ sector, rows }) => (
          <div key={sector} className="flex items-center gap-3">
            <Eyebrow className="w-[9.5rem] shrink-0 truncate text-right leading-4">
              {sector}
            </Eyebrow>
            <div className="flex flex-wrap gap-[3px]">
              {rows.map((forecast) => (
                <Link
                  key={forecast.ticker}
                  href={`/stocks/${forecast.ticker}`}
                  title={`${forecast.company ?? forecast.ticker} (${forecast.ticker.replace(/\.NS$/, "")}) — ${evidenceStateLabel(evidenceState(forecast))}`}
                  className={cx(
                    "h-3.5 w-3.5 transition-transform hover:scale-125",
                    STATE_CELL[evidenceState(forecast)],
                  )}
                >
                  <span className="sr-only">
                    {forecast.company ?? forecast.ticker}:{" "}
                    {evidenceStateLabel(evidenceState(forecast))}
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
        {LEGEND_ORDER.filter((state) => counts[state] > 0).map((state) => (
          <div
            key={state}
            className="flex items-center gap-1.5"
            title={evidenceStateExplainer(state)}
          >
            <span
              aria-hidden
              className={cx("h-2.5 w-2.5 shrink-0", STATE_CELL[state])}
            />
            <dt className="text-[0.66rem] uppercase tracking-[0.1em] text-dim">
              {evidenceStateLabel(state)}
            </dt>
            <dd className="text-[0.7rem] text-text">{counts[state]}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function groupBySector(forecasts: CurrentForecast[]) {
  const map = new Map<string, CurrentForecast[]>();
  for (const forecast of forecasts) {
    const sector = forecast.sector ?? "Unclassified";
    const bucket = map.get(sector);
    if (bucket) bucket.push(forecast);
    else map.set(sector, [forecast]);
  }

  return Array.from(map, ([sector, rows]) => ({
    sector,
    // Lit cells first inside a sector, so a graded name is never buried at the
    // end of a 22-cell row where it reads as noise.
    rows: rows.sort(
      (a, b) =>
        LEGEND_ORDER.indexOf(evidenceState(a)) -
          LEGEND_ORDER.indexOf(evidenceState(b)) ||
        a.ticker.localeCompare(b.ticker),
    ),
  })).sort(
    (a, b) => b.rows.length - a.rows.length || a.sector.localeCompare(b.sector),
  );
}

function countByState(
  forecasts: CurrentForecast[],
): Record<EvidenceState, number> {
  const counts = {
    STRONG: 0,
    WEAK: 0,
    INSUFFICIENT: 0,
    NO_FORECAST: 0,
  } satisfies Record<EvidenceState, number>;
  for (const forecast of forecasts) counts[evidenceState(forecast)] += 1;
  return counts;
}
