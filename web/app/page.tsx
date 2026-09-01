import Link from "next/link";

import { ForecastTable } from "@/components/forecasts/table";
import { UniverseStrip } from "@/components/forecasts/universe-strip";
import { Eyebrow, Note, Panel, Readout } from "@/components/ui";
import { getForecasts } from "@/lib/api";
import { evidenceState, pctPoints, signed } from "@/lib/format";
import type { CurrentForecast } from "@/lib/types";

// Route segment config must be a literal — Next cannot statically analyse an
// imported constant here. Keep in step with REVALIDATE_SECONDS in lib/api.ts.
export const revalidate = 3600;

/*
 * Vercel Hobby caps a function at 60s. A warm Render answers this route in
 * ~1.1s; a spun-down one takes ~82s and will blow the cap. That is the
 * intended failure: a revalidation that times out leaves the previous static
 * page in place, so the reader gets slightly stale data instead of an error.
 */
export const maxDuration = 60;

export const metadata = {
  title: "Forecasts",
  description:
    "A 30-session forecast for every stock in a fixed NIFTY 100 universe, each carrying the held-out evidence behind it.",
};

export default async function ForecastsPage() {
  const data = await getForecasts();
  const forecasts = data.forecasts;
  const summary = summarise(forecasts);

  return (
    <div className="space-y-7">
      <header className="max-w-3xl">
        <h1 className="font-display text-[1.15rem] font-bold tracking-tight text-bright">
          A forecast for every stock
        </h1>
        <p className="mt-2 font-prose text-[0.86rem] leading-relaxed text-mid">
          One 30-session prediction per name across a{" "}
          <span className="text-text">fixed universe</span> of {summary.total}{" "}
          NIFTY 100 stocks, selected on data quality alone and never on measured
          accuracy. Each forecast is a return{" "}
          <span className="text-text">in excess of a sector benchmark</span>,
          and each is shown with the held-out evidence behind it. These
          forecasts are <span className="text-text">not ranked against each
          other</span>: too few clear the evidence gate for an ordering to mean
          anything, so every stock is read on its own terms.
        </p>
      </header>

      {/*
        The strip is the hero rather than a row of headline figures, because
        the headline here is a ratio and a ratio is what a stat card is worst
        at conveying. Three lit cells out of eighty-four says the thing in one
        glance that the callout below has to spend a paragraph on.
      */}
      <UniverseStrip forecasts={forecasts} />

      {summary.nullResult ? (
        <Note tone="neg" title="The model does not beat its baselines">
          Of {summary.total} names, <strong>{summary.graded}</strong>{" "}
          {summary.graded === 1 ? "carries" : "carry"} any held-out evidence at
          all — and <strong>3.12</strong> is what chance alone produces from
          these three checks, so that count is not a finding.{" "}
          <strong>{summary.belowBaseline}</strong> score below their own
          majority-class baseline on direction, and{" "}
          <strong>{summary.degenerate}</strong> tie it exactly by predicting a
          single direction for every row. Read everything here as a research
          output with no demonstrated edge: the calibrated intervals hold up,
          the point forecast does not.{" "}
          <Link href="/research">What was tried, and what it measured →</Link>
        </Note>
      ) : null}

      <section
        aria-label="Universe summary"
        className="grid grid-cols-1 gap-x-6 gap-y-5 sm:grid-cols-2 xl:grid-cols-4"
      >
        <Readout
          label="Forecast"
          value={`${summary.withForecast} / ${summary.total}`}
          tone={summary.withForecast === summary.total ? "neutral" : "dim"}
          sub="Names the pipeline produced a prediction for. The rest have too little history, or a benchmark index that has stopped publishing."
        />
        <Readout
          label="Evidence"
          value={
            <span>
              {summary.strong}
              <span className="text-dim"> strong </span>
              {summary.weak}
              <span className="text-dim"> weak</span>
            </span>
          }
          tone={summary.strong > 0 ? "pos" : "dim"}
          sub={`${summary.insufficient} carry no held-out evidence at all. Two of three checks earn WEAK; all three must run and pass for STRONG.`}
        />
        <Readout
          label="Directional accuracy"
          value={pctPoints(summary.meanHit)}
          tone={summary.hitEdge > 0 ? "pos" : "neg"}
          sub={`${summary.hitEdge >= 0 ? "+" : "−"}${Math.abs(summary.hitEdge).toFixed(1)}pp against a ${summary.meanBaseline.toFixed(1)}% majority-class baseline. Accuracy means nothing without the baseline beside it.`}
        />
        <Readout
          label="Mean rank IC"
          value={signed(summary.meanIc)}
          tone={summary.meanIc > 0 ? "pos" : "neg"}
          sub={`${summary.positiveIc} positive, ${summary.negativeIc} negative. Beats a random walk on ${summary.beatsRw} of ${summary.total}.`}
        />
      </section>

      <ForecastTable forecasts={forecasts} />

      <Panel className="px-4 py-3">
        <Eyebrow as="h2">How to read this</Eyebrow>
        <p className="mt-2 max-w-[92ch] font-prose text-[0.8rem] leading-relaxed text-dim">
          {data.methodology}
        </p>
      </Panel>
    </div>
  );
}

/**
 * Aggregates computed here rather than in the API so the page and the table
 * are guaranteed to be describing the same rows. Every mean skips nulls
 * instead of coercing them to zero — most of this table is null, and a mean
 * that counted "not measured" as 0.000 would understate the signal rather than
 * report its absence.
 */
function summarise(forecasts: CurrentForecast[]) {
  const total = forecasts.length;
  const mean = (values: number[]) =>
    values.length === 0
      ? Number.NaN
      : values.reduce((a, b) => a + b, 0) / values.length;

  const ics = forecasts
    .map((f) => f.eval_rank_ic)
    .filter((v): v is number => v !== null && Number.isFinite(v));
  const hits = forecasts
    .map((f) => f.eval_hit_rate)
    .filter((v): v is number => v !== null && Number.isFinite(v));
  const bases = forecasts
    .map((f) => f.eval_baseline_hit_rate)
    .filter((v): v is number => v !== null && Number.isFinite(v));

  const meanHit = mean(hits);
  const meanBaseline = mean(bases);

  const paired = forecasts.filter(
    (f) => f.eval_hit_rate !== null && f.eval_baseline_hit_rate !== null,
  );

  const states = forecasts.map(evidenceState);
  const strong = states.filter((s) => s === "STRONG").length;
  const weak = states.filter((s) => s === "WEAK").length;

  const belowBaseline = paired.filter(
    (f) => (f.eval_hit_rate as number) < (f.eval_baseline_hit_rate as number),
  ).length;
  const degenerate = paired.filter(
    (f) =>
      Math.abs(
        (f.eval_hit_rate as number) - (f.eval_baseline_hit_rate as number),
      ) < 1e-6,
  ).length;

  return {
    total,
    withForecast: states.filter((s) => s !== "NO_FORECAST").length,
    graded: strong + weak,
    strong,
    weak,
    insufficient: states.filter((s) => s === "INSUFFICIENT").length,
    meanIc: ics.length ? mean(ics) : Number.NaN,
    positiveIc: ics.filter((v) => v > 0).length,
    negativeIc: ics.filter((v) => v < 0).length,
    meanHit,
    meanBaseline,
    hitEdge: meanHit - meanBaseline,
    beatsRw: forecasts.filter((f) => f.eval_beats_random_walk === true).length,
    belowBaseline,
    degenerate,
    // The honest headline: a positive mean IC and a majority of stocks
    // beating their own baseline are both required before this stops firing.
    nullResult:
      !(ics.length > 0 && mean(ics) > 0) || belowBaseline > paired.length / 2,
  };
}
