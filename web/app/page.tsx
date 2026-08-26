import Link from "next/link";

import { LeaderboardBoard } from "@/components/leaderboard/board";
import { UniverseStrip } from "@/components/leaderboard/universe-strip";
import { Eyebrow, Note, Panel, Readout } from "@/components/ui";
import { getLeaderboard } from "@/lib/api";
import { pctPoints, signed } from "@/lib/format";
import type { LeaderboardEntry } from "@/lib/types";

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
  title: "Leaderboard",
  description:
    "NIFTY 100 stocks ranked by predicted 30-session excess return, gated behind held-out evidence.",
};

export default async function LeaderboardPage() {
  const data = await getLeaderboard();
  const entries = data.entries;
  const summary = summarise(entries);

  return (
    <div className="space-y-7">
      <header className="max-w-3xl">
        <h1 className="font-display text-[1.15rem] font-bold tracking-tight text-bright">
          Ranked long candidates
        </h1>
        <p className="mt-2 font-prose text-[0.86rem] leading-relaxed text-mid">
          NIFTY 100 names ordered by predicted 30-session return{" "}
          <span className="text-text">in excess of a sector benchmark</span>,
          gated behind held-out evidence from purged walk-forward evaluation
          with a 30-session embargo. A name that fails the gate scores exactly
          zero — it is not ranked low, it is not ranked at all.
        </p>
      </header>

      {/*
        The strip is the hero rather than a row of headline figures, because
        the headline here is a ratio and a ratio is what a stat card is worst
        at conveying. Two lit cells out of ninety-five says the thing in one
        glance that the callout below has to spend a paragraph on.
      */}
      <UniverseStrip entries={entries} />

      {summary.nullResult ? (
        <Note tone="neg" title="The model does not beat its baselines">
          Of {summary.total} names, <strong>{summary.ranked}</strong>{" "}
          {summary.ranked === 1 ? "clears" : "clear"} the evidence gate with a
          forecast pointing up. <strong>{summary.belowBaseline}</strong> score
          below their own majority-class baseline on direction, and{" "}
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
          label="Rated"
          value={`${summary.ranked} / ${summary.total}`}
          tone={summary.ranked === 0 ? "dim" : "neutral"}
          sub="Cleared the evidence gate and predicted to outperform. Everything else scores exactly 0.0, for one of several unrelated reasons."
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

      <LeaderboardBoard entries={entries} />

      <Panel className="px-4 py-3">
        <Eyebrow as="h2">How to read this table</Eyebrow>
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
function summarise(entries: LeaderboardEntry[]) {
  const total = entries.length;
  const mean = (values: number[]) =>
    values.length === 0
      ? Number.NaN
      : values.reduce((a, b) => a + b, 0) / values.length;

  const ics = entries
    .map((e) => e.eval_rank_ic)
    .filter((v): v is number => v !== null && Number.isFinite(v));
  const hits = entries
    .map((e) => e.eval_hit_rate)
    .filter((v): v is number => v !== null && Number.isFinite(v));
  const bases = entries
    .map((e) => e.eval_baseline_hit_rate)
    .filter((v): v is number => v !== null && Number.isFinite(v));

  const meanHit = mean(hits);
  const meanBaseline = mean(bases);

  const paired = entries.filter(
    (e) => e.eval_hit_rate !== null && e.eval_baseline_hit_rate !== null,
  );

  const ranked = entries.filter((e) => e.score_basis === "RANKED").length;
  const belowBaseline = paired.filter(
    (e) => (e.eval_hit_rate as number) < (e.eval_baseline_hit_rate as number),
  ).length;
  const degenerate = paired.filter(
    (e) =>
      Math.abs(
        (e.eval_hit_rate as number) - (e.eval_baseline_hit_rate as number),
      ) < 1e-6,
  ).length;

  return {
    total,
    ranked,
    strong: entries.filter((e) => e.forecast_confidence === "STRONG").length,
    weak: entries.filter((e) => e.forecast_confidence === "WEAK").length,
    insufficient: entries.filter(
      (e) => e.forecast_confidence === "INSUFFICIENT" || !e.forecast_confidence,
    ).length,
    meanIc: ics.length ? mean(ics) : Number.NaN,
    positiveIc: ics.filter((v) => v > 0).length,
    negativeIc: ics.filter((v) => v < 0).length,
    meanHit,
    meanBaseline,
    hitEdge: meanHit - meanBaseline,
    beatsRw: entries.filter((e) => e.eval_beats_random_walk === true).length,
    belowBaseline,
    degenerate,
    // The honest headline: a positive mean IC and a majority of stocks
    // beating their own baseline are both required before this stops firing.
    nullResult:
      !(ics.length > 0 && mean(ics) > 0) || belowBaseline > paired.length / 2,
  };
}
