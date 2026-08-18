import Link from "next/link";

import { LeaderboardBoard } from "@/components/leaderboard/board";
import { Callout, Card, Stat } from "@/components/ui";
import { getLeaderboard } from "@/lib/api";
import { pctPoints, signed, timestamp } from "@/lib/format";
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
    <div className="space-y-8">
      <header className="flex flex-wrap items-end justify-between gap-6">
        <div className="max-w-2xl">
          <h1 className="text-2xl font-semibold tracking-tight text-mist-100">
            Leaderboard
          </h1>
          <p className="mt-2 text-sm leading-relaxed text-mist-400">
            NIFTY 100 stocks ranked by predicted 30-session return{" "}
            <em className="not-italic text-mist-300">
              in excess of a sector benchmark
            </em>
            , gated behind held-out evidence from purged walk-forward
            evaluation with a 30-session embargo.
          </p>
        </div>
        <div className="text-right">
          <div className="text-[0.65rem] font-semibold uppercase tracking-[0.09em] text-mist-500">
            Pipeline last run
          </div>
          <div className="nums mt-0.5 text-sm font-semibold text-brand-400">
            {timestamp(data.last_updated)}
          </div>
          <div className="nums mt-0.5 text-xs text-mist-500">
            {data.total} stocks in the universe
          </div>
        </div>
      </header>

      {summary.nullResult ? (
        <Callout tone="negative" title="The model does not beat its baselines.">
          Of {summary.total} stocks, <strong>{summary.ranked}</strong>{" "}
          {summary.ranked === 1 ? "clears" : "clear"} the evidence gate with a
          forecast pointing up.{" "}
          <strong>{summary.belowBaseline}</strong> score below their own
          majority-class baseline on direction and{" "}
          <strong>{summary.degenerate}</strong> tie it exactly by predicting a
          single direction for every row. Treat everything here as a research
          output with no demonstrated edge — the calibrated intervals hold up,
          the point forecast does not.{" "}
          <Link href="/methodology">How this is measured →</Link>
        </Callout>
      ) : null}

      <section
        aria-label="Universe summary"
        className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4"
      >
        <Stat
          label="Ranked"
          value={`${summary.ranked} of ${summary.total}`}
          tone={summary.ranked === 0 ? "muted" : "brand"}
          sub="Cleared the evidence gate and predicted to outperform."
          help="Everything else scores exactly 0.0, for one of several unrelated reasons."
        />
        <Stat
          label="Evidence grade"
          value={
            <span className="text-xl">
              {summary.strong} strong · {summary.weak} weak
            </span>
          }
          sub={`${summary.insufficient} with no held-out evidence at all.`}
          help="Two of three held-out checks are needed for WEAK; all three must run and pass for STRONG."
        />
        <Stat
          label="Directional accuracy"
          value={pctPoints(summary.meanHit)}
          tone={summary.hitEdge > 0 ? "positive" : "negative"}
          sub={`${summary.hitEdge >= 0 ? "+" : "−"}${Math.abs(summary.hitEdge).toFixed(1)}pp against a ${summary.meanBaseline.toFixed(1)}% majority-class baseline.`}
          help="Accuracy means nothing on its own. The baseline is always predicting the more common direction over the same window."
        />
        <Stat
          label="Mean rank IC"
          value={signed(summary.meanIc)}
          tone={summary.meanIc > 0 ? "positive" : "negative"}
          sub={`${summary.positiveIc} positive · ${summary.negativeIc} negative. Beats a random walk on ${summary.beatsRw} of ${summary.total}.`}
          help="Out-of-sample Spearman correlation between predicted and realised excess return. 0 = no skill; 0.02–0.05 is typical for a real technical signal."
        />
      </section>

      <LeaderboardBoard entries={entries} />

      <Card className="p-4">
        <h2 className="text-[0.68rem] font-semibold uppercase tracking-[0.09em] text-mist-500">
          How to read this table
        </h2>
        <p className="mt-2 max-w-4xl text-xs leading-relaxed text-mist-400">
          {data.methodology}
        </p>
      </Card>
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
