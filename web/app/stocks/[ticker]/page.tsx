import Link from "next/link";
import { notFound } from "next/navigation";

import { PriceChart, type PricePoint } from "@/components/charts/price-chart";
import { SignalsPanel } from "@/components/stock/signals-panel";
import { Tabs } from "@/components/tabs";
import {
  Badge,
  Callout,
  Card,
  Empty,
  Prose,
  Stat,
  evidenceTone,
  verdictTone,
} from "@/components/ui";
import {
  getForecast,
  getHeadlines,
  getSignals,
  getStocks,
  NotFoundError,
  soft,
} from "@/lib/api";
import {
  dateOnly,
  daysAgo,
  money,
  pctPoints,
  probability,
  signed,
  signedPct,
  timestamp,
} from "@/lib/format";
import type { Forecast, Headline, SignalRow } from "@/lib/types";

// Route segment config must be a literal — Next cannot statically analyse an
// imported constant here. Keep in step with REVALIDATE_SECONDS in lib/api.ts.
export const revalidate = 3600;
export const maxDuration = 60;

/**
 * Pre-render the whole universe at build time.
 *
 * 95 pages against a free-tier upstream is a slower build, and it is worth it:
 * every stock page then lives on the CDN, so no visitor ever waits on Render's
 * ~82s cold start to read one. `dynamicParams` stays on so a ticker added to
 * the index between deploys still resolves — it renders on demand and is
 * cached from then on.
 *
 * The soft fallback matters. If `/api/stocks` is unreachable at build time,
 * returning `[]` yields a deployable site where every stock page is generated
 * on demand, instead of a failed deploy.
 */
export async function generateStaticParams() {
  const { stocks } = await soft(getStocks(), { stocks: [], total: 0 });
  return stocks.map((stock) => ({ ticker: stock.ticker }));
}

export const dynamicParams = true;

export async function generateMetadata({
  params,
}: {
  params: Promise<{ ticker: string }>;
}) {
  const { ticker } = await params;
  const forecast = await soft(getForecast(ticker), null);
  if (!forecast) return { title: ticker };
  return {
    title: `${forecast.company ?? forecast.ticker}`,
    description: `Predicted 30-session excess return, held-out evidence and critic review for ${forecast.company ?? forecast.ticker} (${forecast.ticker}).`,
  };
}

export default async function StockPage({
  params,
}: {
  params: Promise<{ ticker: string }>;
}) {
  const { ticker } = await params;

  let forecast: Forecast;
  try {
    forecast = await getForecast(ticker);
  } catch (error) {
    if (error instanceof NotFoundError) notFound();
    throw error;
  }

  // Soft: the chart and the headline list are enrichments. Losing either
  // should not take down the forecast, which is the reason for the page.
  const [signals, headlines] = await Promise.all([
    soft(getSignals(ticker), { ticker, signals_df: [], latest_signals: {}, rows: 0 }),
    soft(getHeadlines(ticker), [] as Headline[]),
  ]);

  const history = signals.signals_df;
  const grade = forecast.forecast_confidence ?? "INSUFFICIENT";
  const evaluation = forecast.evaluation;
  const benchmark =
    forecast.benchmark_name ?? forecast.benchmark_ticker ?? "its benchmark";

  return (
    <div className="space-y-6">
      <Link
        href="/"
        className="inline-flex items-center gap-1.5 text-sm text-mist-400 hover:text-brand-300"
      >
        ← Leaderboard
      </Link>

      <header className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight text-mist-100">
            {forecast.company ?? forecast.ticker}
          </h1>
          <div className="mt-1.5 flex flex-wrap items-center gap-2 text-sm text-mist-500">
            <span className="font-mono">{forecast.ticker}</span>
            {forecast.sector ? <span>· {forecast.sector}</span> : null}
            <span>· NSE</span>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone={evidenceTone(grade)}>
            {grade === "INSUFFICIENT" ? "No evidence" : `${grade} evidence`}
          </Badge>
          <Badge tone={verdictTone(forecast.critic_verdict)}>
            {forecast.critic_verdict ?? "no verdict"}
          </Badge>
        </div>
      </header>

      {forecast.critic_verdict === "REJECTED" ? (
        <Callout tone="negative" title="Rejected by the critic.">
          This forecast did not survive review. Read the Agent review tab before
          taking anything on this page at face value.
        </Callout>
      ) : null}

      {grade !== "STRONG" ? (
        <EvidenceWarning
          grade={grade}
          hit={evaluation?.hit_rate ?? null}
          baseline={evaluation?.baseline_hit_rate ?? null}
        />
      ) : null}

      <Tabs
        tabs={[
          {
            id: "overview",
            label: "Overview",
            content: (
              <Overview
                forecast={forecast}
                benchmark={benchmark}
                history={history}
              />
            ),
          },
          {
            id: "signals",
            label: "Signals",
            content:
              history.length === 0 ? (
                <Empty title="No signal history stored">
                  The pipeline has not written signal rows for this ticker.
                </Empty>
              ) : (
                <SignalsPanel
                  latest={signals.latest_signals}
                  history={history}
                />
              ),
          },
          {
            id: "sentiment",
            label: "Sentiment",
            content: <SentimentPanel ticker={forecast.ticker} headlines={headlines} />,
          },
          {
            id: "review",
            label: "Agent review",
            badge:
              forecast.critic_flags.length > 0 ? (
                <span className="nums rounded bg-neg-500/15 px-1.5 text-[0.65rem] font-semibold text-neg-500">
                  {forecast.critic_flags.length}
                </span>
              ) : undefined,
            content: <AgentReview forecast={forecast} />,
          },
        ]}
      />

      <p className="text-xs text-mist-500">
        Forecast written {timestamp(forecast.last_updated)}.
        {evaluation?.evaluated_at
          ? ` Evidence last measured ${timestamp(evaluation.evaluated_at)}.`
          : ""}
      </p>
    </div>
  );
}

/* ── Overview ──────────────────────────────────────────────────────────── */

function Overview({
  forecast,
  benchmark,
  history,
}: {
  forecast: Forecast;
  benchmark: string;
  history: SignalRow[];
}) {
  const excess = forecast.pred_excess_return;
  const evaluation = forecast.evaluation;
  const coverage = forecast.interval_coverage;

  const direction =
    forecast.direction === "OUTPERFORM"
      ? { symbol: "▲", tone: "positive" as const }
      : forecast.direction === "UNDERPERFORM"
        ? { symbol: "▼", tone: "negative" as const }
        : { symbol: "–", tone: "muted" as const };

  const contradicts =
    excess !== null &&
    excess < 0 &&
    forecast.prob_outperform !== null &&
    forecast.prob_outperform > 0.5;

  const points: PricePoint[] = history.map((row) => ({
    date: row.date,
    close: numberOrNull(row.close),
    sma_20: numberOrNull(row.sma_20),
    ema_21: numberOrNull(row.ema_21),
    ema_50: numberOrNull(row.ema_50),
    bb:
      numberOrNull(row.bb_lower) !== null && numberOrNull(row.bb_upper) !== null
        ? [row.bb_lower as number, row.bb_upper as number]
        : null,
    obv: numberOrNull(row.obv),
  }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Stat
          label="Current price"
          value={money(forecast.current_price)}
          tone="neutral"
          sub={`Random-walk baseline forecast: ${money(forecast.random_walk_price)}.`}
          help="The baseline this model has to beat on magnitude is simply repeating today's price."
        />

        <Stat
          label="Implied 30-session target"
          value={
            <span className="flex flex-wrap items-baseline gap-2">
              {money(forecast.forecast_price)}
              <span
                className={
                  direction.tone === "positive"
                    ? "text-sm text-pos-500"
                    : direction.tone === "negative"
                      ? "text-sm text-neg-500"
                      : "text-sm text-mist-500"
                }
              >
                {direction.symbol} {signedPct(excess)}
              </span>
            </span>
          }
          sub={
            <>
              {coverage
                ? `${(coverage * 100).toFixed(0)}% interval: `
                : "Interval: "}
              {money(forecast.interval_low)} – {money(forecast.interval_high)}.
              <br />
              Assumes {benchmark} is flat — the model forecasts relative
              performance only.
            </>
          }
          help="Derived from the excess-return forecast. The model says nothing about where the index itself goes."
        />

        <Stat
          label="P(outperform)"
          value={probability(forecast.prob_outperform)}
          tone={
            (forecast.prob_outperform ?? 0.5) > 0.55
              ? "positive"
              : (forecast.prob_outperform ?? 0.5) < 0.45
                ? "negative"
                : "muted"
          }
          sub={
            <>
              Against {benchmark}
              {forecast.benchmark_sector_specific === false
                ? " (no sector index available, so NIFTY 50 is used)"
                : ""}
              . 50% is a coin flip.
            </>
          }
          help="Calibrated on out-of-sample conformal residuals."
        />

        <Stat
          label="Held-out evidence"
          value={
            forecast.forecast_confidence === "INSUFFICIENT"
              ? "NONE"
              : (forecast.forecast_confidence ?? "—")
          }
          tone={
            forecast.forecast_confidence === "STRONG"
              ? "positive"
              : forecast.forecast_confidence === "WEAK"
                ? "warning"
                : "negative"
          }
          sub={
            <>
              Rank IC {signed(evaluation?.rank_ic)}
              <br />
              Hit {pctPoints(evaluation?.hit_rate)} vs{" "}
              {pctPoints(evaluation?.baseline_hit_rate)} baseline
            </>
          }
          help="From purged walk-forward folds the model never trained on."
        />
      </div>

      {contradicts ? (
        <Callout tone="warning" title="The model contradicts itself here.">
          The point forecast predicts underperformance while the calibrated
          probability sits above a coin flip. That combination means the model
          runs biased low on this ticker rather than that it likes the stock —
          the calibrated probability is the share of residuals above{" "}
          {signedPct(excess !== null ? -excess : null)}, not an independent
          opinion. The composite refuses to rank on the cheerier half, so this
          stock scores zero.
        </Callout>
      ) : null}

      <Card className="p-4">
        <h2 className="mb-4 text-sm font-semibold text-mist-100">
          Price and technical indicators
        </h2>
        <PriceChart data={points} />
      </Card>

      <p className="max-w-4xl text-xs leading-relaxed text-mist-500">
        Performance figures come from purged walk-forward validation with a
        30-session embargo — folds the model never trained on — and are stated
        before transaction costs. The evaluation runs weekly, so the evidence
        beside a price may be up to a week older than the price itself.
      </p>
    </div>
  );
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/* ── Evidence warning ──────────────────────────────────────────────────── */

function EvidenceWarning({
  grade,
  hit,
  baseline,
}: {
  grade: string;
  hit: number | null;
  baseline: number | null;
}) {
  const degenerate =
    hit !== null && baseline !== null && Math.abs(hit - baseline) < 1e-6;

  return (
    <Callout
      tone={grade === "WEAK" ? "warning" : "negative"}
      title={
        grade === "WEAK"
          ? "Weak held-out evidence."
          : "No held-out evidence of skill."
      }
    >
      {grade === "WEAK"
        ? "Two of three held-out checks passed. That is the minimum this system will grade at all, and it halves the composite score rather than endorsing the forecast."
        : "This model failed its held-out checks, so its forecast is excluded from the ranking however large a move it predicts."}
      {hit !== null && baseline !== null ? (
        <>
          {" "}
          Out-of-sample hit rate {pctPoints(hit)} against a{" "}
          {pctPoints(baseline)} majority-class baseline.
        </>
      ) : null}
      {degenerate ? (
        <>
          {" "}
          <strong>
            Those two figures are identical, which means the model predicted a
            single direction for every row
          </strong>{" "}
          — it matched the baseline by never disagreeing with it, not by having
          a view.
        </>
      ) : null}
    </Callout>
  );
}

/* ── Sentiment ─────────────────────────────────────────────────────────── */

function SentimentPanel({
  ticker,
  headlines,
}: {
  ticker: string;
  headlines: Headline[];
}) {
  return (
    <div className="space-y-5">
      <Callout tone="muted" title="Not scored.">
        Headlines are collected but never scored — no sentiment model runs in
        this pipeline, and sentiment is not a model input either way. It was
        removed as a feature because it existed only for the current date: zero
        across every training row and non-zero only for the row being
        predicted, which is a train/serve mismatch at exactly the row that
        matters. A gauge parked at neutral would claim a measurement that was
        never taken, so there is no gauge.
      </Callout>

      <div>
        <h3 className="mb-3 text-sm font-semibold text-mist-100">
          Recent headlines for {ticker}
        </h3>
        {headlines.length === 0 ? (
          <Empty title="No headlines stored">
            Nothing has been collected for this ticker yet.
          </Empty>
        ) : (
          <ul className="space-y-2">
            {headlines.map((item, index) => (
              <Card as="li" key={`${item.headline}-${index}`} className="p-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <span className="min-w-0 flex-1 text-sm leading-relaxed text-mist-200">
                    {item.headline}
                  </span>
                  <div className="flex shrink-0 items-center gap-2">
                    <Badge tone="muted">
                      {item.sentiment_label ?? "unscored"}
                    </Badge>
                    <span className="nums text-xs text-mist-500">
                      {dateOnly(item.date)}
                    </span>
                  </div>
                </div>
              </Card>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

/* ── Agent review ──────────────────────────────────────────────────────── */

function AgentReview({ forecast }: { forecast: Forecast }) {
  const staleness = daysAgo(forecast.evaluation?.evaluated_at);

  return (
    <div className="space-y-6">
      <Card className="p-4">
        <h3 className="text-[0.68rem] font-semibold uppercase tracking-[0.09em] text-brand-400">
          Forecasting agent · signal narrative
        </h3>
        <p className="mt-2.5 max-w-3xl leading-relaxed text-mist-200">
          {forecast.signal_narrative ?? "No narrative was generated."}
        </p>
        <p className="mt-3 text-xs text-mist-500">
          Written from the signal snapshot alone. The model is not shown the
          forecast, and it produces, adjusts and reviews no number.
        </p>
      </Card>

      <div>
        <h3 className="mb-3 text-sm font-semibold text-mist-100">
          Critic review
        </h3>

        <div className="space-y-3 border-l-2 border-ink-500 pl-5">
          <div>
            <Badge tone={verdictTone(forecast.critic_verdict)}>
              Verdict: {forecast.critic_verdict ?? "none"}
            </Badge>
          </div>

          {forecast.critic_reasoning ? (
            <Card className="p-3.5">
              <div className="text-[0.68rem] font-semibold uppercase tracking-[0.09em] text-mist-500">
                Reasoning
              </div>
              <p className="mt-1.5 whitespace-pre-line leading-relaxed text-mist-200">
                {forecast.critic_reasoning}
              </p>
            </Card>
          ) : null}

          {forecast.critic_flags.length > 0 ? (
            <div>
              <div className="mb-1.5 text-[0.68rem] font-semibold uppercase tracking-[0.09em] text-mist-500">
                Flags raised · −5 points each
              </div>
              <ul className="space-y-1.5">
                {forecast.critic_flags.map((flag) => (
                  <Card
                    as="li"
                    key={flag}
                    className="border-l-4 border-l-neg-500 p-3 text-sm text-mist-200"
                  >
                    {flag}
                  </Card>
                ))}
              </ul>
            </div>
          ) : null}

          <div className="text-xs text-mist-500">
            <span className="uppercase tracking-[0.09em]">Verdict set by</span>{" "}
            <Badge tone="warning" className="ml-1">
              {forecast.critic_source ?? "evidence_gate"}
            </Badge>
            <p className="mt-2 max-w-3xl leading-relaxed">
              The evidence gate is deterministic and driven entirely by held-out
              walk-forward metrics. The LLM review may add flags and downgrade a
              verdict; it can never raise one. It sees numbers it has no way to
              verify, so it is allowed to raise doubt and not to certify.
            </p>
          </div>
        </div>
      </div>

      <Prose>
        <h3>Why this evidence may lag the price</h3>
        <p>
          The evaluation behind this verdict is refreshed <strong>weekly</strong>
          , not daily. Hyperparameters are searched with Optuna inside each
          training fold, which is expensive enough that running it every day
          once starved a production server&rsquo;s CPU for over an hour and
          crashed it. Whether a model form has skill does not change day to day,
          so it is measured once a week and each day refits with the
          hyperparameters that search already found.
          {staleness !== null ? (
            <>
              {" "}
              This evidence was measured{" "}
              <strong>
                {staleness === 0
                  ? "today"
                  : `${staleness} day${staleness === 1 ? "" : "s"} ago`}
              </strong>
              .
            </>
          ) : null}
        </p>
      </Prose>
    </div>
  );
}
