import Link from "next/link";
import { notFound } from "next/navigation";

import { PriceChart, type PricePoint } from "@/components/charts/price-chart";
import { SignalsPanel } from "@/components/stock/signals-panel";
import { Tabs } from "@/components/tabs";
import {
  Badge,
  Empty,
  Eyebrow,
  Note,
  Panel,
  Prose,
  Readout,
  SectionHead,
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

/**
 * Next hands back the route segment exactly as it appears in the URL, so a
 * ticker carrying a character that had to be escaped arrives still escaped:
 * `M&M.NS` comes back as `M%26M.NS`. Passing that to `encodeURIComponent` in
 * the API client escapes the percent again and requests `M%2526M.NS`, which
 * is a ticker that does not exist.
 *
 * Exactly one name in the NIFTY 100 contains such a character, which is
 * precisely why this survived every manual check — 94 of 95 pages are correct.
 */
function decodeTicker(raw: string): string {
  try {
    return decodeURIComponent(raw);
  } catch {
    // A malformed escape sequence is not a ticker. Let the API 404 it.
    return raw;
  }
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ ticker: string }>;
}) {
  const ticker = decodeTicker((await params).ticker);
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
  const ticker = decodeTicker((await params).ticker);

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
    <div className="space-y-5">
      <Link
        href="/"
        className="inline-flex items-center gap-1.5 text-[0.7rem] uppercase tracking-[0.14em] text-dim hover:text-bright"
      >
        ← Board
      </Link>

      {/*
        The identity block is a terminal's instrument header: the name, the
        machine-readable key, then the two verdicts that qualify every number
        below it. The badges sit up here rather than beside the figures they
        govern, because they govern all of them.
      */}
      <header className="flex flex-wrap items-end justify-between gap-x-6 gap-y-3 border-b border-rule-hi pb-3">
        <div className="min-w-0">
          <h1 className="font-display text-[1.3rem] font-bold leading-none tracking-tight text-bright">
            {forecast.company ?? forecast.ticker}
          </h1>
          <div className="mt-2 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-[0.68rem] uppercase tracking-[0.12em] text-dim">
            <span className="text-mid">{forecast.ticker}</span>
            <span aria-hidden className="text-rule-hi">
              /
            </span>
            {forecast.sector ? (
              <>
                <span>{forecast.sector}</span>
                <span aria-hidden className="text-rule-hi">
                  /
                </span>
              </>
            ) : null}
            <span>NSE</span>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge tone={evidenceTone(grade)}>
            {grade === "INSUFFICIENT" ? "No evidence" : `${grade} evidence`}
          </Badge>
          <Badge tone={verdictTone(forecast.critic_verdict)}>
            {forecast.critic_verdict ?? "no verdict"}
          </Badge>
        </div>
      </header>

      {forecast.critic_verdict === "REJECTED" ? (
        <Note tone="neg" title="Rejected by the critic">
          This forecast did not survive review. Read the agent review before
          taking anything on this page at face value.
        </Note>
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
            content: (
              <SentimentPanel ticker={forecast.ticker} headlines={headlines} />
            ),
          },
          {
            id: "review",
            label: "Agent review",
            badge:
              forecast.critic_flags.length > 0 ? (
                <span className="border border-neg/50 px-1 text-[0.6rem] font-semibold text-neg">
                  {forecast.critic_flags.length}
                </span>
              ) : undefined,
            content: <AgentReview forecast={forecast} />,
          },
        ]}
      />

      <p className="text-[0.68rem] uppercase tracking-[0.1em] text-dim">
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
      ? { symbol: "▲", cls: "text-pos" }
      : forecast.direction === "UNDERPERFORM"
        ? { symbol: "▼", cls: "text-neg" }
        : { symbol: "–", cls: "text-dim" };

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
      <div className="grid grid-cols-1 gap-x-6 gap-y-5 sm:grid-cols-2 xl:grid-cols-4">
        <Readout
          label="Current price"
          value={money(forecast.current_price)}
          help="The baseline this model has to beat on magnitude is simply repeating today's price."
          sub={`Random-walk baseline forecast: ${money(forecast.random_walk_price)}.`}
        />

        <Readout
          label="Implied 30-session target"
          help="Derived from the excess-return forecast. The model says nothing about where the index itself goes."
          value={
            <span className="flex flex-wrap items-baseline gap-2">
              {money(forecast.forecast_price)}
              <span className={`font-mono text-[0.8rem] ${direction.cls}`}>
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
        />

        <Readout
          label="P(outperform)"
          value={probability(forecast.prob_outperform)}
          help="Calibrated on out-of-sample conformal residuals."
          tone={
            (forecast.prob_outperform ?? 0.5) > 0.55
              ? "pos"
              : (forecast.prob_outperform ?? 0.5) < 0.45
                ? "neg"
                : "dim"
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
        />

        <Readout
          label="Held-out evidence"
          help="From purged walk-forward folds the model never trained on."
          value={
            forecast.forecast_confidence === "INSUFFICIENT"
              ? "NONE"
              : (forecast.forecast_confidence ?? "—")
          }
          tone={
            forecast.forecast_confidence === "STRONG"
              ? "pos"
              : forecast.forecast_confidence === "WEAK"
                ? "bar"
                : "neg"
          }
          sub={
            <>
              Rank IC {signed(evaluation?.rank_ic)}
              <br />
              Hit {pctPoints(evaluation?.hit_rate)} vs{" "}
              {pctPoints(evaluation?.baseline_hit_rate)} baseline
            </>
          }
        />
      </div>

      {contradicts ? (
        <Note tone="bar" title="The model contradicts itself here">
          The point forecast predicts underperformance while the calibrated
          probability sits above a coin flip. That combination means the model
          runs biased low on this ticker rather than that it likes the stock —
          the calibrated probability is the share of residuals above{" "}
          {signedPct(excess !== null ? -excess : null)}, not an independent
          opinion. Both halves are shown rather than reconciled: a model that
          disagrees with itself about a stock is telling you something the
          average of the two would hide.
        </Note>
      ) : null}

      <Panel className="p-4">
        <SectionHead as="h3" title="Price and technical indicators" />
        <PriceChart data={points} />
      </Panel>

      <p className="max-w-[88ch] font-prose text-[0.8rem] leading-relaxed text-dim">
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
    <Note
      tone={grade === "WEAK" ? "bar" : "neg"}
      title={
        grade === "WEAK"
          ? "Weak held-out evidence"
          : "No held-out evidence of skill"
      }
    >
      {grade === "WEAK"
        ? "Two of three held-out checks passed. That is the minimum this system grades at all, and it is not an endorsement — the thresholds are low, and only one of the three checks tests significance."
        : "This model failed at least two of its three held-out checks. The forecast below is still shown, because withholding it would be less informative than showing it with this said plainly — but there is no evidence the model forecasts this stock better than chance."}
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
    </Note>
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
      <Note tone="dim" title="Not scored">
        Headlines are collected but never scored — no sentiment model runs in
        this pipeline, and sentiment is not a model input either way. It was
        removed as a feature because it existed only for the current date: zero
        across every training row and non-zero only for the row being
        predicted, which is a train/serve mismatch at exactly the row that
        matters. A gauge parked at neutral would claim a measurement that was
        never taken, so there is no gauge.
      </Note>

      <div>
        <SectionHead
          as="h3"
          title={`Recent headlines — ${ticker}`}
          count={headlines.length || undefined}
        />
        {headlines.length === 0 ? (
          <Empty title="No headlines stored">
            Nothing has been collected for this ticker yet.
          </Empty>
        ) : (
          <ul className="border-t border-rule">
            {headlines.map((item, index) => (
              <li
                key={`${item.headline}-${index}`}
                className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 border-b border-rule px-1 py-2"
              >
                <span className="min-w-0 flex-1 font-prose text-[0.84rem] leading-relaxed text-text">
                  {item.headline}
                </span>
                <div className="flex shrink-0 items-center gap-2">
                  <Badge tone="dim">{item.sentiment_label ?? "unscored"}</Badge>
                  <span className="text-[0.7rem] text-dim">
                    {dateOnly(item.date)}
                  </span>
                </div>
              </li>
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
      <Panel className="p-4">
        <Eyebrow as="h3">Forecasting agent · signal narrative</Eyebrow>
        <p className="mt-2.5 max-w-[80ch] font-prose text-[0.86rem] leading-relaxed text-text">
          {forecast.signal_narrative ?? "No narrative was generated."}
        </p>
        <p className="mt-3 max-w-[80ch] font-prose text-[0.78rem] leading-relaxed text-dim">
          Written from the signal snapshot alone. The model is not shown the
          forecast, and it produces, adjusts and reviews no number.
        </p>
      </Panel>

      <div>
        <SectionHead as="h3" title="Critic review" />

        <div className="space-y-3 border-l border-rule-hi pl-5">
          <div>
            <Badge tone={verdictTone(forecast.critic_verdict)}>
              Verdict: {forecast.critic_verdict ?? "none"}
            </Badge>
          </div>

          {forecast.critic_reasoning ? (
            <Panel className="p-3.5">
              <Eyebrow>Reasoning</Eyebrow>
              <p className="mt-1.5 max-w-[80ch] whitespace-pre-line font-prose text-[0.86rem] leading-relaxed text-text">
                {forecast.critic_reasoning}
              </p>
            </Panel>
          ) : null}

          {forecast.critic_flags.length > 0 ? (
            <div>
              <Eyebrow className="mb-1.5">Flags raised · −5 points each</Eyebrow>
              <ul className="space-y-1.5">
                {forecast.critic_flags.map((flag) => (
                  <li
                    key={flag}
                    className="border-y border-l-2 border-y-rule border-l-neg bg-shell px-3 py-2 font-prose text-[0.84rem] text-text"
                  >
                    {flag}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          <div className="text-[0.7rem] text-dim">
            <Eyebrow as="span">Verdict set by</Eyebrow>{" "}
            <Badge tone="bar" className="ml-1">
              {forecast.critic_source ?? "evidence_gate"}
            </Badge>
            <p className="mt-2 max-w-[80ch] font-prose text-[0.8rem] leading-relaxed">
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
