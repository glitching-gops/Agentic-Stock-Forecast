import Link from "next/link";

import {
  Eyebrow,
  Note,
  Panel,
  Prose,
  Readout,
  SectionHead,
} from "@/components/ui";
import { getLeaderboard, soft } from "@/lib/api";
import { pctPoints, signed } from "@/lib/format";
import type { LeaderboardEntry } from "@/lib/types";

// Route segment config must be a literal — Next cannot statically analyse an
// imported constant here. Keep in step with REVALIDATE_SECONDS in lib/api.ts.
export const revalidate = 3600;
export const maxDuration = 60;

export const metadata = {
  title: "Methodology",
  description:
    "What the model predicts, how it is evaluated, what it actually measures, and everything it does not model.",
};

export default async function MethodologyPage() {
  // Soft: the prose is the point of this page and must render even if the
  // measured block cannot.
  const data = await soft(getLeaderboard(), null);
  const measured = data ? measure(data.entries) : null;

  return (
    <div className="max-w-5xl space-y-11">
      <header>
        <h1 className="font-display text-[1.15rem] font-bold tracking-tight text-bright">
          Methodology
        </h1>
        <p className="mt-2 max-w-[78ch] font-prose text-[0.86rem] leading-relaxed text-mid">
          ZeRO ranks NSE stocks by their predicted 30-session return{" "}
          <span className="text-text">relative to a sector benchmark</span>.
          This page states how that is measured, what the measurement currently
          says, and what the system does not model. Every figure here is read
          live from the evaluation harness rather than typed in — a number
          written into a page cannot be audited and goes stale in silence.
        </p>
      </header>

      {/* ── Measured performance ─────────────────────────────────────────── */}
      <section>
        <SectionHead
          title="Measured performance"
          description="Purged walk-forward validation with a 30-session embargo. Hyperparameters are tuned inside each training fold, so no configuration is ever chosen with sight of the rows it is later scored on. Before transaction costs."
        />

        {measured === null ? (
          <Panel className="px-6 py-8 text-center text-[0.8rem] text-dim">
            Evaluation metrics could not be loaded.
          </Panel>
        ) : (
          <>
            <div className="grid grid-cols-1 gap-x-6 gap-y-5 sm:grid-cols-2 xl:grid-cols-4">
              <Readout
                label="Mean rank IC"
                value={signed(measured.meanIc)}
                tone={measured.meanIc > 0 ? "pos" : "neg"}
                help="Out-of-sample Spearman correlation between predicted and realised excess return. 0 = no skill; 0.02–0.05 is typical for a genuine technical signal."
                sub={`${measured.positiveIc} positive, ${measured.negativeIc} negative.`}
              />
              <Readout
                label="Directional accuracy"
                value={pctPoints(measured.meanHit)}
                tone={measured.edge > 0 ? "pos" : "neg"}
                sub={`${measured.edge >= 0 ? "+" : "−"}${Math.abs(measured.edge).toFixed(1)}pp against a ${measured.meanBaseline.toFixed(1)}% majority-class baseline.`}
              />
              <Readout
                label="Beats a random walk"
                value={`${measured.beatsRw} / ${measured.total}`}
                tone={measured.beatsRw > measured.total / 2 ? "pos" : "neg"}
                sub="Mean absolute error better than forecasting zero excess return."
              />
              <Readout
                label="Below their own baseline"
                value={`${measured.belowBaseline} / ${measured.paired}`}
                tone="neg"
                sub={`${measured.degenerate} more tie it exactly by predicting one direction for every row.`}
              />
            </div>

            <div className="mt-5">
              {measured.nullResult ? (
                <Note
                  tone="neg"
                  title="The model does not currently beat its baselines"
                >
                  Mean rank IC is at or below zero and directional accuracy
                  trails the majority-class rate on most stocks. Treat every
                  forecast on this site as a research output with{" "}
                  <strong>no demonstrated edge</strong>. This system is, for
                  now, a measuring instrument that reports its own signal is
                  absent — and that is a more useful thing to publish than a
                  number that flatters it.{" "}
                  <Link href="/research">Everything that was tried →</Link>
                </Note>
              ) : (
                <Note tone="bar" title="Read these honestly">
                  A rank IC near 0.05 is a weak signal — real, but not a licence
                  to trade. Where the model does not beat a random walk on
                  magnitude, the ranking is the output and the rupee target is
                  illustrative only.
                </Note>
              )}
            </div>

            <Prose className="mt-5">
              <p>
                <strong>One result does hold up.</strong> The conformal
                intervals are calibrated: the harness measures realised coverage
                against the nominal 80% every week and the two agree closely.
                The uncertainty quantification works even though the point
                forecast does not — which is why the interval and the
                probability are shown on every stock page while the ranking is
                gated behind evidence most stocks never produce. Realised
                coverage is measured in the weekly job but is not yet persisted
                anywhere the API can read, so it is described here rather than
                displayed as a live figure.
              </p>
            </Prose>
          </>
        )}
      </section>

      {/* ── What it predicts ─────────────────────────────────────────────── */}
      <section>
        <SectionHead title="What the model actually predicts" />
        <Prose>
          <p>
            The target is the{" "}
            <strong>
              30-session log return in excess of the stock&rsquo;s benchmark
              index
            </strong>{" "}
            — not an absolute price.
          </p>
          <p>
            That choice matters. An earlier version predicted the closing price
            30 sessions ahead, which made the error look small for the wrong
            reason: prices are persistent, so simply repeating today&rsquo;s
            price scores well on MAPE. It also capped every forecast at the
            highest price seen in training, because gradient-boosted trees
            cannot extrapolate beyond their training range.
          </p>
          <p>
            The rupee target on each stock page is <strong>derived</strong> from
            the excess-return forecast and assumes the benchmark is flat over
            the horizon. The model forecasts relative performance; it has
            nothing to say about where the market goes.
          </p>
        </Prose>
      </section>

      {/* ── Pipeline ─────────────────────────────────────────────────────── */}
      <section>
        <SectionHead
          title="How it works"
          description="Four stages, in order. The numbering is the data's own dependency chain — nothing downstream can run before what precedes it."
        />
        <div className="border-t border-rule">
          <Step n="1" title="Universe — point-in-time construction">
            <p>
              The tradable universe comes from a rule that references{" "}
              <strong>no model output</strong>: NIFTY 100 membership as of the
              date in question, a 20-day median traded value above ₹25 crore,
              and at least 750 sessions (~3 years) of listed price history.
              Membership is stored as dated intervals, so the system can ask
              &ldquo;who was in the index then?&rdquo; rather than &ldquo;who is
              in it now?&rdquo;
            </p>
            <p>
              <strong>Known limitation:</strong> membership is only recorded
              from the first sync onward. Evaluation windows starting before
              that date fall back to present-day membership and are therefore
              survivorship-biased.
            </p>
          </Step>

          <Step n="2" title="Trading data agent — signals">
            <p>
              Ten years of daily OHLCV and 24 engineered signals: momentum
              (RSI-14, Stochastic %K, Williams %R, ROC-10, lag-1/5 returns),
              trend (SMA-20, EMA-9/21/50, SMA-50 deviation, 52-week proximity),
              volatility (Bollinger width and bands, ATR-14), volume (OBV,
              volume ROC), regime (Hurst exponent over log prices),
              sector-relative momentum over 5/10/20 sessions, and quarterly EPS
              surprise.
            </p>
            <p>
              Earnings surprise attaches to the first session{" "}
              <strong>strictly after</strong> the announcement, because Indian
              results are commonly declared post-close. Prices are stored raw
              and adjusted separately, so a split or dividend cannot splice two
              adjustment bases into one series.
            </p>
          </Step>

          <Step n="3" title="Forecasting agent — model and uncertainty">
            <p>
              <strong>XGBoost</strong>, heavily regularised, trained per stock on
              the excess-return target. Optuna searches hyperparameters{" "}
              <strong>inside each training fold</strong>, seeded so a tuning run
              is reproducible.
            </p>
            <p>
              That search runs <strong>weekly, not daily</strong>. Every ticker
              was originally re-tuned with a full nested search every day, which
              starved a production server&rsquo;s single core for over an hour
              and eventually crashed it. Whether a model form has skill does not
              change day to day, so it is measured once a week; each day refits
              with the hyperparameters that search already found. The evidence
              badge states when it was last measured, which can be up to a week
              before the price beside it.
            </p>
            <p>
              <strong>Split-conformal prediction</strong> calibrates an 80%
              interval on out-of-sample residuals and yields a probability that
              the stock beats its benchmark. Coverage is measured, not assumed.
            </p>
            <p>
              The LLM writes a plain-English read of the signals. It produces,
              adjusts and reviews no number, and it is not shown the forecast
              when writing the narrative.
            </p>
            <p>
              An LSTM and a Ridge meta-learner were previously advertised here.
              Both are archived: the LSTM never wrote a checkpoint and so never
              produced a forecast at all, and the meta-learner was the source of
              the inflated accuracy figures. They return only if an experiment
              shows they beat a linear baseline out-of-sample.
            </p>
          </Step>

          <Step n="4" title="Critic agent — the evidence gate">
            <p>Two jobs, deliberately not mixed.</p>
            <p>
              <strong>The evidence gate</strong> is deterministic and tested. It
              asks one question — has this model shown skill on folds it never
              trained on? — through three checks: out-of-sample rank IC against
              a +0.02 floor, the IC t-statistic against a 2.0 floor, and hit
              rate at least 1.0pp above the majority-class baseline. Two of
              three passing grades the forecast <strong>WEAK</strong>; all three
              must run and pass for <strong>STRONG</strong>; anything less is{" "}
              <strong>INSUFFICIENT</strong>.
            </p>
            <p>
              One check used to be enough. On 2026-08-15 that handed a
              validation badge to a stock whose rank IC was +0.049 while its hit
              rate sat 4.7pp <strong>below</strong> its baseline, its IC
              t-statistic was indistinguishable from noise, and its error was
              worse than a random walk&rsquo;s — with twelve more names on the
              same basis. A badge that one weak correlation can buy is not
              reporting evidence, it is laundering it. Requiring two checks cut
              WEAK from 13 names to 5.
            </p>
            <p>
              <strong>The LLM review</strong> looks for contradictions in the
              signal snapshot and may add flags. It can only{" "}
              <strong>downgrade</strong>. It sees numbers it has no way to
              verify, so it is allowed to raise doubt and never to certify.
            </p>
          </Step>
        </div>
      </section>

      {/* ── Composite ────────────────────────────────────────────────────── */}
      <section>
        <SectionHead
          title="The composite score"
          description="A ranking heuristic in [0, 100]. Not an expected return, and not a price target."
        />
        <div className="overflow-x-auto">
          <table className="w-full min-w-[620px] border-collapse text-[0.78rem]">
            <thead>
              <tr className="bg-inset text-left">
                <th className="border-y border-rule px-2 py-1">
                  <Eyebrow as="span">Component</Eyebrow>
                </th>
                <th className="border-y border-rule px-2 py-1">
                  <Eyebrow as="span">Effect</Eyebrow>
                </th>
                <th className="border-y border-rule px-2 py-1">
                  <Eyebrow as="span">Description</Eyebrow>
                </th>
              </tr>
            </thead>
            <tbody className="text-mid">
              {[
                [
                  "Signal",
                  "0–60 pts",
                  "Predicted 30-session excess return, saturating at +10% so one extreme forecast cannot dominate the board.",
                ],
                [
                  "Conviction",
                  "0–40 pts",
                  "How far P(outperform) sits from a coin flip — counted only where the point forecast also predicts an upward move.",
                ],
                [
                  "Evidence grade",
                  "×1.0 / ×0.5 / ×0",
                  "STRONG / WEAK / INSUFFICIENT from purged walk-forward folds. A model that failed its held-out checks scores zero however large a move it predicts.",
                ],
                [
                  "Critic flags",
                  "−5 pts each",
                  "Contradictions raised by the LLM signal review, floored at zero.",
                ],
              ].map(([component, effect, description]) => (
                <tr key={component} className="border-b border-rule/70">
                  <td className="px-2 py-1.5 text-bright">{component}</td>
                  <td className="whitespace-nowrap px-2 py-1.5">{effect}</td>
                  <td className="px-2 py-1.5 font-prose leading-relaxed">
                    {description}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <Prose className="mt-5">
          <p>
            Both components floor at zero, which makes the composite a{" "}
            <strong>long-only</strong> ranking: a confidently predicted
            underperformer scores exactly the same as a confidently predicted
            flat one. That is why the leaderboard groups its zeros by{" "}
            <code>score_basis</code> instead of numbering them off — one value
            covers five unrelated situations, and sorting on it alone conflates
            &ldquo;no view&rdquo; with &ldquo;negative view&rdquo;.
          </p>
          <p>
            Conviction used to be computed independently of the point forecast,
            and that was not long-only at all. PNB ranked{" "}
            <strong>third</strong> on the live board on 2026-08-17 while
            forecasting a 1.69% underperformance: the signal component floored
            to 0.00 as intended, but conviction still collected 10.75 points
            from a calibrated probability of 0.567. The two inputs disagreed and
            the score quietly sided with the cheerier one. They disagree for a
            real reason — the calibrated probability is the share of residuals
            above the negated forecast, so a value over 0.5 means the model runs
            biased low for that ticker — but that is a statement that the model
            contradicts itself, not a ranking signal. It now ranks nowhere.
          </p>
        </Prose>
      </section>

      {/* ── Evaluation ───────────────────────────────────────────────────── */}
      <section>
        <SectionHead title="Evaluation" />
        <Prose>
          <ul>
            <li>
              <strong>Purged walk-forward validation.</strong> Each label spans
              30 sessions, so training rows whose label reaches into the test
              window are removed, plus a further 30-session embargo. Without
              this, the last 30 training rows carry labels drawn from inside the
              test window. Five folds, a minimum training window of 500 rows.
            </li>
            <li>
              <strong>Nested tuning.</strong> Hyperparameters are searched inside
              each training fold only. The tuner is structurally incapable of
              seeing test rows.
            </li>
            <li>
              <strong>Baselines always reported.</strong> Directional accuracy
              beside the majority-class rate; error beside the naive zero-excess
              forecast. Neither figure is shown alone anywhere on this site.
            </li>
            <li>
              <strong>Overlap-corrected t-statistics.</strong> Consecutive
              30-session labels are ~97% overlapping, so the effective sample is
              roughly <em>n / 30</em>. Treating all rows as independent inflates
              every t-statistic by about 5.5×. Panel comparisons on{" "}
              <Link href="/research">the research page</Link> are therefore
              quoted on non-overlapping rebalance dates only.
            </li>
            <li>
              <strong>Measured interval coverage.</strong> Conformal intervals
              claim 80%; realised coverage is checked against that claim rather
              than assumed.
            </li>
          </ul>
        </Prose>
      </section>

      {/* ── Limitations ──────────────────────────────────────────────────── */}
      <section>
        <SectionHead
          title="Known limitations"
          description="Stated plainly, because a system that hides these is not worth trusting."
        />
        <Prose>
          <ul>
            <li>
              <strong>No demonstrated edge.</strong> As measured, the model does
              not beat a majority-class baseline on direction or a zero-excess
              forecast on magnitude. The calibrated intervals hold up; the point
              forecast does not.
            </li>
            <li>
              <strong>Degenerate direction on some models.</strong> Roughly a
              fifth of tickers emit a single predicted direction for every row,
              which matches the majority-class baseline exactly without having a
              view. Those rows are marked on the leaderboard.
            </li>
            <li>
              <strong>No transaction costs.</strong> Indian delivery round trips
              run roughly 30–60 bps before market impact. On a monthly rebalance
              that is comparable to the entire measured edge.
            </li>
            <li>
              <strong>No portfolio backtest.</strong> Forecast accuracy and
              investment profitability are different questions; only the first
              is currently measured.
            </li>
            <li>
              <strong>Realised outcomes are recorded but not yet reported
              here.</strong> Each published forecast is resolved against the
              same stored target it was scored on, appended once and never
              updated — a published claim cannot be quietly improved after the
              fact. The record exists; no page reads it back yet.
            </li>
            <li>
              <strong>Survivorship bias.</strong> Point-in-time index membership
              is only recorded from the first universe sync onward.
            </li>
            <li>
              <strong>News sentiment is not a model feature and is not scored
              at all.</strong> As a feature it only ever existed for the current
              date — zero across the training set, non-zero only for the row
              being predicted, a train/serve mismatch at exactly the row that
              matters. The scorer is gone too. Headlines are collected and shown
              unscored. Both return only when a dated news archive exists.
            </li>
            <li>
              <strong>T+1 settlement is not modelled.</strong> A signal computed
              at the 15:30 close is actionable at the next open at the earliest.
            </li>
            <li>
              <strong>Fundamentals arrive as restated, not as filed.</strong>{" "}
              The vendor serves the latest version of each statement, so a
              figure may have been revised after the date the model is told it
              was known. Every revision observed since instrumentation began is
              appended to a log rather than overwritten, which makes the size of
              the bias measurable going forward — but not backwards.
            </li>
          </ul>
        </Prose>
      </section>

      {/* ── Stack ────────────────────────────────────────────────────────── */}
      <section>
        <SectionHead title="Stack" />
        <div className="grid grid-cols-1 gap-px bg-rule sm:grid-cols-2">
          <StackCard
            title="Modelling"
            items={[
              "XGBoost + Optuna (seeded, nested tuning)",
              "Split-conformal prediction",
              "scipy / scikit-learn",
              "pytest — leakage and regression suite",
            ]}
          />
          <StackCard
            title="Agents"
            items={[
              "LangGraph (orchestration)",
              "Groq — llama-3.1-8b-instant",
              "Narrative and signal review only, never numbers",
            ]}
          />
          <StackCard
            title="Data"
            items={[
              "yfinance — OHLCV, benchmarks, macro",
              "NSE archives — index constituents",
              "ta — technical indicators",
              "Supabase PostgreSQL",
            ]}
          />
          <StackCard
            title="Infrastructure"
            items={[
              "Next.js on Vercel — this site, statically generated",
              "FastAPI on Render — reads only, no scheduled compute",
              "GitHub Actions — daily forecast, weekly evaluation",
            ]}
          />
        </div>
      </section>
    </div>
  );
}

/**
 * A disclosure per pipeline stage.
 *
 * The number is not decoration: these four stages are a strict dependency
 * chain, and the order is the one fact about them a reader most needs.
 */
function Step({
  n,
  title,
  children,
}: {
  n: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <details className="group border-b border-rule">
      <summary className="flex cursor-pointer list-none items-center gap-3 px-1 py-2.5 hover:bg-raise [&::-webkit-details-marker]:hidden">
        <span className="w-5 shrink-0 text-[0.7rem] text-dim">{n}</span>
        <span className="flex-1 text-[0.8rem] font-semibold text-bright">
          {title}
        </span>
        <span
          aria-hidden
          className="text-[0.7rem] text-dim transition-transform group-open:rotate-90"
        >
          ▶
        </span>
      </summary>
      <div className="border-t border-rule bg-inset px-4 py-4 pl-9">
        <Prose>{children}</Prose>
      </div>
    </details>
  );
}

function StackCard({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="bg-shell p-4">
      <Eyebrow as="h3">{title}</Eyebrow>
      <ul className="mt-2.5 space-y-1 text-[0.78rem] text-text">
        {items.map((item) => (
          <li key={item} className="flex gap-2">
            <span aria-hidden className="text-rule-hi">
              ·
            </span>
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

function measure(entries: LeaderboardEntry[]) {
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

  const paired = entries.filter(
    (e) => e.eval_hit_rate !== null && e.eval_baseline_hit_rate !== null,
  );
  const belowBaseline = paired.filter(
    (e) => (e.eval_hit_rate as number) < (e.eval_baseline_hit_rate as number),
  ).length;

  const meanHit = mean(hits);
  const meanBaseline = mean(bases);
  const meanIc = mean(ics);

  return {
    total: entries.length,
    paired: paired.length,
    meanIc,
    positiveIc: ics.filter((v) => v > 0).length,
    negativeIc: ics.filter((v) => v < 0).length,
    meanHit,
    meanBaseline,
    edge: meanHit - meanBaseline,
    beatsRw: entries.filter((e) => e.eval_beats_random_walk === true).length,
    belowBaseline,
    degenerate: paired.filter(
      (e) =>
        Math.abs(
          (e.eval_hit_rate as number) - (e.eval_baseline_hit_rate as number),
        ) < 1e-6,
    ).length,
    nullResult:
      !(ics.length > 0 && meanIc > 0) || belowBaseline > paired.length / 2,
  };
}
