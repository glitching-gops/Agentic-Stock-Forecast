import Link from "next/link";

import {
  Badge,
  Eyebrow,
  Note,
  Panel,
  Prose,
  Rail,
  Readout,
  SectionHead,
} from "@/components/ui";
import { cx, signed } from "@/lib/format";
import {
  BAR,
  COMPARATORS,
  FOLD_DISPERSION,
  LESSONS,
  LORA_FOLDS,
  MEASURED_ON,
  PANEL,
  PROBE,
  RETIRED,
  VALUATION_FIXED_SAMPLE,
  VALUATION_SWEEP,
  axisRows,
  type Comparator,
} from "@/lib/research";

export const metadata = {
  title: "Research",
  description:
    "Every model scored against the pre-registered bar on the NIFTY 100 panel, including the two results that cleared it and were retired.",
};

/* ── The axis ──────────────────────────────────────────────────────────── */

/**
 * The domain every rail on this page shares.
 *
 * It is fixed rather than fitted to the data, and every rail uses it, which is
 * the whole point: the amber threshold mark lands at the same x on every row,
 * so a column of adjacent rails stacks its marks into one continuous vertical
 * line. That line is the pre-registered bar, drawn once, and the reader can see
 * in a single glance that almost nothing reaches it. Fitting the domain per row
 * would destroy the comparison the page exists to make.
 */
const T_MIN = -1.5;
const T_MAX = 4;

function axisPos(t: number): number {
  return ((Math.max(T_MIN, Math.min(T_MAX, t)) - T_MIN) / (T_MAX - T_MIN)) * 100;
}

const TICKS = [-1, 0, 1, 2, 3, 4];

/**
 * One row of the axis. Deliberately gapless — see the note on the domain above.
 */
function AxisRow({
  label,
  detail,
  t,
  tag,
  struck = false,
}: {
  label: string;
  detail?: string;
  t: number;
  /** A short annotation after the label — "retired", "quoted". */
  tag?: string;
  /** Strike the label. Reserved for a measurement that is not claimed. */
  struck?: boolean;
}) {
  const cleared = t >= BAR;
  return (
    <div className="grid grid-cols-[8rem_1fr_3rem] items-center gap-3 border-b border-rule/60 py-0.5 sm:grid-cols-[13rem_1fr_3.5rem]">
      <div className="flex min-w-0 items-baseline gap-1.5">
        <span
          className={cx(
            "truncate text-[0.72rem]",
            struck ? "text-dim line-through" : cleared ? "text-bar" : "text-text",
          )}
          title={detail}
        >
          {label}
        </span>
        {tag ? (
          <span className="shrink-0 text-[0.58rem] uppercase tracking-[0.1em] text-dim">
            {tag}
          </span>
        ) : null}
      </div>
      <Rail value={t} threshold={BAR} min={T_MIN} max={T_MAX} label={`t ${signed(t, 2)}`} />
      <span
        className={cx(
          "text-right text-[0.72rem] tabular-nums",
          cleared ? "text-bar" : t < 0 ? "text-dim" : "text-mid",
        )}
      >
        {signed(t, 2)}
      </span>
    </div>
  );
}

/* ── Cells ─────────────────────────────────────────────────────────────── */

function Th({
  children,
  className,
  title,
}: {
  children: React.ReactNode;
  className?: string;
  title?: string;
}) {
  return (
    <th scope="col" className={cx("px-2 py-1.5 text-left font-normal", className)}>
      <Eyebrow as="span" title={title}>
        {children}
      </Eyebrow>
    </th>
  );
}

const NUM = "px-2 py-1.5 text-right tabular-nums";

/** A signed t-statistic, coloured only when it clears the bar. */
function TCell({ t }: { t: number | undefined }) {
  if (t === undefined) return <td className={cx(NUM, "text-dim")}>—</td>;
  return (
    <td className={cx(NUM, t >= BAR ? "text-bar" : t < 0 ? "text-dim" : "text-text")}>
      {signed(t, 2)}
    </td>
  );
}

/** Percent-point deltas against the MAE floor. Lower is better, so pos/neg invert. */
function floorLabel(v: number | undefined): string {
  if (v === undefined) return "—";
  if (Math.abs(v) < 0.05) return "+0.0%";
  return `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)}%`;
}

function cost(seconds: number): string {
  if (seconds < 90) return `${seconds.toFixed(1)}s`;
  return `${Math.round(seconds / 60)}m`;
}

const FAMILY_LABEL: Record<Comparator["family"], string> = {
  naive: "naive",
  linear: "linear",
  tree: "tree",
  foundation: "foundation",
  adapter: "adapter",
  probe: "probe",
  finetune: "fine-tune",
};

/* ── Page ──────────────────────────────────────────────────────────────── */

export default function ResearchPage() {
  const rows = axisRows();
  const best = rows[rows.length - 1];
  const scoredCount = rows.length + RETIRED.length + PROBE.length;

  return (
    <div className="space-y-12 pb-16">
      {/* Header */}
      <header className="border-b border-rule-hi pb-6">
        <div className="flex flex-wrap items-center gap-2">
          <Eyebrow as="span">Phase 2</Eyebrow>
          <span className="text-dim">/</span>
          <Badge tone="dim">Closed</Badge>
          <span className="text-dim">/</span>
          <span className="text-[0.7rem] text-dim">measured {MEASURED_ON}</span>
        </div>
        <h1 className="mt-3 max-w-3xl font-display text-[1.6rem] font-bold leading-tight tracking-tight text-bright sm:text-[2rem]">
          Everything that was tried, against the bar it had to clear
        </h1>
        <p className="mt-3 max-w-[70ch] font-prose text-[0.9rem] leading-relaxed text-mid">
          The success criterion was fixed with the owner on 21 August 2026,
          before the last two model families had been scored, so that a marginal
          number could not be re-read as a win afterwards. It is one sentence: a
          comparator succeeds if its rebalance rank IC is positive with{" "}
          <strong className="font-semibold text-bright">t &gt; 2</strong>.
        </p>

        <div className="mt-6 grid grid-cols-2 gap-y-5 sm:grid-cols-4">
          <Readout
            label="Configurations scored"
            value={scoredCount}
            sub="Models, contexts and ablations, all on one panel and one set of folds."
            tone="dim"
          />
          <Readout
            label="The bar"
            value="t > 2"
            sub="Pre-registered, on non-overlapping rebalance windows."
            tone="bar"
          />
          <Readout
            label="Best surviving"
            value={signed(best.rebT, 2)}
            sub={`${best.label} — a ridge and a tree still lead every foundation model at matched context.`}
            tone="dim"
          />
          <Readout
            label="Cleared and survived"
            value="0"
            sub="Two results cleared the bar. Neither survived being re-measured."
          />
        </div>
      </header>

      {/* The axis — the signature of this page */}
      <section>
        <SectionHead
          title="Every comparator on one axis"
          count={`${rows.length + RETIRED.length}`}
          description={
            <>
              The amber line is the bar. It sits at the same position on every
              rail, so it reads as one continuous mark down the block — which is
              the finding: on {PANEL.rows.toLocaleString("en-IN")} rows across{" "}
              {PANEL.tickers} tickers and {PANEL.rebalances} independent
              rebalance windows, almost nothing gets near it.
            </>
          }
        />

        <Panel className="px-4 py-4 sm:px-5">
          {/* Tick scale, sharing the rails' domain exactly. */}
          <div className="grid grid-cols-[8rem_1fr_3rem] gap-3 sm:grid-cols-[13rem_1fr_3.5rem]">
            <Eyebrow>Comparator</Eyebrow>
            <div className="relative h-4">
              {TICKS.map((t) => (
                <span
                  key={t}
                  className={cx(
                    "absolute top-0 -translate-x-1/2 text-[0.6rem] tabular-nums",
                    t === BAR ? "text-bar" : "text-dim",
                  )}
                  style={{ left: `${axisPos(t)}%` }}
                >
                  {t > 0 ? `+${t}` : t}
                </span>
              ))}
            </div>
            <Eyebrow className="text-right">reb t</Eyebrow>
          </div>

          <div className="mt-1">
            {rows.map((c) => (
              <AxisRow key={c.id} label={c.label} detail={c.note} t={c.rebT} />
            ))}
          </div>

          {/* The two that crossed it. Same domain, so the amber line runs on. */}
          <div className="mt-6">
            <Eyebrow className="mb-1">Cleared the bar, and was retired</Eyebrow>
            {RETIRED.map((r) => (
              <AxisRow
                key={r.id}
                label={r.label}
                detail={r.verdict}
                t={r.headlineT}
                tag="retired"
                struck
              />
            ))}
          </div>

          <p className="mt-4 max-w-[74ch] font-prose text-[0.78rem] leading-relaxed text-dim">
            The two struck rows are the honest part of this chart. Both cleared
            the criterion on first measurement. Both were then attacked — one by
            sweeping a hyperparameter nobody had thought was load-bearing, one by
            breaking a pooled statistic down by fold — and neither is claimed.
          </p>
        </Panel>
      </section>

      {/* The full table */}
      <section>
        <SectionHead
          title="The comparator table"
          description={
            <>
              One panel, one splitter, one report. {PANEL.folds} purged folds
              with a {PANEL.embargo}-session embargo, a {PANEL.horizon}-session
              horizon, and a median of {PANEL.medianNamesPerDate} names per date.
              Sorted by mean absolute error, so the artifacts at the top are read
              before the models below them.
            </>
          }
        />
        <Panel className="overflow-x-auto">
          <table className="w-full min-w-[46rem] border-collapse text-[0.74rem]">
            <thead>
              <tr className="border-b border-rule-hi">
                <Th>Comparator</Th>
                <Th>Family</Th>
                <Th
                  className="text-right"
                  title="Mean per-date rank IC over every out-of-sample date. Overlapping windows — it does not support a t-statistic."
                >
                  daily IC
                </Th>
                <Th
                  className="text-right"
                  title="Rank IC over the 64 non-overlapping rebalance dates. This is the one the bar applies to."
                >
                  reb IC
                </Th>
                <Th className="text-right" title="t-statistic of the rebalance IC.">
                  reb t
                </Th>
                <Th className="text-right">MAE</Th>
                <Th
                  className="text-right"
                  title="MAE against the zero-forecast floor. Lower is better, so a negative number here is not necessarily skill."
                >
                  vs floor
                </Th>
                <Th className="text-right">cost</Th>
              </tr>
            </thead>
            <tbody>
              {COMPARATORS.map((c) => {
                const isFloor = c.id === "zero";
                return (
                  <tr
                    key={c.id}
                    className={cx(
                      "border-b border-rule/60 last:border-b-0",
                      isFloor && "bg-raise",
                    )}
                  >
                    <th
                      scope="row"
                      className="px-2 py-1.5 text-left font-normal"
                      title={c.note}
                    >
                      <span className={isFloor ? "text-bright" : "text-text"}>
                        {c.label}
                      </span>
                    </th>
                    <td className="px-2 py-1.5 text-[0.66rem] uppercase tracking-[0.08em] text-dim">
                      {FAMILY_LABEL[c.family]}
                    </td>
                    <td className={cx(NUM, "text-mid")}>
                      {c.dailyIc === undefined ? "—" : signed(c.dailyIc, 4)}
                    </td>
                    <td className={cx(NUM, "text-mid")}>
                      {c.rebIc === undefined ? "—" : signed(c.rebIc, 4)}
                    </td>
                    <TCell t={c.rebT} />
                    <td className={cx(NUM, isFloor ? "text-bright" : "text-text")}>
                      {c.mae.toFixed(5)}
                    </td>
                    <td className={cx(NUM, "text-dim")}>{floorLabel(c.vsFloor)}</td>
                    <td className={cx(NUM, c.seconds > 1000 ? "text-mid" : "text-dim")}>
                      {cost(c.seconds)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </Panel>

        <div className="mt-6 grid gap-5 lg:grid-cols-3">
          <Panel className="px-4 py-4">
            <Eyebrow>Three rows beat the floor, and none of it is skill</Eyebrow>
            <p className="mt-2 font-prose text-[0.8rem] leading-relaxed text-mid">
              <code className="font-mono text-text">train_mean</code>,{" "}
              <code className="font-mono text-text">reversal_5d</code> and{" "}
              <code className="font-mono text-text">momentum_20d</code> all land
              at 0.0667 against the floor&apos;s 0.0668. But{" "}
              <code className="font-mono text-text">train_mean</code> is a
              constant — it exists to detect exactly this. The universe drifted
              positive against its benchmarks, so a constant non-zero prediction
              beats a constant zero on magnitude while carrying no ordering at
              all. The other two post a rebalance t of +0.29 and −0.01: they are
              capturing the level, not the ranking.
            </p>
          </Panel>
          <Panel className="px-4 py-4">
            <Eyebrow>The edge was context, not architecture</Eyebrow>
            <p className="mt-2 font-prose text-[0.8rem] leading-relaxed text-mid">
              Chronos-2&apos;s +1.86 is measured in the single most expensive
              cell on the page, and collapses to +0.88 when the context drops
              from 2048 to 512. At matched context the architecture comparison
              finally reads cleanly — 120M encoder-only at +0.88 against 231M
              decoder-only at +0.57 — and both sit below a ridge on 15 columns.
              Cost is quadratic in context, not linear: 98 minutes against 7, a
              factor of 14.8 for four times the window.
            </p>
          </Panel>
          <Panel className="px-4 py-4">
            <Eyebrow>Cross-learning is negative at both contexts</Eyebrow>
            <p className="mt-2 font-prose text-[0.8rem] leading-relaxed text-mid">
              Conditioning each forecast on the rest of that date&apos;s
              cross-section is the feature Chronos-2 is sold on. It takes +1.86
              to +0.65 at context 2048, and +0.88 to −0.09 at 512. Two
              independent architectures, different corpora and different
              objectives, both below the floor with a non-positive rebalance IC
              — that is evidence about the target, not about either model.
            </p>
          </Panel>
        </div>
      </section>

      {/* Retirement 1 — valuation */}
      <section>
        <SectionHead
          title="Retired — valuation, t +3.32"
          right={<Badge tone="dim">23 Aug 2026</Badge>}
          description="For one day this was the only number in the project that had cleared the pre-registered bar: a gradient-boosted tree with point-in-time valuation features, rebalance IC +0.0708 at t +3.32, with a 100-draw placebo null putting it 4.35 standard deviations above the null mean. Four experiments on the identical 77,585-row panel retired it."
        />

        <div className="grid gap-6 lg:grid-cols-2">
          <Panel className="px-4 py-4">
            <Eyebrow>A. It is the maximum of an unstable grid</Eyebrow>
            <p className="mt-2 max-w-[62ch] font-prose text-[0.8rem] leading-relaxed text-mid">
              The headline was measured at{" "}
              <code className="font-mono text-text">min_train = 380</code>.
              Everything else in this project is reported at the harness default
              of 500, where the same comparator on the same rows scores +1.00.
            </p>
            <div className="mt-4">
              {VALUATION_SWEEP.map((s) => (
                <AxisRow
                  key={s.minTrain}
                  label={`min_train ${s.minTrain}`}
                  t={s.t}
                  tag={s.minTrain === 380 ? "quoted" : undefined}
                />
              ))}
            </div>
            <p className="mt-3 max-w-[62ch] font-prose text-[0.78rem] leading-relaxed text-dim">
              A spread of 3.79 t-units across the grid, with the headline a lone
              spike between neighbours of +1.30 and +1.18. A p-value computed at
              one cell of a grid that unstable describes the cell, not the
              feature.
            </p>
          </Panel>

          <Panel className="px-4 py-4">
            <Eyebrow>C &amp; D. A sample effect, not a valuation effect</Eyebrow>
            <p className="mt-2 max-w-[62ch] font-prose text-[0.8rem] leading-relaxed text-mid">
              Withholding each figure for an extra 90, 180 and 365 days on top of
              the 60-day filing lag decayed the edge monotonically, which looked
              like clean evidence of a staleness effect. The row count had
              quietly fallen from 77,585 to 54,304, because a longer lag costs
              the earliest rows their coverage. Re-scored on the fixed sample
              that survives all four settings:
            </p>
            <table className="mt-4 w-full border-collapse text-[0.74rem]">
              <thead>
                <tr className="border-b border-rule-hi">
                  <Th>Extra lag</Th>
                  <Th className="text-right">t @ 380</Th>
                  <Th className="text-right">t @ 460</Th>
                  <Th className="text-right">t @ 500</Th>
                </tr>
              </thead>
              <tbody>
                {VALUATION_FIXED_SAMPLE.map((r) => (
                  <tr key={r.extraLagDays} className="border-b border-rule/60 last:border-b-0">
                    <th scope="row" className="px-2 py-1.5 text-left font-normal text-text">
                      {r.extraLagDays === 0 ? "none" : `+${r.extraLagDays}d`}
                    </th>
                    <TCell t={r.t380} />
                    <TCell t={r.t460} />
                    <TCell t={r.t500} />
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-3 max-w-[62ch] font-prose text-[0.78rem] leading-relaxed text-dim">
              The edge is gone at every lag, lag zero included. It was never
              about staleness — the entire result lived in the specific rows the
              long-lag sweep happened to drop.
            </p>
          </Panel>
        </div>

        <div className="mt-6">
          <Note tone="bar" title="Why the placebo null had to move too">
            A feature that is <strong>persistent per ticker</strong> earns a
            positive t-statistic from nothing at all. Measured, not theorised:
            two random constants assigned per ticker — carrying zero information
            about returns — scored the pooled tree at a mean rebalance t of{" "}
            <strong>+0.77</strong> over 24 draws. The tree identifies the ticker
            from the constant and learns which names paid during training. The
            within-date rank of earnings yield correlates{" "}
            <strong>+0.813</strong> with itself 250 sessions later, against −0.03
            to −0.01 for the technical features, so valuation is exactly this
            kind of feature and must be read against a placebo null rather than
            against t = 2. That null moves with{" "}
            <code>min_train</code> as well — from −0.05 at 500 to +0.90 at 380 —
            which is why raw t-statistics are not comparable across settings.
          </Note>
        </div>
      </section>

      {/* Retirement 2 — LoRA */}
      <section>
        <SectionHead
          title="Retired — LoRA fine-tuning, t +2.37"
          right={<Badge tone="dim">25 Aug 2026</Badge>}
          description="Rank-16 LoRA on Chronos-2's query and value projections — 1,179,648 trainable parameters — with five separate fine-tunes, one nested inside each purged fold. A single global fit before the split would have been leakage with a 120M-parameter model standing in for the tuner. Pooled over all 64 rebalances it scored +0.0336 at t +2.37, the first number in the project to clear the bar."
        />

        <Panel className="overflow-x-auto">
          <table className="w-full min-w-[42rem] border-collapse text-[0.74rem]">
            <thead>
              <tr className="border-b border-rule-hi">
                <Th>Fold</Th>
                <Th>Test window</Th>
                <Th className="text-right">Train rows</Th>
                <Th
                  className="text-right"
                  title="On a standardised target, a train MSE of 1.00 means the model never left 'predict the mean'."
                >
                  Train MSE
                </Th>
                <Th className="text-right">Target dispersion</Th>
                <Th className="text-right">reb IC</Th>
                <Th className="text-right">reb t</Th>
              </tr>
            </thead>
            <tbody>
              {LORA_FOLDS.map((f) => (
                <tr key={f.fold} className="border-b border-rule/60 last:border-b-0">
                  <th scope="row" className="px-2 py-1.5 text-left font-normal text-text">
                    {f.fold}
                  </th>
                  <td className="px-2 py-1.5 text-mid">{f.period}</td>
                  <td className={cx(NUM, "text-mid")}>
                    {f.trainRows.toLocaleString("en-IN")}
                  </td>
                  <td className={cx(NUM, f.trainMse < 0.9 ? "text-bright" : "text-dim")}>
                    {f.trainMse.toFixed(2)}
                  </td>
                  <td className={cx(NUM, "text-dim")}>
                    {FOLD_DISPERSION[f.fold].toFixed(3)}
                  </td>
                  <td
                    className={cx(NUM, f.rebIc < 0 ? "text-neg" : "text-mid")}
                  >
                    {signed(f.rebIc, 4)}
                  </td>
                  <td
                    className={cx(
                      NUM,
                      f.rebT >= BAR ? "text-bar" : f.rebT < 0 ? "text-neg" : "text-mid",
                    )}
                  >
                    {signed(f.rebT, 2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>

        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <Prose>
            <h3>The effect declines in training data and inverts</h3>
            <p>
              Only folds 0 and 1 fitted their training data at all, and they are
              exactly the folds carrying the positive IC. 1.18M parameters can
              memorise 2,922 rows and cannot beat the mean against 17,000.
              Pooled over folds 2 to 4 the whole thing is <strong>t +0.40</strong>;
              pooled over 0 and 1 it is +2.90. A learned signal that degrades the
              more you train it is not a learned signal.
            </p>
            <h3>The confound was separated, not assumed</h3>
            <p>
              Fold 0 differs from fold 4 in two ways at once — least training
              data <em>and</em> earliest test window. Retraining fold 4 on 2,922
              rows taken adjacent to its own test window, fold 0&apos;s exact
              volume, gives train MSE 0.48 and <strong>reb t +0.38</strong>,
              against fold 0&apos;s +3.02. Same data volume, eight times the
              t-statistic. It is the period, not the sample size.
            </p>
          </Prose>
          <div className="space-y-5">
            <Note tone="neg" title="Two unrelated methods, the same early window">
              Fold 0&apos;s test window contains the COVID crash, and target
              cross-sectional dispersion falls monotonically across the folds —
              0.108, 0.104, 0.100, 0.088, 0.077 — with the apparent skill falling
              with it. This is the <strong>second</strong> positive result in the
              project carried entirely by the earliest part of the panel, after
              valuation&apos;s +3.32. Two unrelated methods producing apparent
              signal concentrated in early, high-dispersion data and absent in
              recent data is evidence about the <strong>panel</strong>, not about
              either method.
            </Note>
            <Note tone="dim" title="The run that was discarded rather than reported">
              An earlier fine-tune at a lower learning rate scored −0.61 and
              looked like a clean null. Its train MSE had never left 1.00, which
              on a standardised target means it never learned anything at all —
              the overfit diagnostic drives the same loop from 1.14 to 0.17 on
              256 rows, so gradients flow and the loop works. Reporting −0.61
              would have been a hyperparameter choice masquerading as a model
              verdict.
            </Note>
          </div>
        </div>
      </section>

      {/* The probe */}
      <section>
        <SectionHead
          title="The cheap question, asked before the expensive one"
          description="A zero-shot failure has two explanations the results table cannot separate: the representation carries nothing about 30-session excess return, or it carries something the generic forecast head does not express as a relative-price move. Only the second is worth fine-tuning for, because LoRA adapts a representation rather than conjuring one. So the model was frozen, the encoder state the forecast head reads was extracted, and a ridge was fitted straight to the target."
        />
        <Panel className="overflow-x-auto">
          <table className="w-full min-w-[36rem] border-collapse text-[0.74rem]">
            <thead>
              <tr className="border-b border-rule-hi">
                <Th>Context</Th>
                <Th>Selection rule</Th>
                <Th className="text-right">daily IC</Th>
                <Th className="text-right">reb IC</Th>
                <Th className="text-right">reb t</Th>
                <Th className="text-right">vs floor</Th>
              </tr>
            </thead>
            <tbody>
              {PROBE.map((p) => (
                <tr
                  key={`${p.context}-${p.label}`}
                  className="border-b border-rule/60 last:border-b-0"
                >
                  <th scope="row" className="px-2 py-1.5 text-left font-normal text-text">
                    {p.context}
                  </th>
                  <td className="px-2 py-1.5 text-mid">{p.label}</td>
                  <td className={cx(NUM, "text-mid")}>{signed(p.dailyIc, 4)}</td>
                  <td className={cx(NUM, "text-mid")}>{signed(p.rebIc, 4)}</td>
                  <TCell t={p.rebT} />
                  <td className={cx(NUM, "text-dim")}>{floorLabel(p.vsFloor)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Panel>
        <div className="mt-5 grid gap-6 lg:grid-cols-2">
          <Prose>
            <p>
              The probe does not beat the pretrained head, and at context 2048 it
              is negative — which is the context where zero-shot Chronos-2 is
              strongest. Read together: a linear read-out of the frozen
              representation recovers no more of the target than the head already
              does, and less where the head does best.
            </p>
            <p>
              Selecting the ridge penalty on ranking IC rather than error made it{" "}
              <strong>worse</strong> at both contexts, +0.0211 to −0.0033 at 512.
              Optimising a noisy inner ranking metric overfits the selection
              itself. Both rules are reported rather than the better one quoted.
            </p>
          </Prose>
          <Note tone="dim" title="The caveat that keeps this honest">
            A linear probe tests <strong>linear</strong> decodability. A
            non-linear head could in principle find structure a ridge cannot, so
            this is evidence against spending days on fine-tuning, not proof. The
            fine-tune was run anyway — it is the section above.
          </Note>
        </div>
      </section>

      {/* Lessons */}
      <section>
        <SectionHead
          title="What the phase actually produced"
          count={LESSONS.length}
          description="Every one of these was learned by a measurement going wrong, not by reasoning about it beforehand. Three separate guards were in place when the valuation result got as far as being called the project's one success — a purged panel splitter, a pre-registered threshold, and a persistence placebo — and what caught it was none of the three. It was re-running the same measurement at a different arbitrary setting."
        />
        <div className="grid gap-px bg-rule sm:grid-cols-2">
          {LESSONS.map((l, i) => (
            // An odd count leaves a hole in the two-column grid, and the hole
            // reads as a missing card rather than as the end of the list.
            <div
              key={l.title}
              className={cx(
                "bg-shell px-4 py-4",
                i === LESSONS.length - 1 &&
                  LESSONS.length % 2 === 1 &&
                  "sm:col-span-2",
              )}
            >
              <h3 className="text-[0.72rem] font-semibold uppercase tracking-[0.14em] text-bright">
                {l.title}
              </h3>
              <p className="mt-2 max-w-[60ch] font-prose text-[0.8rem] leading-relaxed text-mid">
                {l.body}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Provenance */}
      <section>
        <Note tone="neutral" title="Where these numbers come from">
          Every figure on this page is transcribed from{" "}
          <code>experiment_runs.metrics</code>, written by the weekly evaluation
          job and rendered locally by <code>tools/run_baselines.py</code>. Unlike
          the rest of the site they are <strong>not fetched live</strong>: the
          research log is keyed by a configuration hash and a data hash and means
          nothing without both, and Phase 2 is closed, so a closed record is
          served as a closed record. The board and every stock page read the API
          directly. See the{" "}
          <Link href="/methodology">methodology</Link> for how a forecast becomes
          a published rank, and the{" "}
          <Link href="/">board</Link> for what the live pipeline currently says.
        </Note>
      </section>
    </div>
  );
}
