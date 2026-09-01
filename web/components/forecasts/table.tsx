"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { Badge, Eyebrow, evidenceTone, verdictTone } from "@/components/ui";
import {
  cx,
  decimal,
  evidenceState,
  evidenceStateExplainer,
  evidenceStateLabel,
  probability,
  signed,
  signedPct,
} from "@/lib/format";
import type { CurrentForecast, EvidenceState } from "@/lib/types";

/**
 * Every forecast in the frozen universe, filtered in the browser.
 *
 * The whole table is 84 rows and arrives with the page, so a filter change
 * costs no network at all — no round trip to a free-tier instance that may be
 * asleep. The API's own filter parameters still exist and still work; they are
 * just not the right tool at this size.
 *
 * THIS IS NOT A LEADERBOARD, and the difference is structural rather than
 * cosmetic. The table it replaced put a "Rated" section on top, ordered by a
 * composite score, and collapsed the rest under a disclosure. Two names sat in
 * the lit section and ninety-three in the drawer, which reads as a podium with
 * a long tail — but the evidence gate clears three of ninety-six tickers, and
 * 3.12 is what chance produces (Poisson p = 0.60). There was no podium.
 *
 * So the ordering is alphabetical and fixed, the same order the API returns,
 * and grouping is by WHAT IS KNOWN about each forecast rather than by how it
 * placed. Sorting by a measured column stays available because a reader
 * exploring the data should be able to ask "which names have the largest
 * predicted move" — but it is a question the reader poses, not a verdict the
 * page hands them on arrival.
 */

const SORTS = {
  ticker: "Ticker",
  pred_return: "Return",
  prob_up: "P(up)",
  eval_rank_ic: "Rank IC",
  eval_rank_ic_t: "IC t-stat",
  eval_hit_rate: "Hit rate",
} as const;

type SortKey = keyof typeof SORTS;

/** Most informative first: what we know, then what we do not. */
const STATE_ORDER: EvidenceState[] = [
  "STRONG",
  "WEAK",
  "INSUFFICIENT",
  "NO_FORECAST",
];

export function ForecastTable({ forecasts }: { forecasts: CurrentForecast[] }) {
  const [sector, setSector] = useState("");
  const [evidence, setEvidence] = useState("");
  const [verdict, setVerdict] = useState("");
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("ticker");

  const sectors = useMemo(
    () =>
      Array.from(
        new Set(forecasts.map((f) => f.sector).filter((s): s is string => !!s)),
      ).sort(),
    [forecasts],
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return forecasts.filter((forecast) => {
      if (sector && forecast.sector !== sector) return false;
      if (evidence && evidenceState(forecast) !== evidence) return false;
      if (verdict && forecast.critic_verdict !== verdict) return false;
      if (
        q &&
        !`${forecast.company ?? ""} ${forecast.ticker}`.toLowerCase().includes(q)
      ) {
        return false;
      }
      return true;
    });
  }, [forecasts, sector, evidence, verdict, query]);

  const sorted = useMemo(
    () => [...filtered].sort(comparator(sortKey)),
    [filtered, sortKey],
  );

  const groups = STATE_ORDER.map((state) => ({
    state,
    rows: sorted.filter((f) => evidenceState(f) === state),
  })).filter((group) => group.rows.length > 0);

  const anyFilter = Boolean(sector || evidence || verdict || query);

  return (
    <div className="space-y-5">
      <Filters
        sectors={sectors}
        sector={sector}
        setSector={setSector}
        evidence={evidence}
        setEvidence={setEvidence}
        verdict={verdict}
        setVerdict={setVerdict}
        query={query}
        setQuery={setQuery}
        sortKey={sortKey}
        setSortKey={setSortKey}
        matched={filtered.length}
        total={forecasts.length}
        onReset={() => {
          setSector("");
          setEvidence("");
          setVerdict("");
          setQuery("");
          setSortKey("ticker");
        }}
      />

      {groups.length === 0 ? (
        <div className="border border-rule bg-shell px-4 py-6 text-center text-[0.78rem] text-dim">
          {anyFilter
            ? "No stock matches these filters."
            : "No forecasts have been published yet."}
        </div>
      ) : (
        <section aria-labelledby="forecasts-heading" className="space-y-1.5">
          <div className="mb-2 border-b border-rule-hi pb-1.5">
            <h2
              id="forecasts-heading"
              className="text-[0.72rem] font-semibold uppercase tracking-[0.18em] text-bright"
            >
              Forecasts
              <span className="ml-2 font-normal text-dim">
                {filtered.length}
              </span>
            </h2>
            <p className="mt-2 max-w-4xl font-prose text-[0.8rem] leading-relaxed text-mid">
              Grouped by what the held-out evaluation supports, not by how the
              stocks compare. Every stock in the universe gets a forecast; the
              group says how much weight it has earned.
            </p>
          </div>

          {groups.map((group) => (
            <EvidenceGroup
              key={group.state}
              state={group.state}
              rows={group.rows}
              // The groups that matter open on arrival; the large INSUFFICIENT
              // block does not, because a 70-row table above the fold buries
              // the distinction the grouping exists to draw.
              open={group.state === "STRONG" || group.state === "WEAK"}
            />
          ))}
        </section>
      )}
    </div>
  );
}

/* ── Sorting ───────────────────────────────────────────────────────────── */

/**
 * Ticker ascending, or a measured column descending with nulls last.
 *
 * Nulls last rather than as zero. Roughly a fifth of these cells are genuinely
 * unmeasured, and sorting them as 0.0 would interleave "no evidence was
 * gathered" with "the evidence came out neutral" — two statements a reader
 * must be able to tell apart.
 */
function comparator(key: SortKey) {
  if (key === "ticker") {
    return (a: CurrentForecast, b: CurrentForecast) =>
      a.ticker.localeCompare(b.ticker);
  }
  return (a: CurrentForecast, b: CurrentForecast) => {
    const left = a[key];
    const right = b[key];
    const leftNull = left === null || left === undefined;
    const rightNull = right === null || right === undefined;
    if (leftNull && rightNull) return a.ticker.localeCompare(b.ticker);
    if (leftNull) return 1;
    if (rightNull) return -1;
    if (left === right) return a.ticker.localeCompare(b.ticker);
    return (right as number) - (left as number);
  };
}

/* ── Filters ───────────────────────────────────────────────────────────── */

function Filters(props: {
  sectors: string[];
  sector: string;
  setSector: (v: string) => void;
  evidence: string;
  setEvidence: (v: string) => void;
  verdict: string;
  setVerdict: (v: string) => void;
  query: string;
  setQuery: (v: string) => void;
  sortKey: SortKey;
  setSortKey: (v: SortKey) => void;
  matched: number;
  total: number;
  onReset: () => void;
}) {
  const dirty = props.matched !== props.total;

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-y border-rule bg-shell px-3 py-2">
      <Field label="Find">
        <input
          type="search"
          value={props.query}
          onChange={(e) => props.setQuery(e.target.value)}
          placeholder="name or ticker"
          className="w-36 border border-rule bg-inset px-2 py-0.5 text-[0.74rem] text-bright placeholder:text-dim focus:border-rule-hi focus:outline-none"
        />
      </Field>

      <Field label="Sector">
        <Select value={props.sector} onChange={props.setSector}>
          <option value="">all</option>
          {props.sectors.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </Select>
      </Field>

      <Field label="Evidence">
        <Select value={props.evidence} onChange={props.setEvidence}>
          <option value="">any</option>
          <option value="STRONG">strong</option>
          <option value="WEAK">weak</option>
          <option value="INSUFFICIENT">none</option>
          <option value="NO_FORECAST">no forecast</option>
        </Select>
      </Field>

      <Field label="Critic">
        <Select value={props.verdict} onChange={props.setVerdict}>
          <option value="">any</option>
          <option value="APPROVED">approved</option>
          <option value="FLAGGED">flagged</option>
          <option value="REJECTED">rejected</option>
        </Select>
      </Field>

      <Field label="Sort">
        <Select
          value={props.sortKey}
          onChange={(v) => props.setSortKey(v as SortKey)}
        >
          {Object.entries(SORTS).map(([key, label]) => (
            <option key={key} value={key}>
              {label}
            </option>
          ))}
        </Select>
      </Field>

      <div className="ml-auto flex items-center gap-3">
        <span className="text-[0.7rem] text-dim">
          {props.matched}/{props.total}
        </span>
        {dirty ? (
          <button
            type="button"
            onClick={props.onReset}
            className="border border-rule-hi px-2 py-0.5 text-[0.68rem] uppercase tracking-[0.1em] text-text hover:bg-bright hover:text-void"
          >
            Reset
          </button>
        ) : null}
      </div>
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="flex items-center gap-2">
      <Eyebrow as="span">{label}</Eyebrow>
      {children}
    </label>
  );
}

function Select({
  value,
  onChange,
  children,
}: {
  value: string;
  onChange: (v: string) => void;
  children: React.ReactNode;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="border border-rule bg-inset px-1.5 py-0.5 text-[0.74rem] text-bright focus:border-rule-hi focus:outline-none"
    >
      {children}
    </select>
  );
}

/* ── Grouped disclosure ────────────────────────────────────────────────── */

function EvidenceGroup({
  state,
  rows,
  open,
}: {
  state: EvidenceState;
  rows: CurrentForecast[];
  open: boolean;
}) {
  return (
    <details className="group border border-rule bg-shell" open={open}>
      <summary className="flex cursor-pointer list-none items-start gap-2.5 px-3 py-2 hover:bg-raise [&::-webkit-details-marker]:hidden">
        <span
          aria-hidden
          className="mt-[3px] text-[0.6rem] text-dim transition-transform group-open:rotate-90"
        >
          ▶
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-baseline gap-2">
            <span className="text-[0.76rem] font-semibold uppercase tracking-[0.12em] text-bright">
              {evidenceStateLabel(state)}
            </span>
            <span className="text-[0.7rem] text-dim">{rows.length}</span>
          </span>
          <span className="mt-1 block max-w-4xl font-prose text-[0.78rem] leading-relaxed text-dim">
            {evidenceStateExplainer(state)}
          </span>
        </span>
      </summary>
      <div className="border-t border-rule">
        <ForecastRows rows={rows} />
      </div>
    </details>
  );
}

/* ── Table ─────────────────────────────────────────────────────────────── */

function ForecastRows({ rows }: { rows: CurrentForecast[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[1120px] border-collapse text-[0.76rem]">
        <thead>
          <tr className="border-b border-rule-hi bg-inset text-left">
            <Th className="w-24">Ticker</Th>
            <Th className="min-w-[150px]">Name</Th>
            <Th className="w-36">Sector</Th>
            <Th align="right" help="Last close, in rupees.">
              Last
            </Th>
            <Th
              align="right"
              help="The model's point forecast of the price in 30 sessions. No assumption about the index is needed: the forecast is the stock's own return."
            >
              Target
            </Th>
            <Th
              align="right"
              help="The model's actual output: the stock's predicted 30-session log return. Absolute, not relative to an index."
            >
              Return
            </Th>
            <Th
              align="right"
              help="Calibrated probability that the stock rises. Read it against 57.7%, the measured unconditional rate on this universe — NOT against 50%. A value near 0.5 is bearish."
            >
              P(up)
            </Th>
            <Th
              align="right"
              help="Out-of-sample Spearman correlation between predicted and realised return. 0 means no skill."
            >
              IC
            </Th>
            <Th
              align="right"
              help="t-statistic of that IC, and the only one of the three checks that tests significance. Read it with the sign: the gate tests |t|, so a large negative value marks a stock the model gets reliably WRONG."
            >
              IC t
            </Th>
            <Th
              align="right"
              help="Out-of-sample directional accuracy minus the majority-class baseline on the same window. This is the number that matters, not the raw hit rate."
            >
              Hit−base
            </Th>
            <Th className="w-40">Evidence</Th>
          </tr>
        </thead>
        <tbody>
          {rows.map((forecast) => (
            <Row key={forecast.ticker} forecast={forecast} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Th({
  children,
  align = "left",
  className,
  help,
}: {
  children: React.ReactNode;
  align?: "left" | "right";
  className?: string;
  help?: string;
}) {
  return (
    <th
      scope="col"
      title={help}
      className={cx(
        "px-2.5 py-1.5 text-[0.62rem] font-medium uppercase tracking-[0.12em] text-dim",
        align === "right" && "text-right",
        className,
      )}
    >
      {children}
      {help ? <span className="ml-0.5 opacity-50">?</span> : null}
    </th>
  );
}

function Row({ forecast }: { forecast: CurrentForecast }) {
  const excess = forecast.pred_return;
  const hit = forecast.eval_hit_rate;
  const base = forecast.eval_baseline_hit_rate;
  const ict = forecast.eval_rank_ic_t;

  /*
   * The model contradicting itself is worth showing, not smoothing over.
   * `prob_positive(pred)` is the share of calibration residuals above `-pred`,
   * so a probability above 0.5 on a negative point forecast means the model is
   * biased low for that ticker. The composite used to hide this by flooring;
   * with no composite the disagreement is simply displayed.
   */
  const contradicts =
    excess !== null &&
    excess < 0 &&
    forecast.prob_up !== null &&
    forecast.prob_up > 0.5;

  /*
   * A hit rate that equals its baseline to four decimals is not a near miss —
   * it means the model emitted one direction for every row and matched the
   * majority class by never disagreeing with it. 22 of 95 tickers do this.
   */
  const degenerate =
    hit !== null && base !== null && Math.abs(hit - base) < 1e-6;

  /*
   * The gate takes |t|, so a t of −2.3 PASSES a check while meaning the model
   * is reliably wrong about this stock. All four tickers that clear it sit
   * there. Marking it is the only way a reader sees the difference between a
   * check passed and a check passed backwards.
   */
  const antiSignal = ict !== null && ict <= -2.0;

  return (
    <tr className="border-b border-rule/70 last:border-b-0 hover:bg-raise">
      <td className="px-2.5 py-1">
        <Link
          href={`/stocks/${forecast.ticker}`}
          className="font-semibold text-bright underline-offset-2 hover:underline"
        >
          {forecast.ticker.replace(/\.NS$/, "")}
        </Link>
      </td>

      <td className="max-w-[220px] truncate px-2.5 py-1 text-text">
        {forecast.company ?? "—"}
      </td>

      <td className="max-w-[9rem] truncate px-2.5 py-1 text-dim">
        {forecast.sector ?? "—"}
      </td>

      <td className="px-2.5 py-1 text-right text-text">
        {decimal(forecast.current_price)}
      </td>

      <td className="px-2.5 py-1 text-right text-mid">
        {decimal(forecast.forecast_price)}
        {forecast.interval_low !== null && forecast.interval_high !== null ? (
          <span className="ml-1.5 text-[0.66rem] text-dim">
            [{decimal(forecast.interval_low, 0)}–
            {decimal(forecast.interval_high, 0)}]
          </span>
        ) : null}
      </td>

      <td
        className={cx(
          "px-2.5 py-1 text-right font-semibold",
          excess === null ? "text-dim" : excess >= 0 ? "text-pos" : "text-neg",
        )}
      >
        {signedPct(excess)}
      </td>

      <td className="px-2.5 py-1 text-right text-text">
        {contradicts ? (
          <span
            className="mr-1 text-bar"
            title="The calibrated probability sits above a coin flip while the point forecast says down. The model runs biased low for this ticker, and the two halves disagree."
          >
            !
          </span>
        ) : null}
        {probability(forecast.prob_up)}
      </td>

      <td
        className={cx(
          "px-2.5 py-1 text-right",
          forecast.eval_rank_ic === null
            ? "text-dim"
            : forecast.eval_rank_ic >= 0
              ? "text-text"
              : "text-neg/85",
        )}
      >
        {signed(forecast.eval_rank_ic)}
      </td>

      <td className="px-2.5 py-1 text-right">
        {antiSignal ? (
          <span
            className="mr-1 text-bar"
            title="Passes the significance check by magnitude while the sign says the model is reliably WRONG about this stock. The gate tests |t| and does not distinguish the two."
          >
            !
          </span>
        ) : null}
        <span
          className={cx(
            ict === null ? "text-dim" : antiSignal ? "text-neg" : "text-text",
          )}
        >
          {signed(ict, 2)}
        </span>
      </td>

      <td className="px-2.5 py-1 text-right">
        {hit === null || base === null ? (
          <span className="text-dim">—</span>
        ) : (
          <>
            {degenerate ? (
              <span
                className="mr-1 text-bar"
                title="Hit rate equals the majority-class baseline exactly: the model predicted one direction for every row, so it matched the baseline by never disagreeing with it."
              >
                =
              </span>
            ) : null}
            <span className={hit > base ? "text-pos" : "text-dim"}>
              {(hit - base >= 0 ? "+" : "−") + Math.abs(hit - base).toFixed(1)}
              pp
            </span>
          </>
        )}
      </td>

      <td className="px-2.5 py-1">
        <span className="flex flex-wrap items-center gap-1">
          <Badge tone={evidenceTone(forecast.forecast_confidence)}>
            {forecast.forecast_confidence === "INSUFFICIENT"
              ? "none"
              : (forecast.forecast_confidence ?? "—")}
          </Badge>
          <Badge tone={verdictTone(forecast.critic_verdict)}>
            {forecast.critic_verdict ?? "—"}
          </Badge>
        </span>
      </td>
    </tr>
  );
}
