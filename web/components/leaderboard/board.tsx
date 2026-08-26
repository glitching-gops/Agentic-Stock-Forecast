"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { Badge, Eyebrow, Meter, evidenceTone, verdictTone } from "@/components/ui";
import {
  cx,
  decimal,
  probability,
  scoreBasisExplainer,
  scoreBasisLabel,
  signed,
  signedPct,
} from "@/lib/format";
import type { LeaderboardEntry } from "@/lib/types";

/**
 * The leaderboard, filtered and sorted entirely in the browser.
 *
 * The whole table is 95 rows and arrives with the page, so a filter change
 * costs no network at all — no round trip to a free-tier instance that may be
 * asleep. The API's own filter parameters still exist and still work; they are
 * just not the right tool at this size.
 *
 * The one thing NOT recomputed here is `rank`. It is issued by the API as a
 * SQL window function over the full filtered set, with ties sharing a rank.
 * Re-deriving that client-side would mean maintaining the same tie semantics
 * in two languages, so instead the rank column is shown only while the
 * ordering it refers to is in effect, and hidden the moment the reader sorts
 * by something else.
 */

const SORTS = {
  composite_score: "Score",
  pred_excess_return: "Excess return",
  prob_outperform: "P(outperform)",
  eval_rank_ic: "Rank IC",
  eval_hit_rate: "Hit rate",
} as const;

type SortKey = keyof typeof SORTS;

/** Order in which the unranked groups are disclosed — most informative first. */
const BASIS_ORDER = [
  "NOT_LONG",
  "FLAGGED_OUT",
  "NO_EVIDENCE",
  "NO_FORECAST",
] as const;

export function LeaderboardBoard({ entries }: { entries: LeaderboardEntry[] }) {
  const [sector, setSector] = useState("");
  const [evidence, setEvidence] = useState("");
  const [verdict, setVerdict] = useState("");
  const [query, setQuery] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("composite_score");

  const sectors = useMemo(
    () =>
      Array.from(
        new Set(entries.map((e) => e.sector).filter((s): s is string => !!s)),
      ).sort(),
    [entries],
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return entries.filter((entry) => {
      if (sector && entry.sector !== sector) return false;
      if (evidence && entry.forecast_confidence !== evidence) return false;
      if (verdict && entry.critic_verdict !== verdict) return false;
      if (
        q &&
        !`${entry.company ?? ""} ${entry.ticker}`.toLowerCase().includes(q)
      ) {
        return false;
      }
      return true;
    });
  }, [entries, sector, evidence, verdict, query]);

  const sorted = useMemo(
    () => [...filtered].sort(byDescendingWithNullsLast(sortKey)),
    [filtered, sortKey],
  );

  const ranked = sorted.filter((e) => e.score_basis === "RANKED");
  const unranked = sorted.filter((e) => e.score_basis !== "RANKED");

  const groups = BASIS_ORDER.map((basis) => ({
    basis: basis as string,
    rows: unranked.filter((e) => e.score_basis === basis),
  }))
    .concat({
      basis: "__other__",
      rows: unranked.filter(
        (e) =>
          !BASIS_ORDER.includes(e.score_basis as (typeof BASIS_ORDER)[number]),
      ),
    })
    .filter((group) => group.rows.length > 0);

  const showRank = sortKey === "composite_score";
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
        total={entries.length}
        onReset={() => {
          setSector("");
          setEvidence("");
          setVerdict("");
          setQuery("");
          setSortKey("composite_score");
        }}
      />

      {/* ── Rated ────────────────────────────────────────────────────────── */}
      <section aria-labelledby="ranked-heading">
        <div className="mb-2 flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 border-b border-rule-hi pb-1.5">
          <h2
            id="ranked-heading"
            className="text-[0.72rem] font-semibold uppercase tracking-[0.18em] text-bright"
          >
            Rated
            <span className="ml-2 font-normal text-dim">{ranked.length}</span>
          </h2>
          {!showRank && ranked.length > 0 ? (
            <p className="text-[0.68rem] text-dim">
              Rank is assigned on score — hidden while sorted by{" "}
              {SORTS[sortKey].toLowerCase()}.
            </p>
          ) : null}
        </div>

        {ranked.length === 0 ? (
          <div className="border border-rule bg-shell px-4 py-6 text-center text-[0.78rem] text-dim">
            {anyFilter
              ? "No rated name matches these filters."
              : "No name currently clears the evidence gate with a positive forecast."}
          </div>
        ) : (
          <EntryTable rows={ranked} showRank={showRank} />
        )}
      </section>

      {/* ── Not rated ────────────────────────────────────────────────────── */}
      {groups.length > 0 ? (
        <section aria-labelledby="unranked-heading" className="space-y-1.5">
          <div className="mb-2 border-b border-rule-hi pb-1.5">
            <h2
              id="unranked-heading"
              className="text-[0.72rem] font-semibold uppercase tracking-[0.18em] text-bright"
            >
              Not rated
              <span className="ml-2 font-normal text-dim">
                {unranked.length}
              </span>
            </h2>
            <p className="mt-2 max-w-4xl font-prose text-[0.8rem] leading-relaxed text-mid">
              Every name below scores exactly 0.0, and that one value covers
              several unrelated situations. They are grouped by the reason
              rather than numbered off in an order the score cannot support.
            </p>
          </div>

          {groups.map((group) => (
            <BasisGroup
              key={group.basis}
              basis={group.basis}
              rows={group.rows}
            />
          ))}
        </section>
      ) : null}
    </div>
  );
}

/* ── Sorting ───────────────────────────────────────────────────────────── */

/**
 * Descending, nulls last — matching the API's `ORDER BY <key> DESC NULLS LAST`
 * so a client-side sort never reorders rows differently from the server.
 */
function byDescendingWithNullsLast(key: SortKey) {
  return (a: LeaderboardEntry, b: LeaderboardEntry) => {
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

function BasisGroup({
  basis,
  rows,
}: {
  basis: string;
  rows: LeaderboardEntry[];
}) {
  const label = basis === "__other__" ? "Other" : scoreBasisLabel(basis);
  const explainer =
    basis === "__other__"
      ? "These rows carry a score basis this interface does not recognise."
      : scoreBasisExplainer(basis);

  return (
    <details className="group border border-rule bg-shell">
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
              {label}
            </span>
            <span className="text-[0.7rem] text-dim">{rows.length}</span>
          </span>
          <span className="mt-1 block max-w-4xl font-prose text-[0.78rem] leading-relaxed text-dim">
            {explainer}
          </span>
        </span>
      </summary>
      <div className="border-t border-rule">
        <EntryTable rows={rows} showRank={false} flush />
      </div>
    </details>
  );
}

/* ── Table ─────────────────────────────────────────────────────────────── */

function EntryTable({
  rows,
  showRank,
  flush = false,
}: {
  rows: LeaderboardEntry[];
  showRank: boolean;
  flush?: boolean;
}) {
  return (
    <div className={cx("overflow-x-auto", !flush && "border border-rule")}>
      <table className="w-full min-w-[1120px] border-collapse text-[0.76rem]">
        <thead>
          <tr className="border-b border-rule-hi bg-inset text-left">
            {showRank ? <Th className="w-9 text-right">#</Th> : null}
            <Th className="w-24">Ticker</Th>
            <Th className="min-w-[150px]">Name</Th>
            <Th className="w-36">Sector</Th>
            <Th align="right" help="Last close, in rupees.">
              Last
            </Th>
            <Th
              align="right"
              help="Implied price assuming the benchmark index is flat. The model forecasts relative performance and says nothing about where the index goes."
            >
              Target
            </Th>
            <Th
              align="right"
              help="The model's actual output: predicted 30-session log return relative to the stock's benchmark index."
            >
              Excess
            </Th>
            <Th
              align="right"
              help="Calibrated probability of beating the benchmark. 50% is a coin flip."
            >
              P(out)
            </Th>
            <Th
              align="right"
              help="Out-of-sample Spearman correlation between predicted and realised excess return. 0 means no skill."
            >
              IC
            </Th>
            <Th
              align="right"
              help="Out-of-sample directional accuracy minus the majority-class baseline on the same window. This is the number that matters, not the raw hit rate."
            >
              Hit−base
            </Th>
            <Th className="w-40">Evidence</Th>
            <Th
              className="w-32"
              help="Ranking heuristic in [0,100]: predicted excess return and conviction, multiplied by an evidence grade. Not an expected return."
            >
              Score
            </Th>
          </tr>
        </thead>
        <tbody>
          {rows.map((entry) => (
            <Row key={entry.ticker} entry={entry} showRank={showRank} />
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

function Row({
  entry,
  showRank,
}: {
  entry: LeaderboardEntry;
  showRank: boolean;
}) {
  const excess = entry.pred_excess_return;
  const hit = entry.eval_hit_rate;
  const base = entry.eval_baseline_hit_rate;

  /*
   * The model contradicting itself is worth showing, not smoothing over.
   * `prob_positive(pred)` is the share of calibration residuals above `-pred`,
   * so a probability above 0.5 on a negative point forecast means the model is
   * biased low for that ticker. This is exactly the disagreement that put PNB
   * at rank 3 with a predicted 1.69% underperformance before the conviction
   * gate closed it — the gate stops it from ranking, and this marks it so the
   * reader sees the disagreement rather than just its consequence.
   */
  const contradicts =
    excess !== null &&
    excess < 0 &&
    entry.prob_outperform !== null &&
    entry.prob_outperform > 0.5;

  /*
   * A hit rate that equals its baseline to four decimals is not a near miss —
   * it means the model emitted one direction for every row and matched the
   * majority class by never disagreeing with it. 22 of 95 tickers do this.
   */
  const degenerate =
    hit !== null && base !== null && Math.abs(hit - base) < 1e-6;

  return (
    <tr className="border-b border-rule/70 last:border-b-0 hover:bg-raise">
      {showRank ? (
        <td className="px-2.5 py-1 text-right text-dim">{entry.rank ?? "—"}</td>
      ) : null}

      <td className="px-2.5 py-1">
        <Link
          href={`/stocks/${entry.ticker}`}
          className="font-semibold text-bright underline-offset-2 hover:underline"
        >
          {entry.ticker.replace(/\.NS$/, "")}
        </Link>
      </td>

      <td className="max-w-[220px] truncate px-2.5 py-1 text-text">
        {entry.company ?? "—"}
      </td>

      <td className="max-w-[9rem] truncate px-2.5 py-1 text-dim">
        {entry.sector ?? "—"}
      </td>

      <td className="px-2.5 py-1 text-right text-text">
        {decimal(entry.current_price)}
      </td>

      <td className="px-2.5 py-1 text-right text-mid">
        {decimal(entry.forecast_price)}
        {entry.interval_low !== null && entry.interval_high !== null ? (
          <span className="ml-1.5 text-[0.66rem] text-dim">
            [{decimal(entry.interval_low, 0)}–{decimal(entry.interval_high, 0)}]
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
            title="The calibrated probability says up while the point forecast says down. The model is biased low for this ticker; the composite refuses to rank on the cheerier half."
          >
            !
          </span>
        ) : null}
        {probability(entry.prob_outperform)}
      </td>

      <td
        className={cx(
          "px-2.5 py-1 text-right",
          entry.eval_rank_ic === null
            ? "text-dim"
            : entry.eval_rank_ic >= 0
              ? "text-text"
              : "text-neg/85",
        )}
      >
        {signed(entry.eval_rank_ic)}
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
          <Badge tone={evidenceTone(entry.forecast_confidence)}>
            {entry.forecast_confidence === "INSUFFICIENT"
              ? "none"
              : (entry.forecast_confidence ?? "—")}
          </Badge>
          <Badge tone={verdictTone(entry.critic_verdict)}>
            {entry.critic_verdict ?? "—"}
          </Badge>
        </span>
      </td>

      <td className="px-2.5 py-1">
        <Meter value={entry.composite_score} />
      </td>
    </tr>
  );
}
