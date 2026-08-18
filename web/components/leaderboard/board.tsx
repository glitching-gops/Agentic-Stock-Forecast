"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { Badge, Card, ScoreMeter, evidenceTone, verdictTone } from "@/components/ui";
import {
  cx,
  money,
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
 * in two languages, so instead the rank badge is shown only while the ordering
 * it refers to is in effect, and hidden the moment the reader sorts by
 * something else.
 */

const SORTS = {
  composite_score: "Composite score",
  pred_excess_return: "Predicted excess return",
  prob_outperform: "P(outperform)",
  eval_rank_ic: "Out-of-sample rank IC",
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
        (e) => !BASIS_ORDER.includes(e.score_basis as (typeof BASIS_ORDER)[number]),
      ),
    })
    .filter((group) => group.rows.length > 0);

  const showRank = sortKey === "composite_score";
  const anyFilter = Boolean(sector || evidence || verdict || query);

  return (
    <div className="space-y-6">
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

      {/* ── Ranked ───────────────────────────────────────────────────────── */}
      <section aria-labelledby="ranked-heading">
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
          <h2
            id="ranked-heading"
            className="text-sm font-semibold uppercase tracking-[0.09em] text-mist-300"
          >
            Ranked
            <span className="ml-2 font-mono text-xs font-normal text-mist-500">
              {ranked.length}
            </span>
          </h2>
          {!showRank && ranked.length > 0 ? (
            <p className="text-xs text-mist-500">
              Ranks are assigned on composite score; hidden while sorted by{" "}
              {SORTS[sortKey].toLowerCase()}.
            </p>
          ) : null}
        </div>

        {ranked.length === 0 ? (
          <Card className="px-6 py-8 text-center text-sm text-mist-400">
            {anyFilter
              ? "No ranked stock matches these filters."
              : "No stock currently clears the evidence gate with a positive forecast."}
          </Card>
        ) : (
          <EntryTable rows={ranked} showRank={showRank} />
        )}
      </section>

      {/* ── Not ranked ───────────────────────────────────────────────────── */}
      {groups.length > 0 ? (
        <section aria-labelledby="unranked-heading" className="space-y-3">
          <div>
            <h2
              id="unranked-heading"
              className="text-sm font-semibold uppercase tracking-[0.09em] text-mist-300"
            >
              Not ranked
              <span className="ml-2 font-mono text-xs font-normal text-mist-500">
                {unranked.length}
              </span>
            </h2>
            <p className="mt-1 max-w-3xl text-xs leading-relaxed text-mist-500">
              Every stock below scores exactly 0.0, and that one value covers
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
    <Card className="flex flex-wrap items-end gap-3 p-3">
      <Field label="Search">
        <input
          type="search"
          value={props.query}
          onChange={(e) => props.setQuery(e.target.value)}
          placeholder="Company or ticker"
          className="w-44 rounded-md border border-ink-500 bg-ink-800 px-2.5 py-1.5 text-sm text-mist-100 placeholder:text-mist-500 focus:border-brand-500/60 focus:outline-none"
        />
      </Field>

      <Field label="Sector">
        <Select value={props.sector} onChange={props.setSector}>
          <option value="">All sectors</option>
          {props.sectors.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </Select>
      </Field>

      <Field label="Evidence">
        <Select value={props.evidence} onChange={props.setEvidence}>
          <option value="">Any grade</option>
          <option value="STRONG">Strong</option>
          <option value="WEAK">Weak</option>
          <option value="INSUFFICIENT">None</option>
        </Select>
      </Field>

      <Field label="Critic verdict">
        <Select value={props.verdict} onChange={props.setVerdict}>
          <option value="">Any verdict</option>
          <option value="APPROVED">Approved</option>
          <option value="FLAGGED">Flagged</option>
          <option value="REJECTED">Rejected</option>
        </Select>
      </Field>

      <Field label="Sort by">
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

      <div className="ml-auto flex items-center gap-3 pb-1">
        <span className="nums text-xs text-mist-500">
          {props.matched} of {props.total}
        </span>
        {dirty ? (
          <button
            type="button"
            onClick={props.onReset}
            className="rounded-md border border-ink-500 px-2 py-1 text-xs font-medium text-mist-300 hover:bg-ink-600"
          >
            Reset
          </button>
        ) : null}
      </div>
    </Card>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[0.65rem] font-semibold uppercase tracking-[0.09em] text-mist-500">
        {label}
      </span>
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
      className="rounded-md border border-ink-500 bg-ink-800 px-2.5 py-1.5 text-sm text-mist-100 focus:border-brand-500/60 focus:outline-none"
    >
      {children}
    </select>
  );
}

/* ── Grouped disclosure ────────────────────────────────────────────────── */

function BasisGroup({ basis, rows }: { basis: string; rows: LeaderboardEntry[] }) {
  const label = basis === "__other__" ? "Other" : scoreBasisLabel(basis);
  const explainer =
    basis === "__other__"
      ? "These rows carry a score basis this interface does not recognise."
      : scoreBasisExplainer(basis);

  return (
    <details className="group rounded-xl border border-ink-500/70 bg-ink-700/40">
      <summary className="flex cursor-pointer list-none items-start gap-3 px-4 py-3 [&::-webkit-details-marker]:hidden">
        <span
          aria-hidden
          className="mt-1 text-mist-500 transition-transform group-open:rotate-90"
        >
          ▶
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-baseline gap-2">
            <span className="text-sm font-semibold text-mist-100">{label}</span>
            <span className="nums rounded-md bg-ink-600 px-1.5 py-0.5 text-[0.68rem] font-semibold text-mist-400">
              {rows.length}
            </span>
          </span>
          <span className="mt-1 block max-w-3xl text-xs leading-relaxed text-mist-500">
            {explainer}
          </span>
        </span>
      </summary>
      <div className="border-t border-ink-500/70">
        <EntryTable rows={rows} showRank={false} />
      </div>
    </details>
  );
}

/* ── Table ─────────────────────────────────────────────────────────────── */

function EntryTable({
  rows,
  showRank,
}: {
  rows: LeaderboardEntry[];
  showRank: boolean;
}) {
  return (
    <div className="overflow-x-auto rounded-xl border border-ink-500/70">
      <table className="w-full min-w-[1020px] border-collapse text-sm">
        <thead>
          <tr className="border-b border-ink-500/70 bg-ink-800/70 text-left">
            {showRank ? <Th className="w-14 text-center">#</Th> : null}
            <Th className="min-w-[230px]">Stock</Th>
            <Th align="right">Price</Th>
            <Th
              align="right"
              help="Implied price assuming the benchmark index is flat. The model forecasts relative performance and says nothing about where the index goes."
            >
              Implied target
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
              Rank IC
            </Th>
            <Th
              align="right"
              help="Out-of-sample directional accuracy against the majority-class baseline on the same window."
            >
              Hit vs base
            </Th>
            <Th>Evidence</Th>
            <Th
              className="min-w-[150px]"
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
  align?: "left" | "right" | "center";
  className?: string;
  help?: string;
}) {
  return (
    <th
      scope="col"
      title={help}
      className={cx(
        "px-3 py-2.5 text-[0.68rem] font-semibold uppercase tracking-[0.07em] text-mist-500",
        align === "right" && "text-right",
        align === "center" && "text-center",
        className,
      )}
    >
      {children}
      {help ? <span className="ml-1 opacity-60">ⓘ</span> : null}
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
    <tr className="border-b border-ink-500/40 last:border-b-0 hover:bg-ink-600/40">
      {showRank ? (
        <td className="nums px-3 py-2.5 text-center text-sm font-semibold text-mist-300">
          {entry.rank ?? "—"}
        </td>
      ) : null}

      <td className="px-3 py-2.5">
        <Link
          href={`/stocks/${entry.ticker}`}
          className="block truncate font-medium text-mist-100 hover:text-brand-300"
        >
          {entry.company ?? entry.ticker}
        </Link>
        <div className="mt-0.5 truncate text-xs text-mist-500">
          <span className="font-mono">{entry.ticker.replace(/\.NS$/, "")}</span>
          {entry.sector ? ` · ${entry.sector}` : ""}
        </div>
      </td>

      <td className="nums px-3 py-2.5 text-right text-mist-200">
        {money(entry.current_price)}
      </td>

      <td className="nums px-3 py-2.5 text-right text-mist-300">
        {money(entry.forecast_price)}
        {entry.interval_low !== null && entry.interval_high !== null ? (
          <div className="text-[0.68rem] text-mist-500">
            {money(entry.interval_low)} – {money(entry.interval_high)}
          </div>
        ) : null}
      </td>

      <td
        className={cx(
          "nums px-3 py-2.5 text-right font-semibold",
          excess === null
            ? "text-mist-500"
            : excess >= 0
              ? "text-pos-500"
              : "text-neg-500",
        )}
      >
        {signedPct(excess)}
      </td>

      <td className="nums px-3 py-2.5 text-right text-mist-200">
        <span className="inline-flex items-center gap-1">
          {contradicts ? (
            <span
              className="text-warn-500"
              title="The calibrated probability says up while the point forecast says down. The model is biased low for this ticker; the composite refuses to rank on the cheerier half."
            >
              ⚠
            </span>
          ) : null}
          {probability(entry.prob_outperform)}
        </span>
      </td>

      <td
        className={cx(
          "nums px-3 py-2.5 text-right",
          entry.eval_rank_ic === null
            ? "text-mist-500"
            : entry.eval_rank_ic >= 0
              ? "text-mist-200"
              : "text-neg-500/90",
        )}
      >
        {signed(entry.eval_rank_ic)}
      </td>

      <td className="nums px-3 py-2.5 text-right text-mist-300">
        {hit === null || base === null ? (
          "—"
        ) : (
          <span className="inline-flex items-center gap-1">
            {degenerate ? (
              <span
                className="text-warn-500"
                title="Hit rate equals the majority-class baseline exactly: the model predicted one direction for every row, so it matched the baseline by never disagreeing with it."
              >
                ≡
              </span>
            ) : null}
            <span className={hit > base ? "text-pos-500" : "text-mist-400"}>
              {(hit - base >= 0 ? "+" : "−") + Math.abs(hit - base).toFixed(1)}
              pp
            </span>
          </span>
        )}
      </td>

      <td className="px-3 py-2.5">
        <div className="flex flex-wrap items-center gap-1">
          <Badge tone={evidenceTone(entry.forecast_confidence)}>
            {entry.forecast_confidence === "INSUFFICIENT"
              ? "None"
              : (entry.forecast_confidence ?? "—")}
          </Badge>
          <Badge tone={verdictTone(entry.critic_verdict)}>
            {entry.critic_verdict ?? "—"}
          </Badge>
        </div>
      </td>

      <td className="px-3 py-2.5">
        <ScoreMeter value={entry.composite_score} />
      </td>
    </tr>
  );
}
