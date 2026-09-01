/**
 * TypeScript mirrors of the Pydantic response models in `api/schemas/`.
 *
 * These are hand-written rather than generated, so they can drift. The pairing
 * is asserted at runtime in one place only — `lib/api.ts` narrows unknown JSON
 * into these shapes and tolerates missing fields — because the failure mode we
 * actually care about is a column the pipeline has not written yet, not a
 * wholesale schema change.
 *
 * Nearly everything is nullable on purpose. Most rows in this dataset have no
 * forecast and no evidence; a type that pretended otherwise would push
 * `undefined` handling into every component instead of into the formatters.
 */

export type Verdict = "APPROVED" | "FLAGGED" | "REJECTED";
export type EvidenceGrade = "STRONG" | "WEAK" | "INSUFFICIENT";

/**
 * What is known about a forecast, replacing the old `score_basis`.
 *
 * That field existed to disambiguate a composite of 0.0, which covered "never
 * evaluated", "predicted to underperform" and "flagged out" at once. There is
 * no composite now, so three of its five values described the ranking rather
 * than the forecast and are gone with it. What remains is the evidence grade
 * the API already publishes, plus the one case it cannot express: a stock the
 * pipeline produced no prediction for at all.
 *
 * Derived on the client by `evidenceState`, never sent as a field — a second
 * server column carrying a value derivable from two others is a column that
 * can disagree with them.
 */
export type EvidenceState =
  | "STRONG"
  | "WEAK"
  | "INSUFFICIENT"
  | "NO_FORECAST";

export interface StockInfo {
  ticker: string;
  company: string;
  sector: string;
}

export interface StockList {
  stocks: StockInfo[];
  total: number;
}

/**
 * One stock's current forecast. NO RANK AND NO SCORE — the API publishes
 * neither, and adding either here would invite a component to invent one.
 */
export interface CurrentForecast {
  ticker: string;
  company: string | null;
  sector: string | null;

  current_price: number | null;
  /** Implied price ASSUMING THE BENCHMARK IS FLAT. */
  forecast_price: number | null;
  direction: string | null;
  /** Implied percentage change. Was `upside_pct`; a forecast can point down. */
  change_pct: number | null;
  /** Predicted 30-session log return in excess of the benchmark. */
  pred_return: number | null;

  interval_low: number | null;
  interval_high: number | null;
  interval_coverage: number | null;
  prob_up: number | null;
  random_walk_price: number | null;

  benchmark_ticker: string | null;
  benchmark_name: string | null;
  benchmark_sector_specific: boolean | null;

  critic_verdict: Verdict | string | null;
  forecast_confidence: EvidenceGrade | string | null;
  signal_narrative: string | null;

  eval_rank_ic: number | null;
  /**
   * Read WITH the sign. The gate tests |t|, so a large NEGATIVE value marks a
   * ticker the model gets reliably wrong, not one it gets right.
   */
  eval_rank_ic_t: number | null;
  eval_hit_rate: number | null;
  eval_baseline_hit_rate: number | null;
  eval_beats_random_walk: boolean | null;
  model_version: string | null;
  /** Measured weekly, so this may be older than `last_updated`. */
  evaluated_at: string | null;
  last_updated: string | null;
}

export interface ForecastListResponse {
  forecasts: CurrentForecast[];
  total: number;
  last_updated: string;
  filters_applied: Record<string, string>;
  methodology: string;
}

export interface EvaluationEvidence {
  rank_ic: number | null;
  hit_rate: number | null;
  /** hit_rate is only meaningful relative to this. */
  baseline_hit_rate: number | null;
  beats_random_walk: boolean | null;
  model_version: string | null;
  evaluated_at: string | null;
}

export interface Forecast {
  ticker: string;
  company: string | null;
  sector: string | null;

  pred_return: number | null;
  benchmark_ticker: string | null;
  benchmark_name: string | null;
  /** False when the stock falls back to NIFTY 50 for want of a sector index. */
  benchmark_sector_specific: boolean | null;

  current_price: number | null;
  forecast_price: number | null;
  direction: "OUTPERFORM" | "UNDERPERFORM" | "UNAVAILABLE" | string | null;
  change_pct: number | null;
  random_walk_price: number | null;

  interval_low: number | null;
  interval_high: number | null;
  interval_coverage: number | null;
  prob_up: number | null;

  evaluation: EvaluationEvidence | null;
  forecast_confidence: EvidenceGrade | string | null;
  signal_narrative: string | null;
  critic_verdict: Verdict | string | null;
  critic_reasoning: string | null;
  critic_flags: string[];
  critic_source: string | null;

  forecast_available: boolean;
  forecast_error: string | null;
  universe_rule: string | null;
  last_updated: string | null;
}

/**
 * One row of `signals`. The table is wide (24 features plus bookkeeping) and
 * the API returns `SELECT *`, so this is indexed rather than exhaustively
 * typed; the named fields are the ones the UI actually reads.
 */
export interface SignalRow {
  date: string;
  close: number | null;
  volume: number | null;
  sma_20: number | null;
  ema_9: number | null;
  ema_21: number | null;
  ema_50: number | null;
  bb_upper: number | null;
  bb_lower: number | null;
  rsi: number | null;
  macd_hist: number | null;
  stoch_k: number | null;
  williams_r: number | null;
  roc_10: number | null;
  [key: string]: string | number | boolean | null | undefined;
}

export interface SignalsResponse {
  ticker: string;
  signals_df: SignalRow[];
  latest_signals: Record<string, number | string | boolean | null>;
  rows: number;
}

export interface Headline {
  headline: string;
  sentiment_label: string | null;
  sentiment_score: number | null;
  date: string | null;
}
