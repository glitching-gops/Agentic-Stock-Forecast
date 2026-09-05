/**
 * The Phase 2 record — HARDCODED, and deliberately so.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * SOURCE OF TRUTH: `experiment_runs.metrics["baselines"]`, written by the
 * weekly job in `pipeline/baselines.py::compare_baselines()`. The same numbers
 * render from `tools/run_baselines.py`.
 *
 * DRIFT RISK: these figures are transcribed, not fetched, so they will go stale
 * the moment the panel changes and nothing here will notice. That is an
 * accepted trade rather than an oversight. The API exposes no endpoint over
 * `experiment_runs`, and adding one would put a research log — a table whose
 * rows are keyed by config_hash and data_hash and mean nothing without both —
 * behind a public read path in order to render a page that describes a CLOSED
 * phase. A closed phase does not move.
 *
 * WHAT TO DO WHEN IT DOES MOVE: re-run `tools/run_baselines.py`, copy the
 * table, and update `MEASURED_ON` and `PANEL` below with it. If Phase 2 is ever
 * reopened, replace this file with a fetch.
 * ─────────────────────────────────────────────────────────────────────────────
 */

/** The date the tables below were measured. Not the date they were written. */
export const MEASURED_ON = "22 Aug 2026";

/**
 * The pre-registered success criterion, fixed with the owner on 21 Aug 2026 —
 * BEFORE the TimesFM result was seen, so that a marginal number could not be
 * read as a win afterwards.
 */
export const BAR = 2;

export const PANEL = {
  tickers: 100,
  rows: 224_338,
  labelled: 221_338,
  dates: 2_429,
  medianNamesPerDate: 94,
  folds: 5,
  rebalances: 64,
  horizon: 30,
  embargo: 30,
} as const;

export type Family =
  | "naive"
  | "linear"
  | "tree"
  | "foundation"
  | "adapter"
  | "probe"
  | "finetune";

export interface Comparator {
  /** The key this ran under in `experiment_runs`. */
  id: string;
  label: string;
  family: Family;
  /** Mean per-date rank IC. Undefined for a comparator with no ordering. */
  dailyIc?: number;
  /** Rank IC over the 64 NON-OVERLAPPING rebalance dates. */
  rebIc?: number;
  /** t-statistic of `rebIc`. The only number the bar is applied to. */
  rebT?: number;
  mae: number;
  /** MAE relative to the `zero` floor, as a percentage. Lower is better. */
  vsFloor?: number;
  seconds: number;
  note: string;
}

/**
 * The full 100-ticker panel run. Assembled from three separate runs and merged
 * only after 20 shared comparator instances matched to the last digit, which
 * is what proves all three scored the same rows on the same folds.
 *
 * Ordered by MAE ascending — by the magnitude question — rather than by the
 * ranking t-statistic, because the top of this table is where the artifacts
 * are and they should be read before the models below them.
 */
export const COMPARATORS: Comparator[] = [
  {
    id: "zero",
    label: "zero",
    family: "naive",
    mae: 0.06678,
    seconds: 0.5,
    note: "Predict no excess return at all. This is the floor every other row is measured against.",
  },
  {
    id: "train_mean",
    label: "train_mean",
    family: "naive",
    mae: 0.06671,
    vsFloor: -0.1,
    seconds: 0.5,
    note: "One constant per fold. It exists to detect exactly the artifact it produced here — it beats the floor on MAE while carrying no cross-sectional information whatsoever.",
  },
  {
    id: "reversal_5d",
    label: "reversal_5d",
    family: "naive",
    dailyIc: 0.0073,
    rebIc: -0.0002,
    rebT: -0.01,
    mae: 0.06671,
    vsFloor: -0.1,
    seconds: 1.1,
    note: "Negated 5-session return. Beats the floor on MAE and ranks nothing: reb_t is −0.01.",
  },
  {
    id: "momentum_20d",
    label: "momentum_20d",
    family: "naive",
    dailyIc: 0.0176,
    rebIc: 0.0061,
    rebT: 0.29,
    mae: 0.06672,
    vsFloor: -0.1,
    seconds: 1.1,
    note: "20-session return carried forward. Same story as reversal — it is capturing the level, not the ordering.",
  },
  {
    id: "linear_factor",
    label: "linear_factor",
    family: "linear",
    dailyIc: 0.0155,
    rebIc: 0.0236,
    rebT: 1.36,
    mae: 0.06687,
    vsFloor: 0.1,
    seconds: 1.2,
    note: "A ridge on 15 cross-sectionally standardised columns. The strongest honest thing in the project, and still short of the bar.",
  },
  {
    id: "chronos2_xl_2048",
    label: "chronos2_xl @2048",
    family: "foundation",
    dailyIc: 0.0169,
    rebIc: 0.0123,
    rebT: 0.65,
    mae: 0.0679,
    vsFloor: 1.7,
    seconds: 5952,
    note: "Chronos-2 120M with cross_learning on — each forecast conditioned on the rest of that date's cross-section. The feature the model is sold on, and negative at both contexts.",
  },
  {
    id: "chronos2_xl_512",
    label: "chronos2_xl @512",
    family: "foundation",
    dailyIc: -0.0125,
    rebIc: -0.0018,
    rebT: -0.09,
    mae: 0.06819,
    vsFloor: 2.1,
    seconds: 388,
    note: "Cross-learning at the shorter context. −0.09.",
  },
  {
    id: "chronos2_2048",
    label: "chronos2 @2048",
    family: "foundation",
    dailyIc: 0.04,
    rebIc: 0.0289,
    rebT: 1.86,
    mae: 0.0682,
    vsFloor: 2.1,
    seconds: 5894,
    note: "The best foundation-model cell measured, in the single most expensive configuration — and it does not survive a context change.",
  },
  {
    id: "series_drift",
    label: "series_drift",
    family: "adapter",
    dailyIc: -0.0048,
    rebIc: -0.0051,
    rebT: -0.21,
    mae: 0.06822,
    vsFloor: 2.2,
    seconds: 4.9,
    note: "Extrapolate the historical mean drift of the relative-price series.",
  },
  {
    id: "chronos2_512",
    label: "chronos2 @512",
    family: "foundation",
    dailyIc: 0.0173,
    rebIc: 0.0131,
    rebT: 0.88,
    mae: 0.06891,
    vsFloor: 3.2,
    seconds: 398,
    note: "The same 120M checkpoint at a quarter of the context. +1.86 collapses to +0.88 — the edge was context, not architecture.",
  },
  {
    id: "pooled_xgb",
    label: "pooled_xgb",
    family: "tree",
    dailyIc: 0.0122,
    rebIc: 0.0273,
    rebT: 1.73,
    mae: 0.06935,
    vsFloor: 3.8,
    seconds: 5.6,
    note: "An untuned gradient-boosted tree on the pooled panel. Ranks better than every foundation model at matched context, and is 3.8% worse than predicting nothing on magnitude.",
  },
  {
    id: "timesfm25_512",
    label: "timesfm25 @512",
    family: "foundation",
    dailyIc: 0.0003,
    rebIc: 0.0084,
    rebT: 0.57,
    mae: 0.06984,
    vsFloor: 4.6,
    seconds: 702,
    note: "Google's 231M decoder-only model. A different architecture, corpus and objective from Chronos, and it agrees with it.",
  },
  {
    id: "series_last_return",
    label: "series_last_return",
    family: "adapter",
    dailyIc: -0.0161,
    rebIc: -0.0115,
    rebT: -0.52,
    mae: 0.09634,
    vsFloor: 44.3,
    seconds: 4.9,
    note: "Repeat the last 30-session relative return. 44% worse than predicting nothing: relative momentum at this horizon reverses.",
  },
  {
    id: "series_zero",
    label: "series_zero",
    family: "adapter",
    mae: 0.06678,
    vsFloor: 0.0,
    seconds: 4.4,
    note: "The calibration case. It reproduces `zero` to the fifth decimal through the full series adapter, which is what proves the adapter is not what is being measured.",
  },
];

/* ── The two results that cleared the bar, and were retired ─────────────── */

export interface Retired {
  id: string;
  label: string;
  /** The t-statistic as originally quoted. */
  headlineT: number;
  headlineIc: number;
  measured: string;
  /** One line, on the axis. */
  verdict: string;
}

export const RETIRED: Retired[] = [
  {
    id: "valuation",
    label: "pooled_xgb + valuation",
    headlineT: 3.32,
    headlineIc: 0.0708,
    measured: "23 Aug 2026",
    verdict: "A sample effect. Absent at every lag on a fixed sample, lag zero included.",
  },
  {
    id: "lora",
    label: "chronos2 + LoRA",
    headlineT: 2.37,
    headlineIc: 0.0336,
    measured: "25 Aug 2026",
    verdict: "Carried entirely by the two smallest folds. Inverts on the most recent one.",
  },
];

/**
 * The valuation result across `min_train`, on IDENTICAL rows.
 *
 * The headline was measured at 380. Everything else in the project is reported
 * at the harness default of 500.
 */
export const VALUATION_SWEEP: { minTrain: number; t: number }[] = [
  { minTrain: 300, t: 0.37 },
  { minTrain: 340, t: 1.3 },
  { minTrain: 380, t: 3.32 },
  { minTrain: 420, t: 1.18 },
  { minTrain: 460, t: 1.72 },
  { minTrain: 500, t: 1.0 },
  { minTrain: 540, t: -0.46 },
];

/**
 * Re-scored on the FIXED sample that survives a 365-day extra lag, so the row
 * count no longer moves with the setting. The edge is gone at every lag.
 */
export const VALUATION_FIXED_SAMPLE: {
  extraLagDays: number;
  t380: number;
  t460: number;
  t500: number;
}[] = [
  { extraLagDays: 0, t380: 0.77, t460: 0.25, t500: -0.56 },
  { extraLagDays: 90, t380: 0.3, t460: -0.38, t500: -0.39 },
  { extraLagDays: 180, t380: 1.01, t460: -0.43, t500: -0.16 },
  { extraLagDays: 365, t380: 0.66, t460: -0.44, t500: -0.45 },
];

/** LoRA, one fine-tune nested inside each purged fold. */
export const LORA_FOLDS: {
  fold: number;
  period: string;
  trainRows: number;
  trainMse: number;
  rebIc: number;
  rebT: number;
}[] = [
  {
    fold: 0,
    period: "2018-11 → 2020-05",
    trainRows: 2922,
    trainMse: 0.41,
    rebIc: 0.0879,
    rebT: 3.02,
  },
  {
    fold: 1,
    period: "2020-05 → 2021-12",
    trainRows: 6312,
    trainMse: 0.73,
    rebIc: 0.0471,
    rebT: 0.88,
  },
  {
    fold: 2,
    period: "2021-12 → 2023-06",
    trainRows: 9753,
    trainMse: 1.02,
    rebIc: 0.0138,
    rebT: 0.42,
  },
  {
    fold: 3,
    period: "2023-06 → 2024-12",
    trainRows: 13347,
    trainMse: 1.0,
    rebIc: 0.0185,
    rebT: 0.78,
  },
  {
    fold: 4,
    period: "2024-12 → 2026-07",
    trainRows: 16970,
    trainMse: 1.01,
    rebIc: -0.0307,
    rebT: -1.24,
  },
];

/** Target cross-sectional dispersion, fold 0 → fold 4. */
export const FOLD_DISPERSION = [0.108, 0.104, 0.1, 0.088, 0.077];

/** A linear read-out of the frozen Chronos-2 encoder state. */
export const PROBE: {
  context: number;
  label: string;
  rebIc: number;
  rebT: number;
  dailyIc: number;
  vsFloor: number;
}[] = [
  {
    context: 512,
    label: "probe_mse",
    rebIc: 0.0211,
    rebT: 0.87,
    dailyIc: 0.0249,
    vsFloor: -0.0,
  },
  {
    context: 512,
    label: "probe_ic",
    rebIc: -0.0033,
    rebT: -0.16,
    dailyIc: 0.0005,
    vsFloor: 3.6,
  },
  {
    context: 2048,
    label: "probe_mse",
    rebIc: -0.0031,
    rebT: -0.13,
    dailyIc: -0.004,
    vsFloor: 0.1,
  },
  {
    context: 2048,
    label: "probe_ic",
    rebIc: -0.013,
    rebT: -0.63,
    dailyIc: -0.0027,
    vsFloor: 4.2,
  },
];

/**
 * The durable output of the phase. Every one of these was learned by a
 * measurement going wrong, not by reasoning about it beforehand.
 */
export const LESSONS: { title: string; body: string }[] = [
  {
    title: "One hyperparameter cell is not a result",
    body: "The valuation finding scored t +3.32 at min_train 380 and +1.00 at the harness default of 500, on identical rows — a 3.79 spread with the headline a lone spike between neighbours of +1.30 and +1.18. Sweep the setting before believing the number. The placebo null moves with the setting too, so raw t-statistics are not comparable across cells; only each cell against its own null.",
  },
  {
    title: "A sweep that changes the row count measures two things",
    body: "Adding filing lag decayed the valuation edge monotonically and read as clean evidence of a staleness effect. The sample had quietly shrunk from 77,585 rows to 54,304, because a longer lag costs the earliest rows their coverage. Re-scored on a fixed sample, the edge was absent at every lag including zero.",
  },
  {
    title: "Break a pooled t-statistic down by fold before quoting it",
    body: "LoRA's pooled +2.37 hid a per-fold sequence of +0.0879, +0.0471, +0.0138, +0.0185, −0.0307 against 2,922 → 16,970 training rows. Real skill does not degrade monotonically in sample size.",
  },
  {
    title: "Walk-forward confounds training size with test period",
    body: "Fold 0 has the least data and the earliest window, so a per-fold trend has two explanations. Separate them: retraining fold 4 on fold 0's exact data volume, taken adjacent to its own test window, scored +0.38 against fold 0's +3.02. It is the period, not the sample size.",
  },
  {
    title: "Check the model fitted its training data before believing its test score",
    body: "The first LoRA run scored −0.61 and looked like a clean null. Train MSE had never left 1.00, which on a standardised target means it never learned anything at all — the null described the learning rate, not the data. Reporting it would have been a hyperparameter choice masquerading as a model verdict.",
  },
  {
    title: "A constant prediction must earn no ranking result",
    body: "Sorting by score to form quantiles has to decide what a tie means. A stable sort answers 'whatever order the rows arrived in', which here was alphabetical — and three constant comparators each reported an alpha of +0.00914 at t +1.19, all of it the return of the alphabetically-first fifth of the universe.",
  },
  {
    title: "Only non-overlapping windows support inference",
    body: "Consecutive dates share 29 of their 30 forward sessions, so ~2,400 dates hold ~64 independent windows and a naive t-statistic is inflated roughly fivefold. The daily IC and the rebalance t describe different samples and routinely carry opposite signs — chronos2_xl @512 reports −0.0125 and −0.09.",
  },
];

/** Everything scored, ordered for the axis: worst t first, best last. */
export function axisRows() {
  const scored = COMPARATORS.filter(
    (c): c is Comparator & { rebT: number } => c.rebT !== undefined,
  );
  return [...scored].sort((a, b) => a.rebT - b.rebT);
}

/* ── Phase 4: what the ordering would have cost ────────────────────────────
 *
 * Transcribed from `tools/run_portfolio.py`, run 2026-09-04 and recorded in
 * experiment_runs. NOT a track record and NOT a recommendation: P0 removed the
 * portfolio from this product and it stays removed. This is the same null the
 * rest of the page reports, restated in money so it can be read by someone who
 * does not think in rank ICs.
 *
 * Every book is historical, dated, and scored against what actually happened
 * next. Nothing here is a current or forward-looking holding.
 */

export const COSTS = {
  /** Zerodha equity-delivery, NSE, verified 4 Sep 2026. */
  sttEachSide: 0.001,
  stampDutyBuy: 0.00015,
  roundTrip: 0.002225,
  /** 252 / 30 — the horizon is 30 SESSIONS, not 30 days. */
  rebalancesPerYear: 8.4,
  /** At the ~0.6 turnover these orderings actually generate. */
  annualDragTypical: 0.011,
} as const;

export type Book = {
  id: string;
  label: string;
  /** Annualised, net of the measured Indian cost stack. */
  netAnnual: number;
  /** Against the equal-weighted panel — buy everything. Null for market-neutral. */
  vsFloor: number | null;
  sharpe: number;
  maxDrawdown: number;
  turnover: number;
  /**
   * A long-short book holds no net market exposure, which changes what several
   * columns MEAN rather than just their value. Stated rather than inferred from
   * `vsFloor === null`, so the suppressions below key on a fact about the book.
   */
  marketNeutral: boolean;
  /** The control row: sorts by beta, holds no company-specific view. */
  reference?: boolean;
};

/**
 * The out-of-sample window the 64 rebalances actually cover. NAMED, because a
 * +18.8%/yr floor cannot be judged without it and the whole beta argument
 * depends on this period being a strong bull market.
 */
export const PORTFOLIO_WINDOW = {
  from: "Nov 2018",
  to: "Jun 2026",
  years: 7.6,
  rebalances: 64,
} as const;

/** The equal-weighted panel over that window. The floor every long-only row is read against. */
export const PORTFOLIO_FLOOR = 0.188;

/**
 * ORDER IS DELIBERATE AND MUST NOT BE RE-SORTED BY RETURN.
 *
 * `beta_market` is pinned first as the reference row: it sorts by beta and holds
 * no company-specific view, so every row below it is read as a deviation from a
 * strategy with no opinion about any company. Sorting this table best-first
 * rebuilds the leaderboard P0 removed — a ranked table invites a reader to
 * compare rows the measurement cannot separate, and that is exactly what these
 * rows cannot do.
 *
 * Market-neutral books come last, and are not comparable to the long-only rows
 * above on return, drawdown or Calmar.
 */
export const BOOKS: Book[] = [
  { id: "beta_market_lo", label: "beta_market · long-only", netAnnual: 0.2679, vsFloor: 0.0799, sharpe: 0.83, maxDrawdown: 0.592, turnover: 0.02, marketNeutral: false, reference: true },
  { id: "news_factor_lo", label: "news_factor · long-only", netAnnual: 0.2775, vsFloor: 0.0895, sharpe: 1.35, maxDrawdown: 0.211, turnover: 0.61, marketNeutral: false },
  { id: "pooled_xgb_lo", label: "pooled_xgb · long-only", netAnnual: 0.2643, vsFloor: 0.0762, sharpe: 1.21, maxDrawdown: 0.245, turnover: 0.69, marketNeutral: false },
  { id: "regime_factor_lo", label: "regime_factor · long-only", netAnnual: 0.2563, vsFloor: 0.0682, sharpe: 1.30, maxDrawdown: 0.239, turnover: 0.61, marketNeutral: false },
  { id: "linear_factor_lo", label: "linear_factor · long-only", netAnnual: 0.2479, vsFloor: 0.0599, sharpe: 1.19, maxDrawdown: 0.248, turnover: 0.63, marketNeutral: false },
  { id: "momentum_lo", label: "momentum_20d · long-only", netAnnual: 0.2017, vsFloor: 0.0137, sharpe: 0.77, maxDrawdown: 0.571, turnover: 0.79, marketNeutral: false },
  { id: "reversal_lo", label: "reversal_5d · long-only", netAnnual: 0.1900, vsFloor: 0.0020, sharpe: 0.73, maxDrawdown: 0.499, turnover: 0.76, marketNeutral: false },
  { id: "beta_market_ls", label: "beta_market · long-short", netAnnual: 0.1185, vsFloor: null, sharpe: 0.49, maxDrawdown: 0.599, turnover: 0.02, marketNeutral: true },
  { id: "pooled_xgb_ls", label: "pooled_xgb · long-short", netAnnual: 0.0845, vsFloor: null, sharpe: 0.57, maxDrawdown: 0.220, turnover: 0.66, marketNeutral: true },
  { id: "momentum_ls", label: "momentum_20d · long-short", netAnnual: -0.0443, vsFloor: null, sharpe: -0.33, maxDrawdown: 0.762, turnover: 0.79, marketNeutral: true },
  { id: "reversal_ls", label: "reversal_5d · long-short", netAnnual: -0.0759, vsFloor: null, sharpe: -0.59, maxDrawdown: 1.101, turnover: 0.77, marketNeutral: true },
];

/**
 * The control and the best row, so the prose beside the table stops carrying
 * literals. Every other figure on this page comes from a constant; a hardcoded
 * one in prose lies silently the moment BOOKS is updated.
 */
export const BETA_REFERENCE = BOOKS.find((b) => b.reference)!;
export const BEST_LONG_ONLY = BOOKS.filter((b) => !b.marketNeutral).reduce(
  (a, b) => (b.vsFloor! > a.vsFloor! ? b : a),
);

/** The rank ICs the panel's comparators actually reached, for the break-even note. */
export const PANEL_IC_RANGE = {
  best: 0.0464,
  bestLabel: "beta_market",
  worst: 0.0083,
  worstLabel: "news_factor",
} as const;

/** What rank IC would be needed just to cover costs, by assumed impact cost. */
export const BREAK_EVEN: { impactBps: number; roundTrip: number; annualDrag: number; ic: number }[] = [
  { impactBps: 0, roundTrip: 0.002225, annualDrag: 0.0151, ic: 0.0051 },
  { impactBps: 10, roundTrip: 0.004225, annualDrag: 0.0288, ic: 0.0097 },
  { impactBps: 25, roundTrip: 0.007225, annualDrag: 0.0497, ic: 0.0166 },
  { impactBps: 50, roundTrip: 0.012225, annualDrag: 0.0856, ic: 0.0282 },
];

/** The deflated Sharpe, under both trial counts. The larger N is the honest one. */
export const DEFLATED = {
  /**
   * Two-sided 95% normal critical value, and the same number
   * `evaluation.deflated_sharpe_note` tests against. Kept here so the page
   * cannot quote a threshold the code no longer uses.
   */
  threshold: 1.96,
  best: "news_factor · long-only",
  sharpeAnnual: 1.348,
  sharpePerRebalance: 0.465,
  nRebalances: 64,
  /** Measured spread of per-rebalance Sharpes across the 14 books run. */
  sharpeSpread: 0.199,
  trials: [
    { label: "P4's own variants", n: 24, expectedMax: 0.393, deflated: 0.57, clears: false },
    { label: "every trial on this panel", n: 40, expectedMax: 0.435, deflated: 0.24, clears: false, honest: true },
  ],
} as const;

/**
 * Phase 5 — what happens to the ordering once the beta channel is removed from
 * the target, within each date. The question P4 forced, and the answer is not
 * the one it was asked in expectation of.
 */
export const NEUTRALISED = {
  /** Share of within-date target variance the beta channel actually explains. */
  betaChannelR2: 0.065,
  floor: { label: "beta_market", rawIc: 0.0464, residualIc: -0.0045 },
  best: { label: "pooled_xgb", rawIc: 0.0389, residualIc: 0.041, residualT: 2.71 },
  /** The same residual t across the min_train grid. The headline is the spike. */
  sweep: [
    { minTrain: 380, t: 2.08, headline: false },
    { minTrain: 420, t: 0.93, headline: false },
    { minTrain: 460, t: 1.26, headline: false },
    { minTrain: 500, t: 2.71, headline: true },
    { minTrain: 540, t: 1.67, headline: false },
    { minTrain: 580, t: 1.47, headline: false },
  ],
  /** A pre-registered regime split, at one cell of the same grid. */
  regimeArtifactT: 5.21,
  /** Best beta-neutral book, deflated at the cumulative trial count. */
  hedgedBestSharpe: 0.79,
  deflatedAtCumulative: -1.03,
  cumulativeTrials: 103,
} as const;

/** Planted edges of known size, to prove the simulator can see one. */
export const SYNTHETIC: { plantedIc: number; measuredIc: number; netSharpe: number }[] = [
  { plantedIc: 0.0, measuredIc: -0.0227, netSharpe: -0.18 },
  { plantedIc: 0.02, measuredIc: -0.0040, netSharpe: 0.04 },
  { plantedIc: 0.05, measuredIc: 0.0249, netSharpe: 0.59 },
  { plantedIc: 0.10, measuredIc: 0.0748, netSharpe: 1.37 },
  { plantedIc: 0.20, measuredIc: 0.1698, netSharpe: 2.97 },
];
