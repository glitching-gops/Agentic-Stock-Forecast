import type { EvidenceState } from "./types";

/**
 * Formatters. Every one of them takes `null | undefined` and returns an em
 * dash, because most cells in this dataset are genuinely empty and a page that
 * printed `0.00` for "not measured" would be making a claim the pipeline
 * cannot support.
 */

const DASH = "—";

/**
 * Indian digit grouping (1,23,456.78), not the Western 123,456.78 the Streamlit
 * app used. These are NSE prices in rupees read by an Indian audience.
 */
const inr = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const inrCompact = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});

export function money(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? DASH
    : inr.format(value);
}

export function moneyCompact(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? DASH
    : inrCompact.format(value);
}

/** A fraction (0.0169) rendered as a signed percentage (+1.69%). */
export function signedPct(
  value: number | null | undefined,
  digits = 2,
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return DASH;
  }
  return `${value >= 0 ? "+" : "−"}${Math.abs(value * 100).toFixed(digits)}%`;
}

/** A value already expressed in percent (59.71) rendered as 59.7%. */
export function pctPoints(
  value: number | null | undefined,
  digits = 1,
): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? DASH
    : `${value.toFixed(digits)}%`;
}

/** A probability in [0,1] rendered as 62%. */
export function probability(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? DASH
    : `${(value * 100).toFixed(0)}%`;
}

/** Rank IC and similar small signed correlations: +0.041 / −0.234. */
export function signed(
  value: number | null | undefined,
  digits = 3,
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return DASH;
  }
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
}

export function decimal(
  value: number | null | undefined,
  digits = 2,
): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? DASH
    : value.toFixed(digits);
}

export function compactNumber(value: number | null | undefined): string {
  return value === null || value === undefined || !Number.isFinite(value)
    ? DASH
    : new Intl.NumberFormat("en-IN", {
        notation: "compact",
        maximumFractionDigits: 1,
      }).format(value);
}

/**
 * Timestamps arrive as naive strings from Postgres (`2026-08-17 07:02:07.267`).
 * They are IST wall-clock, with no offset attached, so they are formatted as
 * given rather than run through a timezone conversion that would silently
 * shift them by 5h30m.
 */
export function timestamp(value: string | null | undefined): string {
  if (!value) return DASH;

  const match = value.match(
    /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2}))?/,
  );
  if (!match) return value;

  const [, y, m, d, hh, mm] = match;
  const months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];
  const day = `${Number(d)} ${months[Number(m) - 1]} ${y}`;
  return hh ? `${day}, ${hh}:${mm} IST` : day;
}

export function dateOnly(value: string | null | undefined): string {
  if (!value) return DASH;
  return timestamp(value).split(",")[0];
}

/** Whole days between `value` and now, or null if unparseable. */
export function daysAgo(value: string | null | undefined): number | null {
  if (!value) return null;
  const parsed = Date.parse(value.replace(" ", "T"));
  if (Number.isNaN(parsed)) return null;
  return Math.floor((Date.now() - parsed) / 86_400_000);
}

/* ── Domain vocabulary ─────────────────────────────────────────────────── */

/**
 * What is known about a forecast, and therefore how much weight it carries.
 *
 * This replaces `score_basis`, which existed to disambiguate a composite of
 * 0.0. Three of its five values (RANKED, NOT_LONG, FLAGGED_OUT) described the
 * ranking rather than the stock, and went with it. The remaining distinction
 * — "no forecast at all" against "a forecast with no held-out support" — is
 * real and is preserved here, because a reader who cannot tell those apart is
 * being shown an absence dressed as a measurement.
 *
 * Derived, never sent. A server column duplicating a value computable from two
 * others is a column that can disagree with them.
 */
export function evidenceState(forecast: {
  pred_excess_return: number | null;
  forecast_confidence: string | null;
}): EvidenceState {
  if (forecast.pred_excess_return === null) return "NO_FORECAST";
  const grade = forecast.forecast_confidence;
  if (grade === "STRONG" || grade === "WEAK") return grade;
  return "INSUFFICIENT";
}

export const EVIDENCE_STATE_LABEL: Record<EvidenceState, string> = {
  STRONG: "Strong evidence",
  WEAK: "Weak evidence",
  INSUFFICIENT: "No held-out evidence",
  NO_FORECAST: "No forecast produced",
};

export const EVIDENCE_STATE_EXPLAINER: Record<EvidenceState, string> = {
  STRONG:
    "All three held-out checks passed on purged walk-forward folds: rank IC, the IC t-statistic, and directional accuracy above the majority-class baseline. No ticker currently reaches this.",
  WEAK:
    "Two of the three held-out checks passed. That is the minimum this system grades at all, and it is not an endorsement: the thresholds are low, and only one of the three checks tests significance at all.",
  INSUFFICIENT:
    "A forecast exists, but the model failed at least two of its three held-out checks. Read the number as the model's output and nothing more — there is no evidence it forecasts this stock better than chance.",
  NO_FORECAST:
    "The pipeline produced no prediction for this stock at all, usually too little price history or a benchmark index that has stopped publishing. Nothing here is a view; it is an absence.",
};

export function evidenceStateLabel(state: string | null | undefined): string {
  if (!state) return "Not measured";
  return EVIDENCE_STATE_LABEL[state as EvidenceState] ?? state;
}

export function evidenceStateExplainer(
  state: string | null | undefined,
): string {
  if (!state) return "Nothing was recorded about this row.";
  return (
    EVIDENCE_STATE_EXPLAINER[state as EvidenceState] ??
    "No description is recorded for this state."
  );
}

export const EVIDENCE_LABEL: Record<string, string> = {
  STRONG: "Strong",
  WEAK: "Weak",
  INSUFFICIENT: "None",
};

export function cx(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}
