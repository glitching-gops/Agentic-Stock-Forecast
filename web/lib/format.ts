import type { ScoreBasis } from "./types";

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
 * `score_basis` is the single most load-bearing field on the leaderboard: 93 of
 * 95 rows score 0.0, and this is the only thing that distinguishes "we have no
 * forecast" from "the forecast is fine but points down".
 */
export const SCORE_BASIS_LABEL: Record<ScoreBasis, string> = {
  RANKED: "Ranked",
  NO_FORECAST: "No forecast produced",
  NO_EVIDENCE: "No held-out evidence",
  NOT_LONG: "Not a long candidate",
  FLAGGED_OUT: "Zeroed by critic flags",
};

export const SCORE_BASIS_EXPLAINER: Record<ScoreBasis, string> = {
  RANKED:
    "Cleared the evidence gate, and the point forecast predicts outperformance.",
  NO_FORECAST:
    "The model produced no prediction at all for this stock, so there is nothing to score.",
  NO_EVIDENCE:
    "A forecast exists, but the model failed its held-out checks on purged walk-forward folds. Graded INSUFFICIENT, it is multiplied to zero however large the predicted move.",
  NOT_LONG:
    "Held-out evidence is in hand, but the prediction is flat to negative. This board ranks long candidates only, so a downward forecast floors at zero rather than ranking below one — including when the calibrated probability disagrees with it.",
  FLAGGED_OUT:
    "A genuine long signal with evidence behind it, driven to zero by the critic's flags at five points each.",
};

export function scoreBasisLabel(basis: string | null | undefined): string {
  if (!basis) return "Not ranked";
  return SCORE_BASIS_LABEL[basis as ScoreBasis] ?? basis;
}

export function scoreBasisExplainer(basis: string | null | undefined): string {
  if (!basis) return "No score basis was recorded for this row.";
  return (
    SCORE_BASIS_EXPLAINER[basis as ScoreBasis] ??
    "No description is recorded for this score basis."
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
