import "server-only";

import type {
  Forecast,
  ForecastListResponse,
  Headline,
  SignalsResponse,
  StockList,
} from "./types";

/**
 * Server-only data layer. Every call in this file runs on Vercel's side of the
 * wire — during `next build`, during ISR revalidation, or inside a route
 * handler. Nothing here is ever bundled for the browser.
 *
 * That is a deliberate constraint, not an accident of App Router defaults:
 *
 *  - The visitor is never exposed to Render's cold start. A spun-down free-tier
 *    instance takes ~82s to answer; the CDN answers in milliseconds from the
 *    last good render.
 *  - FastAPI's CORS allowlist needs no Vercel entry, because no browser origin
 *    ever calls it. Server-to-server requests carry no Origin header.
 *  - The forecast list is 84 rows / ~60 KB. Filtering it in the browser over
 *    one already-fetched payload is cheaper *and* faster than a round trip per
 *    filter change, so the interactive surface costs zero additional upstream
 *    requests.
 */

const BASE = (process.env.API_BASE_URL ?? "http://localhost:8000").replace(
  /\/+$/,
  "",
);

/** Data changes once a day (pipeline) and once a week (evaluation). */
export const REVALIDATE_SECONDS = 3600;

/*
 * Retry budget, tuned against measured Render behaviour rather than a guess:
 * warm reads land in 0.2–2.0s, and a cold start took 82.5s wall-clock on
 * 2026-08-18. The first attempt is what actually triggers the spin-up, so it is
 * expected to fail when the instance is asleep; the later, longer attempts are
 * the ones that collect the answer.
 *
 * During `next build` there is no wall-clock cap, so the full ~110s budget is
 * available and a cold start is survivable. Inside a Vercel function the
 * platform cap (see `maxDuration` on each page) cuts this short — and that is
 * the correct outcome: a failed revalidation makes Next keep serving the last
 * good page rather than replacing it with an error.
 */
const ATTEMPT_TIMEOUTS_MS = [25_000, 40_000, 40_000];

export class ApiError extends Error {
  constructor(
    readonly path: string,
    readonly status: number | null,
    readonly cause_: unknown,
  ) {
    super(
      status === null
        ? `GET ${path} failed to reach the API: ${String(cause_)}`
        : `GET ${path} returned ${status}`,
    );
    this.name = "ApiError";
  }
}

/** A 404 is a fact about the data ("no forecast for this ticker"), not a fault. */
export class NotFoundError extends ApiError {
  constructor(path: string) {
    super(path, 404, null);
    this.name = "NotFoundError";
  }
}

async function request<T>(
  path: string,
  { revalidate = REVALIDATE_SECONDS }: { revalidate?: number } = {},
): Promise<T> {
  const url = `${BASE}${path}`;
  let lastError: unknown;

  for (const timeout of ATTEMPT_TIMEOUTS_MS) {
    try {
      const response = await fetch(url, {
        signal: AbortSignal.timeout(timeout),
        headers: { accept: "application/json" },
        next: { revalidate },
      });

      // Retrying a 404 just burns the budget — the row genuinely is not there.
      if (response.status === 404) throw new NotFoundError(path);

      if (!response.ok) {
        lastError = new ApiError(path, response.status, null);
        // 4xx other than 404 will not fix itself either.
        if (response.status < 500) throw lastError;
        continue;
      }

      return (await response.json()) as T;
    } catch (error) {
      if (error instanceof NotFoundError) throw error;
      if (error instanceof ApiError && error.status && error.status < 500) {
        throw error;
      }
      lastError = error;
    }
  }

  throw lastError instanceof ApiError
    ? lastError
    : new ApiError(path, null, lastError);
}

/* ── Endpoints ─────────────────────────────────────────────────────────── */

export function getStocks(): Promise<StockList> {
  return request<StockList>("/api/stocks");
}

export interface ForecastQuery {
  sector?: string;
  verdict?: string;
  evidence?: string;
  limit?: number;
}

/**
 * Every current forecast, in ticker order.
 *
 * There is no sort parameter. The endpoint this replaced took one and reached
 * ORDER BY with it as an identifier, which cannot be a bind parameter, so it
 * needed an allowlist acting as a SQL-injection boundary. Removing the ordering
 * removed the boundary with it — and the ordering was the thing worth removing
 * anyway: three of ninety-six tickers clear the evidence gate, which is what
 * chance produces, so ranking the rest published a comparison nothing supports.
 */
export function getForecasts(
  query: ForecastQuery = {},
): Promise<ForecastListResponse> {
  const params = new URLSearchParams({
    // 500 is the API's own ceiling. The universe is 84, so one call is the
    // whole table and the client can filter without touching the network.
    limit: String(query.limit ?? 200),
  });
  if (query.sector) params.set("sector", query.sector);
  if (query.verdict) params.set("verdict", query.verdict);
  if (query.evidence) params.set("evidence", query.evidence);

  return request<ForecastListResponse>(`/api/forecasts?${params}`);
}

export function getForecast(ticker: string): Promise<Forecast> {
  return request<Forecast>(`/api/forecasts/${encodeURIComponent(ticker)}`);
}

export function getSignals(
  ticker: string,
  days = 220,
): Promise<SignalsResponse> {
  return request<SignalsResponse>(
    `/api/signals/${encodeURIComponent(ticker)}?days=${days}`,
  );
}

export function getHeadlines(ticker: string): Promise<Headline[]> {
  return request<Headline[]>(
    `/api/sentiment/${encodeURIComponent(ticker)}/headlines`,
  );
}

/* ── Soft variants ─────────────────────────────────────────────────────── */

/**
 * Resolve `promise`, or `fallback` if the API cannot be reached.
 *
 * Use this only where a missing piece degrades the page rather than
 * invalidating it — headlines, the signal history behind a chart. Do NOT wrap
 * the forecast list or a single forecast: those throwing is what makes a build
 * fail loudly instead of publishing an empty dashboard, and what makes Next
 * keep serving the last good page instead of overwriting it with an error
 * state.
 */
export async function soft<T>(
  promise: Promise<T>,
  fallback: T,
): Promise<T> {
  try {
    return await promise;
  } catch (error) {
    if (!(error instanceof NotFoundError)) {
      console.error("[api] soft failure:", error);
    }
    return fallback;
  }
}
