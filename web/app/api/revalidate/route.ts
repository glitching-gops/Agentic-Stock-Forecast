import { revalidatePath } from "next/cache";
import { NextResponse } from "next/server";

import { getStocks, soft } from "@/lib/api";

/**
 * On-demand revalidation, called by the pipeline once its data has landed.
 *
 * Hourly ISR alone is a poor fit here, for two reasons that compound:
 *
 *  1. The data changes on a schedule, not continuously — the daily pipeline
 *     writes once a weekday evening, the evaluation once a week. 23 of every
 *     24 hourly revalidations therefore refetch data that has not moved.
 *  2. Render's free tier sleeps after 15 minutes idle. An hourly revalidation
 *     is guaranteed to arrive at a cold instance, and an ~82s spin-up does not
 *     fit inside a 60s function — so the revalidation that would pick up new
 *     data is exactly the one most likely to time out.
 *
 * Calling this from the workflow that just finished writing the data inverts
 * that: the runner has already woken Render, has no 60s ceiling, and knows
 * when there is something new to publish. `revalidate = 3600` stays on every
 * page as a safety net for when this is not called.
 *
 * Usage from .github/workflows, after the pipeline step:
 *
 *     curl -fsS -X POST "$SITE_URL/api/revalidate" \
 *          -H "x-revalidate-secret: ${{ secrets.REVALIDATE_SECRET }}"
 */
export async function POST(request: Request) {
  const expected = process.env.REVALIDATE_SECRET;

  if (!expected) {
    return NextResponse.json(
      { error: "REVALIDATE_SECRET is not configured" },
      { status: 503 },
    );
  }

  if (!timingSafeEqual(request.headers.get("x-revalidate-secret"), expected)) {
    return NextResponse.json({ error: "unauthorised" }, { status: 401 });
  }

  revalidatePath("/", "layout");
  revalidatePath("/");
  revalidatePath("/methodology");

  // Stock pages are per-ticker, so each path has to be named. Soft-failing
  // here still leaves them on the hourly schedule.
  const { stocks } = await soft(getStocks(), { stocks: [], total: 0 });
  for (const stock of stocks) {
    revalidatePath(`/stocks/${stock.ticker}`);
  }

  return NextResponse.json({
    revalidated: true,
    stockPages: stocks.length,
    at: new Date().toISOString(),
  });
}

/**
 * Constant-time comparison, mirroring the `secrets.compare_digest` the API
 * uses for its admin key (audit finding F15). A `===` on a secret leaks its
 * length and prefix through response timing.
 */
function timingSafeEqual(candidate: string | null, expected: string): boolean {
  if (candidate === null) return false;

  const a = new TextEncoder().encode(candidate);
  const b = new TextEncoder().encode(expected);
  const length = Math.max(a.length, b.length);

  let diff = a.length ^ b.length;
  for (let i = 0; i < length; i += 1) {
    diff |= (a[i] ?? 0) ^ (b[i] ?? 0);
  }
  return diff === 0;
}
