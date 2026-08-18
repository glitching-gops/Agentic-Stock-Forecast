import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-xl py-24 text-center">
      <h1 className="text-2xl font-semibold tracking-tight text-mist-100">
        Not found
      </h1>
      <p className="mt-3 text-sm leading-relaxed text-mist-400">
        No forecast exists for that ticker. It may have left the NIFTY 100, or
        fallen below the liquidity floor or listing-history minimum the universe
        rule applies.
      </p>
      <Link
        href="/"
        className="mt-6 inline-block rounded-md border border-ink-500 px-3 py-1.5 text-sm font-medium text-mist-200 hover:bg-ink-600"
      >
        Back to the leaderboard
      </Link>
    </div>
  );
}
