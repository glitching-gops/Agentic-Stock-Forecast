import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-xl py-24">
      <div className="text-[0.64rem] uppercase tracking-[0.16em] text-dim">
        404
      </div>
      <h1 className="mt-2 font-display text-[1.2rem] font-bold tracking-tight text-bright">
        Not found
      </h1>
      <p className="mt-3 max-w-[62ch] font-prose text-[0.86rem] leading-relaxed text-mid">
        No forecast exists for that ticker. It may have left the NIFTY 100, or
        fallen below the liquidity floor or listing-history minimum the universe
        rule applies.
      </p>
      <Link
        href="/"
        className="mt-6 inline-flex border border-rule-hi px-2.5 py-1 text-[0.72rem] uppercase tracking-[0.12em] text-text transition-colors hover:bg-bright hover:text-void"
      >
        Back to the board
      </Link>
    </div>
  );
}
