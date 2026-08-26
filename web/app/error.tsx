"use client";

/**
 * Rendered when a page's data fetch fails outright — in practice, when Render
 * is cold or down and no cached page exists to fall back to. Says which of the
 * two it is rather than showing a bare error, because the free-tier spin-up is
 * the overwhelmingly likely cause and it resolves itself in about a minute.
 */
export default function Error({ reset }: { error: Error; reset: () => void }) {
  return (
    <div className="mx-auto max-w-xl py-24">
      <div className="text-[0.64rem] uppercase tracking-[0.16em] text-bar">
        Upstream timeout
      </div>
      <h1 className="mt-2 font-display text-[1.2rem] font-bold tracking-tight text-bright">
        The forecast API did not answer
      </h1>
      <p className="mt-3 max-w-[62ch] font-prose text-[0.86rem] leading-relaxed text-mid">
        The backend runs on a free-tier instance that sleeps after 15 minutes of
        inactivity and takes about a minute to wake. Give it a moment and try
        again.
      </p>
      <button
        type="button"
        onClick={reset}
        className="mt-6 border border-rule-hi px-2.5 py-1 text-[0.72rem] uppercase tracking-[0.12em] text-text transition-colors hover:bg-bright hover:text-void"
      >
        Retry
      </button>
    </div>
  );
}
