"use client";

/**
 * Rendered when a page's data fetch fails outright — in practice, when Render
 * is cold or down and no cached page exists to fall back to. Says which of the
 * two it is rather than showing a bare error, because the free-tier spin-up is
 * the overwhelmingly likely cause and it resolves itself in about a minute.
 */
export default function Error({ reset }: { error: Error; reset: () => void }) {
  return (
    <div className="mx-auto max-w-xl py-24 text-center">
      <h1 className="text-2xl font-semibold tracking-tight text-mist-100">
        The forecast API did not answer
      </h1>
      <p className="mt-3 text-sm leading-relaxed text-mist-400">
        The backend runs on a free-tier instance that sleeps after 15 minutes of
        inactivity and takes about a minute to wake. Give it a moment and try
        again.
      </p>
      <button
        type="button"
        onClick={reset}
        className="mt-6 rounded-md border border-ink-500 px-3 py-1.5 text-sm font-medium text-mist-200 hover:bg-ink-600"
      >
        Retry
      </button>
    </div>
  );
}
