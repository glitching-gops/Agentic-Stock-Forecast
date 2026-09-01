"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

import { cx, daysAgo, timestamp } from "@/lib/format";
import type { StockInfo } from "@/lib/types";

const LINKS = [
  { href: "/", label: "Forecasts" },
  { href: "/research", label: "Research" },
  { href: "/methodology", label: "Method" },
  { href: "/about", label: "About" },
] as const;

export function SiteNav({
  stocks,
  lastRun,
  universe,
}: {
  stocks: StockInfo[];
  lastRun: string | null;
  universe: number | null;
}) {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-40 border-b border-rule-hi bg-void/95 backdrop-blur-md">
      <div className="mx-auto flex h-12 max-w-[1560px] items-center gap-4 px-4 sm:px-6">
        <Link
          href="/"
          className="flex shrink-0 items-baseline gap-2"
          aria-label="ZeRO home"
        >
          <span className="font-display text-[0.95rem] font-bold tracking-tight text-bright">
            ZeRO
          </span>
          <span className="hidden text-[0.62rem] uppercase tracking-[0.2em] text-dim md:inline">
            Forecast Terminal
          </span>
        </Link>

        <nav className="flex min-w-0 items-center gap-px overflow-x-auto">
          {LINKS.map((link) => {
            const active =
              link.href === "/"
                ? pathname === "/" || pathname.startsWith("/stocks")
                : pathname.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                aria-current={active ? "page" : undefined}
                className={cx(
                  "whitespace-nowrap px-2.5 py-1 text-[0.72rem] uppercase tracking-[0.12em] transition-colors",
                  active
                    ? "inv font-semibold"
                    : "text-dim hover:bg-raise hover:text-text",
                )}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        <div className="ml-auto">
          <StockSwitcher stocks={stocks} />
        </div>
      </div>

      <StatusLine lastRun={lastRun} universe={universe} />
    </header>
  );
}

/**
 * The status line.
 *
 * Everything on it is a fact about what the reader is looking at rather than
 * decoration: when the pipeline last wrote, how many names it covers, and the
 * two parameters that define the target. The caret is the only motion on the
 * site and it carries meaning — it blinks while the data is current and goes
 * dark once the last run is older than the daily cadence, so a stale board
 * announces itself without a banner.
 */
function StatusLine({
  lastRun,
  universe,
}: {
  lastRun: string | null;
  universe: number | null;
}) {
  const age = daysAgo(lastRun);
  const live = age !== null && age <= 2;

  return (
    <div className="border-t border-rule bg-inset">
      <div className="mx-auto flex max-w-[1560px] flex-wrap items-center gap-x-4 gap-y-0.5 px-4 py-1 text-[0.64rem] uppercase tracking-[0.12em] text-dim sm:px-6">
        <span className="flex items-center gap-1.5">
          <span
            aria-hidden
            className={cx(
              "inline-block h-2 w-1.5",
              live ? "caret bg-pos" : "bg-rule-hi",
            )}
          />
          {lastRun ? (
            <>
              Last run{" "}
              <span className="text-mid">{timestamp(lastRun)}</span>
              {!live && age !== null ? (
                <span className="text-bar">· stale {age}d</span>
              ) : null}
            </>
          ) : (
            <span className="text-bar">Pipeline status unavailable</span>
          )}
        </span>
        <Sep />
        <span>
          Universe <span className="text-mid">{universe ?? "—"}</span>
        </span>
        <Sep />
        <span>
          Horizon <span className="text-mid">30 sessions</span>
        </span>
        <Sep />
        <span>
          Target <span className="text-mid">excess vs sector benchmark</span>
        </span>
      </div>
    </div>
  );
}

function Sep() {
  return (
    <span aria-hidden className="text-rule-hi">
      /
    </span>
  );
}

/**
 * Type-ahead over the whole universe.
 *
 * The Streamlit original was a 95-item dropdown that only appeared on the
 * stock page — and reaching that page cost a blocking call to `/api/stocks`
 * before anything rendered. Here the list is embedded in the layout at build
 * time, so switching stocks is a client-side filter over data already present
 * and every route can offer it.
 */
function StockSwitcher({ stocks }: { stocks: StockInfo[] }) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [highlight, setHighlight] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase();
    const pool = q
      ? stocks.filter(
          (s) =>
            s.company.toLowerCase().includes(q) ||
            s.ticker.toLowerCase().includes(q) ||
            s.sector.toLowerCase().includes(q),
        )
      : stocks;
    return pool.slice(0, 40);
  }, [stocks, query]);

  useEffect(() => setHighlight(0), [query]);

  useEffect(() => {
    if (!open) return;
    function onPointerDown(event: MouseEvent) {
      if (!containerRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, [open]);

  // Cmd/Ctrl-K from anywhere.
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setOpen(true);
        requestAnimationFrame(() => inputRef.current?.focus());
      }
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  if (stocks.length === 0) return null;

  return (
    <div ref={containerRef} className="relative">
      <input
        ref={inputRef}
        type="text"
        role="combobox"
        aria-expanded={open}
        aria-controls="stock-switcher-list"
        aria-label="Find a stock"
        placeholder="FIND ⌘K"
        value={query}
        onChange={(event) => {
          setQuery(event.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown") {
            event.preventDefault();
            setOpen(true);
            setHighlight((h) => Math.min(h + 1, matches.length - 1));
          } else if (event.key === "ArrowUp") {
            event.preventDefault();
            setHighlight((h) => Math.max(h - 1, 0));
          } else if (event.key === "Enter" && matches[highlight]) {
            event.preventDefault();
            window.location.href = `/stocks/${matches[highlight].ticker}`;
          } else if (event.key === "Escape") {
            setOpen(false);
            inputRef.current?.blur();
          }
        }}
        className="w-32 border border-rule bg-inset px-2 py-1 text-[0.72rem] uppercase tracking-[0.1em] text-bright placeholder:text-dim focus:w-52 focus:border-rule-hi focus:outline-none sm:w-40 sm:focus:w-64"
      />

      {open ? (
        <ul
          id="stock-switcher-list"
          role="listbox"
          className="absolute right-0 z-50 mt-1 max-h-[min(28rem,70vh)] w-[min(24rem,calc(100vw-2rem))] overflow-y-auto border border-rule-hi bg-shell py-1 shadow-2xl shadow-black/70"
        >
          {matches.length === 0 ? (
            <li className="px-3 py-3 text-[0.75rem] text-dim">
              No name matches “{query}”.
            </li>
          ) : (
            matches.map((stock, index) => (
              <li
                key={stock.ticker}
                role="option"
                aria-selected={index === highlight}
              >
                <Link
                  href={`/stocks/${stock.ticker}`}
                  onClick={() => {
                    setOpen(false);
                    setQuery("");
                  }}
                  onMouseEnter={() => setHighlight(index)}
                  className={cx(
                    "flex items-baseline justify-between gap-3 px-3 py-1",
                    index === highlight ? "inv" : "text-text",
                  )}
                >
                  <span className="truncate text-[0.78rem]">
                    {stock.company}
                  </span>
                  <span
                    className={cx(
                      "shrink-0 text-[0.68rem]",
                      index === highlight ? "opacity-70" : "text-dim",
                    )}
                  >
                    {stock.ticker.replace(/\.NS$/, "")}
                  </span>
                </Link>
              </li>
            ))
          )}
        </ul>
      ) : null}
    </div>
  );
}
