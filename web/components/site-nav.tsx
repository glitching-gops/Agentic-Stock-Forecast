"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

import { cx } from "@/lib/format";
import type { StockInfo } from "@/lib/types";

const LINKS = [
  { href: "/", label: "Leaderboard" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/methodology", label: "Methodology" },
  { href: "/about", label: "About" },
] as const;

export function SiteNav({ stocks }: { stocks: StockInfo[] }) {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-40 border-b border-ink-500/70 bg-ink-900/85 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-[1400px] items-center gap-3 px-4 sm:px-6">
        <Link
          href="/"
          className="flex shrink-0 items-baseline gap-2"
          aria-label="ZeRO home"
        >
          <span className="text-base font-bold tracking-tight text-brand-400">
            ZeRO
          </span>
          <span className="hidden text-[0.7rem] font-medium uppercase tracking-[0.14em] text-mist-500 sm:inline">
            Stock Forecast
          </span>
        </Link>

        <nav className="ml-2 flex min-w-0 items-center gap-0.5 overflow-x-auto">
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
                  "whitespace-nowrap rounded-md px-2.5 py-1.5 text-sm font-medium transition-colors",
                  active
                    ? "bg-ink-600 text-mist-100"
                    : "text-mist-400 hover:bg-ink-700 hover:text-mist-200",
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
    </header>
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
        placeholder="Find a stock…"
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
        className="w-36 rounded-md border border-ink-500 bg-ink-700 px-2.5 py-1.5 text-sm text-mist-100 placeholder:text-mist-500 focus:w-56 focus:border-brand-500/60 focus:outline-none sm:w-48 sm:focus:w-72"
      />

      {open ? (
        <ul
          id="stock-switcher-list"
          role="listbox"
          className="absolute right-0 z-50 mt-1.5 max-h-[min(28rem,70vh)] w-[min(22rem,calc(100vw-2rem))] overflow-y-auto rounded-lg border border-ink-500 bg-ink-800 py-1 shadow-2xl shadow-black/60"
        >
          {matches.length === 0 ? (
            <li className="px-3 py-3 text-sm text-mist-500">
              Nothing matches “{query}”.
            </li>
          ) : (
            matches.map((stock, index) => (
              <li key={stock.ticker} role="option" aria-selected={index === highlight}>
                <Link
                  href={`/stocks/${stock.ticker}`}
                  onClick={() => {
                    setOpen(false);
                    setQuery("");
                  }}
                  onMouseEnter={() => setHighlight(index)}
                  className={cx(
                    "flex items-baseline justify-between gap-3 px-3 py-1.5",
                    index === highlight ? "bg-ink-600" : "",
                  )}
                >
                  <span className="truncate text-sm text-mist-100">
                    {stock.company}
                  </span>
                  <span className="shrink-0 font-mono text-[0.7rem] text-mist-500">
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
