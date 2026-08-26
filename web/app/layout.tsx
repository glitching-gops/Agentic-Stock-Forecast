import type { Metadata } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Martian_Mono } from "next/font/google";
import Link from "next/link";

import { SiteNav } from "@/components/site-nav";
import { getLeaderboard, getStocks, soft } from "@/lib/api";

import "./globals.css";

/*
 * Everything is monospaced, and the hierarchy comes from WIDTH rather than
 * from a serif/sans contrast.
 *
 * The justification is the content: every heading in this interface is a label
 * on a measurement, not a sentence, so a proportional display face would imply
 * prose where the material is instrumentation. Martian Mono is wide and
 * engineered; IBM Plex Mono beside it reads as the data it is. The one place
 * that argument fails is long-form reading, so Plex Sans is loaded for the
 * methodology and research write-ups and used nowhere else.
 */
const display = Martian_Mono({
  variable: "--font-martian-mono",
  subsets: ["latin"],
  weight: ["500", "700"],
});

const mono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

const prose = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  subsets: ["latin"],
  weight: ["400", "600"],
});

// Route segment config must be a literal — Next cannot statically analyse an
// imported constant here. Keep in step with REVALIDATE_SECONDS in lib/api.ts.
export const revalidate = 3600;

export const metadata: Metadata = {
  title: {
    default: "ZeRO — Agentic Stock Forecast",
    template: "%s · ZeRO",
  },
  description:
    "NIFTY 100 stocks ranked by predicted 30-session excess return against a sector benchmark, gated behind held-out evidence from purged walk-forward evaluation.",
};

export default async function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  /*
   * Both soft, and for the same reason: the chrome is a convenience. Losing
   * the switcher or the status line should not take the site down, and every
   * page fetches the data it actually needs for itself.
   *
   * The leaderboard call is free in practice — the home page requests the same
   * URL with the same options, so Next's fetch cache serves one upstream hit
   * for both. Note that `soft` is deliberately NOT used on the page's own
   * copy: a failure there must break the build rather than publish an empty
   * dashboard.
   */
  const [{ stocks }, board] = await Promise.all([
    soft(getStocks(), { stocks: [], total: 0 }),
    soft(getLeaderboard(), null),
  ]);

  return (
    <html
      lang="en"
      className={`${display.variable} ${mono.variable} ${prose.variable} h-full`}
    >
      <body className="scan flex min-h-full flex-col">
        <SiteNav
          stocks={stocks}
          lastRun={board?.last_updated ?? null}
          universe={board?.total ?? null}
        />
        <main className="relative z-10 mx-auto w-full max-w-[1560px] flex-1 px-4 py-7 sm:px-6">
          {children}
        </main>
        <SiteFooter />
      </body>
    </html>
  );
}

/**
 * The bottom status line. A terminal has one, and it is the right home for a
 * disclaimer that must be on every route without competing with the data
 * above it.
 */
function SiteFooter() {
  return (
    <footer className="relative z-10 mt-10 border-t border-rule-hi bg-inset">
      <div className="mx-auto max-w-[1560px] px-4 py-6 sm:px-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:gap-10">
          <p className="max-w-[74ch] font-prose text-[0.78rem] leading-relaxed text-dim">
            <span className="font-mono font-semibold uppercase tracking-[0.14em] text-mid">
              Not financial advice.
            </span>{" "}
            ZeRO is a research project. Every performance figure here is
            out-of-sample and stated before transaction costs. As measured, the
            model does not beat a majority-class baseline on direction or a
            zero-excess forecast on magnitude — the calibrated intervals hold
            up, the point forecast does not.{" "}
            <Link
              href="/methodology"
              className="text-text underline underline-offset-2"
            >
              Limitations
            </Link>
            .
          </p>
          <div className="shrink-0 space-y-1 text-[0.7rem] uppercase tracking-[0.12em] text-dim lg:ml-auto lg:text-right">
            <div>Venu Gopal Battula</div>
            <div>
              <a
                href="https://github.com/glitching-gops/Agentic-Stock-Forecast"
                target="_blank"
                rel="noreferrer"
                className="text-text underline underline-offset-2"
              >
                Source on GitHub
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
}
