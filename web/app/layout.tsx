import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";

import { SiteNav } from "@/components/site-nav";
import { getStocks, soft } from "@/lib/api";

import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

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
  // Soft: the switcher is a convenience. Losing it should not take the site
  // down, and every page fetches the data it actually needs for itself.
  const { stocks } = await soft(getStocks(), { stocks: [], total: 0 });

  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full`}
    >
      <body className="flex min-h-full flex-col font-sans">
        <SiteNav stocks={stocks} />
        <main className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-8 sm:px-6">
          {children}
        </main>
        <SiteFooter />
      </body>
    </html>
  );
}

function SiteFooter() {
  return (
    <footer className="mt-12 border-t border-ink-500/70 bg-ink-800/60">
      <div className="mx-auto max-w-[1400px] space-y-4 px-4 py-8 text-xs leading-relaxed text-mist-500 sm:px-6">
        <p className="max-w-4xl">
          <strong className="font-semibold text-mist-400">
            Not financial advice.
          </strong>{" "}
          ZeRO is a research project. Every performance figure on this site is
          out-of-sample and stated before transaction costs. As measured, the
          model does not beat a majority-class baseline on direction or a
          zero-excess forecast on magnitude — the calibrated intervals hold up,
          the point forecast does not. See{" "}
          <Link
            href="/methodology"
            className="text-brand-400 underline underline-offset-2"
          >
            Methodology
          </Link>{" "}
          for the full list of limitations.
        </p>
        <p>
          Built by Venu Gopal Battula ·{" "}
          <a
            href="https://github.com/glitching-gops/Agentic-Stock-Forecast"
            target="_blank"
            rel="noreferrer"
            className="text-brand-400 underline underline-offset-2"
          >
            Source on GitHub
          </a>
        </p>
      </div>
    </footer>
  );
}
