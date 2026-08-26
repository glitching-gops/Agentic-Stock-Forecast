import Link from "next/link";

export const metadata = {
  title: "About",
};

/**
 * Intentionally blank.
 *
 * The owner is writing this page himself once the whole app exists. Everything
 * a reader needs in order to judge the numbers — how they are measured, what
 * they currently say, and what the system does not model — lives on
 * /methodology and /research, so nothing load-bearing is waiting on this page.
 */
export default function AboutPage() {
  return (
    <div className="mx-auto max-w-xl py-24">
      <div className="text-[0.64rem] uppercase tracking-[0.16em] text-dim">
        Unwritten
      </div>
      <h1 className="mt-2 font-display text-[1.2rem] font-bold tracking-tight text-bright">
        About
      </h1>
      <p className="mt-3 max-w-[62ch] font-prose text-[0.86rem] leading-relaxed text-mid">
        This page is still to be written.
      </p>
      <p className="mt-4 max-w-[62ch] font-prose text-[0.86rem] leading-relaxed text-dim">
        Looking for how the forecasts are produced and evaluated?{" "}
        <Link href="/methodology" className="text-bright underline underline-offset-2">
          Read the methodology
        </Link>
        , or see{" "}
        <Link href="/research" className="text-bright underline underline-offset-2">
          what has been tried and measured
        </Link>
        .
      </p>
    </div>
  );
}
