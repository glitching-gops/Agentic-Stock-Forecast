export const metadata = {
  title: "About",
};

/**
 * Intentionally blank.
 *
 * The owner is writing this page himself once the whole app exists. Everything
 * a reader needs in order to judge the numbers — how they are measured, what
 * they currently say, and what the system does not model — lives on
 * /methodology, so nothing load-bearing is waiting on this page.
 */
export default function AboutPage() {
  return (
    <div className="mx-auto max-w-2xl py-24 text-center">
      <h1 className="text-2xl font-semibold tracking-tight text-mist-100">
        About
      </h1>
      <p className="mt-3 text-sm leading-relaxed text-mist-500">
        This page is still to be written.
      </p>
      <p className="mt-6 text-sm leading-relaxed text-mist-400">
        Looking for how the forecasts are produced and evaluated?{" "}
        <a
          href="/methodology"
          className="text-brand-400 underline underline-offset-2"
        >
          Read the methodology
        </a>
        .
      </p>
    </div>
  );
}
