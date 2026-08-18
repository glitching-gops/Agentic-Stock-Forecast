import type { ReactNode } from "react";

import { cx } from "@/lib/format";

/**
 * Presentational primitives. All server components — none of them holds state,
 * so none of them needs to ship JavaScript.
 *
 * Help text is delivered through the native `title` attribute rather than a
 * custom tooltip. This app leans on explanatory copy heavily (almost every
 * number needs a caveat attached), and a hand-rolled tooltip on that many
 * elements is a large amount of JS and a large accessibility surface to get
 * wrong. Anything a reader genuinely must see is rendered as visible text
 * instead of hidden behind a hover.
 */

export function Card({
  children,
  className,
  as: Tag = "div",
}: {
  children: ReactNode;
  className?: string;
  as?: "div" | "section" | "article" | "li";
}) {
  return (
    <Tag
      className={cx(
        "rounded-xl border border-ink-500/70 bg-ink-700/60 backdrop-blur-[2px]",
        className,
      )}
    >
      {children}
    </Tag>
  );
}

export function SectionHeading({
  title,
  description,
  right,
  id,
}: {
  title: string;
  description?: ReactNode;
  right?: ReactNode;
  id?: string;
}) {
  return (
    <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
      <div className="max-w-3xl">
        <h2
          id={id}
          className="text-lg font-semibold tracking-tight text-mist-100"
        >
          {title}
        </h2>
        {description ? (
          <p className="mt-1.5 text-sm leading-relaxed text-mist-400">
            {description}
          </p>
        ) : null}
      </div>
      {right}
    </div>
  );
}

/** A labelled figure. `sub` is for the baseline a headline number must beat. */
export function Stat({
  label,
  value,
  sub,
  tone = "neutral",
  help,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tone?: Tone;
  help?: string;
}) {
  return (
    <Card className="p-4">
      <div
        className="text-[0.68rem] font-semibold uppercase tracking-[0.09em] text-mist-500"
        title={help}
      >
        {label}
        {help ? <span className="ml-1 text-mist-500/70">ⓘ</span> : null}
      </div>
      <div
        className={cx(
          "nums mt-2 text-2xl font-semibold tracking-tight",
          TONE_TEXT[tone],
        )}
      >
        {value}
      </div>
      {sub ? (
        <div className="mt-1.5 text-xs leading-relaxed text-mist-400">
          {sub}
        </div>
      ) : null}
    </Card>
  );
}

export type Tone =
  | "neutral"
  | "brand"
  | "positive"
  | "negative"
  | "warning"
  | "muted";

const TONE_TEXT: Record<Tone, string> = {
  neutral: "text-mist-100",
  brand: "text-brand-400",
  positive: "text-pos-500",
  negative: "text-neg-500",
  warning: "text-warn-500",
  muted: "text-mist-400",
};

const TONE_BADGE: Record<Tone, string> = {
  neutral: "border-ink-400 bg-ink-600 text-mist-300",
  brand: "border-brand-500/40 bg-brand-500/12 text-brand-300",
  positive: "border-pos-500/40 bg-pos-500/12 text-pos-500",
  negative: "border-neg-500/40 bg-neg-500/12 text-neg-500",
  warning: "border-warn-500/40 bg-warn-500/12 text-warn-500",
  muted: "border-ink-500 bg-ink-600/60 text-mist-500",
};

export function Badge({
  children,
  tone = "neutral",
  title,
  className,
}: {
  children: ReactNode;
  tone?: Tone;
  title?: string;
  className?: string;
}) {
  return (
    <span
      title={title}
      className={cx(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-md border px-1.5 py-0.5 text-[0.68rem] font-semibold uppercase tracking-wide",
        TONE_BADGE[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function verdictTone(verdict: string | null | undefined): Tone {
  if (verdict === "APPROVED") return "positive";
  if (verdict === "FLAGGED") return "warning";
  if (verdict === "REJECTED") return "negative";
  return "muted";
}

export function evidenceTone(grade: string | null | undefined): Tone {
  if (grade === "STRONG") return "positive";
  if (grade === "WEAK") return "warning";
  return "muted";
}

/**
 * A block of prose that qualifies the numbers next to it. Used liberally and
 * on purpose: the measured finding here is a null result, and a dashboard that
 * renders a null result without saying so is misleading by omission.
 */
export function Callout({
  tone = "neutral",
  title,
  children,
}: {
  tone?: Tone;
  title?: string;
  children: ReactNode;
}) {
  const accent: Record<Tone, string> = {
    neutral: "border-l-ink-400",
    brand: "border-l-brand-500",
    positive: "border-l-pos-500",
    negative: "border-l-neg-500",
    warning: "border-l-warn-500",
    muted: "border-l-ink-500",
  };
  return (
    <div
      className={cx(
        "rounded-r-lg border border-l-4 border-ink-500/70 bg-ink-700/50 px-4 py-3",
        accent[tone],
      )}
    >
      {title ? (
        <div className={cx("text-sm font-semibold", TONE_TEXT[tone])}>
          {title}
        </div>
      ) : null}
      <div
        className={cx(
          "text-sm leading-relaxed text-mist-300",
          title && "mt-1",
          "[&_a]:text-brand-400 [&_a]:underline [&_a]:underline-offset-2",
          "[&_strong]:font-semibold [&_strong]:text-mist-100",
        )}
      >
        {children}
      </div>
    </div>
  );
}

/** Horizontal 0–100 meter for the composite score. */
export function ScoreMeter({
  value,
  className,
}: {
  value: number | null | undefined;
  className?: string;
}) {
  const pct =
    value === null || value === undefined || !Number.isFinite(value)
      ? 0
      : Math.max(0, Math.min(100, value));
  const zero = pct === 0;

  return (
    <div className={cx("flex items-center gap-2", className)}>
      <div className="h-1.5 w-full min-w-14 overflow-hidden rounded-full bg-ink-600">
        <div
          className={cx(
            "h-full rounded-full",
            zero ? "bg-ink-500" : "bg-brand-500",
          )}
          style={{ width: `${zero ? 100 : pct}%` }}
        />
      </div>
      <span
        className={cx(
          "nums w-10 shrink-0 text-right text-xs font-semibold",
          zero ? "text-mist-500" : "text-mist-100",
        )}
      >
        {pct.toFixed(1)}
      </span>
    </div>
  );
}

export function Prose({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cx(
        "max-w-3xl space-y-4 text-sm leading-relaxed text-mist-300",
        "[&_a]:text-brand-400 [&_a]:underline [&_a]:underline-offset-2",
        "[&_code]:rounded [&_code]:bg-ink-600 [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[0.85em] [&_code]:text-brand-300",
        "[&_strong]:font-semibold [&_strong]:text-mist-100",
        "[&_li]:pl-1 [&_ul]:list-disc [&_ul]:space-y-2 [&_ul]:pl-5",
        "[&_h3]:pt-2 [&_h3]:text-base [&_h3]:font-semibold [&_h3]:text-mist-100",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function Empty({
  title,
  children,
}: {
  title: string;
  children?: ReactNode;
}) {
  return (
    <Card className="px-6 py-10 text-center">
      <div className="text-sm font-semibold text-mist-300">{title}</div>
      {children ? (
        <div className="mx-auto mt-2 max-w-md text-sm text-mist-500">
          {children}
        </div>
      ) : null}
    </Card>
  );
}
