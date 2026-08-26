import type { ReactNode } from "react";

import { cx } from "@/lib/format";

/**
 * Presentational primitives. All server components — none of them holds state,
 * so none of them ships JavaScript.
 *
 * The vocabulary is a terminal's, not a dashboard's: panels are hairline
 * rules rather than raised cards, emphasis is inverse video rather than an
 * accent colour, and every label is a fixed-width eyebrow so columns line up
 * across blocks that know nothing about each other.
 *
 * Help text is delivered through the native `title` attribute rather than a
 * custom tooltip. Almost every number here needs a caveat attached, and a
 * hand-rolled tooltip on that many elements is a large amount of JS and a
 * large accessibility surface to get wrong. Anything a reader genuinely must
 * see is rendered as visible text instead of hidden behind a hover.
 */

export type Tone = "neutral" | "bar" | "pos" | "neg" | "dim";

const TONE_TEXT: Record<Tone, string> = {
  neutral: "text-bright",
  bar: "text-bar",
  pos: "text-pos",
  neg: "text-neg",
  dim: "text-dim",
};

const TONE_EDGE: Record<Tone, string> = {
  neutral: "border-rule-hi text-text",
  bar: "border-bar/45 text-bar",
  pos: "border-pos/45 text-pos",
  neg: "border-neg/45 text-neg",
  dim: "border-rule text-dim",
};

/* ── Structure ─────────────────────────────────────────────────────────── */

/** A hairline-bounded region. No radius, no fill gradient, no shadow. */
export function Panel({
  children,
  className,
  as: Tag = "div",
}: {
  children: ReactNode;
  className?: string;
  as?: "div" | "section" | "article" | "li";
}) {
  return (
    <Tag className={cx("border border-rule bg-shell", className)}>
      {children}
    </Tag>
  );
}

/**
 * A fixed-width uppercase label. The tracking and size are constants across
 * the whole app on purpose — this is the character cell everything aligns to,
 * so a one-off size here would break alignment somewhere else.
 */
export function Eyebrow({
  children,
  className,
  title,
  as: Tag = "div",
}: {
  children: ReactNode;
  className?: string;
  title?: string;
  as?: "div" | "span" | "h2" | "h3" | "dt" | "legend";
}) {
  return (
    <Tag
      title={title}
      className={cx(
        "text-[0.64rem] font-medium uppercase tracking-[0.16em] text-dim",
        className,
      )}
    >
      {children}
      {title ? <span className="ml-1 opacity-50">?</span> : null}
    </Tag>
  );
}

/**
 * A section head. The rule runs the full width and the label sits on it —
 * the device is a terminal's section divider rather than a heading with
 * margin, so sections butt against each other without accumulating space.
 */
export function SectionHead({
  title,
  count,
  description,
  right,
  id,
  as: Tag = "h2",
}: {
  title: string;
  count?: ReactNode;
  description?: ReactNode;
  right?: ReactNode;
  id?: string;
  as?: "h2" | "h3";
}) {
  return (
    <div className="mb-3">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1 border-b border-rule-hi pb-1.5">
        <Tag
          id={id}
          className="text-[0.72rem] font-semibold uppercase tracking-[0.18em] text-bright"
        >
          {title}
          {count !== undefined ? (
            <span className="ml-2 font-normal text-dim">{count}</span>
          ) : null}
        </Tag>
        {right}
      </div>
      {description ? (
        <p className="mt-2 max-w-4xl font-prose text-[0.8rem] leading-relaxed text-mid">
          {description}
        </p>
      ) : null}
    </div>
  );
}

/* ── Figures ───────────────────────────────────────────────────────────── */

/**
 * A labelled figure. `sub` is where the baseline a headline number must beat
 * goes — a figure quoted here without one is usually a figure that cannot be
 * judged.
 */
export function Readout({
  label,
  value,
  sub,
  tone = "neutral",
  help,
  className,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tone?: Tone;
  help?: string;
  className?: string;
}) {
  return (
    <div className={cx("border-l border-rule-hi pl-3", className)}>
      <Eyebrow title={help}>{label}</Eyebrow>
      <div
        className={cx(
          "mt-1.5 font-display text-[1.35rem] leading-none tracking-tight",
          TONE_TEXT[tone],
        )}
      >
        {value}
      </div>
      {sub ? (
        <div className="mt-2 text-[0.7rem] leading-relaxed text-dim">{sub}</div>
      ) : null}
    </div>
  );
}

export function Badge({
  children,
  tone = "dim",
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
        "inline-flex items-center whitespace-nowrap border px-1.5 py-px text-[0.62rem] font-medium uppercase tracking-[0.1em]",
        TONE_EDGE[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function verdictTone(verdict: string | null | undefined): Tone {
  if (verdict === "APPROVED") return "pos";
  if (verdict === "FLAGGED") return "bar";
  if (verdict === "REJECTED") return "neg";
  return "dim";
}

export function evidenceTone(grade: string | null | undefined): Tone {
  if (grade === "STRONG") return "pos";
  if (grade === "WEAK") return "bar";
  return "dim";
}

/* ── Rails ─────────────────────────────────────────────────────────────── */

/**
 * A value plotted against a threshold on a shared axis.
 *
 * This is the one device the whole site is built around, and it exists
 * because every claim in this project has the same shape: a measurement, a
 * bar that was fixed before the measurement was taken, and the gap between
 * them. Printing "+1.86" alone asks the reader to remember what beats what;
 * printing it on the rail shows the answer.
 *
 * The threshold is drawn in phosphor amber and nothing else on the page is
 * allowed to use that hue, so a reader learns the mark once.
 */
export function Rail({
  value,
  threshold,
  min,
  max,
  label,
  className,
}: {
  value: number | null | undefined;
  threshold: number;
  min: number;
  max: number;
  label?: string;
  className?: string;
}) {
  const span = max - min;
  const pos = (v: number) => ((Math.max(min, Math.min(max, v)) - min) / span) * 100;

  const has = value !== null && value !== undefined && Number.isFinite(value);
  const cleared = has && (value as number) >= threshold;

  return (
    <div
      className={cx("relative h-4 w-full min-w-24", className)}
      title={label}
      aria-hidden
    >
      {/* Baseline */}
      <div className="absolute inset-x-0 top-1/2 h-px -translate-y-1/2 bg-rule-hi" />

      {/* Zero, where it falls inside the domain — a signed statistic needs it. */}
      {min < 0 && max > 0 ? (
        <div
          className="absolute inset-y-1 w-px bg-rule-hi"
          style={{ left: `${pos(0)}%` }}
        />
      ) : null}

      {/* The pre-registered bar. */}
      <div
        className="absolute inset-y-0 w-px bg-bar"
        style={{ left: `${pos(threshold)}%` }}
      />

      {/* The measurement: a bar from zero (or from the floor) to the value. */}
      {has ? (
        <>
          <div
            className={cx(
              "absolute top-1/2 h-[3px] -translate-y-1/2",
              cleared ? "bg-bar" : "bg-mid",
            )}
            style={{
              left: `${Math.min(pos(min < 0 ? 0 : min), pos(value as number))}%`,
              width: `${Math.abs(pos(value as number) - pos(min < 0 ? 0 : min))}%`,
            }}
          />
          <div
            className={cx(
              "absolute top-1/2 h-2 w-[3px] -translate-y-1/2",
              cleared ? "bg-bar" : "bg-bright",
            )}
            style={{ left: `${pos(value as number)}%` }}
          />
        </>
      ) : null}
    </div>
  );
}

/**
 * Horizontal 0–100 meter for the composite score.
 *
 * A zero fills the track in rule grey rather than drawing nothing, because an
 * empty track and a zero score look identical and 93 of 95 rows are zero.
 */
export function Meter({
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
      <div className="h-[5px] w-full min-w-12 bg-inset">
        <div
          className={cx("h-full", zero ? "bg-rule" : "bg-bright")}
          style={{ width: `${zero ? 100 : pct}%` }}
        />
      </div>
      <span
        className={cx(
          "w-9 shrink-0 text-right text-[0.7rem]",
          zero ? "text-dim" : "text-bright",
        )}
      >
        {pct.toFixed(1)}
      </span>
    </div>
  );
}

/* ── Prose ─────────────────────────────────────────────────────────────── */

/**
 * A block of text that qualifies the numbers beside it. Used liberally and on
 * purpose: the measured finding is a null result, and an interface that
 * renders a null result without saying so is misleading by omission.
 */
export function Note({
  tone = "neutral",
  title,
  children,
}: {
  tone?: Tone;
  title?: string;
  children: ReactNode;
}) {
  const edge: Record<Tone, string> = {
    neutral: "border-l-rule-hi",
    bar: "border-l-bar",
    pos: "border-l-pos",
    neg: "border-l-neg",
    dim: "border-l-rule",
  };
  return (
    <div
      className={cx("border-y border-l-2 border-y-rule bg-shell px-4 py-3", edge[tone])}
    >
      {title ? (
        <div
          className={cx(
            "text-[0.72rem] font-semibold uppercase tracking-[0.14em]",
            TONE_TEXT[tone],
          )}
        >
          {title}
        </div>
      ) : null}
      <div
        className={cx(
          "font-prose text-[0.83rem] leading-relaxed text-mid",
          title && "mt-1.5",
          "[&_a]:text-bright [&_a]:underline [&_a]:underline-offset-2",
          "[&_strong]:font-semibold [&_strong]:text-bright",
          "[&_code]:font-mono [&_code]:text-[0.9em] [&_code]:text-text",
        )}
      >
        {children}
      </div>
    </div>
  );
}

/** Long-form reading. The one place a proportional face is allowed. */
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
        "max-w-[68ch] space-y-4 font-prose text-[0.86rem] leading-[1.75] text-mid",
        "[&_a]:text-bright [&_a]:underline [&_a]:underline-offset-2",
        "[&_code]:bg-raise [&_code]:px-1 [&_code]:font-mono [&_code]:text-[0.85em] [&_code]:text-text",
        "[&_strong]:font-semibold [&_strong]:text-bright",
        "[&_li]:pl-1 [&_ul]:list-[square] [&_ul]:space-y-2 [&_ul]:pl-5",
        "[&_h3]:font-mono [&_h3]:text-[0.72rem] [&_h3]:font-semibold [&_h3]:uppercase [&_h3]:tracking-[0.16em] [&_h3]:text-bright [&_h3]:pt-2",
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
    <Panel className="px-6 py-10 text-center">
      <div className="text-[0.78rem] font-semibold uppercase tracking-[0.14em] text-mid">
        {title}
      </div>
      {children ? (
        <div className="mx-auto mt-2 max-w-md font-prose text-[0.82rem] leading-relaxed text-dim">
          {children}
        </div>
      ) : null}
    </Panel>
  );
}

/** Bordered action, sized to the character cell. */
export function ActionLink({
  href,
  children,
  className,
}: {
  href: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <a
      href={href}
      className={cx(
        "inline-flex items-center border border-rule-hi px-2.5 py-1 text-[0.72rem] uppercase tracking-[0.12em] text-text transition-colors hover:bg-bright hover:text-void",
        className,
      )}
    >
      {children}
    </a>
  );
}
