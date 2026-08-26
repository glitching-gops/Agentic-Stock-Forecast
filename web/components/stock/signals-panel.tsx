"use client";

import { Badge, Eyebrow, SectionHead, type Tone } from "@/components/ui";
import { cx, dateOnly, decimal } from "@/lib/format";
import type { SignalRow } from "@/lib/types";

type Reading = "Bullish" | "Bearish" | "Neutral";

const READING_TONE: Record<Reading, Tone> = {
  Bullish: "pos",
  Bearish: "neg",
  Neutral: "dim",
};

/**
 * Conventional overbought/oversold readings on the latest signal row.
 *
 * These thresholds are the textbook ones and are carried over unchanged from
 * the Streamlit view. They are a description of the indicator, NOT the model's
 * opinion: the model consumes these columns as raw features and reaches its
 * own conclusion, which the Overview tab reports. A column of Bullish badges
 * here next to an INSUFFICIENT evidence grade is not a contradiction — it
 * means the indicators look constructive and the model has failed to turn that
 * into out-of-sample skill.
 */
interface ReadingRow {
  label: string;
  value: number | null;
  reading: Reading;
  note: string;
}

function readings(latest: Record<string, unknown>): ReadingRow[] {
  const num = (key: string): number | null => {
    const value = latest[key];
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  };

  const close = num("close");
  const compare = (
    a: number | null,
    b: number | null,
    above: Reading,
    below: Reading,
  ): Reading => {
    if (a === null || b === null) return "Neutral";
    return a > b ? above : a < b ? below : "Neutral";
  };

  const band = (
    value: number | null,
    high: number,
    low: number,
    aboveHigh: Reading,
    belowLow: Reading,
  ): Reading => {
    if (value === null) return "Neutral";
    return value > high ? aboveHigh : value < low ? belowLow : "Neutral";
  };

  return [
    {
      label: "RSI (14)",
      value: num("rsi"),
      reading: band(num("rsi"), 70, 30, "Bearish", "Bullish"),
      note: "Above 70 overbought, below 30 oversold.",
    },
    {
      label: "MACD histogram",
      value: num("macd_hist"),
      reading: compare(num("macd_hist"), 0, "Bullish", "Bearish"),
      note: "Sign of the MACD minus its signal line.",
    },
    {
      label: "Stochastic %K",
      value: num("stoch_k"),
      reading: band(num("stoch_k"), 80, 20, "Bearish", "Bullish"),
      note: "Above 80 overbought, below 20 oversold.",
    },
    {
      label: "Williams %R",
      value: num("williams_r"),
      reading: band(num("williams_r"), -20, -80, "Bearish", "Bullish"),
      note: "Above −20 overbought, below −80 oversold.",
    },
    {
      label: "ROC (10)",
      value: num("roc_10"),
      reading: compare(num("roc_10"), 0, "Bullish", "Bearish"),
      note: "10-session rate of change.",
    },
    {
      label: "Price vs SMA-20",
      value: close,
      reading: compare(close, num("sma_20"), "Bullish", "Bearish"),
      note: `SMA-20 at ${decimal(num("sma_20"))}.`,
    },
    {
      label: "Price vs EMA-50",
      value: close,
      reading: compare(close, num("ema_50"), "Bullish", "Bearish"),
      note: `EMA-50 at ${decimal(num("ema_50"))}.`,
    },
    {
      label: "EMA-9 vs EMA-21",
      value: num("ema_9"),
      reading: compare(num("ema_9"), num("ema_21"), "Bullish", "Bearish"),
      note: `EMA-21 at ${decimal(num("ema_21"))}.`,
    },
    {
      label: "Price vs upper band",
      value: close,
      reading:
        close !== null &&
        num("bb_upper") !== null &&
        close > (num("bb_upper") as number)
          ? "Bearish"
          : "Neutral",
      note: `Upper band at ${decimal(num("bb_upper"))}.`,
    },
    {
      label: "Price vs lower band",
      value: close,
      reading:
        close !== null &&
        num("bb_lower") !== null &&
        close < (num("bb_lower") as number)
          ? "Bullish"
          : "Neutral",
      note: `Lower band at ${decimal(num("bb_lower"))}.`,
    },
  ];
}

export function SignalsPanel({
  latest,
  history,
}: {
  latest: Record<string, unknown>;
  history: SignalRow[];
}) {
  const rows = readings(latest);

  return (
    <div className="space-y-8">
      <section>
        <SectionHead
          as="h3"
          title="Latest technical readings"
          description="Conventional overbought/oversold thresholds applied to the most recent session. These describe the indicators — the model consumes them as raw features and its own conclusion is on the Overview tab."
        />

        {/*
          A table, not a grid of cards. Ten readings share one set of columns,
          and the comparison a reader wants is down a column: which of these
          agree. Cards put a border between every pair of numbers being
          compared.
        */}
        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] border-collapse text-[0.76rem]">
            <thead>
              <tr className="bg-inset">
                <th className="border-y border-rule px-2 py-1 text-left">
                  <Eyebrow>Indicator</Eyebrow>
                </th>
                <th className="border-y border-rule px-2 py-1 text-right">
                  <Eyebrow>Value</Eyebrow>
                </th>
                <th className="border-y border-rule px-2 py-1 text-left">
                  <Eyebrow>Reading</Eyebrow>
                </th>
                <th className="border-y border-rule px-2 py-1 text-left">
                  <Eyebrow>Threshold</Eyebrow>
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.label} className="border-b border-rule/70">
                  <td className="px-2 py-1 text-text">{row.label}</td>
                  <td className="px-2 py-1 text-right text-bright">
                    {decimal(row.value)}
                  </td>
                  <td className="px-2 py-1">
                    <Badge tone={READING_TONE[row.reading]}>
                      {row.reading}
                    </Badge>
                  </td>
                  <td className="px-2 py-1 text-dim">{row.note}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <SignalHeatmap history={history} />
    </div>
  );
}

/**
 * 30 sessions × N features, min–max normalised per feature.
 *
 * Built from CSS grid cells rather than a charting library: it is a matrix of
 * coloured rectangles, and shipping a plotting runtime to draw rectangles is
 * a poor trade. Normalisation is per row and within the visible window only,
 * so a cell says "high for this feature over these 30 sessions" and nothing
 * about its absolute level — which is why the actual value is on every cell's
 * tooltip.
 */
function SignalHeatmap({ history }: { history: SignalRow[] }) {
  const window = history.slice(-30);
  if (window.length === 0) return null;

  const excluded = new Set([
    "date",
    "ticker",
    "target",
    "target_return",
    "target_excess_return",
    "benchmark_return",
    "benchmark_ticker",
    "benchmark_sector_specific",
  ]);

  const features = Object.keys(window[window.length - 1])
    .filter((key) => !excluded.has(key))
    .filter((key) =>
      window.some(
        (row) => typeof row[key] === "number" && Number.isFinite(row[key] as number),
      ),
    )
    .sort();

  if (features.length === 0) return null;

  return (
    <section>
      <SectionHead
        as="h3"
        title={`Signal heatmap — last ${window.length} sessions`}
        description={
          <>
            Each row is min–max normalised across the window shown, so
            brightness means “high or low{" "}
            <span className="text-text">for this feature, recently</span>” and
            carries no absolute scale. Forward-looking columns — the target and
            its benchmark components — are excluded so the grid can never leak
            a label.
          </>
        }
      />

      <div className="overflow-x-auto border border-rule bg-inset p-3">
        <div className="min-w-[640px] space-y-[2px]">
          {features.map((feature) => {
            const values = window.map((row) =>
              typeof row[feature] === "number" &&
              Number.isFinite(row[feature] as number)
                ? (row[feature] as number)
                : null,
            );
            const present = values.filter((v): v is number => v !== null);
            const min = Math.min(...present);
            const max = Math.max(...present);
            const span = max - min;

            return (
              <div key={feature} className="flex items-center gap-2">
                <div className="w-36 shrink-0 truncate text-right text-[0.64rem] text-dim">
                  {feature}
                </div>
                <div className="flex flex-1 gap-px">
                  {values.map((value, index) => {
                    const t =
                      value === null || span === 0 ? 0.5 : (value - min) / span;
                    return (
                      <div
                        key={window[index].date}
                        title={`${feature} · ${dateOnly(window[index].date)} · ${decimal(value, 4)}`}
                        className={cx(
                          "h-4 flex-1",
                          value === null && "opacity-25",
                        )}
                        style={{ backgroundColor: rampColor(t) }}
                      />
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-4 flex items-center gap-2 pl-[9.5rem] text-[0.64rem] uppercase tracking-[0.12em] text-dim">
          <span>low</span>
          <div
            aria-hidden
            className="h-2 w-32"
            style={{
              background:
                "linear-gradient(90deg, var(--color-inset), var(--color-bright))",
            }}
          />
          <span>high</span>
        </div>
      </div>
    </section>
  );
}

/**
 * A luminance ramp, not a hue ramp.
 *
 * The Streamlit original ran cyan → navy → amber. Two of those now mean
 * something specific elsewhere on the site, and a min–max normalised value has
 * no meaningful midpoint for a diverging scale to diverge around — it is a
 * rank within a window. Brightness carries that with no colour spent, and the
 * grid reads as the phosphor matrix it is.
 */
function rampColor(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const [r1, g1, b1] = [6, 6, 7]; // --color-inset
  const [r2, g2, b2] = [236, 236, 242]; // --color-bright
  const mix = (a: number, b: number) => Math.round(a + (b - a) * clamped);
  return `rgb(${mix(r1, r2)} ${mix(g1, g2)} ${mix(b1, b2)})`;
}
