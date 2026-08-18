"use client";

import { Badge, Card, type Tone } from "@/components/ui";
import { cx, dateOnly, decimal } from "@/lib/format";
import type { SignalRow } from "@/lib/types";

type Reading = "Bullish" | "Bearish" | "Neutral";

const READING_TONE: Record<Reading, Tone> = {
  Bullish: "positive",
  Bearish: "negative",
  Neutral: "muted",
};

/**
 * Conventional overbought/oversold readings on the latest signal row.
 *
 * These thresholds are the textbook ones and are carried over unchanged from
 * the Streamlit view. They are a description of the indicator, NOT the model's
 * opinion: the model consumes these columns as raw features and reaches its
 * own conclusion, which the Overview tab reports. A row of green badges here
 * next to an INSUFFICIENT evidence grade is not a contradiction — it means the
 * indicators look constructive and the model has failed to turn that into
 * out-of-sample skill.
 */
interface ReadingCard {
  label: string;
  value: number | null;
  reading: Reading;
  note: string;
}

function readings(latest: Record<string, unknown>): ReadingCard[] {
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
        close !== null && num("bb_upper") !== null && close > (num("bb_upper") as number)
          ? "Bearish"
          : "Neutral",
      note: `Upper band at ${decimal(num("bb_upper"))}.`,
    },
    {
      label: "Price vs lower band",
      value: close,
      reading:
        close !== null && num("bb_lower") !== null && close < (num("bb_lower") as number)
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
  const cards = readings(latest);

  return (
    <div className="space-y-8">
      <section>
        <h3 className="mb-1 text-sm font-semibold text-mist-100">
          Latest technical readings
        </h3>
        <p className="mb-4 max-w-3xl text-xs leading-relaxed text-mist-500">
          Conventional overbought/oversold thresholds applied to the most recent
          session. These describe the indicators — the model consumes them as
          raw features and its own conclusion is on the Overview tab.
        </p>

        <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2 xl:grid-cols-3">
          {cards.map((card) => (
            <Card key={card.label} className="p-3">
              <div className="flex items-start justify-between gap-2">
                <span className="text-xs font-medium text-mist-400">
                  {card.label}
                </span>
                <Badge tone={READING_TONE[card.reading]}>{card.reading}</Badge>
              </div>
              <div className="nums mt-1.5 text-lg font-semibold text-mist-100">
                {decimal(card.value)}
              </div>
              <div className="mt-0.5 text-[0.68rem] text-mist-500">
                {card.note}
              </div>
            </Card>
          ))}
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
      window.some((row) => typeof row[key] === "number" && Number.isFinite(row[key] as number)),
    )
    .sort();

  if (features.length === 0) return null;

  return (
    <section>
      <h3 className="mb-1 text-sm font-semibold text-mist-100">
        Signal heatmap — last {window.length} sessions
      </h3>
      <p className="mb-4 max-w-3xl text-xs leading-relaxed text-mist-500">
        Each row is min–max normalised across the window shown, so colour means
        “high or low <em className="not-italic text-mist-400">for this feature,
        recently</em>” and carries no absolute scale. Forward-looking columns —
        the target and its benchmark components — are excluded so the grid can
        never leak a label.
      </p>

      <div className="overflow-x-auto rounded-xl border border-ink-500/70 bg-ink-700/40 p-3">
        <div className="min-w-[640px] space-y-[3px]">
          {features.map((feature) => {
            const values = window.map((row) =>
              typeof row[feature] === "number" && Number.isFinite(row[feature] as number)
                ? (row[feature] as number)
                : null,
            );
            const present = values.filter((v): v is number => v !== null);
            const min = Math.min(...present);
            const max = Math.max(...present);
            const span = max - min;

            return (
              <div key={feature} className="flex items-center gap-2">
                <div className="w-36 shrink-0 truncate text-right font-mono text-[0.65rem] text-mist-500">
                  {feature}
                </div>
                <div className="flex flex-1 gap-[2px]">
                  {values.map((value, index) => {
                    const t = value === null || span === 0 ? 0.5 : (value - min) / span;
                    return (
                      <div
                        key={window[index].date}
                        title={`${feature} · ${dateOnly(window[index].date)} · ${decimal(value, 4)}`}
                        className={cx(
                          "h-4 flex-1 rounded-[2px]",
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

        <div className="mt-4 flex items-center gap-2 pl-[9.5rem] text-[0.65rem] text-mist-500">
          <span>low</span>
          <div
            className="h-2 w-32 rounded-full"
            style={{
              background:
                "linear-gradient(90deg, var(--color-brand-500), var(--color-ink-700), var(--color-warn-500))",
            }}
          />
          <span>high</span>
        </div>
      </div>
    </section>
  );
}

/** Cyan → deep navy → amber, the ramp the Streamlit heatmap used. */
function rampColor(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const stops: [number, number, number][] = [
    [0, 180, 216],
    [15, 24, 52],
    [255, 183, 3],
  ];
  const scaled = clamped * (stops.length - 1);
  const index = Math.min(Math.floor(scaled), stops.length - 2);
  const local = scaled - index;
  const [r1, g1, b1] = stops[index];
  const [r2, g2, b2] = stops[index + 1];
  const mix = (a: number, b: number) => Math.round(a + (b - a) * local);
  return `rgb(${mix(r1, r2)} ${mix(g1, g2)} ${mix(b1, b2)})`;
}
