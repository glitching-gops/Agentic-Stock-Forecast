"use client";

import { useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { compactNumber, cx, dateOnly, money } from "@/lib/format";

export interface PricePoint {
  date: string;
  close: number | null;
  sma_20: number | null;
  ema_21: number | null;
  ema_50: number | null;
  /** [lower, upper] — Recharts renders a two-element value as a band. */
  bb: [number, number] | null;
  /**
   * On-balance volume, not raw volume.
   *
   * The `signals` table stores no volume column — it holds engineered features,
   * and raw volume is not one of them. The Streamlit chart this replaces asked
   * for `volume` behind an `if "volume" in columns` guard that never passed, so
   * it drew an empty subplot on every stock and had done since it was written.
   * OBV is the volume-derived series that is actually present, and is a feature
   * the model consumes, so it is the honest thing to put in that panel.
   */
  obv: number | null;
}

const WINDOWS = [
  { label: "3M", sessions: 63 },
  { label: "6M", sessions: 126 },
  { label: "1Y", sessions: 252 },
] as const;

/*
 * The chart is monochrome, and the series separate by DASH PATTERN and weight
 * rather than by hue.
 *
 * That is a palette decision, not a stylistic one. Amber means "a threshold",
 * jade and coral mean "a realised move up or down"; a moving average is none
 * of those, so giving one a colour here would spend a reserved hue on a line
 * that carries no such claim. Four greys plus four dash patterns separate
 * cleanly at this line weight, and the legend prints the pattern rather than
 * naming it.
 */
const SERIES = [
  {
    key: "close",
    label: "Close",
    color: "var(--color-bright)",
    width: 1.75,
    dash: undefined,
  },
  {
    key: "sma_20",
    label: "SMA-20",
    color: "var(--color-mid)",
    width: 1,
    dash: "5 3",
  },
  {
    key: "ema_21",
    label: "EMA-21",
    color: "var(--color-mid)",
    width: 1,
    dash: "1 3",
  },
  {
    key: "ema_50",
    label: "EMA-50",
    color: "var(--color-dim)",
    width: 1,
    dash: "7 3 1 3",
  },
] as const;

export function PriceChart({ data }: { data: PricePoint[] }) {
  const [windowIndex, setWindowIndex] = useState(0);
  const sessions = WINDOWS[windowIndex].sessions;
  const view = data.slice(-sessions);

  if (view.length === 0) {
    return (
      <p className="px-4 py-8 text-center text-[0.78rem] text-dim">
        No price history is stored for this stock.
      </p>
    );
  }

  const available = WINDOWS.filter(
    (w, i) => i === 0 || data.length > w.sessions * 0.6,
  );

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <Legend />
        <div className="flex gap-px">
          {available.map((w) => {
            const index = WINDOWS.indexOf(w);
            return (
              <button
                key={w.label}
                type="button"
                onClick={() => setWindowIndex(index)}
                className={cx(
                  "px-2 py-0.5 text-[0.7rem] uppercase tracking-[0.1em] transition-colors",
                  index === windowIndex
                    ? "inv font-semibold"
                    : "text-dim hover:bg-raise hover:text-text",
                )}
              >
                {w.label}
              </button>
            );
          })}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <ComposedChart data={view} syncId="price" margin={CHART_MARGIN}>
          <CartesianGrid
            stroke="var(--color-rule)"
            strokeOpacity={0.8}
            vertical={false}
          />
          {/* Both panels share one x-axis via syncId; only the lower one labels it. */}
          <XAxis dataKey="date" {...X_AXIS} tick={false} height={0} />
          <YAxis
            {...Y_AXIS}
            domain={["auto", "auto"]}
            width={62}
            tickFormatter={(v: number) =>
              `₹${Math.round(v).toLocaleString("en-IN")}`
            }
          />
          <Tooltip content={<PriceTooltip />} cursor={CURSOR} />

          {/* Bollinger band as a filled range behind the lines. */}
          <Area
            dataKey="bb"
            stroke="none"
            fill="var(--color-rule-hi)"
            fillOpacity={0.45}
            isAnimationActive={false}
            connectNulls
            activeDot={false}
          />

          {SERIES.map((series) => (
            <Line
              key={series.key}
              type="monotone"
              dataKey={series.key}
              name={series.label}
              stroke={series.color}
              strokeWidth={series.width}
              strokeDasharray={series.dash}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>

      <div className="mt-1 text-[0.64rem] font-medium uppercase tracking-[0.16em] text-dim">
        On-balance volume
      </div>
      <ResponsiveContainer width="100%" height={96}>
        <AreaChart data={view} syncId="price" margin={{ ...CHART_MARGIN, top: 4 }}>
          <XAxis dataKey="date" {...X_AXIS} />
          {/*
            No ticks. OBV is a cumulative running total whose zero point is
            wherever the series happened to start, so its level carries no
            meaning — only its slope does. Three axis labels reading "1.2KCr"
            because the variation is four orders of magnitude below the level
            is noise dressed as precision. The width is kept so the panel stays
            aligned with the price chart above it; the value is on the tooltip.
          */}
          <YAxis {...Y_AXIS} width={62} domain={["auto", "auto"]} tick={false} />
          <Tooltip content={<ObvTooltip />} cursor={CURSOR} />
          <Area
            type="monotone"
            dataKey="obv"
            name="OBV"
            stroke="var(--color-mid)"
            strokeWidth={1}
            fill="var(--color-mid)"
            fillOpacity={0.09}
            isAnimationActive={false}
            connectNulls
            activeDot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}

const CHART_MARGIN = { top: 8, right: 8, bottom: 0, left: 0 };
const CURSOR = { stroke: "var(--color-rule-hi)", strokeWidth: 1 };

const X_AXIS = {
  tick: { fill: "var(--color-dim)", fontSize: 10 },
  tickLine: false,
  axisLine: { stroke: "var(--color-rule)" },
  minTickGap: 48,
  tickFormatter: (value: string) => dateOnly(value),
} as const;

const Y_AXIS = {
  tick: { fill: "var(--color-dim)", fontSize: 10 },
  tickLine: false,
  axisLine: false,
} as const;

interface TooltipProps {
  active?: boolean;
  payload?: { payload: PricePoint }[];
}

function TooltipShell({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="border border-rule-hi bg-shell px-3 py-2 text-[0.72rem] shadow-2xl shadow-black/70">
      <div className="mb-1.5 uppercase tracking-[0.12em] text-dim">{label}</div>
      {children}
    </div>
  );
}

/** A legend swatch that draws the series' own dash pattern rather than a hue. */
function Stroke({
  color,
  dash,
  width,
}: {
  color: string;
  dash?: string;
  width: number;
}) {
  return (
    <svg aria-hidden width="18" height="4" className="shrink-0 overflow-visible">
      <line
        x1="0"
        y1="2"
        x2="18"
        y2="2"
        stroke={color}
        strokeWidth={width}
        strokeDasharray={dash}
      />
    </svg>
  );
}

function PriceTooltip({ active, payload }: TooltipProps) {
  const point = active ? payload?.[0]?.payload : undefined;
  if (!point) return null;

  return (
    <TooltipShell label={dateOnly(point.date)}>
      <dl className="space-y-0.5">
        {SERIES.map((series) => (
          <div key={series.key} className="flex items-center gap-3">
            <Stroke
              color={series.color}
              dash={series.dash}
              width={series.width}
            />
            <dt className="flex-1 text-dim">{series.label}</dt>
            <dd className="text-bright">{money(point[series.key])}</dd>
          </div>
        ))}
        {point.bb ? (
          <div className="flex items-center gap-3 pt-1 text-dim">
            <span aria-hidden className="w-[18px] shrink-0" />
            <dt className="flex-1">Bollinger</dt>
            <dd>
              {money(point.bb[0])} – {money(point.bb[1])}
            </dd>
          </div>
        ) : null}
      </dl>
    </TooltipShell>
  );
}

function ObvTooltip({ active, payload }: TooltipProps) {
  const point = active ? payload?.[0]?.payload : undefined;
  if (!point) return null;
  return (
    <TooltipShell label={dateOnly(point.date)}>
      <div className="flex gap-3">
        <span className="text-dim">On-balance volume</span>
        <span className="text-bright">{compactNumber(point.obv)}</span>
      </div>
    </TooltipShell>
  );
}

function Legend() {
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[0.7rem] text-dim">
      {SERIES.map((series) => (
        <li key={series.key} className="flex items-center gap-1.5">
          <Stroke color={series.color} dash={series.dash} width={series.width} />
          {series.label}
        </li>
      ))}
      <li className="flex items-center gap-1.5">
        <span
          aria-hidden
          className="h-2.5 w-[18px] shrink-0 bg-rule-hi opacity-45"
        />
        Bollinger band
      </li>
    </ul>
  );
}
