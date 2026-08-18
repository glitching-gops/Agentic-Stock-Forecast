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

const SERIES = [
  { key: "close", label: "Close", color: "var(--color-brand-400)", width: 2 },
  { key: "sma_20", label: "SMA-20", color: "var(--color-warn-500)", width: 1.25 },
  { key: "ema_21", label: "EMA-21", color: "var(--color-mist-300)", width: 1.25 },
  { key: "ema_50", label: "EMA-50", color: "var(--color-mist-500)", width: 1.25 },
] as const;

export function PriceChart({ data }: { data: PricePoint[] }) {
  const [windowIndex, setWindowIndex] = useState(0);
  const sessions = WINDOWS[windowIndex].sessions;
  const view = data.slice(-sessions);

  if (view.length === 0) {
    return (
      <p className="px-4 py-8 text-center text-sm text-mist-500">
        No price history is stored for this stock.
      </p>
    );
  }

  const available = WINDOWS.filter((w, i) => i === 0 || data.length > w.sessions * 0.6);

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <Legend />
        <div className="flex gap-1">
          {available.map((w) => {
            const index = WINDOWS.indexOf(w);
            return (
              <button
                key={w.label}
                type="button"
                onClick={() => setWindowIndex(index)}
                className={cx(
                  "rounded-md px-2 py-1 text-xs font-semibold transition-colors",
                  index === windowIndex
                    ? "bg-ink-600 text-mist-100"
                    : "text-mist-500 hover:bg-ink-700 hover:text-mist-300",
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
          <CartesianGrid stroke="var(--color-ink-500)" strokeOpacity={0.5} vertical={false} />
          {/* Both panels share one x-axis via syncId; only the lower one labels it. */}
          <XAxis dataKey="date" {...X_AXIS} tick={false} height={0} />
          <YAxis
            {...Y_AXIS}
            domain={["auto", "auto"]}
            width={62}
            tickFormatter={(v: number) => `₹${Math.round(v).toLocaleString("en-IN")}`}
          />
          <Tooltip content={<PriceTooltip />} cursor={CURSOR} />

          {/* Bollinger band as a filled range behind the lines. */}
          <Area
            dataKey="bb"
            stroke="none"
            fill="var(--color-brand-500)"
            fillOpacity={0.08}
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
              strokeDasharray={series.key === "close" ? undefined : "4 3"}
              dot={false}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>

      <div className="mt-1 text-[0.68rem] font-semibold uppercase tracking-[0.09em] text-mist-500">
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
            stroke="var(--color-mist-400)"
            strokeWidth={1.25}
            fill="var(--color-mist-400)"
            fillOpacity={0.1}
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
const CURSOR = { stroke: "var(--color-ink-400)", strokeWidth: 1 };

const X_AXIS = {
  tick: { fill: "var(--color-mist-500)", fontSize: 11 },
  tickLine: false,
  axisLine: { stroke: "var(--color-ink-500)" },
  minTickGap: 48,
  tickFormatter: (value: string) => dateOnly(value),
} as const;

const Y_AXIS = {
  tick: { fill: "var(--color-mist-500)", fontSize: 11 },
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
    <div className="rounded-lg border border-ink-500 bg-ink-800/95 px-3 py-2 text-xs shadow-xl">
      <div className="mb-1.5 font-semibold text-mist-300">{label}</div>
      {children}
    </div>
  );
}

function PriceTooltip({ active, payload }: TooltipProps) {
  const point = active ? payload?.[0]?.payload : undefined;
  if (!point) return null;

  return (
    <TooltipShell label={dateOnly(point.date)}>
      <dl className="nums space-y-0.5">
        {SERIES.map((series) => (
          <div key={series.key} className="flex items-center gap-3">
            <span
              aria-hidden
              className="h-0.5 w-3 shrink-0 rounded"
              style={{ backgroundColor: series.color }}
            />
            <dt className="flex-1 text-mist-500">{series.label}</dt>
            <dd className="font-medium text-mist-100">
              {money(point[series.key])}
            </dd>
          </div>
        ))}
        {point.bb ? (
          <div className="flex items-center gap-3 pt-1 text-mist-500">
            <span aria-hidden className="w-3 shrink-0" />
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
      <div className="nums flex gap-3">
        <span className="text-mist-500">On-balance volume</span>
        <span className="font-medium text-mist-100">
          {compactNumber(point.obv)}
        </span>
      </div>
    </TooltipShell>
  );
}

function Legend() {
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-mist-400">
      {SERIES.map((series) => (
        <li key={series.key} className="flex items-center gap-1.5">
          <span
            aria-hidden
            className="h-0.5 w-4 rounded"
            style={{ backgroundColor: series.color }}
          />
          {series.label}
        </li>
      ))}
      <li className="flex items-center gap-1.5">
        <span
          aria-hidden
          className="h-2.5 w-4 rounded-sm"
          style={{ backgroundColor: "color-mix(in srgb, var(--color-brand-500) 20%, transparent)" }}
        />
        Bollinger band
      </li>
    </ul>
  );
}
