"use client";

import { useId, useState, type ReactNode } from "react";

import { cx } from "@/lib/format";

/**
 * Tab panels whose content is rendered on the server.
 *
 * Only the switching is client-side — each `content` arrives as an already
 * rendered RSC payload, so the panels themselves ship no component code and no
 * data fetching moves into the browser. All panels are mounted and the
 * inactive ones hidden, which keeps in-panel state (a chart's zoom window)
 * alive across tab switches.
 */
export interface TabSpec {
  id: string;
  label: string;
  content: ReactNode;
  badge?: ReactNode;
}

export function Tabs({ tabs }: { tabs: TabSpec[] }) {
  const [active, setActive] = useState(tabs[0]?.id);
  const uid = useId();

  return (
    <div>
      {/*
        The selected tab is inverse video rather than an underline in an accent
        colour. Inverse video is the terminal's own emphasis and it costs
        nothing from a palette where every hue is spoken for.
      */}
      <div
        role="tablist"
        aria-label="Stock detail sections"
        className="flex gap-px overflow-x-auto border-b border-rule-hi"
      >
        {tabs.map((tab) => {
          const selected = tab.id === active;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              id={`${uid}-tab-${tab.id}`}
              aria-selected={selected}
              aria-controls={`${uid}-panel-${tab.id}`}
              onClick={() => setActive(tab.id)}
              className={cx(
                "flex items-center gap-2 whitespace-nowrap px-3 py-1.5 text-[0.72rem] uppercase tracking-[0.12em] transition-colors",
                selected
                  ? "inv font-semibold"
                  : "text-dim hover:bg-raise hover:text-text",
              )}
            >
              {tab.label}
              {tab.badge}
            </button>
          );
        })}
      </div>

      {tabs.map((tab) => (
        <div
          key={tab.id}
          role="tabpanel"
          id={`${uid}-panel-${tab.id}`}
          aria-labelledby={`${uid}-tab-${tab.id}`}
          hidden={tab.id !== active}
          className="pt-5"
        >
          {tab.content}
        </div>
      ))}
    </div>
  );
}
