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
      <div
        role="tablist"
        aria-label="Stock detail sections"
        className="flex gap-1 overflow-x-auto border-b border-ink-500/70"
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
                "-mb-px flex items-center gap-2 whitespace-nowrap border-b-2 px-3.5 py-2.5 text-sm font-medium transition-colors",
                selected
                  ? "border-brand-500 text-mist-100"
                  : "border-transparent text-mist-500 hover:text-mist-300",
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
          className="pt-6"
        >
          {tab.content}
        </div>
      ))}
    </div>
  );
}
