"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import { ResultPanel } from "@/components/ResultPanel";
import type { UploadItem } from "@/lib/types";

type AnalysisSlideshowProps = {
  items: UploadItem[];
  active: number;
  onActiveChange: (index: number) => void;
};

export function AnalysisSlideshow({
  items,
  active,
  onActiveChange,
}: AnalysisSlideshowProps) {
  const current = items[active];
  const many = items.length > 1;

  if (!current) return null;

  const go = (index: number) => {
    const next = (index + items.length) % items.length;
    onActiveChange(next);
  };

  return (
    <div className="animate-rise">
      <div className="grid items-start gap-8 lg:grid-cols-[minmax(0,0.92fr)_minmax(0,1.08fr)]">
        <div>
          <div className="overflow-hidden rounded-3xl bg-cream shadow-[0_18px_40px_-28px_rgba(44,28,19,0.45)]">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              key={current.id}
              src={current.previewUrl}
              alt={current.file.name}
              className="animate-fade aspect-[4/5] w-full object-cover sm:aspect-[5/6]"
            />
          </div>
          <div className="mt-3 flex items-center justify-between gap-3 px-1">
            <p className="truncate text-sm text-taupe">{current.file.name}</p>
            {many ? (
              <p className="shrink-0 text-xs uppercase tracking-[0.16em] text-taupe">
                {active + 1} of {items.length}
              </p>
            ) : null}
          </div>
        </div>

        <div className="rounded-3xl bg-card px-5 py-6 shadow-[0_16px_40px_-30px_rgba(44,28,19,0.4)] sm:px-7 sm:py-8">
          {current.error ? (
            <p className="text-sm text-clay-dark">{current.error}</p>
          ) : current.result ? (
            <ResultPanel result={current.result} />
          ) : (
            <p className="text-sm text-taupe">No result for this image.</p>
          )}
        </div>
      </div>

      {many ? (
        <div className="mt-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex gap-2 overflow-x-auto pb-1">
            {items.map((item, index) => (
              <button
                key={item.id}
                type="button"
                onClick={() => onActiveChange(index)}
                aria-label={`Show ${item.file.name}`}
                className={`h-16 w-16 shrink-0 overflow-hidden rounded-2xl border-2 transition-colors ${
                  index === active ? "border-clay" : "border-transparent opacity-70 hover:opacity-100"
                }`}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={item.previewUrl} alt="" className="h-full w-full object-cover" />
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => go(active - 1)}
              className="inline-flex items-center gap-1 rounded-full bg-cream px-4 py-2 text-sm text-bark transition-colors hover:bg-sand"
            >
              <ChevronLeft size={16} /> Previous
            </button>
            <button
              type="button"
              onClick={() => go(active + 1)}
              className="inline-flex items-center gap-1 rounded-full bg-umber px-4 py-2 text-sm text-paper transition-colors hover:bg-bark"
            >
              Next <ChevronRight size={16} />
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
