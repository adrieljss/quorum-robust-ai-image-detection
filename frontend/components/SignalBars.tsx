import { SIGNAL_META, type Signals } from "@/lib/types";

export function SignalBars({ signals }: { signals: Signals }) {
  return (
    <div>
      <div className="mb-4 flex items-end justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.16em] text-taupe">Model signals</p>
          <p className="mt-1 text-sm text-bark">
            Four independent readings. Higher values lean toward machine-generated.
          </p>
        </div>
      </div>
      <ul className="space-y-4">
        {SIGNAL_META.map((meta) => {
          const raw = signals[meta.key];
          const pct = Math.round(clamp(raw, 0, 1) * 100);
          return (
            <li key={meta.key}>
              <div className="mb-1.5 flex items-baseline justify-between gap-3">
                <div>
                  <p className="text-sm font-medium text-umber">{meta.label}</p>
                  <p className="text-xs text-taupe">{meta.hint}</p>
                </div>
                <span className="font-display text-lg leading-none text-bark">{pct}</span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-cream">
                <div
                  className="h-full rounded-full bg-clay transition-[width] duration-700 ease-out"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
