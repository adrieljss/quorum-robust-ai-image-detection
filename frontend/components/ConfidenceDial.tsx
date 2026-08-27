type ConfidenceDialProps = {
  value: number;
  label: string;
};

export function ConfidenceDial({ value, label }: ConfidenceDialProps) {
  const pct = Math.round(clamp(value, 0, 1) * 100);
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - pct / 100);

  return (
    <div className="flex items-center gap-4">
      <div className="relative h-[108px] w-[108px]">
        <svg viewBox="0 0 108 108" className="h-full w-full -rotate-90">
          <circle
            cx="54"
            cy="54"
            r={radius}
            fill="none"
            stroke="#eadbc8"
            strokeWidth="8"
          />
          <circle
            cx="54"
            cy="54"
            r={radius}
            fill="none"
            stroke="#c45a22"
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className="transition-[stroke-dashoffset] duration-700 ease-out"
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="font-display text-2xl leading-none text-umber">{pct}</span>
          <span className="mt-0.5 text-[10px] uppercase tracking-[0.16em] text-taupe">
            %
          </span>
        </div>
      </div>
      <div>
        <p className="text-xs uppercase tracking-[0.16em] text-taupe">Confidence</p>
        <p className="mt-1 max-w-[11rem] text-sm leading-snug text-bark">{label}</p>
      </div>
    </div>
  );
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}
