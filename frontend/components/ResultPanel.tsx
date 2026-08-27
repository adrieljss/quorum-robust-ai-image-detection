import { ConfidenceDial } from "@/components/ConfidenceDial";
import { SignalBars } from "@/components/SignalBars";
import type { BackendResult, Reliability, Verdict } from "@/lib/types";

export function ResultPanel({ result }: { result: BackendResult }) {
  const verdict = normalizeVerdict(result.verdict);
  const reliability = normalizeReliability(result.reliability);
  const copy = verdictCopy(verdict);

  return (
    <div className="flex flex-col gap-7">
      <div>
        <p className="text-xs uppercase tracking-[0.18em] text-taupe">
          {formatContentType(result.content_type)}
        </p>
        <h2 className="font-display mt-2 text-3xl leading-tight text-umber sm:text-[2.1rem]">
          {copy.title}
        </h2>
        <p className="mt-3 max-w-xl text-[0.98rem] leading-relaxed text-bark">
          {result.explanation}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <span
          className={`rounded-full px-3 py-1 text-xs font-medium ${copy.badgeClass}`}
        >
          {copy.badge}
        </span>
        <span className="rounded-full bg-cream px-3 py-1 text-xs text-bark">
          Evidence {reliability}
        </span>
      </div>

      <div className="grid gap-6 border-y border-line py-6 sm:grid-cols-[auto_1fr] sm:items-center">
        <ConfidenceDial
          value={result.confidence}
          label={copy.confidenceNote}
        />
        <ProvenanceNote provenance={result.provenance} />
      </div>

      <SignalBars signals={result.signals} />
    </div>
  );
}

function ProvenanceNote({
  provenance,
}: {
  provenance: BackendResult["provenance"];
}) {
  const parts: string[] = [];
  if (provenance?.exif_software) {
    parts.push(`Software tag: ${provenance.exif_software}`);
  }
  if (provenance?.c2pa) {
    parts.push(`C2PA: ${provenance.c2pa}`);
  }

  return (
    <div className="rounded-2xl bg-cream/70 px-4 py-3">
      <p className="text-xs uppercase tracking-[0.16em] text-taupe">Provenance</p>
      <p className="mt-1 text-sm leading-snug text-bark">
        {parts.length ? parts.join(" · ") : "No embedded provenance found in this file."}
      </p>
    </div>
  );
}

function normalizeVerdict(value: string): Verdict {
  if (value === "likely_authentic" || value === "likely_real") return "likely_authentic";
  if (value === "uncertain") return "uncertain";
  return "likely_ai";
}

function normalizeReliability(value: string): Reliability {
  if (value === "high" || value === "low") return value;
  return "medium";
}

function formatContentType(value: string) {
  if (!value) return "Image";
  return value.replace(/[-_]/g, " ");
}

function verdictCopy(verdict: Verdict) {
  if (verdict === "likely_ai") {
    return {
      title: "Likely machine-generated",
      badge: "AI leaning",
      badgeClass: "bg-[#f3ddd0] text-clay-dark",
      confidenceNote: "How sure the fused model is of this reading.",
    };
  }
  if (verdict === "likely_authentic") {
    return {
      title: "Likely camera-captured",
      badge: "Authentic leaning",
      badgeClass: "bg-[#dce6d8] text-moss",
      confidenceNote: "How sure the fused model is of this reading.",
    };
  }
  return {
    title: "Not enough to decide",
    badge: "Uncertain",
    badgeClass: "bg-cream text-bark",
    confidenceNote: "The branches disagree, or the file is too degraded.",
  };
}
