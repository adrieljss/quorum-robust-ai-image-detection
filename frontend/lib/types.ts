export type Verdict = "likely_ai" | "likely_authentic" | "uncertain";
export type Reliability = "high" | "medium" | "low";

export type Provenance = {
  c2pa: string | null;
  exif_software: string | null;
};

export type Signals = {
  general: number;
  face: number;
  text: number;
  regularity: number;
};

/** Exact per-image payload expected from the Flask API. */
export type BackendResult = {
  verdict: Verdict | string;
  confidence: number;
  provenance: Provenance;
  signals: Signals;
  content_type: string;
  explanation: string;
  degradation_estimate?: string;
  reliability: Reliability | string;
};

export type AnalyzeResponse = {
  results: BackendResult[];
};

export type UploadItem = {
  id: string;
  file: File;
  previewUrl: string;
  result: BackendResult | null;
  error: string | null;
};

export const SIGNAL_META: {
  key: keyof Signals;
  label: string;
  hint: string;
}[] = [
  {
    key: "general",
    label: "General",
    hint: "Overall photographic structure and texture.",
  },
  {
    key: "face",
    label: "Face",
    hint: "Aligned facial landmarks and lighting.",
  },
  {
    key: "text",
    label: "Text",
    hint: "Letterforms, signage, and glyph consistency.",
  },
  {
    key: "regularity",
    label: "Regularity",
    hint: "Repeating patterns and spectral fingerprints.",
  },
];
