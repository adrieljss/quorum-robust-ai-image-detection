import type { BackendResult } from "./types";

const SAMPLES: BackendResult[] = [
  {
    verdict: "likely_ai",
    confidence: 0.87,
    provenance: { c2pa: null, exif_software: "Stable Diffusion" },
    signals: { general: 0.91, face: 0.83, text: 0.62, regularity: 0.4 },
    content_type: "portrait",
    explanation:
      "Catchlights in the eyes are inconsistent, and background signage does not resolve into readable letterforms. Skin texture is unusually even across planes of the face.",
    degradation_estimate: "heavy_jpeg",
    reliability: "medium",
  },
  {
    verdict: "likely_authentic",
    confidence: 0.81,
    provenance: { c2pa: null, exif_software: "Apple iPhone 14 Pro" },
    signals: { general: 0.18, face: 0.22, text: 0.11, regularity: 0.27 },
    content_type: "scene",
    explanation:
      "Optical blur falls off naturally with distance, and high-frequency grain is consistent with a camera sensor. Embedded software metadata matches a common device pipeline.",
    degradation_estimate: "mild_jpeg",
    reliability: "high",
  },
  {
    verdict: "uncertain",
    confidence: 0.54,
    provenance: { c2pa: null, exif_software: null },
    signals: { general: 0.58, face: 0.41, text: 0.33, regularity: 0.61 },
    content_type: "object",
    explanation:
      "The image has been heavily recompressed, which weakens several cues. Regularity is elevated, but face and text branches do not agree strongly enough for a firm call.",
    degradation_estimate: "heavy_jpeg",
    reliability: "low",
  },
  {
    verdict: "likely_ai",
    confidence: 0.74,
    provenance: { c2pa: null, exif_software: null },
    signals: { general: 0.77, face: 0.5, text: 0.86, regularity: 0.69 },
    content_type: "text-heavy",
    explanation:
      "Rendered lettering shows even stroke weight and implausible dictionary hits. The general branch also leans synthetic, though no face was a decisive factor.",
    degradation_estimate: "resize",
    reliability: "medium",
  },
];

export function mockResultForIndex(index: number): BackendResult {
  return structuredClone(SAMPLES[index % SAMPLES.length]);
}

export async function mockAnalyze(files: File[]): Promise<BackendResult[]> {
  await wait(1400 + Math.min(files.length, 4) * 380);
  return files.map((_, index) => mockResultForIndex(index));
}

function wait(ms: number) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}
