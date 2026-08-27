import { mockAnalyze } from "./mock";
import type { AnalyzeResponse, BackendResult } from "./types";

/**
 * DEMO SWITCH
 * Set this to `false` once the Flask API is running.
 * When false, uploads are POSTed as multipart form data to `/api/analyze`
 * (rewritten to Flask by next.config.ts).
 */
export const USE_MOCK_ANALYSIS = true;

const ACCEPTED_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/gif",
  "image/jpg",
]);

export const MAX_FILE_BYTES = 15 * 1024 * 1024;

export function isAcceptedImage(file: File) {
  if (ACCEPTED_TYPES.has(file.type)) return true;
  return /\.(jpe?g|png|webp|gif)$/i.test(file.name);
}

export async function analyzeImages(files: File[]): Promise<BackendResult[]> {
  if (USE_MOCK_ANALYSIS) {
    return mockAnalyze(files);
  }

  const body = new FormData();
  files.forEach((file) => body.append("images", file));

  const response = await fetch("/api/analyze", {
    method: "POST",
    body,
  });

  if (!response.ok) {
    const detail = await safeError(response);
    throw new Error(detail || `Analysis failed (${response.status})`);
  }

  const payload = (await response.json()) as AnalyzeResponse | BackendResult[];
  const results = Array.isArray(payload) ? payload : payload.results;

  if (!Array.isArray(results) || results.length !== files.length) {
    throw new Error("The API did not return one result per uploaded image.");
  }

  return results;
}

async function safeError(response: Response) {
  try {
    const data = (await response.json()) as { error?: string; message?: string };
    return data.error || data.message || "";
  } catch {
    return "";
  }
}
