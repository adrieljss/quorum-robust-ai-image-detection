"use client";

import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { AnalysisSlideshow } from "@/components/AnalysisSlideshow";
import { SiteHeader } from "@/components/SiteHeader";
import { UploadDropzone } from "@/components/UploadDropzone";
import { analyzeImages, isAcceptedImage, MAX_FILE_BYTES } from "@/lib/api";
import type { UploadItem } from "@/lib/types";

type Phase = "ready" | "analyzing" | "results" | "error";

export function AssessWorkspace() {
  const [items, setItems] = useState<UploadItem[]>([]);
  const [phase, setPhase] = useState<Phase>("ready");
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState(0);

  useEffect(() => {
    return () => {
      items.forEach((item) => URL.revokeObjectURL(item.previewUrl));
    };
    // Revoke only on unmount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (phase !== "results" || items.length < 2) return;

    const onKey = (event: KeyboardEvent) => {
      if (event.key === "ArrowRight") {
        setActive((value) => (value + 1) % items.length);
      }
      if (event.key === "ArrowLeft") {
        setActive((value) => (value - 1 + items.length) % items.length);
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [phase, items.length]);

  const addFiles = (files: File[]) => {
    const accepted: UploadItem[] = [];
    let message: string | null = null;

    files.forEach((file) => {
      if (!isAcceptedImage(file)) {
        message = "Some files were skipped because they are not supported images.";
        return;
      }
      if (file.size > MAX_FILE_BYTES) {
        message = "Some files were skipped because they are larger than 15 MB.";
        return;
      }
      accepted.push({
        id: `${file.name}-${file.size}-${file.lastModified}-${crypto.randomUUID()}`,
        file,
        previewUrl: URL.createObjectURL(file),
        result: null,
        error: null,
      });
    });

    if (accepted.length) {
      setItems((current) => [...current, ...accepted]);
      setPhase("ready");
    }
    setError(message);
  };

  const removeItem = (id: string) => {
    setItems((current) => {
      const next = current.filter((item) => item.id !== id);
      const removed = current.find((item) => item.id === id);
      if (removed) URL.revokeObjectURL(removed.previewUrl);
      return next;
    });
  };

  const reset = () => {
    items.forEach((item) => URL.revokeObjectURL(item.previewUrl));
    setItems([]);
    setActive(0);
    setError(null);
    setPhase("ready");
  };

  const analyze = async () => {
    if (!items.length) return;
    setPhase("analyzing");
    setError(null);
    try {
      const results = await analyzeImages(items.map((item) => item.file));
      setItems((current) =>
        current.map((item, index) => ({
          ...item,
          result: results[index] ?? null,
          error: results[index] ? null : "No result was returned for this image.",
        })),
      );
      setActive(0);
      setPhase("results");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Analysis could not be completed.");
      setPhase("error");
    }
  };

  return (
    <div className="min-h-screen">
      <SiteHeader current="assess" />
      <main className="mx-auto w-full max-w-6xl px-5 pb-20 sm:px-8">
        <div className="max-w-2xl pb-8 pt-2">
          <p className="text-xs uppercase tracking-[0.18em] text-taupe">Workspace</p>
          <h1 className="font-display mt-2 text-4xl text-umber sm:text-5xl">
            Assess images
          </h1>
          <p className="mt-3 text-[1.02rem] leading-relaxed text-bark">
            Upload a photograph, or a small set. Each file is read on its own — general
            structure, faces, text, and regularity — then shown with a clear verdict.
          </p>
        </div>

        {phase === "results" ? (
          <div>
            <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm text-taupe">
                {items.length === 1
                  ? "One image assessed."
                  : `${items.length} images assessed. Use the strip or arrow keys to move between them.`}
              </p>
              <button
                type="button"
                onClick={reset}
                className="rounded-full bg-cream px-4 py-2 text-sm text-bark transition-colors hover:bg-sand"
              >
                New assessment
              </button>
            </div>
            <AnalysisSlideshow items={items} active={active} onActiveChange={setActive} />
          </div>
        ) : null}

        {phase === "analyzing" ? (
          <div className="animate-rise rounded-3xl bg-card px-6 py-10 shadow-[0_16px_40px_-30px_rgba(44,28,19,0.4)] sm:px-10">
            <p className="text-xs uppercase tracking-[0.18em] text-taupe">Reading files</p>
            <h2 className="font-display mt-2 text-3xl text-umber">Looking closely</h2>
            <p className="mt-2 max-w-lg text-sm leading-relaxed text-bark">
              Each image is passing through the general, face, text, and regularity
              branches. This usually takes a few seconds.
            </p>
            <ul className="mt-8 grid gap-3 sm:grid-cols-2">
              {items.map((item, index) => (
                <li
                  key={item.id}
                  className="flex items-center gap-3 rounded-2xl bg-cream/80 px-3 py-3"
                  style={{ animationDelay: `${index * 80}ms` }}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={item.previewUrl}
                    alt=""
                    className="h-14 w-14 rounded-xl object-cover"
                  />
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-umber">{item.file.name}</p>
                    <p className="mt-0.5 flex items-center gap-2 text-xs text-taupe">
                      <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-clay" />
                      Queued for assessment
                    </p>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        ) : null}

        {phase === "error" ? (
          <div className="animate-rise rounded-3xl border border-line bg-card px-6 py-10 sm:px-10">
            <p className="text-xs uppercase tracking-[0.18em] text-taupe">Could not finish</p>
            <h2 className="font-display mt-2 text-3xl text-umber">Something went wrong</h2>
            <p className="mt-3 max-w-lg text-sm leading-relaxed text-bark">
              {error}
            </p>
            <div className="mt-6 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={analyze}
                className="rounded-full bg-umber px-5 py-2.5 text-sm font-medium text-paper hover:bg-bark"
              >
                Try again
              </button>
              <button
                type="button"
                onClick={reset}
                className="rounded-full bg-cream px-5 py-2.5 text-sm text-bark hover:bg-sand"
              >
                Start over
              </button>
            </div>
          </div>
        ) : null}

        {phase === "ready" ? (
          <div className="space-y-6">
            <UploadDropzone onFiles={addFiles} />
            {error ? <p className="text-sm text-clay-dark">{error}</p> : null}
            {items.length ? (
              <div>
                <div className="mb-3 flex items-center justify-between">
                  <p className="text-sm text-bark">
                    {items.length} {items.length === 1 ? "image" : "images"} ready
                  </p>
                  <button
                    type="button"
                    onClick={reset}
                    className="text-sm text-taupe underline-offset-2 hover:text-umber hover:underline"
                  >
                    Clear all
                  </button>
                </div>
                <ul className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  {items.map((item) => (
                    <li key={item.id} className="group relative overflow-hidden rounded-2xl bg-cream">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={item.previewUrl}
                        alt={item.file.name}
                        className="aspect-square w-full object-cover"
                      />
                      <button
                        type="button"
                        onClick={() => removeItem(item.id)}
                        className="absolute right-2 top-2 rounded-full bg-card/90 p-1 text-bark opacity-100 shadow-sm transition-opacity sm:opacity-0 sm:group-hover:opacity-100"
                        aria-label={`Remove ${item.file.name}`}
                      >
                        <X size={14} />
                      </button>
                    </li>
                  ))}
                </ul>
                <button
                  type="button"
                  onClick={analyze}
                  className="mt-6 rounded-full bg-clay px-6 py-3 text-sm font-medium text-paper transition-colors hover:bg-clay-dark"
                >
                  Assess {items.length === 1 ? "this image" : "these images"}
                </button>
              </div>
            ) : (
              <p className="text-sm text-taupe">
                Nothing selected yet. Results stay on this page until you start a new
                assessment.
              </p>
            )}
          </div>
        ) : null}
      </main>
    </div>
  );
}
