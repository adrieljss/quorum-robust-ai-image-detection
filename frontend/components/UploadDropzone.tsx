"use client";

import { useRef, useState } from "react";
import { ImagePlus } from "lucide-react";

type UploadDropzoneProps = {
  onFiles: (files: File[]) => void;
  disabled?: boolean;
};

export function UploadDropzone({ onFiles, disabled }: UploadDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const take = (list: FileList | File[] | null) => {
    if (!list || disabled) return;
    onFiles(Array.from(list));
  };

  return (
    <div
      onDragOver={(event) => {
        event.preventDefault();
        if (!disabled) setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(event) => {
        event.preventDefault();
        setOver(false);
        take(event.dataTransfer.files);
      }}
      className={`rounded-3xl border border-dashed px-6 py-12 text-center transition-colors sm:px-10 sm:py-16 ${
        over ? "border-clay bg-cream" : "border-sand bg-card"
      } ${disabled ? "opacity-60" : ""}`}
    >
      <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-cream text-bark">
        <ImagePlus size={22} strokeWidth={1.6} />
      </div>
      <p className="font-display mt-5 text-2xl text-umber">Add one image, or several</p>
      <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-taupe">
        Drag files here, or choose them from your computer. JPEG, PNG, WebP, and GIF
        up to 15&nbsp;MB each.
      </p>
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        className="mt-6 rounded-full bg-umber px-5 py-2.5 text-sm font-medium text-paper transition-colors hover:bg-bark disabled:cursor-not-allowed"
      >
        Choose images
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/gif"
        multiple
        className="hidden"
        onChange={(event) => {
          take(event.target.files);
          event.target.value = "";
        }}
      />
    </div>
  );
}
