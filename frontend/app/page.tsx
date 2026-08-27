import Link from "next/link";
import { SiteHeader } from "@/components/SiteHeader";

export default function HomePage() {
  return (
    <div className="min-h-screen">
      <SiteHeader current="home" />
      <main className="mx-auto w-full max-w-6xl px-5 pb-20 sm:px-8">
        <section className="grid items-end gap-12 pb-16 pt-6 lg:grid-cols-[1.15fr_0.85fr] lg:gap-16 lg:pt-10">
          <div className="animate-rise">
            <p className="text-xs uppercase tracking-[0.2em] text-taupe">For photographs</p>
            <h1 className="font-display mt-4 max-w-xl text-[2.7rem] leading-[1.08] text-umber sm:text-6xl">
              A careful reading of an image.
            </h1>
            <p className="mt-6 max-w-lg text-[1.08rem] leading-relaxed text-bark">
              Quorum looks at each file on its own terms — overall structure, faces,
              lettering, and repeating patterns — then offers a plain-language verdict.
              It is meant to be used slowly, with the photograph still in view.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link
                href="/assess"
                className="rounded-full bg-clay px-6 py-3 text-sm font-medium text-paper transition-colors hover:bg-clay-dark"
              >
                Open the workspace
              </Link>
              <p className="text-sm text-taupe">One image or a small batch. No account needed.</p>
            </div>
          </div>

          <aside
            className="animate-rise rounded-3xl bg-card p-5 shadow-[0_22px_50px_-32px_rgba(44,28,19,0.45)] sm:p-6"
            style={{ animationDelay: "80ms" }}
          >
            <p className="text-xs uppercase tracking-[0.16em] text-taupe">Example reading</p>
            <h2 className="font-display mt-3 text-2xl text-umber">Likely machine-generated</h2>
            <p className="mt-2 text-sm leading-relaxed text-bark">
              Catchlights in the eyes are inconsistent, and background signage does not
              resolve into readable letterforms.
            </p>
            <div className="mt-5 space-y-3">
              {EXAMPLE_SIGNALS.map((row) => (
                <div key={row.label}>
                  <div className="mb-1 flex justify-between text-xs text-taupe">
                    <span>{row.label}</span>
                    <span className="text-bark">{row.value}</span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-cream">
                    <div
                      className="h-full rounded-full bg-clay"
                      style={{ width: `${row.value}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </aside>
        </section>

        <section className="border-t border-line pt-12">
          <p className="text-xs uppercase tracking-[0.18em] text-taupe">Four readings</p>
          <div className="mt-6 grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
            {FEATURES.map((feature) => (
              <article key={feature.title}>
                <h3 className="font-display text-xl text-umber">{feature.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-bark">{feature.body}</p>
              </article>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

const EXAMPLE_SIGNALS = [
  { label: "General", value: 91 },
  { label: "Face", value: 83 },
  { label: "Text", value: 62 },
  { label: "Regularity", value: 40 },
];

const FEATURES = [
  {
    title: "General",
    body: "Looks at the whole frame: lighting, texture, and whether the photograph behaves like a camera capture.",
  },
  {
    title: "Face",
    body: "When a face is present, alignment and lighting are checked on their own, apart from the rest of the scene.",
  },
  {
    title: "Text",
    body: "Lettering on signs, shirts, and screens is a frequent giveaway. Quorum reads those regions separately.",
  },
  {
    title: "Regularity",
    body: "Repeating structure and spectral traces that often remain after an image has been compressed or resized.",
  },
];
