import Link from "next/link";

type SiteHeaderProps = {
  current?: "home" | "assess";
};

export function SiteHeader({ current = "home" }: SiteHeaderProps) {
  return (
    <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-5 py-6 sm:px-8">
      <Link href="/" className="group flex items-baseline gap-2">
        <span className="font-display text-[1.65rem] leading-none tracking-tight text-umber">
          Quorum
        </span>
        <span className="hidden text-[0.7rem] uppercase tracking-[0.18em] text-taupe sm:inline">
          Image assessment
        </span>
      </Link>
      <nav className="flex items-center gap-2 text-sm">
        <Link
          href="/"
          className={`rounded-full px-3 py-1.5 transition-colors ${
            current === "home" ? "text-umber" : "text-taupe hover:text-umber"
          }`}
        >
          Overview
        </Link>
        <Link
          href="/assess"
          className={`rounded-full px-4 py-2 font-medium transition-colors ${
            current === "assess"
              ? "bg-umber text-paper"
              : "bg-cream text-bark hover:bg-sand"
          }`}
        >
          Open workspace
        </Link>
      </nav>
    </header>
  );
}
