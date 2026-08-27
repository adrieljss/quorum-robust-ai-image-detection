# Agent notes — Quorum frontend

You are working in `frontend/` only unless the user also asks to change `docs/FRONTEND.md`. Install npm packages in this directory. The app is started with `npm run dev` from `frontend/`.

## Product

Quorum is an image-assessment UI. Users upload one or more images. Each image is sent to a Flask API that sits in front of four ML branches (general, face, text, regularity). Results are shown per image. Multiple images use a slideshow (`components/AnalysisSlideshow.tsx`).

This frontend currently **mocks** those results. The ML and Flask layers are another agent's job.

## Stack

Next.js 15 App Router, React 19, Tailwind CSS 4 (`@import "tailwindcss"` in `app/globals.css`), `lucide-react` icons. No extra UI kit.

## Integration (read this first)

1. Contract and field visibility: `../docs/FRONTEND.md`
2. Types: `lib/types.ts` (`BackendResult`)
3. **The only fetch / mock switch:** `lib/api.ts` → `USE_MOCK_ANALYSIS`
4. Proxy: `next.config.ts` rewrites `/api/:path*` to Flask (`API_PROXY_TARGET` or `http://localhost:5000`)

When wiring the backend:

- Set `USE_MOCK_ANALYSIS` to `false` in `lib/api.ts`.
- Keep `FormData` field name `images`.
- Expect one result per file, same order, scores in `[0, 1]`.
- Do not implement Flask here.

## Where things live

| Concern | File |
|---|---|
| Landing | `app/page.tsx` |
| Workspace route | `app/assess/page.tsx` |
| Upload, analyze, states | `components/AssessWorkspace.tsx` |
| Drop zone | `components/UploadDropzone.tsx` |
| Slideshow | `components/AnalysisSlideshow.tsx` |
| Verdict + provenance + explanation | `components/ResultPanel.tsx` |
| Four signal bars | `components/SignalBars.tsx` |
| Mock payloads | `lib/mock.ts` |

## Design constraints

Warm palette: paper `#f7f1e8`, cream, sand, clay `#c45a22`, umber `#2c1c13`. Generous spacing, rounded-3xl cards, light shadows. No glassmorphism, neon, heavy gradients, or generic “AI startup” chrome. Keep empty, loading, error, and result states equally considered.

`degradation_estimate` is accepted in JSON but not rendered. Do not add it unless the user asks.

## Do not

- Move `package.json` / `node_modules` to the repo root.
- Train models or add Python in this folder.
- Break the 1:1 upload-to-result ordering.
- Replace the landing + workspace split without a product reason.
