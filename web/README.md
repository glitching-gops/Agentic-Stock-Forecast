# ZeRO — web

The Next.js frontend. Deploys to Vercel; reads everything through the FastAPI
service on Render, never directly from Supabase, so the ranking and methodology
semantics have exactly one source of truth.

```bash
npm install
echo 'API_BASE_URL=https://your-service.onrender.com' > .env.local
npm run dev
```

## Why every fetch is server-side

`lib/api.ts` imports `server-only`, so nothing in it can be pulled into a client
bundle. Consequences worth knowing before changing it:

- **Visitors never wait on a cold start.** Render's free tier sleeps after 15
  minutes idle and takes ~82s to wake. Pages are static, so the CDN answers
  while Render is still asleep.
- **`ALLOWED_ORIGINS` needs no Vercel entry.** No browser origin calls the API.
- **The leaderboard is filtered and sorted in the browser** over the single
  95-row payload the server already fetched. Filter changes cost no network.

The one thing the client must not recompute is `rank`: the API issues it as a
SQL window function over the full filtered set, with ties sharing a rank.

## Freshness

Every page carries `revalidate = 3600` as a floor. The real refresh path is
`POST /api/revalidate` with an `x-revalidate-secret` header, called by the daily
GitHub Actions workflow once it has finished writing data — see the comment in
`app/api/revalidate/route.ts` for why the hourly schedule alone is a poor fit
for a backend that sleeps.

## Environment

| Variable | Where | Purpose |
|---|---|---|
| `API_BASE_URL` | Vercel + `.env.local` | FastAPI base URL. Server-side only. |
| `REVALIDATE_SECRET` | Vercel + GH Actions | Shared secret for the revalidate hook. |
