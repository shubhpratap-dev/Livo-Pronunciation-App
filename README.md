# Say It Clearly — Pronunciation Scoring App

Built for the Livo AI SWE assessment. Upload 30–45s of English speech, get a
per-word pronunciation score and highlighted transcript.

See `ARCHITECTURE.md` (or the submitted `ARCHITECTURE.docx`) for the full
system design, model choices, scoring methodology, and DPDP compliance notes.

## Run locally

Requires Python 3.11+ and `ffmpeg` installed on your system.

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 — the same server serves both the frontend and
the API, so there's nothing else to configure.

First startup downloads the `base.en` faster-whisper model (~150MB) and a
small NLTK dataset for g2p — this needs internet access once, then it's
cached locally.

## Run with Docker

```bash
docker build -t livo-pronunciation .
docker run -p 8000:8000 livo-pronunciation
```

## Deploy (Render — recommended)

1. Push this folder to a new GitHub repo (see commands below).
2. Go to https://dashboard.render.com → **New** → **Blueprint**, and point it
   at your repo. Render will read `render.yaml` automatically and create a
   Docker web service named `livo-pronunciation-scorer`.
   - Alternatively: **New** → **Web Service** → connect the repo → environment
     **Docker** → it'll pick up the `Dockerfile` automatically.
3. Wait for the build (~5–8 min — it bakes the whisper model into the image
   at build time, see `Dockerfile`, so cold starts are fast).
4. Once live, open the Render URL — the frontend and API are both served
   from it directly.

### Push to GitHub

```bash
cd livo-pronunciation-app
git add -A
git commit -m "Livo AI SWE assessment: pronunciation scoring app"
gh repo create livo-pronunciation-app --public --source=. --push
# or, without gh CLI:
git remote add origin https://github.com/<you>/livo-pronunciation-app.git
git branch -M main
git push -u origin main
```

## Other hosts

Any Docker-friendly host works the same way (Fly.io, Railway):
- **Fly.io**: `fly launch` (it detects the Dockerfile), then `fly deploy`.
- **Railway**: New Project → Deploy from GitHub repo → it detects the
  Dockerfile automatically.

Avoid pure static hosts (Vercel/Netlify/Cloudflare Pages) for the *backend* —
this needs a persistent Python process to hold the whisper model in memory,
which serverless/edge functions on those platforms aren't built for. If you
want to use one of them anyway, deploy the frontend there as a static site
and point `API_BASE` in `static/index.html` at a separately-hosted backend
(Render/Fly/Railway).

## API

`POST /api/analyze` — multipart form field `file` (audio, 30–45s) →
JSON `{ transcript, overall_score, duration_seconds, words: [...] }`.

`GET /api/health` — liveness check for the hosting platform.
