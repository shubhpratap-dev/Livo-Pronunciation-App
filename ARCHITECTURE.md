# System Architecture — Pronunciation Scoring App

*(See the submitted ARCHITECTURE.docx for the formatted 2-page version. This
file mirrors the same content for repo readers.)*

## Overview
Learner uploads 30–45s of English speech and gets back a transcript with
per-word pronunciation scoring — clear / rushed / unclear-or-mispronounced —
plus the correct reference pronunciation for flagged words. Scoring is
**reference-free**: there's no fixed script to align against, so the design
is built around that constraint.

## Components
| Component | Tech | Role |
|---|---|---|
| Frontend | Vanilla HTML/CSS/JS, no build step | Upload UI, client-side duration pre-check, renders highlighted transcript |
| API server | FastAPI + Uvicorn | Upload handling, validation, orchestration |
| Audio decode | pydub + ffmpeg | Normalizes any input to 16kHz mono WAV; measures true duration server-side |
| STT | faster-whisper (`base.en`, CTranslate2, CPU int8) | Transcription with word timestamps + per-word confidence |
| Pronunciation reference | g2p_en (CMUdict) | Expected ARPABET phonemes + syllable count per word |
| Scoring | Custom Python (`app/scoring.py`) | Combines ASR confidence + speech-rate into a 0–100 score |
| Hosting | Render (Docker web service) | One service serves both API and static frontend |

**Flow:** Browser → `POST /api/analyze` → decode/validate duration →
faster-whisper transcription → per-word scoring against g2p reference +
speech rate → JSON → browser renders color-coded transcript with tooltips.

## Models/APIs and why
- **faster-whisper over the OpenAI Whisper API**: runs entirely on the
  deployment's own server, so audio never leaves it — simplifies DPDP data
  transfer. No per-request cost. Trade-off: slightly less accurate than
  Whisper-large or the hosted API on accented/noisy speech.
- **g2p_en/CMUdict over a phoneme-CTC acoustic model** (e.g. wav2vec2-espeak):
  instant, deterministic, no GPU needed — fits a CPU-only starter tier.
  Trade-off: gives the *expected* pronunciation, not a direct acoustic
  measurement of what was actually produced — this is the biggest accuracy
  gap in the system (see Trade-offs).
- **No LLM in the scoring path**: nothing for it to reason over acoustically;
  it would just restate the ASR/timing signals in prose, or hallucinate.

## Scoring and highlighting
Per-word score = `0.7 × confidence_score + 0.3 × rhythm_score`
- **Confidence (70%)**: faster-whisper's per-word probability, the standard
  reference-free proxy for pronunciation clarity.
- **Rhythm (30%)**: expected duration (syllables from CMUdict × a fluent-speech
  constant) vs actual duration — flags rushed/slurred words even when ASR
  still happened to guess the word right.
- **Reference phonemes**: shown on hover, not scored — lets the learner see
  the correct articulation next to any flagged word.

Overall score = duration-weighted average across words, so long
mispronounced words count more than short clean filler words.

## DPDP compliance
- **In scope**: uploaded audio (biometric/personal data) + derived transcript.
  No accounts, no other identifiers collected.
- **Storage**: none. Audio lives in a per-request temp dir, deleted in a
  `finally` block regardless of success/failure. No DB, no server-side logs
  of audio/transcript/scores.
- **Retention**: zero — data doesn't outlive one HTTP request.
- **Consent**: on-screen notice that audio is processed only for the request
  and never stored. A production version handling minors or needing an audit
  trail would add explicit recorded consent per the Act's notice requirements.
- **Data residency**: no third-party STT API call, so the only residency
  decision is the hosting region — deploy to a Render region in/near India to
  avoid routing through a US-based inference API.
- **Deletion**: nothing persisted, nothing to delete. A future version with
  accounts/history would need an explicit erasure endpoint + defined retention.
- **Data Principal rights**: with no stored personal data, access/correction/
  erasure requests resolve to "nothing is held" — that confirmation itself
  needs to be a documented, fast response in production.

## Trade-offs and what's next
- Biggest gap: no direct acoustic phoneme comparison (true GOP scoring) —
  needs a GPU or much larger CPU model, not a fit for a starter tier.
- Whisper confidence can be miscalibrated for rare/uncommon words. Next:
  a second higher-beam-width decode pass for low-confidence words only.
- Speech-rate constant is currently fixed, not per-speaker. Next: calibrate
  from the speaker's own high-confidence words early in the recording.
- With another week: GPU-gated phoneme-CTC model, per-speaker rate
  calibration, and an opt-in "save last 5 attempts" feature — which would
  require actually solving the retention/consent/deletion questions this
  version currently avoids by storing nothing at all.
