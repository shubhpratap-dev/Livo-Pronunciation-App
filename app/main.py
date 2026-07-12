"""
Livo AI SWE Assessment -- Pronunciation Scoring App
Entry point: FastAPI app that serves the static frontend and exposes
POST /api/analyze for audio upload + scoring.

Privacy-by-design (see ARCHITECTURE.md / DPDP section):
  - Uploaded audio is written to a temp file only for the duration of the
    request, inside a per-request temp directory, and is deleted in a
    `finally` block regardless of success or failure.
  - No audio bytes, transcripts, or scores are persisted to any database or
    log. Nothing about the request outlives the HTTP response.
"""

import logging
import os
import shutil
import tempfile
import time
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.utils import mediainfo

from .scoring import analyze_segments

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("livo-pronunciation")

MIN_DURATION_S = 30
MAX_DURATION_S = 45
MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15MB safety cap regardless of duration
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base.en")

app = FastAPI(title="Livo AI Pronunciation Scorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # single-origin deployment (frontend served by this same app);
    allow_methods=["*"],  # loosened here only because there is no auth/session state to protect
    allow_headers=["*"],
)

log.info("Loading faster-whisper model '%s' (CPU, int8)...", MODEL_SIZE)
_model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
log.info("Model loaded.")


@app.get("/api/health")
def health():
    return {"status": "ok", "model": MODEL_SIZE}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    request_id = uuid.uuid4().hex[:8]
    tmp_dir = tempfile.mkdtemp(prefix=f"livo-{request_id}-")
    raw_path = os.path.join(tmp_dir, "upload")
    wav_path = os.path.join(tmp_dir, "audio.wav")

    try:
        size = 0
        with open(raw_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "File too large (max 15MB).")
                f.write(chunk)

        if size == 0:
            raise HTTPException(400, "Empty file upload.")

        # Decode to a normalized wav (mono, 16kHz) regardless of input
        # container/codec (webm/mp3/m4a/wav/...).
        try:
            audio = AudioSegment.from_file(raw_path)
        except Exception as e:
            log.warning("[%s] decode failed: %s", request_id, e)
            raise HTTPException(400, "Could not decode audio file. Supported: wav, mp3, m4a, webm, ogg.")

        duration_s = len(audio) / 1000.0
        if duration_s < MIN_DURATION_S or duration_s > MAX_DURATION_S:
            raise HTTPException(
                400,
                f"Audio must be {MIN_DURATION_S}-{MAX_DURATION_S} seconds long "
                f"(got {duration_s:.1f}s).",
            )

        audio.set_frame_rate(16000).set_channels(1).export(wav_path, format="wav")

        segments, info = _model.transcribe(
            wav_path,
            language="en",
            word_timestamps=True,
            vad_filter=True,
        )
        segments = list(segments)  # materialize generator

        if info.language != "en" and info.language_probability > 0.6:
            raise HTTPException(400, "Detected non-English speech. This tool scores English pronunciation only.")

        result = analyze_segments(segments, duration_s)

        if not result.words:
            raise HTTPException(422, "No speech detected in the recording. Please try again.")

        return {
            "request_id": request_id,
            "transcript": result.transcript,
            "overall_score": result.overall_score,
            "duration_seconds": result.duration_seconds,
            "words": [w.__dict__ for w in result.words],
        }

    finally:
        # Hard privacy guarantee: nothing from this request survives past
        # the response, success or failure.
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --- Static frontend (single-origin deployment: one URL for everything) ---
STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "static")


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
