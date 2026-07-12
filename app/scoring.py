"""
Pronunciation scoring engine.

Design (reference-free, since the learner speaks freely rather than reading
a fixed script):

  1. Transcribe with word-level timestamps + per-word confidence
     (faster-whisper gives an average log-probability per word).
  2. Estimate expected syllable count per word from CMUdict, compare against
     actual spoken duration -> flags words that were rushed / slurred
     relative to how long they *should* take to say clearly.
  3. Look up the "reference" pronunciation (ARPABET phonemes) for each word
     from CMUdict via g2p, so the learner can see exactly what the correct
     articulation looks like next to any flagged word.
  4. Combine (1) and (2) into a 0-100 per-word score; an overall score is the
     duration-weighted average of the per-word scores.

This is a deliberate, explainable proxy for GOP (Goodness of Pronunciation)
scoring that runs entirely on CPU with small open models -- no phoneme-level
acoustic model needed. See ARCHITECTURE.md for the trade-off discussion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from g2p_en import G2p

_g2p = G2p()

# CMUdict vowel phonemes carry a stress digit (0/1/2) as their last char.
_VOWEL_RE = re.compile(r"[AEIOU][A-Z]*\d$")

# Typical comfortable speaking duration per syllable, in seconds, for a
# clearly-articulated word. Tunable; sourced from average English speech
# rate literature (~3-4 syllables/sec for fluent, unhurried speech).
SECONDS_PER_SYLLABLE = 0.22

# Below this per-word average log-probability, flag as "unclear".
CONFIDENCE_FLOOR = -0.55

# If actual duration is below this fraction of expected duration, flag as
# "rushed" (a common marker of slurred/mispronounced words).
RUSHED_RATIO = 0.45


@dataclass
class WordResult:
    word: str
    start: float
    end: float
    confidence: float          # avg log-prob from ASR, roughly -1 (bad) .. 0 (great)
    expected_phonemes: str      # ARPABET, e.g. "DH AH0"
    syllables_expected: int
    score: float                # 0-100
    issue: Optional[str] = None  # None | "unclear" | "rushed" | "mispronounced"
    note: str = ""


@dataclass
class AnalysisResult:
    transcript: str
    overall_score: float
    duration_seconds: float
    words: list[WordResult] = field(default_factory=list)


def _phonemes_for_word(word: str) -> tuple[str, int]:
    """Return (ARPABET string, syllable count) for a single word using g2p_en,
    which falls back to a CMUdict lookup internally."""
    clean = re.sub(r"[^A-Za-z']", "", word)
    if not clean:
        return "", 0
    phones = [p for p in _g2p(clean) if p != " "]
    syllables = sum(1 for p in phones if _VOWEL_RE.search(p))
    return " ".join(phones), max(syllables, 1)


def _score_word(confidence: float, expected_secs: float, actual_secs: float) -> tuple[float, Optional[str], str]:
    # Confidence component: map log-prob roughly in [-1.2, 0] -> [0, 100]
    conf_score = max(0.0, min(100.0, (confidence + 1.2) / 1.2 * 100))

    # Rhythm component: penalize words spoken much faster than they can be
    # clearly articulated (a strong slurred/mumbled signal); don't penalize
    # slow, careful speech.
    ratio = actual_secs / expected_secs if expected_secs > 0 else 1.0
    rhythm_score = 100.0 if ratio >= RUSHED_RATIO else max(0.0, ratio / RUSHED_RATIO * 100)

    score = 0.7 * conf_score + 0.3 * rhythm_score

    issue = None
    note = ""
    if confidence < CONFIDENCE_FLOOR and ratio < RUSHED_RATIO:
        issue = "mispronounced"
        note = "Low recognition confidence and unusually short duration -- likely misarticulated."
    elif confidence < CONFIDENCE_FLOOR:
        issue = "unclear"
        note = "The recognizer struggled to confidently match this word to any English word."
    elif ratio < RUSHED_RATIO:
        issue = "rushed"
        note = "Spoken much faster than a clear articulation of this word typically takes."

    return round(score, 1), issue, note


def analyze_segments(segments: list, duration_seconds: float) -> AnalysisResult:
    """segments: faster-whisper Segment objects with .words (word_timestamps=True)."""
    words: list[WordResult] = []
    transcript_parts: list[str] = []

    for seg in segments:
        seg_words = getattr(seg, "words", None) or []
        for w in seg_words:
            token = w.word.strip()
            if not token:
                continue
            transcript_parts.append(token)

            phonemes, syll_count = _phonemes_for_word(token)
            expected_secs = syll_count * SECONDS_PER_SYLLABLE
            actual_secs = max(w.end - w.start, 0.01)

            confidence = w.probability if hasattr(w, "probability") else 0.0
            # faster-whisper gives probability in [0,1]; convert to a
            # log-prob-like scale consistent with CONFIDENCE_FLOOR tuning.
            import math
            log_conf = math.log(max(confidence, 1e-4))

            score, issue, note = _score_word(log_conf, expected_secs, actual_secs)

            words.append(
                WordResult(
                    word=token,
                    start=round(w.start, 2),
                    end=round(w.end, 2),
                    confidence=round(confidence, 3),
                    expected_phonemes=phonemes,
                    syllables_expected=syll_count,
                    score=score,
                    issue=issue,
                    note=note,
                )
            )

    if words:
        total_dur = sum(max(w.end - w.start, 0.01) for w in words)
        overall = sum(w.score * max(w.end - w.start, 0.01) for w in words) / total_dur
    else:
        overall = 0.0

    return AnalysisResult(
        transcript=" ".join(transcript_parts),
        overall_score=round(overall, 1),
        duration_seconds=round(duration_seconds, 2),
        words=words,
    )
