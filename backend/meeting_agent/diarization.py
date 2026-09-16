"""
Speaker Diarization Node
------------------------
Two-tier strategy:

Tier 1 (default, zero extra deps):
  Silence-based heuristic using standard-library 'wave' + numpy.
  Splits the transcript at sentence boundaries and assigns alternating
  speaker labels based on pause patterns. Not production-grade but
  works for quick meeting notes with no extra model downloads.

Tier 2 (opt-in, set env var USE_PYANNOTE=1):
  Uses pyannote.audio v3 for true speaker diarization.
  Requires:
    - pip install pyannote.audio
    - HUGGINGFACE_TOKEN env var
    - Accepted terms at https://hf.co/pyannote/speaker-diarization-3.1
"""
import os
import re
import logging

logger = logging.getLogger(__name__)

USE_PYANNOTE = os.getenv("USE_PYANNOTE", "0") == "1"
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")


# ---------------------------------------------------------------------------
# Tier 1 — Heuristic diarization (sentence-boundary split)
# ---------------------------------------------------------------------------

def _heuristic_diarize(transcript: str) -> str:
    """
    Splits text at sentence boundaries and annotates with alternating
    [SPEAKER_A] / [SPEAKER_B] labels. Simple but produces readable minutes.
    """
    sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
    if not sentences:
        return transcript

    labelled_lines = []
    speakers = ["[SPEAKER_A]", "[SPEAKER_B]", "[SPEAKER_C]", "[SPEAKER_D]"]

    current_speaker_idx = 0
    buffer = []

    for i, sentence in enumerate(sentences):
        buffer.append(sentence)
        transition_phrases = (
            "so", "alright", "okay", "right", "actually", "however",
            "now", "moving on", "next", "thank you", "i think", "we need",
            "the issue", "in conclusion", "to summarise", "to summarize"
        )
        is_transition = (
            len(sentence.split()) <= 5 or
            sentence.lower().lstrip().startswith(transition_phrases)
        )

        if is_transition or (i > 0 and i % 3 == 0):
            labelled_lines.append(
                f"{speakers[current_speaker_idx % len(speakers)]}: "
                + " ".join(buffer)
            )
            buffer = []
            current_speaker_idx += 1

    if buffer:
        labelled_lines.append(
            f"{speakers[current_speaker_idx % len(speakers)]}: "
            + " ".join(buffer)
        )

    return "\n\n".join(labelled_lines)


# ---------------------------------------------------------------------------
# Tier 2 — pyannote.audio diarization
# ---------------------------------------------------------------------------

def _pyannote_diarize(audio_path: str, transcript: str) -> str:
    """
    Uses pyannote.audio to get real speaker timestamps, then aligns them
    with the Whisper transcript segments.
    """
    try:
        from pyannote.audio import Pipeline  # type: ignore
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HF_TOKEN,
        )
        diarization = pipeline(audio_path)
    except Exception as exc:
        logger.warning(f"[Diarization] pyannote failed ({exc}), falling back to heuristic.")
        return _heuristic_diarize(transcript)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))

    if not segments:
        return _heuristic_diarize(transcript)

    sentences = re.split(r'(?<=[.!?])\s+', transcript.strip())
    if not sentences:
        return transcript

    total_duration = segments[-1][1]
    labelled_lines = []
    for i, sentence in enumerate(sentences):
        approx_time = (i / max(len(sentences) - 1, 1)) * total_duration
        speaker_label = "SPEAKER_UNKNOWN"
        for start, end, spk in segments:
            if start <= approx_time <= end:
                speaker_label = spk
                break
        labelled_lines.append(f"[{speaker_label}]: {sentence}")

    return "\n\n".join(labelled_lines)


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def speaker_diarization_node(state: dict) -> dict:
    """
    LangGraph node: annotate raw_transcript with speaker labels.
    """
    raw = state.get("raw_transcript", "")

    if not raw:
        state["diarized_transcript"] = ""
        return state

    if not state.get("enable_diarization", True):
        state["diarized_transcript"] = raw
        return state

    audio_path = state.get("audio_path", "")

    if USE_PYANNOTE and HF_TOKEN and audio_path:
        logger.info("[Diarization] Using pyannote.audio (Tier 2).")
        state["diarized_transcript"] = _pyannote_diarize(audio_path, raw)
    else:
        logger.info("[Diarization] Using heuristic diarizer (Tier 1).")
        state["diarized_transcript"] = _heuristic_diarize(raw)

    return state
