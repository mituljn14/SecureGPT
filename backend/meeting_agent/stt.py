"""
Whisper STT Node
----------------
Uses the local `openai-whisper` library (runs 100 % on device — no cloud calls).

Model selection via env var WHISPER_MODEL (default: "base").
  tiny   ~39 MB  — fastest, CPU-friendly
  base   ~74 MB  — good accuracy/speed balance  ← default
  small  ~244 MB — better accuracy
  medium ~769 MB — near-SOTA accuracy
"""
import os
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")

# Lazy-load so the model is only downloaded once on first use
_whisper_model = None
_ffmpeg_ready = False


def _ensure_ffmpeg() -> bool:
    """Make sure a binary literally named `ffmpeg` is resolvable on PATH.

    whisper.load_audio() shells out to `ffmpeg` by name, so a bundled copy
    under a versioned filename is not enough on its own. This machine has no
    system ffmpeg, so rather than requiring a system-wide install we reuse
    the binary shipped inside the venv by imageio-ffmpeg: copied once to
    meeting_agent/bin/ffmpeg.exe and prepended to PATH for this process.

    Returns False if no ffmpeg can be arranged, so the caller can raise a
    clear error instead of whisper dying inside a subprocess.
    """
    global _ffmpeg_ready
    if _ffmpeg_ready or shutil.which("ffmpeg"):
        _ffmpeg_ready = True
        return True

    try:
        import imageio_ffmpeg
        src = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:
        logger.warning(f"[STT] no bundled ffmpeg available: {exc}")
        return False

    if not src.exists():
        return False

    bin_dir = Path(__file__).resolve().parent / "bin"
    bin_dir.mkdir(exist_ok=True)
    dst = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not dst.exists():
        shutil.copy2(src, dst)
        logger.info(f"[STT] staged bundled ffmpeg at {dst}")

    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
    _ffmpeg_ready = shutil.which("ffmpeg") is not None
    return _ffmpeg_ready


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        # Whisper decodes audio by invoking ffmpeg, so a missing binary
        # surfaces as an opaque subprocess failure. Check it up front.
        if not _ensure_ffmpeg():
            raise RuntimeError(
                "ffmpeg was not found and no bundled copy could be staged. "
                "Install it system-wide, or `pip install imageio-ffmpeg` "
                "into this venv."
            )
        try:
            import whisper  # openai-whisper
            logger.info(f"[STT] Loading Whisper model: {WHISPER_MODEL_NAME}")
            _whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
            logger.info("[STT] Model loaded successfully.")
        except ImportError:
            raise RuntimeError(
                "openai-whisper is not installed. "
                "Run: pip install openai-whisper"
            )
    return _whisper_model


def whisper_stt_node(state: dict) -> dict:
    """
    LangGraph node: transcribe the audio file at state['audio_path'].

    Sets:
        state['raw_transcript'] — plain text transcript
        state['error']          — error message on failure (pipeline continues)
    """
    audio_path = state.get("audio_path", "")
    if not audio_path or not os.path.exists(audio_path):
        state["raw_transcript"] = ""
        state["error"] = f"Audio file not found: {audio_path}"
        logger.error(state["error"])
        return state

    try:
        model = _get_model()
        logger.info(f"[STT] Transcribing: {audio_path}")
        result = model.transcribe(audio_path, fp16=False)
        transcript = result.get("text", "").strip()
        state["raw_transcript"] = transcript
        logger.info(f"[STT] Transcription complete — {len(transcript)} chars.")
    except Exception as exc:
        state["raw_transcript"] = ""
        state["error"] = f"Whisper transcription failed: {exc}"
        logger.error(state["error"])

    return state
