"""Speech-to-text with faster-whisper (an efficient Whisper implementation)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from core.errors import TranscriptionError
from core.models import Segment, Word, segments_from_json, segments_to_json
from core.utils import run_command

ProgressFn = Callable[[float, str], None]


def extract_audio(video_path: Path, out_path: Path) -> Path:
    """16 kHz mono WAV: the format Whisper wants, and small enough for long videos."""
    run_command([
        "ffmpeg", "-y", "-v", "error", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_path),
    ])
    return out_path


def _file_key(path: Path, model: str) -> str:
    """Cheap cache key: size + a hash of the first and last MB."""
    size = path.stat().st_size
    h = hashlib.sha1(f"{size}:{model}".encode())
    with open(path, "rb") as f:
        h.update(f.read(1_000_000))
        if size > 2_000_000:
            f.seek(-1_000_000, 2)
            h.update(f.read(1_000_000))
    return h.hexdigest()[:16]


def _synthesize_words(text: str, start: float, end: float) -> list[Word]:
    """Fallback when the model returns no word timings: spread words by character length."""
    tokens = text.split()
    if not tokens:
        return []
    total_chars = sum(len(t) for t in tokens)
    t, words = start, []
    for tok in tokens:
        span = (end - start) * len(tok) / total_chars
        words.append(Word(t, t + span, tok))
        t += span
    return words


def transcribe(
    audio_path: Path,
    model_size: str = "base",
    compute_type: str = "int8",
    language: str | None = None,
    cache_dir: Path | None = None,
    progress: ProgressFn | None = None,
) -> tuple[list[Segment], str]:
    """Return (segments, detected_language). Results are cached on disk per audio file."""
    cache_file = None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"transcript_{_file_key(audio_path, model_size)}.json"
        if cache_file.exists():
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            return segments_from_json(data["segments"]), data["language"]

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise TranscriptionError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from None

    if progress:
        progress(0.0, f"Loading Whisper '{model_size}' model (first run downloads it)…")
    try:
        model = WhisperModel(model_size, device="auto", compute_type=compute_type)
        raw_segments, info = model.transcribe(
            str(audio_path), word_timestamps=True, vad_filter=True, language=language,
            condition_on_previous_text=False,
        )
        total = max(info.duration, 1.0)
        segments: list[Segment] = []
        for seg in raw_segments:  # generator: transcription happens as we iterate
            text = seg.text.strip()
            if not text:
                continue
            words = [
                Word(w.start, w.end, w.word.strip()) for w in (seg.words or []) if w.word.strip()
            ] or _synthesize_words(text, seg.start, seg.end)
            segments.append(Segment(seg.start, seg.end, text, words))
            if progress:
                progress(min(seg.end / total, 1.0), f"Transcribing… {int(seg.end / total * 100)}%")
    except TranscriptionError:
        raise
    except Exception as e:
        raise TranscriptionError(f"Transcription failed: {str(e)[:300]}") from None

    if not segments:
        raise TranscriptionError(
            "No speech was detected. ClipGenius works on videos with talking (conversations, "
            "interviews, podcasts, commentary)."
        )

    if cache_file:
        cache_file.write_text(
            json.dumps({"language": info.language, "segments": segments_to_json(segments)}),
            encoding="utf-8",
        )
    return segments, info.language
