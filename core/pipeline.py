"""Orchestrates: source video -> audio -> transcript -> moments -> rendered clips."""
from __future__ import annotations

import io
import shutil
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from core import clipper, moments, subtitles, transcribe
from core.config import ASPECT_SIZES, NUM_CLIPS, Settings
from core.errors import ClipGeniusError
from core.models import Segment
from core.utils import probe_duration, safe_filename

ProgressFn = Callable[[float, str], None]
MIN_SECONDS_FOR_ANALYSIS = 20


@dataclass
class Clip:
    id: int
    title: str
    start: float
    end: float
    category: str
    reason: str
    scores: dict[str, float]
    overall: float
    aspect: str = "9:16"
    subtitles: bool = True
    path: Path | None = None
    version: int = 0


@dataclass
class Job:
    dir: Path
    video_path: Path
    duration: float
    segments: list[Segment]
    language: str
    clips: list[Clip] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    used_llm: bool = False

    @property
    def word_count(self) -> int:
        return sum(len(s.words) for s in self.segments)


def new_job_dir(settings: Settings) -> Path:
    d = settings.work_dir / f"job_{uuid.uuid4().hex[:10]}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cleanup_job(job_dir: Path | None) -> None:
    if job_dir:
        shutil.rmtree(job_dir, ignore_errors=True)


def _scaled(progress: ProgressFn | None, lo: float, hi: float) -> ProgressFn | None:
    if not progress:
        return None
    return lambda f, msg: progress(lo + (hi - lo) * max(0.0, min(f, 1.0)), msg)


def analyze(
    video_path: Path, job_dir: Path, settings: Settings,
    progress: ProgressFn | None = None, language: str | None = None,
) -> Job:
    """Transcribe the video and pick the best moments. Does not render yet."""
    say = progress or (lambda f, m: None)
    say(0.0, "Checking video…")
    duration = probe_duration(video_path)
    if duration < MIN_SECONDS_FOR_ANALYSIS:
        raise ClipGeniusError("The video is too short (under 20 seconds) to cut clips from.")

    say(0.05, "Extracting audio…")
    audio = transcribe.extract_audio(video_path, job_dir / "audio.wav")

    segments, lang = transcribe.transcribe(
        audio, settings.whisper_model, settings.whisper_compute_type, language,
        cache_dir=job_dir, progress=_scaled(progress, 0.10, 0.75),
    )
    audio.unlink(missing_ok=True)  # big and no longer needed

    say(0.78, "AI is finding the most engaging moments…" if settings.has_llm
        else "Scoring moments with the built-in scorer…")
    result = moments.detect_moments(
        segments, NUM_CLIPS, settings.anthropic_api_key, settings.llm_model
    )
    if not result.moments:
        raise ClipGeniusError("No usable moments were found. " + " ".join(result.warnings))

    # Present best-first, but keep the numbering stable.
    ordered = sorted(result.moments, key=lambda m: m.overall, reverse=True)
    clips = [
        Clip(i + 1, m.title, round(m.start, 2), round(m.end, 2), m.category, m.reason,
             m.scores, m.overall)
        for i, m in enumerate(ordered)
    ]
    return Job(job_dir, video_path, duration, segments, lang, clips, result.warnings, result.used_llm)



def render(job: Job, clip: Clip) -> Clip:
    """(Re)render one clip with its current settings."""
    if clip.start < 0 or clip.end > job.duration + 0.5:
        raise ClipGeniusError(
            f"Times must be within the video (0 - {job.duration:.0f}s)."
        )
    if not 3 <= clip.end - clip.start <= 180:
        raise ClipGeniusError("A clip must be between 3 and 180 seconds long.")

    ass = None
    if clip.subtitles:
        width, height = ASPECT_SIZES[clip.aspect]
        words = subtitles.words_in_range(job.segments, clip.start, clip.end)
        if words:
            ass = subtitles.build_ass(words, clip.start, clip.end, width, height, highlight=True)

    old = clip.path
    clip.version += 1
    out = job.dir / "clips" / f"clip_{clip.id:02d}_v{clip.version}.mp4"
    clipper.render_clip(job.video_path, out, clip.start, min(clip.end, job.duration), clip.aspect, ass)
    clip.path = out
    if old and old != out:
        old.unlink(missing_ok=True)
    return clip


def render_all(job: Job, progress: ProgressFn | None = None) -> None:
    total = len(job.clips)
    for i, clip in enumerate(job.clips):
        if progress:
            progress(i / total, f"Rendering clip {i + 1} of {total}: {clip.title}")
        render(job, clip)
    if progress:
        progress(1.0, "Done!")


def clip_filename(clip: Clip) -> str:
    return f"{clip.id:02d}_{safe_filename(clip.title)}_{clip.aspect.replace(':', 'x')}.mp4"


def build_zip(job: Job) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:  # mp4 is already compressed
        for clip in job.clips:
            if clip.path and clip.path.exists():
                zf.write(clip.path, clip_filename(clip))
    return buf.getvalue()
