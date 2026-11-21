"""Cut a clip with FFmpeg, reframe to the chosen aspect ratio, and burn in subtitles."""
from __future__ import annotations

import shutil
from pathlib import Path

from core.config import ASPECT_SIZES
from core.errors import ClipGeniusError
from core.utils import run_command


def build_filter(width: int, height: int, subtitle_file: str | None) -> str:
    """Blurred-background reframe: the full picture is kept (no cropping of faces),
    centred over a blurred, zoomed copy of itself that fills the frame."""
    graph = (
        "[0:v]split=2[bgsrc][fgsrc];"
        f"[bgsrc]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},boxblur=20:5[bg];"
        f"[fgsrc]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p"
    )
    if subtitle_file:
        graph += f",ass={subtitle_file}"
    return graph + "[v]"


def render_clip(
    source: Path,
    out_path: Path,
    start: float,
    end: float,
    aspect: str,
    ass_text: str | None = None,
) -> Path:
    if aspect not in ASPECT_SIZES:
        raise ClipGeniusError(f"Unsupported aspect ratio: {aspect}")
    if end - start < 1:
        raise ClipGeniusError("A clip must be at least 1 second long.")

    width, height = ASPECT_SIZES[aspect]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = out_path.parent / f".work_{out_path.stem}"
    work.mkdir(exist_ok=True)
    try:
        sub_name = None
        if ass_text:
            # Run FFmpeg inside `work` and refer to the file by bare name:
            # avoids all path-escaping problems in filter strings (Windows drive letters etc.).
            (work / "subs.ass").write_text(ass_text, encoding="utf-8")
            sub_name = "subs.ass"

        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-ss", f"{start:.3f}", "-i", str(source.resolve()), "-t", f"{end - start:.3f}",
            "-filter_complex", build_filter(width, height, sub_name),
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
            str(out_path.resolve()),
        ]
        run_command(cmd, cwd=work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out_path
