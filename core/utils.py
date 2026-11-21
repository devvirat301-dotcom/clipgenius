"""Small helpers: formatting, subprocess wrapper, temp-file cleanup."""
from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

from core.errors import FFmpegError


def format_ts(seconds: float) -> str:
    total = int(round(max(seconds, 0)))
    m, s = divmod(total, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def safe_filename(name: str, max_len: int = 60) -> str:
    cleaned = re.sub(r"[^\w\- ]+", "", name, flags=re.UNICODE).strip().replace(" ", "_")
    return (cleaned or "clip")[:max_len]


def run_command(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a command, return stdout, raise FFmpegError with the useful tail of stderr."""
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
    except FileNotFoundError:
        raise FFmpegError(
            f"'{cmd[0]}' was not found. Install FFmpeg and make sure it is on your PATH."
        ) from None
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-6:])
        raise FFmpegError(f"{cmd[0]} failed:\n{tail}")
    return proc.stdout


def probe_duration(path: Path) -> float:
    out = run_command(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ]
    )
    try:
        return float(out.strip())
    except ValueError:
        raise FFmpegError("Could not read the video duration. Is the file a valid video?") from None


def cleanup_old_jobs(root: Path, max_age_hours: float = 6) -> int:
    """Delete job folders under `root` older than max_age_hours. Returns how many were removed."""
    removed = 0
    cutoff = time.time() - max_age_hours * 3600
    for child in root.glob("job_*"):
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed
