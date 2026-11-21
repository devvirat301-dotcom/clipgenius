"""YouTube URL validation, metadata lookup and (optional, gated) download.

Authorised methods used here:
  * URL parsing: local, no network.
  * Metadata (title, channel, thumbnail): YouTube's public oEmbed endpoint.
  * Duration: read via yt-dlp in *metadata-only* mode if it is installed.
  * Downloading the video file: OFF by default. See README "Legal & ToS".
    Only enable it for videos you own or have permission to use.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.errors import DownloadError, InvalidURLError, VideoUnavailableError

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtube-nocookie.com", "www.youtube-nocookie.com", "youtu.be", "www.youtu.be",
}


@dataclass
class VideoInfo:
    video_id: str
    url: str
    title: str
    channel: str
    thumbnail: str
    duration: float | None = None  # seconds; None when unknown


def extract_video_id(url: str) -> str:
    """Return the 11-character video id or raise InvalidURLError."""
    url = (url or "").strip()
    if not url:
        raise InvalidURLError("Please paste a YouTube link.")
    if "://" not in url:
        url = "https://" + url
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in _HOSTS:
        raise InvalidURLError("That doesn't look like a YouTube link (youtube.com or youtu.be).")

    candidate = None
    parts = [p for p in parsed.path.split("/") if p]
    if host.endswith("youtu.be"):
        candidate = parts[0] if parts else None
    elif parsed.path == "/watch":
        candidate = urllib.parse.parse_qs(parsed.query).get("v", [None])[0]
    elif len(parts) >= 2 and parts[0] in {"shorts", "embed", "live", "v"}:
        candidate = parts[1]

    if not candidate or not _ID_RE.match(candidate):
        raise InvalidURLError(
            "Couldn't find a video id in that link. Playlists and channel pages aren't supported."
        )
    return candidate


def canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def fetch_video_info(url: str, timeout: float = 10.0) -> VideoInfo:
    video_id = extract_video_id(url)
    watch_url = canonical_url(video_id)
    oembed = "https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(watch_url, safe="")
    try:
        with urllib.request.urlopen(oembed, timeout=timeout) as resp:  # noqa: S310 (fixed https host)
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise VideoUnavailableError(
                "This video is private or has embedding disabled, so its details can't be read."
            ) from None
        if e.code == 404:
            raise VideoUnavailableError("Video not found. It may be deleted or the link is wrong.") from None
        raise VideoUnavailableError(f"YouTube returned an error ({e.code}). Try again later.") from None
    except (urllib.error.URLError, TimeoutError):
        raise VideoUnavailableError(
            "Couldn't reach YouTube. Check your internet connection."
        ) from None

    return VideoInfo(
        video_id=video_id,
        url=watch_url,
        title=data.get("title", "Untitled"),
        channel=data.get("author_name", "Unknown channel"),
        thumbnail=data.get("thumbnail_url", f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"),
        duration=_try_duration(watch_url),
    )


def _try_duration(watch_url: str) -> float | None:
    try:
        import yt_dlp
    except ImportError:
        return None
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
            info = ydl.extract_info(watch_url, download=False)
        return float(info["duration"]) if info.get("duration") else None
    except Exception:  # metadata only; never block the UI on this
        return None


def download_video(
    url: str, dest_dir: Path, progress: Callable[[float, str], None] | None = None,
    max_height: int = 720,
) -> Path:
    """Download with yt-dlp. Callers MUST have confirmed the user has permission."""
    try:
        import yt_dlp
    except ImportError:
        raise DownloadError("yt-dlp is not installed. Run: pip install yt-dlp") from None

    video_id = extract_video_id(url)
    dest_dir.mkdir(parents=True, exist_ok=True)

    def hook(d: dict) -> None:
        if progress and d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if total:
                progress(d.get("downloaded_bytes", 0) / total, "Downloading video…")

    opts = {
        "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(dest_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([canonical_url(video_id)])
    except Exception as e:
        raise DownloadError(
            "Couldn't download this video (it may be private, age-restricted, live, or blocked). "
            f"Try uploading the video file instead.\nDetails: {str(e)[:200]}"
        ) from None

    files = sorted(dest_dir.glob("source.*"))
    if not files:
        raise DownloadError("Download finished but no file was produced.")
    return files[0]
