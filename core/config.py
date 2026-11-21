"""Central configuration. All secrets come from environment variables."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:  # optional convenience: load a local .env file if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

NUM_CLIPS = 10
MIN_CLIP_SECONDS = 20
MAX_CLIP_SECONDS = 60

# Output sizes (kept modest so it runs on a school computer).
ASPECT_SIZES = {
    "9:16": (720, 1280),
    "1:1": (720, 720),
    "16:9": (1280, 720),
}

SCORE_WEIGHTS = {
    "entertainment": 0.25,
    "interest": 0.20,
    "emotion": 0.20,
    "hook": 0.20,
    "engagement": 0.15,
}


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str
    llm_model: str
    whisper_model: str
    whisper_compute_type: str
    work_dir: Path
    enable_url_download: bool

    @property
    def has_llm(self) -> bool:
        return bool(self.anthropic_api_key)


def load_settings() -> Settings:
    work_dir = Path(os.getenv("CLIPGENIUS_WORK_DIR", Path(tempfile.gettempdir()) / "clipgenius"))
    work_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        llm_model=os.getenv("CLIPGENIUS_LLM_MODEL", "claude-sonnet-5-5"),
        whisper_model=os.getenv("WHISPER_MODEL", "base"),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        work_dir=work_dir,
        enable_url_download=_flag("ENABLE_URL_DOWNLOAD", False),
    )
