# ✂️ ClipGenius AI

Turn a long video into the **10 most engaging short clips** (Shorts / Reels / TikTok):
transcribe → AI finds the best moments → FFmpeg cuts, reframes and subtitles them → preview, edit, download.

## 1. Architecture

```
 Video source ──► Audio (FFmpeg, 16 kHz mono)
 (upload or        │
  permitted URL)   ▼
                 Whisper (faster-whisper) ──► transcript with word timestamps
                                                   │
                  ┌────────────────────────────────┘
                  ▼
   Candidate builder: every 20–60 s window that starts/ends at a sentence end or pause
                  ▼
   Local keyword scorer (free) ──► shortlist of ~30
                  ▼
   LLM (ONE call) scores 5 criteria + writes titles   [falls back to keyword scorer]
                  ▼
   Selection: best 10, no overlapping / repetitive moments
                  ▼
   FFmpeg: cut → 9:16 / 1:1 / 16:9 (blurred-background fit) → burn ASS subtitles
                  ▼
   Streamlit dashboard: preview · edit · re-render · download · ZIP
```

Why it is cheap and fast on long videos: audio is extracted once, Whisper runs locally (free),
the LLM sees only a 30-candidate shortlist in a single request, and transcripts are cached per job.

## 2. Folder structure

```
clipgenius/
├── app.py                  Streamlit dashboard (UI only)
├── requirements.txt
├── .env.example            copy to .env and add your key
├── .streamlit/config.toml  dark theme + upload limit
├── core/
│   ├── config.py           settings from environment variables
│   ├── models.py           Word / Segment / Moment data classes
│   ├── errors.py           friendly error types
│   ├── utils.py            ffmpeg runner, time formatting, temp-file cleanup
│   ├── youtube.py          URL validation, metadata (oEmbed), gated download
│   ├── transcribe.py       audio extraction + Whisper
│   ├── moments.py          candidates, scoring, LLM ranking, de-duplication
│   ├── subtitles.py        ASS captions with highlighted keywords
│   ├── clipper.py          FFmpeg cutting / reframing / subtitle burn-in
│   └── pipeline.py         glues everything together, ZIP builder
└── tests/test_core.py      real tests (no API key or internet needed)
```

## 3. Install

1. **Python 3.10+** and **FFmpeg** (must be on your PATH; check with `ffmpeg -version`)
   - Windows: `winget install ffmpeg` · macOS: `brew install ffmpeg` · Linux: `sudo apt install ffmpeg`
2. ```bash
   cd clipgenius
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env               # then edit .env
   streamlit run app.py
   ```
3. The first run downloads the Whisper model (`base` ≈ 140 MB). A GPU is not required.

## 4. Configuration & limitations (please read)

| Item | What you need | If missing |
|---|---|---|
| **AI scoring** | `ANTHROPIC_API_KEY` in `.env` (console.anthropic.com; paid per use, ~one request per video) | Falls back to a keyword scorer. It works, but it is much less accurate at humour, emotion and hooks. The dashboard says which mode was used. |
| **Whisper** | nothing (runs locally) | — |
| **Video file** | An uploaded file, or a URL download you are permitted to make | See below |

### Legal & ToS
YouTube's Terms of Service do not allow downloading videos except through features YouTube provides
(for example, **YouTube Studio → Content → ⋮ → Download** for your *own* videos).
So by default ClipGenius reads only public **metadata** from a YouTube link (title, channel,
thumbnail) and asks you to **upload the video file**. Every run requires a checkbox confirming you own the video
or have permission.
Direct download via `yt-dlp` exists but is **disabled** unless `ENABLE_URL_DOWNLOAD=true`; use it only
for videos you own or that are licensed for reuse, and at your own responsibility.

### Other honest limits
- It needs **speech**: music-only or silent videos give no clips.
- Whisper `base` is fast but makes mistakes with accents/noise; use `small` for better accuracy.
- Clip boundaries snap to sentence ends/pauses, but you can fine-tune them in **Edit clip**.
- Rendering 10 clips takes roughly 1–3 minutes on a typical laptop.
- Temporary files live in your system temp folder (`clipgenius/job_*`), are deleted when you start a new
  job or press **Clean up temporary files**, and jobs older than 6 hours are removed at startup.

## 5. Tests

```bash
python -m unittest discover -s tests -v
```
Covers URL parsing, candidate rules (20–60 s, sentence boundaries), duplicate rejection,
subtitle generation, and real FFmpeg renders in all three aspect ratios.

## 6. Ideas for next versions
Face-tracking crop for 9:16, per-word karaoke animation, React + FastAPI front end with a job queue,
saving projects to a database, snap-to-sentence buttons in the editor.
