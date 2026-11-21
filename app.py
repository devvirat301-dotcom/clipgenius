"""ClipGenius AI - Streamlit front end.  Run with:  streamlit run app.py"""
from __future__ import annotations

import importlib.util
import shutil
import traceback
from dataclasses import replace
from pathlib import Path

import streamlit as st

from core import pipeline, youtube
from core.config import ASPECT_SIZES, NUM_CLIPS, SCORE_WEIGHTS, load_settings
from core.errors import ClipGeniusError
from core.utils import cleanup_old_jobs, format_ts

st.set_page_config(page_title="ClipGenius AI", page_icon="✂️", layout="wide")
settings = load_settings()

# --------------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #1b1633 0%, #0b0b0f 55%); }
#MainMenu, footer { visibility: hidden; }
.logo { font-size: 2.3rem; font-weight: 800; letter-spacing: -0.03em;
  background: linear-gradient(90deg, #a78bfa, #60a5fa); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; margin-bottom: 0; }
.tagline { color: #9a9ab0; margin-top: -6px; margin-bottom: 1.2rem; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px; animation: fadeUp .45s ease both;
  transition: border-color .2s, transform .2s; }
div[data-testid="stVerticalBlockBorderWrapper"]:hover { border-color: #6d5bd0; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(10px);} to { opacity: 1; transform: none; } }
.stButton > button, .stDownloadButton > button { border-radius: 12px; font-weight: 600; transition: all .2s; }
.stButton > button[kind="primary"] { background: linear-gradient(90deg, #7c3aed, #3b82f6); border: 0; color: white; }
.stButton > button[kind="primary"]:hover { filter: brightness(1.12); transform: translateY(-1px); }
.badge { display: inline-block; padding: 4px 12px; border-radius: 999px; font-weight: 700; font-size: .95rem;
  background: linear-gradient(90deg, #7c3aed, #3b82f6); color: #fff; }
.chip { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem; color: #c4b5fd;
  border: 1px solid #3b3566; margin-right: 6px; }
.bar { display: flex; align-items: center; gap: 8px; font-size: .78rem; color: #9a9ab0; margin: 2px 0; }
.bar .lbl { width: 92px; } .bar .track { flex: 1; height: 6px; border-radius: 6px; background: #22222c; overflow: hidden; }
.bar .fill { height: 100%; background: linear-gradient(90deg, #7c3aed, #3b82f6); border-radius: 6px; }
.bar .val { width: 28px; text-align: right; color: #d0d0e0; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("job", None)
ss.setdefault("zip_bytes", None)
if "startup_cleanup" not in ss:
    cleanup_old_jobs(settings.work_dir, max_age_hours=6)
    ss.startup_cleanup = True


@st.cache_data(ttl=3600, show_spinner=False)
def cached_video_info(url: str):
    return youtube.fetch_video_info(url)


def reset_job() -> None:
    if ss.job:
        pipeline.cleanup_job(ss.job.dir)
    ss.job = None
    ss.zip_bytes = None


# --------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    whisper_choice = st.selectbox(
        "Whisper model", ["tiny", "base", "small", "medium"],
        index=["tiny", "base", "small", "medium"].index(settings.whisper_model)
        if settings.whisper_model in ["tiny", "base", "small", "medium"] else 1,
        help="Bigger = more accurate but slower. 'base' is a good start on a school PC.",
    )
    language = st.text_input("Spoken language code (optional)", placeholder="auto-detect, or e.g. en, hi, es")
    default_aspect = st.selectbox("Default format", list(ASPECT_SIZES), index=0)
    default_subs = st.toggle("Subtitles on by default", value=True)

    st.markdown("### 🩺 System check")
    checks = {
        "FFmpeg": shutil.which("ffmpeg") is not None,
        "faster-whisper": importlib.util.find_spec("faster_whisper") is not None,
        "AI scoring (API key)": settings.has_llm,
        "yt-dlp (optional)": importlib.util.find_spec("yt_dlp") is not None,
    }
    for name, ok in checks.items():
        st.write(("✅ " if ok else "⚠️ ") + name)
    if not settings.has_llm:
        st.caption("No ANTHROPIC_API_KEY found: the built-in keyword scorer will be used.")

    st.divider()
    if st.button("🗑️ Clean up temporary files", use_container_width=True):
        reset_job()
        st.rerun()

# ---------------------------------------------------------------------- header
st.markdown('<p class="logo">✂️ ClipGenius AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="tagline">Turn long videos into your 10 best Shorts, Reels &amp; TikToks.</p>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- input panel
with st.container(border=True):
    mode = st.radio("Video source", ["YouTube URL", "Upload a video file"], horizontal=True,
                    label_visibility="collapsed")
    url, upload, info, can_run = "", None, None, False

    if mode == "YouTube URL":
        url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=…",
                            label_visibility="collapsed")
        if url.strip():
            try:
                info = cached_video_info(url.strip())
            except ClipGeniusError as e:
                st.error(str(e))
        if info:
            c1, c2 = st.columns([1, 3])
            c1.image(info.thumbnail, use_container_width=True)
            c2.markdown(f"**{info.title}**")
            c2.caption(f"{info.channel}" + (f" · {format_ts(info.duration)}" if info.duration else ""))
            if settings.enable_url_download:
                ok = st.checkbox("I own this video or have the creator's permission to make clips from it.")
                can_run = ok
            else:
                st.info(
                    "Direct download from YouTube is **off** (YouTube's Terms of Service restrict it). "
                    "Download your own video from YouTube Studio, then switch to **Upload a video file** above. "
                    "Teachers/admins can enable URL download by setting `ENABLE_URL_DOWNLOAD=true` "
                    "(only for videos you own or have permission to use)."
                )
    else:
        upload = st.file_uploader(
            "Upload", type=["mp4", "mov", "mkv", "webm", "avi", "m4v"], label_visibility="collapsed"
        )
        if upload:
            ok = st.checkbox("I own this video or have permission to make clips from it.")
            can_run = ok

    run = st.button(f"✨ Generate {NUM_CLIPS} Clips", type="primary", disabled=not can_run,
                    use_container_width=True)

# ------------------------------------------------------------------ generation
if run:
    reset_job()
    job_dir = pipeline.new_job_dir(settings)
    settings_run = replace(settings, whisper_model=whisper_choice)
    bar = st.progress(0.0, text="Starting…")

    def progress(frac: float, msg: str) -> None:
        bar.progress(min(max(frac, 0.0), 1.0), text=msg)

    try:
        with st.status("AI analysis in progress…", expanded=True) as status:
            if mode == "YouTube URL":
                st.write("⬇️ Downloading video (permission confirmed)…")
                video = youtube.download_video(
                    info.url, job_dir, progress=lambda f, m: progress(f * 0.1, m)
                )
            else:
                st.write("📁 Saving uploaded video…")
                video = job_dir / f"source{Path(upload.name).suffix.lower()}"
                with open(video, "wb") as f:
                    shutil.copyfileobj(upload, f)

            st.write("🎙️ Transcribing speech with Whisper…")
            job = pipeline.analyze(video, job_dir, settings_run, lambda f, m: progress(0.1 + f * 0.6, m),
                                   language=language.strip() or None)
            for clip in job.clips:
                clip.aspect, clip.subtitles = default_aspect, default_subs

            st.write(f"🧠 Found {len(job.clips)} moments. Rendering clips with FFmpeg…")
            pipeline.render_all(job, lambda f, m: progress(0.7 + f * 0.3, m))
            ss.job = job
            status.update(label="✅ All clips are ready!", state="complete", expanded=False)
        bar.empty()
    except ClipGeniusError as e:
        pipeline.cleanup_job(job_dir)
        bar.empty()
        st.error(f"**Something went wrong:** {e}")
    except Exception as e:  # unexpected: show a helpful message plus details for the teacher
        pipeline.cleanup_job(job_dir)
        bar.empty()
        st.error(f"Unexpected error: {e}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())

# --------------------------------------------------------------------- results
job = ss.job
if job:
    with st.container(border=True):
        st.markdown("#### 🧠 AI analysis")
        a, b, c, d = st.columns(4)
        a.metric("Video length", format_ts(job.duration))
        b.metric("Words transcribed", f"{job.word_count:,}")
        c.metric("Language", job.language.upper())
        d.metric("Scoring", "AI (LLM)" if job.used_llm else "Keyword scorer")
        for w in job.warnings:
            st.warning(w)

    h1, h2 = st.columns([3, 1])
    h1.markdown(f"### 🎬 Your {len(job.clips)} clips, ranked by engagement")
    if h2.button("📦 Prepare ZIP of all clips", use_container_width=True):
        with st.spinner("Zipping…"):
            ss.zip_bytes = pipeline.build_zip(job)
    if ss.zip_bytes:
        h2.download_button("⬇️ Download ZIP", ss.zip_bytes, "clipgenius_clips.zip", "application/zip",
                           on_click="ignore", use_container_width=True, type="primary")

    cols = st.columns(2)
    for idx, clip in enumerate(job.clips):
        with cols[idx % 2], st.container(border=True):
            top, right = st.columns([4, 1])
            top.markdown(f"**#{idx + 1} · {clip.title}**")
            right.markdown(f'<span class="badge">{clip.overall:.0f}</span>', unsafe_allow_html=True)
            st.markdown(
                f'<span class="chip">{clip.category}</span>'
                f'<span class="chip">{format_ts(clip.start)} – {format_ts(clip.end)}</span>'
                f'<span class="chip">{clip.end - clip.start:.0f}s</span>',
                unsafe_allow_html=True,
            )
            if clip.path and clip.path.exists():
                st.video(str(clip.path))
                st.download_button(
                    "⬇️ Download clip", clip.path.read_bytes(), pipeline.clip_filename(clip),
                    "video/mp4", key=f"dl_{clip.id}", on_click="ignore", use_container_width=True,
                )
            st.caption(clip.reason)
            with st.expander("Score breakdown"):
                for k in SCORE_WEIGHTS:
                    v = clip.scores.get(k, 0)
                    st.markdown(
                        f'<div class="bar"><span class="lbl">{k.title()}</span>'
                        f'<span class="track"><span class="fill" style="width:{v:.0f}%;display:block"></span></span>'
                        f'<span class="val">{v:.0f}</span></div>',
                        unsafe_allow_html=True,
                    )
            with st.expander("✏️ Edit clip"):
                title = st.text_input("Title", clip.title, key=f"title_{clip.id}")
                t1, t2 = st.columns(2)
                start = t1.number_input("Start (seconds)", 0.0, float(job.duration), float(clip.start),
                                        step=0.5, key=f"start_{clip.id}")
                end = t2.number_input("End (seconds)", 0.0, float(job.duration), float(clip.end),
                                      step=0.5, key=f"end_{clip.id}")
                o1, o2 = st.columns(2)
                aspect = o1.selectbox("Format", list(ASPECT_SIZES), key=f"aspect_{clip.id}",
                                      index=list(ASPECT_SIZES).index(clip.aspect))
                subs = o2.toggle("Subtitles", clip.subtitles, key=f"subs_{clip.id}")
                if st.button("Apply & re-render", key=f"apply_{clip.id}", use_container_width=True):
                    backup = (clip.title, clip.start, clip.end, clip.aspect, clip.subtitles)
                    clip.title, clip.start, clip.end = title.strip() or clip.title, start, end
                    clip.aspect, clip.subtitles = aspect, subs
                    try:
                        with st.spinner("Re-rendering…"):
                            pipeline.render(job, clip)
                        ss.zip_bytes = None
                        st.rerun()
                    except ClipGeniusError as e:
                        clip.title, clip.start, clip.end, clip.aspect, clip.subtitles = backup
                        st.error(str(e))
