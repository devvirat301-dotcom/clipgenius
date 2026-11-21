"""Generate styled ASS subtitles with highlighted keywords from word timestamps."""
from __future__ import annotations

import re

from core.models import Segment, Word

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "of", "to", "in", "on", "at", "for",
    "with", "from", "by", "as", "is", "are", "was", "were", "be", "been", "am", "it", "its", "this",
    "that", "these", "those", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "our", "their", "do", "does", "did", "have", "has", "had", "will",
    "would", "can", "could", "should", "just", "like", "not", "no", "yes", "yeah", "um", "uh",
    "oh", "well", "really", "very", "what", "when", "where", "who", "how", "there", "here",
    "about", "into", "than", "too", "also", "got", "get", "going", "gonna", "know", "think",
}
EMPHASIS = {
    "never", "always", "secret", "truth", "biggest", "best", "worst", "crazy", "insane",
    "love", "hate", "money", "million", "free", "wrong", "mistake", "impossible", "amazing",
    "shocking", "important", "problem", "fear", "dream", "win", "lost", "died", "why", "stop",
}

WHITE = "&H00FFFFFF"
HIGHLIGHT = "&H0000D4FF"  # ASS colours are BGR: this is a warm yellow
OUTLINE = "&H00000000"


def _clean(word: str) -> str:
    return re.sub(r"[^\w']", "", word, flags=re.UNICODE).lower()


def _importance(word: str) -> int:
    c = _clean(word)
    if not c or c in STOPWORDS:
        return 0
    score = 0
    if c in EMPHASIS:
        score += 3
    if any(ch.isdigit() for ch in c):
        score += 3
    if len(word) > 2 and word.isupper() and word.isalpha():
        score += 2
    if len(c) >= 7:
        score += 1
    return score


def _ass_time(t: float) -> str:
    t = max(t, 0.0)
    cs = int(round(t * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def words_in_range(segments: list[Segment], start: float, end: float) -> list[Word]:
    out: list[Word] = []
    for seg in segments:
        if seg.end < start or seg.start > end:
            continue
        for w in seg.words:
            mid = (w.start + w.end) / 2
            if start <= mid <= end:
                out.append(w)
    return out


def group_words(words: list[Word], max_words: int = 4, max_gap: float = 0.6) -> list[list[Word]]:
    groups: list[list[Word]] = []
    cur: list[Word] = []
    for w in words:
        if cur and (len(cur) >= max_words or w.start - cur[-1].end > max_gap):
            groups.append(cur)
            cur = []
        cur.append(w)
        if len(cur) >= 2 and re.search(r"[.!?]$", w.text):
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def build_ass(
    words: list[Word],
    clip_start: float,
    clip_end: float,
    width: int,
    height: int,
    highlight: bool = True,
) -> str:
    """Return the full text of an .ass file for a clip of size width x height."""
    vertical = height > width
    font_size = int(height * (0.052 if vertical else 0.062 if width == height else 0.072))
    margin_v = int(height * (0.20 if vertical else 0.08))
    margin_h = int(width * 0.06)
    max_words = 6 if width > height else 4
    duration = clip_end - clip_start

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},{WHITE},{WHITE},{OUTLINE},&H80000000,-1,0,0,0,100,100,0,0,1,{max(3, font_size // 14)},0,2,{margin_h},{margin_h},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    groups = group_words(words, max_words=max_words)
    lines = []
    for gi, group in enumerate(groups):
        start = max(group[0].start - clip_start, 0.0)
        end = group[-1].end - clip_start + 0.15
        if gi + 1 < len(groups):  # never overlap the next caption
            end = min(end, groups[gi + 1][0].start - clip_start)
        end = min(end, duration)
        if end - start < 0.1:
            continue

        top = set()
        if highlight:
            ranked = sorted(range(len(group)), key=lambda i: _importance(group[i].text), reverse=True)
            top = {i for i in ranked[:2] if _importance(group[i].text) > 0}

        parts = []
        for i, w in enumerate(group):
            text = _escape(w.text)
            if i in top:
                parts.append(f"{{\\c{HIGHLIGHT}\\fscx110\\fscy110}}{text}{{\\r}}")
            else:
                parts.append(text)
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{' '.join(parts)}"
        )
    return header + "\n".join(lines) + "\n"
