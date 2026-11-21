"""Run with:  python -m unittest discover -s tests -v
Tests the real logic (URL parsing, moment selection, subtitles, FFmpeg rendering).
No API key, Whisper or internet needed."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import clipper, moments, subtitles, youtube
from core.errors import InvalidURLError
from core.models import Moment, Segment, Word
from core.utils import probe_duration


TOPICS = ["farming", "rockets", "chess", "violin", "volcano", "pizza", "robots", "glaciers", "novels",
          "bridges", "jazz", "oceans", "coffee", "dragons", "satellites", "gardens", "pirates", "metals"]


def make_segments(n=40, seg_len=6.0):
    """Synthetic transcript: n sentences, 6 s each, with some 'exciting' ones."""
    segs = []
    for i in range(n):
        start = i * seg_len
        if i % 7 == 3:
            text = "Wow, that is absolutely insane! I can't believe it, haha!"
        elif i % 5 == 0:
            text = f"Why did this happen number {i}? Remember this important lesson."
        else:
            text = f"{TOPICS[i % len(TOPICS)]} {TOPICS[(i * 7 + 3) % len(TOPICS)]} changed everything for {TOPICS[(i * 3 + 1) % len(TOPICS)]} that year."
        toks = text.split()
        step = seg_len / len(toks)
        words = [Word(start + k * step, start + (k + 1) * step, t) for k, t in enumerate(toks)]
        segs.append(Segment(start, start + seg_len, text, words))
    return segs


class TestYouTube(unittest.TestCase):
    def test_valid_urls(self):
        vid = "dQw4w9WgXcQ"
        for u in [
            f"https://www.youtube.com/watch?v={vid}",
            f"https://youtu.be/{vid}?t=10",
            f"youtube.com/shorts/{vid}",
            f"https://m.youtube.com/watch?v={vid}&list=abc",
            f"https://www.youtube.com/embed/{vid}",
        ]:
            self.assertEqual(youtube.extract_video_id(u), vid, u)

    def test_invalid_urls(self):
        for u in ["", "hello", "https://vimeo.com/123", "https://youtube.com/watch?v=short",
                  "https://www.youtube.com/@channel", "https://evil.com/youtu.be/dQw4w9WgXcQ"]:
            with self.assertRaises(InvalidURLError, msg=u):
                youtube.extract_video_id(u)


class TestMoments(unittest.TestCase):
    def test_candidates_respect_length_and_boundaries(self):
        segs = make_segments()
        cands = moments.build_candidates(segs)
        self.assertTrue(cands)
        for c in cands:
            self.assertGreaterEqual(c.duration, 20)
            self.assertLessEqual(c.duration, 60)
            self.assertTrue(moments.SENT_END.search(c.text.strip()))

    def test_heuristics_prefer_exciting_text(self):
        calm = moments.heuristic_scores("This is a calm statement about topic one.", 20, "This is a calm statement.")
        wild = moments.heuristic_scores("Wow that is insane! I can't believe it, haha! Why? Remember this.", 20,
                                        "Why did nobody tell you?")
        self.assertGreater(moments.weighted_overall(wild), moments.weighted_overall(calm))

    def test_select_top_no_overlap(self):
        ms = [Moment(s, s + 30, "t", "c", "", {}, 100 - s, f"unique text number {s} " * 5)
              for s in (0, 10, 40, 80, 85)]
        chosen = moments.select_top(ms, 10)
        for i, a in enumerate(chosen):
            for b in chosen[i + 1:]:
                self.assertLessEqual(moments._overlap_ratio(a, b), 0.15)

    def test_detect_without_key_returns_distinct_moments(self):
        res = moments.detect_moments(make_segments(100), n=10, api_key="")
        self.assertFalse(res.used_llm)
        self.assertEqual(len(res.moments), 10)
        self.assertTrue(res.warnings)  # tells the user no key was used

    def test_llm_json_parsing(self):
        raw = 'Sure!\n```json\n[{"id":0,"entertainment":80,"interest":70,"emotion":10,"hook":90,"engagement":75,' \
              '"category":"funny","title":"T","reason":"R"}, {"id":"x"}]\n```'
        parsed = moments.parse_llm_json(raw)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(moments._clean_scores(parsed[0])["hook"], 90)
        with self.assertRaises(ValueError):
            moments.parse_llm_json("no json here")


class TestSubtitles(unittest.TestCase):
    def test_ass_structure_and_highlight(self):
        words = [Word(10 + i * 0.4, 10.4 + i * 0.4, t)
                 for i, t in enumerate("This is the biggest SECRET about 2024 money.".split())]
        ass = subtitles.build_ass(words, 10, 20, 720, 1280)
        self.assertIn("PlayResX: 720", ass)
        self.assertIn("Dialogue: 0,0:00:00.00", ass)  # times are relative to clip start
        self.assertIn(subtitles.HIGHLIGHT, ass)
        off = subtitles.build_ass(words, 10, 20, 720, 1280, highlight=False)
        self.assertNotIn(subtitles.HIGHLIGHT, off.split("[Events]")[1])


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class TestRendering(unittest.TestCase):
    def test_render_all_aspects_with_subtitles(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            src = tmp / "src.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=24",
                 "-f", "lavfi", "-i", "sine=frequency=440", "-t", "12", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
            words = [Word(2 + i * 0.5, 2.5 + i * 0.5, w)
                     for i, w in enumerate("Hello everyone this is the biggest announcement ever".split())]
            for aspect, (w, h) in {"9:16": (720, 1280), "1:1": (720, 720), "16:9": (1280, 720)}.items():
                ass = subtitles.build_ass(words, 1, 9, w, h)
                out = clipper.render_clip(src, tmp / f"out_{aspect.replace(':', '_')}.mp4", 1, 9, aspect, ass)
                self.assertTrue(out.exists())
                self.assertAlmostEqual(probe_duration(out), 8, delta=0.5)
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                     "stream=width,height", "-of", "csv=p=0", str(out)],
                    capture_output=True, text=True).stdout.strip()
                self.assertEqual(probe, f"{w},{h}")
                audio = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                     "stream=codec_name", "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
                self.assertIn("aac", audio)  # original audio preserved
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
