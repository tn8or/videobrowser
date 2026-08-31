from __future__ import annotations

from pathlib import Path

from videobrowser.gopro import Recording, chapter_at, recording_id_for, stem_match
from videobrowser.gopro import Chapter
from videobrowser.probe import VideoInfo


def test_stem_match():
    match = stem_match(Path("GX010009.mov"))
    assert match is not None
    assert match.group("prefix").upper() == "GX"
    assert match.group("chapter") == "01"
    assert match.group("recording") == "0009"


def test_recording_id():
    assert recording_id_for(Path("/tmp/GH020008.mov")) == "GH-0008"
    assert recording_id_for(Path("/tmp/random.mp4")) == "random"


def test_chapter_at_timeline():
    info = VideoInfo(
        path=Path("a.mov"),
        duration=10.0,
        width=3840,
        height=2160,
        fps=25.0,
        rotation=0,
        codec="hevc",
        has_gpmd=False,
        gpmd_index=None,
    )
    rec = Recording("GX-0009", "GX", "0009")
    rec.chapters = [
        Chapter(Path("GX010009.mov"), "GX", 1, "0009", 10.0, 0.0, info),
        Chapter(Path("GX020009.mov"), "GX", 2, "0009", 8.0, 10.0, info),
    ]
    assert chapter_at(rec, 3.0).chapter == 1
    assert chapter_at(rec, 10.5).chapter == 2
    assert chapter_at(rec, 99.0).chapter == 2
