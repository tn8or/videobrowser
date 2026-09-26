from __future__ import annotations

from pathlib import Path

from videobrowser.gopro import (
    Chapter,
    Recording,
    chapter_at,
    group_recordings,
    iter_video_paths,
    qualified_recording_id,
    recording_id_for,
    stem_match,
)
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


def test_iter_video_paths_recursive(tmp_path: Path):
    nested = tmp_path / "2026-08-02" / "100GOPRO"
    nested.mkdir(parents=True)
    (nested / "GX010510.MP4").write_bytes(b"x")
    (tmp_path / "top.mp4").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("no")
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "GX010511.mp4").write_bytes(b"x")
    (nested / "._GX010510.MP4").write_bytes(b"x")

    names = {p.name for p in iter_video_paths([tmp_path])}
    assert names == {"GX010510.MP4", "top.mp4"}


def test_iter_video_paths_skips_output_dir(tmp_path: Path):
    clips = tmp_path / "library" / "day1"
    clips.mkdir(parents=True)
    (clips / "GX010001.mp4").write_bytes(b"x")
    out = tmp_path / "output"
    (out / "clips").mkdir(parents=True)
    (out / "clips" / "GX-0001_00000_nearby_rider.mp4").write_bytes(b"x")

    names = {p.name for p in iter_video_paths([tmp_path], exclude_roots=[out])}
    assert names == {"GX010001.mp4"}


def test_iter_video_paths_exclude_substrings(tmp_path: Path):
    keep = tmp_path / "2026-08-02" / "100GOPRO"
    skip_year = tmp_path / "2025-06-06" / "100GOPRO"
    racebox = tmp_path / "RaceBox" / "overlays"
    keep.mkdir(parents=True)
    skip_year.mkdir(parents=True)
    racebox.mkdir(parents=True)
    (keep / "GX010001.mp4").write_bytes(b"x")
    (skip_year / "GX010002.mp4").write_bytes(b"x")
    (racebox / "GX010003.mp4").write_bytes(b"x")
    (tmp_path / "GX010025.mp4").write_bytes(b"x")
    (tmp_path / "session_RaceBox.mp4").write_bytes(b"x")

    names = {
        p.name
        for p in iter_video_paths(
            [tmp_path], exclude_substrings=["RaceBox", "2025"]
        )
    }
    assert names == {"GX010001.mp4", "GX010025.mp4"}


def test_iter_video_paths_exclude_is_case_insensitive(tmp_path: Path):
    nested = tmp_path / "racebox" / "day"
    nested.mkdir(parents=True)
    (nested / "GX010001.mp4").write_bytes(b"x")
    (tmp_path / "keep.mp4").write_bytes(b"x")
    names = {p.name for p in iter_video_paths([tmp_path], exclude_substrings=["RACEBOX"])}
    assert names == {"keep.mp4"}


def test_iter_video_paths_skips_symlinks_outside_input(tmp_path: Path):
    library = tmp_path / "library"
    outside = tmp_path / "elsewhere"
    library.mkdir()
    outside.mkdir()
    (library / "keep.mp4").write_bytes(b"x")
    target = outside / "secret.mp4"
    target.write_bytes(b"x")
    (library / "GX010001.mp4").symlink_to(target)

    names = {p.name for p in iter_video_paths([library])}
    assert names == {"keep.mp4"}


def test_iter_video_paths_skips_broken_symlinks(tmp_path: Path):
    (tmp_path / "keep.mp4").write_bytes(b"x")
    (tmp_path / "GX010001.mp4").symlink_to(tmp_path / "missing.mp4")
    names = {p.name for p in iter_video_paths([tmp_path])}
    assert names == {"keep.mp4"}


def test_iter_video_paths_skips_fcpbundle(tmp_path: Path):
    bundle = tmp_path / "trackdays.fcpbundle" / "__AsyncCopying"
    bundle.mkdir(parents=True)
    (bundle / "GX010001.mp4").write_bytes(b"x")
    (tmp_path / "keep.mp4").write_bytes(b"x")
    names = {p.name for p in iter_video_paths([tmp_path])}
    assert names == {"keep.mp4"}


def test_qualified_id_flat_vs_nested(tmp_path: Path):
    flat = tmp_path / "GX010510.mp4"
    nested = tmp_path / "2026-08-02" / "100GOPRO" / "GX010510.mp4"
    assert qualified_recording_id(flat, [tmp_path]) == "GX-0510"
    assert (
        qualified_recording_id(nested, [tmp_path]) == "2026-08-02_100GOPRO_GX-0510"
    )


def test_group_recordings_separates_same_id_in_different_folders(
    tmp_path: Path, monkeypatch
):
    import videobrowser.gopro as gopro

    info = VideoInfo(
        path=Path("a.mp4"),
        duration=1.0,
        width=64,
        height=64,
        fps=25.0,
        rotation=0,
        codec="hevc",
        has_gpmd=False,
        gpmd_index=None,
    )
    monkeypatch.setattr(gopro, "probe", lambda path: info)

    day1 = tmp_path / "2024-08-02" / "100GOPRO"
    day2 = tmp_path / "2025-06-06" / "100GOPRO"
    day1.mkdir(parents=True)
    day2.mkdir(parents=True)
    a = day1 / "GX010510.mp4"
    b = day2 / "GX010510.mp4"
    a.write_bytes(b"x")
    b.write_bytes(b"x")

    recs = group_recordings([a, b], input_roots=[tmp_path])
    ids = sorted(r.recording_id for r in recs)
    assert ids == [
        "2024-08-02_100GOPRO_GX-0510",
        "2025-06-06_100GOPRO_GX-0510",
    ]

