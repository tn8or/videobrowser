import shutil
from pathlib import Path

import pytest

from videobrowser.gopro import Chapter, Recording
from videobrowser.pipeline import _sample_times
from videobrowser.probe import VideoInfo


def _chapter(n: int, duration: float, offset: float) -> Chapter:
    path = Path(f"GH0{n}0002.mp4")
    info = VideoInfo(
        path=path,
        duration=duration,
        width=1920,
        height=1080,
        fps=30.0,
        rotation=0,
        codec="h264",
        has_gpmd=False,
        gpmd_index=None,
        creation_utc=None,
    )
    return Chapter(path=path, prefix="GH", chapter=n, recording="0002", duration=duration, offset=offset, info=info)


def test_samples_span_all_chapters():
    first, second = _chapter(1, 531.0, 0.0), _chapter(2, 323.6, 531.0)
    recording = Recording(recording_id="GH-0002", prefix="GH", recording="0002", chapters=[first, second])
    picks = _sample_times(recording)
    assert len(picks) == 9
    assert {c.chapter for c, _ in picks} == {1, 2}
    for chapter, t in picks:
        assert 0.0 <= t <= chapter.duration


def test_racebox_match_on_the_camera_second_counts_as_racebox(monkeypatch):
    import types

    import videobrowser.pipeline as pipeline
    import videobrowser.racebox_align as align

    info = VideoInfo(Path("R.mp4"), 60.0, 1920, 1080, 30.0, 0, "h264", False, None, 1785571200.0)
    rec = Recording("R", "", "R")
    rec.chapters.append(Chapter(info.path, "", 1, "R", 60.0, 0.0, info))
    track = types.SimpleNamespace(speeds=lambda s, d: ["speed"], turns=lambda s, d: ["turn"])
    config = types.SimpleNamespace(racebox_track=track)
    monkeypatch.setattr(pipeline, "speed_for_recording", lambda r: [])
    monkeypatch.setattr(
        align, "motion_match", lambda *a: align.MotionMatch(1785571200.0, "r=0.8", 0.8, 0.1)
    )
    speeds, turns, start, source = pipeline.gps_for_recording(rec, config, [], 1785571200.0)
    assert (speeds, turns, start, source) == (["speed"], ["turn"], 1785571200.0, "racebox-motion")
    config.racebox_track = None
    assert pipeline.gps_for_recording(rec, config, [], 1785571200.0)[3] == "clock"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")
@pytest.mark.parametrize("reverse,expected", [(False, "front"), (True, "rear")])
def test_facing_from_a_video_moving_into_or_out_of_the_scene(tmp_path, reverse, expected):
    import cv2
    import numpy as np

    from videobrowser.pipeline import facing_for_recording
    from videobrowser.probe import probe
    from videobrowser.profiles import REAR

    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((720, 1280)) * 255).astype(np.uint8), (5, 5), 1.5)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    path = tmp_path / "zoom.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 360))
    n = 300  # 10 s, the scene growing 1% a frame (a cycle of 30 frames)
    order = range(n - 1, -1, -1) if reverse else range(n)
    for i in order:
        m = cv2.getRotationMatrix2D((700.0, 300.0), 0.0, 1.0 + 0.01 * (i % 30))
        writer.write(cv2.resize(cv2.warpAffine(base, m, (1280, 720)), (640, 360)))
    writer.release()
    info = probe(path)
    rec = Recording("Z", "", "Z")
    rec.chapters.append(Chapter(path, "", 1, "Z", info.duration, 0.0, info))
    # The left-side ego mask of a "rear" mount guess must not matter.
    assert facing_for_recording(rec, 640, REAR) == expected


def test_finish_line_passes_place_to_the_frame(monkeypatch):
    import types

    import numpy as np

    import videobrowser.pipeline as pipeline
    import videobrowser.racebox_align as align

    t0 = 1785571200.0
    info = VideoInfo(Path("R.mov"), 600.0, 1920, 1080, 25.0, 0, "hevc", False, None, t0)
    rec = Recording("R", "", "R")
    rec.chapters.append(Chapter(info.path, "", 1, "R", 600.0, 0.0, info))
    laps = np.array([76.1, 74.9, 77.1, 75.6])
    true_start = t0 + 31.7
    crossings = true_start + 100.0 + np.r_[0.0, np.cumsum(laps)]
    utc = t0 + np.arange(0.0, 700.0, 0.1)
    track = align.RaceboxTrack(utc, np.full(len(utc), 170.0), np.zeros(len(utc)), crossings)
    track.speeds = lambda s, d: ["speed"]
    track.turns = lambda s, d: ["turn"]
    config = types.SimpleNamespace(racebox_track=track)
    monkeypatch.setattr(pipeline, "speed_for_recording", lambda r: [])
    # Scene motion put it 30 s off.
    monkeypatch.setattr(align, "motion_match", lambda *a: align.MotionMatch(t0, "r=0.6", 0.6, 0.1))
    passes = list(crossings - true_start - align.LINE_LEAD_S)
    _, _, start, source = pipeline.gps_for_recording(
        rec, config, [], t0, lambda a, b: [p for p in passes if a <= p <= b]
    )
    assert source == "racebox-line"
    assert abs(start - true_start) < 1e-6


def test_bridge_places_before_the_finish_line(monkeypatch):
    import types

    import numpy as np

    import videobrowser.pipeline as pipeline
    import videobrowser.racebox_align as align

    t0 = 1785571200.0
    info = VideoInfo(Path("R.mov"), 600.0, 1920, 1080, 25.0, 0, "hevc", False, None, t0)
    rec = Recording("R", "", "R")
    rec.chapters.append(Chapter(info.path, "", 1, "R", 600.0, 0.0, info))
    true_start = t0 + 28.9
    crossings = true_start + 100.0 + np.r_[0.0, np.cumsum([76.1, 74.9, 77.1, 75.6])]
    utc = t0 + np.arange(0.0, 700.0, 0.1)
    track = align.RaceboxTrack(utc, np.full(len(utc), 170.0), np.zeros(len(utc)), crossings, crossings + 1.2)
    track.speeds = lambda s, d: ["speed"]
    track.turns = lambda s, d: ["turn"]
    config = types.SimpleNamespace(racebox_track=track)
    monkeypatch.setattr(pipeline, "speed_for_recording", lambda r: [])
    monkeypatch.setattr(align, "motion_match", lambda *a: align.MotionMatch(t0, "r=0.6", 0.6, 0.1))
    bridges = list(crossings + 1.2 - true_start + 0.05)
    _, _, start, source = pipeline.gps_for_recording(
        rec, config, [], t0, lambda a, b: [], lambda a, b: [p for p in bridges if a <= p <= b]
    )
    assert source == "racebox-line"
    assert abs(start - (true_start - 0.05)) < 1e-3


def test_bridge_searches_the_whole_recording_when_scene_motion_fails(monkeypatch):
    import types

    import numpy as np

    import videobrowser.pipeline as pipeline
    import videobrowser.racebox_align as align

    t0 = 1785571200.0
    info = VideoInfo(Path("R.mov"), 900.0, 1920, 1080, 25.0, 0, "hevc", False, None, t0)
    rec = Recording("R", "", "R")
    rec.chapters.append(Chapter(info.path, "", 1, "R", 900.0, 0.0, info))
    rng = np.random.default_rng(4)
    laps = 77.0 + rng.normal(0, 0.5, 10)
    true_start = t0 + 7230.5  # the camera clock is local time read as UTC
    crossings = true_start + 50.0 + np.r_[0.0, np.cumsum(laps)]
    landmarks = np.sort(np.r_[crossings + 1.15, crossings + 25.0])
    utc = t0 + np.arange(0.0, 9000.0, 0.5)
    track = align.RaceboxTrack(utc, np.full(len(utc), 170.0), np.zeros(len(utc)), crossings, landmarks)
    track.speeds = lambda s, d: ["speed"]
    track.turns = lambda s, d: ["turn"]
    config = types.SimpleNamespace(racebox_track=track)
    monkeypatch.setattr(pipeline, "speed_for_recording", lambda r: [])
    monkeypatch.setattr(align, "motion_match", lambda *a: align.MotionMatch(None, "weak", 0.2, 0.1))
    bridges = sorted(u - true_start for u in landmarks if 0 < u - true_start < 900)
    _, _, start, source = pipeline.gps_for_recording(rec, config, [], t0, None, lambda a, b: bridges)
    assert source == "racebox-line"
    assert abs(start - true_start) < 1e-3
