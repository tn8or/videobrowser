from pathlib import Path

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
