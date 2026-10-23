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
