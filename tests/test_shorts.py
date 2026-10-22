import json
import shutil
import subprocess
from pathlib import Path

import pytest

from videobrowser.db import ScanDB
from videobrowser.shorts import build_shorts, plan_shorts
from videobrowser.themes import Moment, Segment

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


def _moment(theme, start, length, score):
    return Moment(theme, start, start + length, score, [Segment("A", start, start + length)])


def test_plan_makes_several_action_shorts_and_theme_series():
    moments = [_moment("action", i * 100, 8, score=i) for i in range(10)]
    moments += [_moment("overtake", 5000 + i * 100, 9, score=1) for i in range(3)]
    moments += [_moment("crash", 9000, 10, score=10)]
    moments += [_moment("kneepuck", 9500, 6, score=1)]  # one is not enough for a series
    plans = plan_shorts(moments, max_seconds=40, max_action_shorts=3)
    names = [p.name for p in plans]
    assert names[:3] == ["action-01", "action-02", "action-03"]
    assert "overhalinger-01" in names and "styrt-01" in names
    assert not any(n.startswith("knae") for n in names)
    for plan in plans:
        assert plan.duration <= 40
        starts = [m.utc_start for m in plan.moments]
        assert starts == sorted(starts)
    # The best moment (the crash) leads the first action short.
    assert any(m.theme == "crash" for m in plans[0].moments)


def _video(path: Path, seconds: int, size: str, audio: bool = True) -> None:
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac"]
    subprocess.run(cmd + [str(path)], check=True)


@needs_ffmpeg
def test_build_shorts_end_to_end(tmp_path):
    pytest.importorskip("cairo")
    out = tmp_path / "out"
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    front_src, rear_src = src_dir / "GX010001.MP4", src_dir / "GX010002.MP4"
    _video(front_src, 40, "640x360")
    _video(rear_src, 40, "640x360", audio=False)
    vdir = out / "vertical" / "2026-08-01"
    vdir.mkdir(parents=True)
    front_v = vdir / "F_00005_approaching_rider.mp4"
    _video(front_v, 12, "270x480")

    db = ScanDB.open(out / "scan.sqlite")
    start = 1785571200.0  # 2026-08-01 10:00 Copenhagen
    for rid, profile, path in (("F", "cockpit", front_src), ("R", "rear", rear_src)):
        db.save_recording_meta(rid, profile, 0, start, "2026-08-01", 40.0,
                               [{"path": str(path), "offset": 0.0, "duration": 40.0}])
    db.replace_events("F", [("F", "approaching_rider", 10.0, 13.0, 0.9, "1", None, "{}")])
    db.replace_events("R", [("R", "receding_rider", 13.5, 16.0, 0.8, "2", None, "{}")])
    db.replace_clips("F", [(5.0, 17.0, "approaching_rider", 0.9, None, str(front_v))])
    db.close()

    result = build_shorts(out, audio_sync=False, crashes=False)
    day = out / "shorts" / "2026-08-01"
    data = json.loads((day / "moments.json").read_text())
    assert [m["theme"] for m in data["moments"]] == ["overtake"]
    names = sorted(p.name for p in result["2026-08-01"])
    assert names == ["action-01.mp4"]  # one overtake is not a series yet
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
         "-show_entries", "format=duration", "-of", "json", str(day / "action-01.mp4")],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(probe.stdout)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    assert any(s["codec_type"] == "audio" for s in info["streams"])
    # Front piece (4.5 s from the vertical clip) + rear piece cut from source (3.3 s).
    assert 7.0 < float(info["format"]["duration"]) < 9.0
    page = (day / "index.html").read_text()
    assert "action-01.mp4" in page and "F_00005_approaching_rider.mp4" in page
