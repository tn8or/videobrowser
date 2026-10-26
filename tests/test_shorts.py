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
    assert [p.name for p in result["2026-08-01"]] == ["action-01"]  # one overtake is not a series yet
    folder = day / "action-01"
    files = sorted(p.name for p in folder.glob("*.mp4"))
    # Front piece: the whole vertical clip, copied. Rear piece: cut from source.
    assert files == ["01a_100009_overtake_front_F_00005_approaching_rider.mp4", "01b_100009_overtake_rear.mp4"]
    assert (folder / files[0]).read_bytes() == front_v.read_bytes()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
         "-of", "json", str(folder / files[1])],
        capture_output=True, text=True, check=True,
    )
    video = next(s for s in json.loads(probe.stdout)["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    listing = (folder / "klip.txt").read_text()
    assert f"{files[0]}  (OVERHALING" in listing and "use 4.0-8.5 s" in listing
    assert not list(day.glob("*.mp4"))
    page = (day / "index.html").read_text()
    assert "action-01/" in page and "F_00005_approaching_rider.mp4" in page


@needs_ffmpeg
def test_failed_render_keeps_previous_short_and_drops_stale_ones(tmp_path, monkeypatch):
    import videobrowser.shorts as shorts

    out = tmp_path / "out"
    src = tmp_path / "GX010001.MP4"
    _video(src, 20, "320x180")
    db = ScanDB.open(out / "scan.sqlite")
    db.save_recording_meta("F", "cockpit", 0, 1785571200.0, "2026-08-01", 20.0,
                           [{"path": str(src), "offset": 0.0, "duration": 20.0}])
    db.replace_events("F", [("F", "nearby_rider", 8.0, 10.0, 0.5, "1", None, "{}")])
    db.close()
    day = out / "shorts" / "2026-08-01"
    (day / "action-01").mkdir(parents=True)
    (day / "action-01" / "klip.txt").write_text("previous")
    (day / "overhalinger-03").mkdir()
    (day / "overhalinger-03" / "klip.txt").write_text("stale")
    (day / "action-02.mp4").write_bytes(b"old finished short")

    def boom(plan, day_data, dest, **kwargs):
        dest.mkdir(parents=True)
        (dest / "half.mp4").write_bytes(b"half")
        raise RuntimeError("ffmpeg died")

    monkeypatch.setattr(shorts, "collect_short", boom)
    build_shorts(out, audio_sync=False, crashes=False)
    assert (day / "action-01" / "klip.txt").read_text() == "previous"
    assert not (day / "overhalinger-03").exists()
    assert not (day / "action-02.mp4").exists()
    assert not list(day.glob(".*partial*"))


def test_day_moments_drop_standing_still_and_paddock_riding(monkeypatch):
    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock
    from videobrowser.telemetry import SpeedSample

    day = shorts.DayData("2026-08-01", {"F": CameraClock("F", 1000.0, 200.0, [])}, {"F": "cockpit"}, {"F": 0}, [], [])
    day.events = [
        {"recording_id": "F", "type": "nearby_rider", "t_start": 20.0, "t_end": 22.0, "score": 0.5},
        {"recording_id": "F", "type": "approaching_rider", "t_start": 60.0, "t_end": 63.0, "score": 0.9},
        {"recording_id": "F", "type": "nearby_rider", "t_start": 150.0, "t_end": 152.0, "score": 0.5},
        {"recording_id": "F", "type": "rider_down", "t_start": 8.0, "t_end": 9.0, "score": 2.0},
    ]
    # Paddock until 30 s, on track until 140 s, then parked (with GPS drift).
    speeds = [SpeedSample(t, 25.0 if t < 30 else 110.0 if t < 140 else 2.0) for t in range(200)]
    monkeypatch.setattr(shorts, "_recording_gps", lambda clock: (speeds, []))
    found = {m.theme for m in shorts.day_moments(day, crashes=False)}
    assert found == {"catching_up"}
    day.gps.clear()
    themes = sorted(m.theme for m in shorts.day_moments(day, crashes=False, min_speed=0))
    assert themes == ["action", "action", "catching_up", "crash_other"]


def test_clip_drop_reason_matches_the_shorts(monkeypatch):
    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock
    from videobrowser.telemetry import SpeedSample

    day = shorts.DayData("2026-08-01", {"F": CameraClock("F", 1000.0, 300.0, [])}, {"F": "cockpit"}, {"F": 0}, [], [])
    # Parked until 30 s, paddock until 100 s, then on track; a van at 250 s.
    speeds = [SpeedSample(t, 2.0 if t < 30 else 25.0 if t < 100 else 110.0) for t in range(300)]
    monkeypatch.setattr(shorts, "_recording_gps", lambda clock: (speeds, []))
    day.vehicles["F"] = [(t + 0.5, 1.0 if t >= 250 else 0.0) for t in range(300)]
    assert shorts.clip_drop_reason(day, "F", 5.0, 20.0) == "standing still"
    assert shorts.clip_drop_reason(day, "F", 50.0, 70.0) == "below 40 km/h"
    assert shorts.clip_drop_reason(day, "F", 150.0, 170.0) is None
    assert shorts.clip_drop_reason(day, "F", 255.0, 270.0) == "car in view"
    assert shorts.clip_drop_reason(day, "F", 50.0, 70.0, min_speed=0, max_vehicle_share=0) is None



def test_day_moments_fall_back_to_scene_motion_without_gps(monkeypatch):
    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock

    day = shorts.DayData("2026-08-01", {"F": CameraClock("F", 1000.0, 200.0, [])}, {"F": "cockpit"}, {"F": 0}, [], [])
    day.events = [
        {"recording_id": "F", "type": "rider_down", "t_start": 20.0, "t_end": 21.0, "score": 2.0},
        {"recording_id": "F", "type": "nearby_rider", "t_start": 60.0, "t_end": 62.0, "score": 0.5},
        {"recording_id": "F", "type": "approaching_rider", "t_start": 120.0, "t_end": 123.0, "score": 0.9},
    ]
    # Parked until 40 s, rolling slowly until 100 s, then on track.
    day.motion["F"] = [(t + 0.5, 1.0 if t < 40 else 6.0 if t < 100 else 15.0) for t in range(200)]
    monkeypatch.setattr(shorts, "_recording_gps", lambda clock: ([], []))
    found = {m.theme for m in shorts.day_moments(day, crashes=False)}
    assert found == {"catching_up"}


def test_day_moments_leave_out_moments_with_a_car_in_view(monkeypatch):
    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock

    day = shorts.DayData("2026-08-01", {"F": CameraClock("F", 1000.0, 200.0, [])}, {"F": "cockpit"}, {"F": 0}, [], [])
    day.events = [
        {"recording_id": "F", "type": "nearby_rider", "t_start": 20.0, "t_end": 22.0, "score": 0.5},
        {"recording_id": "F", "type": "approaching_rider", "t_start": 120.0, "t_end": 123.0, "score": 0.9},
    ]
    day.motion["F"] = [(t + 0.5, 15.0) for t in range(200)]
    # A van in view until 40 s (the paddock), then a car for one second.
    day.vehicles["F"] = [(t + 0.5, 1.0 if t < 40 or t == 121 else 0.0) for t in range(200)]
    monkeypatch.setattr(shorts, "_recording_gps", lambda clock: ([], []))
    assert [m.theme for m in shorts.day_moments(day, crashes=False)] == ["catching_up"]
    kept = shorts.day_moments(day, crashes=False, max_vehicle_share=0)
    assert sorted(m.theme for m in kept) == ["action", "catching_up"]


def test_recording_without_gps_takes_racebox_time_speed_and_turns(monkeypatch):
    import numpy as np

    import videobrowser.racebox_align as ra
    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock

    day = shorts.DayData("2026-08-03", {"M": CameraClock("M", 1000.0, 100.0, [])}, {"M": "cockpit"}, {"M": 0}, [], [])
    day.motion["M"] = [(t + 0.5, 10.0) for t in range(100)]
    monkeypatch.setattr(shorts, "_recording_gps", lambda clock: ([], []))
    monkeypatch.setattr(ra, "motion_match", lambda motion, track, guesses: ra.MotionMatch(1012.0, "r=0.90"))
    utc = np.arange(900.0, 1300.0, 0.1)
    track = ra.RaceboxTrack(utc, np.full(len(utc), 120.0), np.full(len(utc), -20.0))
    shorts.place_without_gps(day, track)
    assert day.clocks["M"].start_utc == 1012.0
    speeds, turns = day.gps["M"]
    assert speeds[0].t == pytest.approx(0.0, abs=0.1) and speeds[0].speed_kmh == 120.0
    assert turns[-1].t <= 100.0 and turns[0].yaw_rate == -20.0


def test_collect_short_fails_whole_when_a_piece_fails(tmp_path, monkeypatch):
    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock

    day = shorts.DayData("2026-08-01", {"A": CameraClock("A", 1000.0, 100.0, [])}, {"A": "cockpit"}, {"A": 0}, [], [])

    def broken(*a, **k):
        raise RuntimeError("ffmpeg died")

    monkeypatch.setattr(shorts, "_from_source", broken)
    plan = shorts.ShortPlan("action-01", "action", [_moment("action", 10, 5, 1.0)])
    with pytest.raises(RuntimeError, match="A@10s"):
        shorts.collect_short(plan, day, tmp_path / "action-01")


def test_unplaced_recording_is_stopped_unless_racebox_rides_at_a_clock_reading(monkeypatch):
    import numpy as np

    import videobrowser.racebox_align as ra
    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock

    start = 1785571200.0  # 2026-08-01 10:00 Copenhagen (UTC+2)
    day = shorts.DayData("2026-08-01", {"M": CameraClock("M", start, 600.0, [])}, {"M": "cockpit"}, {"M": 0}, [], [])
    day.motion["M"] = [(t + 0.5, 12.0) for t in range(600)]  # camera in hand: busy, flat
    monkeypatch.setattr(shorts, "_recording_gps", lambda clock: ([], []))
    # Riding 7200 s after the clock (its other reading) for the first 200 s only.
    utc = np.arange(start + 7200 - 100, start + 7200 + 200, 0.1)
    track = ra.RaceboxTrack(utc, np.full(len(utc), 120.0), np.zeros(len(utc)))
    shorts.place_without_gps(day, track)
    assert day.unplaced["M"] == [start, start + 7200]
    early = Moment("action", start + 50, start + 55, 1.0, [Segment("M", 50, 55)])
    late = Moment("action", start + 500, start + 505, 1.0, [Segment("M", 500, 505)])
    assert shorts._moment_pace(early, day, 40.0) == "slow"
    assert shorts._moment_pace(late, day, 40.0) == "stopped"


def test_whole_day_match_gets_camera_clock_starts_after_a_recording_was_placed(monkeypatch):
    import numpy as np

    import videobrowser.racebox_align as ra
    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock

    clocks = {"A": CameraClock("A", 1000.0, 300.0, []), "B": CameraClock("B", 2000.0, 300.0, [])}
    day = shorts.DayData("2026-08-01", clocks, {"A": "cockpit", "B": "cockpit"}, {"A": 0, "B": 0}, [], [])
    day.motion = {rid: [(t + 0.5, 5.0) for t in range(300)] for rid in clocks}
    monkeypatch.setattr(shorts, "_recording_gps", lambda clock: ([], []))
    monkeypatch.setattr(
        ra, "motion_match",
        lambda motion, track, guesses: ra.MotionMatch(975.0 if guesses[0] == 1000.0 else None, "x"),
    )
    seen = []

    def fake_day(recs, track):
        seen.extend(start for start, _ in recs)
        return ra.MotionMatch(975.0, "r=0.70")

    monkeypatch.setattr(ra, "day_match", fake_day)
    utc = np.arange(0.0, 4000.0, 0.1)
    shorts.place_without_gps(day, ra.RaceboxTrack(utc, np.full(len(utc), 100.0), np.zeros(len(utc))))
    assert clocks["A"].start_utc == 975.0  # placed alone
    assert sorted(seen) == [1000.0, 2000.0]  # camera clock, not RaceBox time
    assert clocks["B"].start_utc == 2000.0 and day.gps["B"][0][0].t == pytest.approx(0.0, abs=0.1)


def test_a_crash_needs_the_picture_to_stop_too():
    from videobrowser.shorts import _moving_after

    streaming = [(t + 0.5, 15.0) for t in range(30)]
    parked = [(t + 0.5, 15.0 if t < 12 else 1.0) for t in range(30)]
    assert _moving_after(streaming, 10.0)
    assert not _moving_after(parked, 10.0)
    assert not _moving_after(None, 10.0)  # no scene motion: trust the GPS


def test_collect_short_cuts_in_parallel_and_keeps_order(tmp_path, monkeypatch):
    import threading

    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock

    day = shorts.DayData("2026-08-01", {"A": CameraClock("A", 1000.0, 100.0, [])}, {"A": "cockpit"}, {"A": 0}, [], [])
    started = threading.Barrier(3, timeout=5)

    def cut(day_data, segment, dest):
        started.wait()  # all three cuts must be running at once
        dest.write_bytes(b"clip")
        return dest

    monkeypatch.setattr(shorts, "_from_source", cut)
    plan = shorts.ShortPlan(
        "action-01", "action", [_moment("action", t, 5, 1.0) for t in (10, 30, 50)]
    )
    out = shorts.collect_short(plan, day, tmp_path / "action-01", jobs=3)
    listed = (out / "klip.txt").read_text().splitlines()[2:]
    assert [line.split("_")[0] for line in listed] == ["01", "02", "03"]
    assert len(list(out.glob("*.mp4"))) == 3

    def broken(day_data, segment, dest):
        if segment.t_start == 30:
            raise RuntimeError("ffmpeg died")
        return dest

    monkeypatch.setattr(shorts, "_from_source", broken)
    with pytest.raises(RuntimeError, match="A@30s"):
        shorts.collect_short(plan, day, tmp_path / "action-02", jobs=3)
