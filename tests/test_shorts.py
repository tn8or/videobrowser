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
    for rid in ("F", "R"):  # No GPS in the test videos: the picture shows the bike riding.
        db.save_riding(rid, [(t + 0.5, 1.0) for t in range(40)])
    db.close()

    result = build_shorts(out, audio_sync=False, crashes=False)
    day = out / "shorts" / "2026-08-01"
    data = json.loads((day / "moments.json").read_text())
    assert [m["theme"] for m in data["moments"]] == ["overtake"]
    assert "riding 100% of the time" in data["moments"][0]["why"] and data["left_out"] == []
    assert [p.name for p in result["2026-08-01"]] == ["action-01"]  # one overtake is not a series yet
    folder = day / "action-01"
    files = sorted(p.name for p in folder.glob("*.mp4"))
    # Front piece: the whole vertical clip, copied. Rear piece: cut from source.
    # Pieces are widened to MIN_PIECE_S (8 s), so the overtake starts at 10:00:07.
    assert files == ["01a_100007_overtake_front_F_00005_approaching_rider.mp4", "01b_100007_overtake_rear.mp4"]
    assert (folder / files[0]).read_bytes() == front_v.read_bytes()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
         "-of", "json", str(folder / files[1])],
        capture_output=True, text=True, check=True,
    )
    video = next(s for s in json.loads(probe.stdout)["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    listing = (folder / "klip.txt").read_text()
    assert f"{files[0]}  (OVERHALING" in listing and "use 2.2-10.2 s" in listing
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
    db.save_riding("F", [(t + 0.5, 1.0) for t in range(20)])
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



def test_day_moments_without_gps_need_the_bike_riding_in_the_picture(monkeypatch):
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
    # Scene motion alone never makes track: a camera in the hand moves as much.
    left_out = []
    assert shorts.day_moments(day, crashes=False, left_out=left_out) == []
    assert all(m.why[-1].startswith("left out: ") for m in left_out)
    # A helmet camera: no part of the bike in view, but another rider is,
    # with the scene streaming past at track pace from 100 s.
    day.action["F"] = [(t + 0.5, 1, 0.0) for t in range(200)]
    assert {m.theme for m in shorts.day_moments(day, crashes=False)} == {"catching_up"}
    del day.action["F"]
    # The picture shows the camera riding on the bike from 100 s.
    day.riding["F"] = [(t + 0.5, 1.0 if t >= 100 else 0.0) for t in range(200)]
    moments = shorts.day_moments(day, crashes=False)
    assert {m.theme for m in moments} == {"catching_up"}
    assert "riding 100% of the time" in moments[0].why


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
    day.riding["F"] = [(t + 0.5, 1.0) for t in range(200)]
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


def test_own_off_track_is_judged_on_the_run_up(monkeypatch):
    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock
    from videobrowser.telemetry import SpeedSample

    day = shorts.DayData("2026-08-01", {"F": CameraClock("F", 1000.0, 300.0, [])}, {"F": "cockpit"}, {"F": 0}, [], [])
    day.events = [
        {"recording_id": "F", "type": "own_off_track", "t_start": 100.0, "t_end": 102.0, "score": 1.8},
        {"recording_id": "F", "type": "own_off_track", "t_start": 250.0, "t_end": 252.0, "score": 1.8},
    ]
    # Riding until 100 s, then stopping in the grass; walking about from 200 s.
    speeds = [SpeedSample(t, 110.0 if t < 100 else 0.0) for t in range(300)]
    monkeypatch.setattr(shorts, "_recording_gps", lambda clock: (speeds, []))
    kept = shorts.day_moments(day, crashes=False)
    assert [(m.theme, m.segments[0].t_start) for m in kept] == [("own_off_track", 95.0)]
    assert any(p.name == "afkoersler-01" for p in plan_shorts(kept))


def test_without_gps_the_picture_tells_riding_from_a_camera_in_the_hand(monkeypatch):
    import numpy as np

    import videobrowser.racebox_align as ra
    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock

    day = shorts.DayData("2026-08-01", {"M": CameraClock("M", 1000.0, 600.0, [])}, {"M": "cockpit"}, {"M": 0}, [], [])
    day.events = [
        {"recording_id": "M", "type": "approaching_rider", "t_start": 100.0, "t_end": 103.0, "score": 0.9},
        {"recording_id": "M", "type": "approaching_rider", "t_start": 400.0, "t_end": 403.0, "score": 0.9},
    ]
    # The picture moves a lot all along, in the hand from 300 s.
    day.motion["M"] = [(t + 0.5, 20.0) for t in range(600)]
    day.riding["M"] = [(t + 0.5, 1.0 if t < 300 else 0.0) for t in range(600)]
    monkeypatch.setattr(shorts, "_recording_gps", lambda clock: ([], []))
    # RaceBox laid on wrongly says riding at 400 s and parked at 100 s: the
    # picture wins.
    monkeypatch.setattr(ra, "motion_match", lambda motion, track, guesses: ra.MotionMatch(1000.0, "r=0.60"))
    utc = np.arange(900.0, 1700.0, 0.1)
    track = ra.RaceboxTrack(utc, np.where(utc < 1300.0, 0.0, 120.0), np.zeros(len(utc)))
    shorts.place_without_gps(day, track)
    kept = shorts.day_moments(day, crashes=False)
    assert [m.segments[0].t_start for m in kept] == [97.5]  # widened to 8 s
    assert shorts.clip_drop_reason(day, "M", 380.0, 420.0) == "below 40 km/h"


def test_gopro_gps_still_beats_the_picture(monkeypatch):
    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock
    from videobrowser.telemetry import SpeedSample

    day = shorts.DayData("2026-08-01", {"F": CameraClock("F", 1000.0, 300.0, [])}, {"F": "cockpit"}, {"F": 0}, [], [])
    day.riding["F"] = [(t + 0.5, 0.0) for t in range(300)]  # e.g. a helmet-ish view
    speeds = [SpeedSample(t, 110.0) for t in range(300)]
    monkeypatch.setattr(shorts, "_recording_gps", lambda clock: (speeds, []))
    assert shorts.clip_drop_reason(day, "F", 100.0, 110.0) is None


def test_moments_need_a_rider_or_kerb_in_view_and_more_riders_score_higher(monkeypatch):
    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock
    from videobrowser.telemetry import SpeedSample

    day = shorts.DayData("2026-08-01", {"F": CameraClock("F", 1000.0, 300.0, [])}, {"F": "cockpit"}, {"F": 0}, [], [])
    day.events = [
        {"recording_id": "F", "type": "nearby_rider", "t_start": 20.0, "t_end": 22.0, "score": 1.0},
        {"recording_id": "F", "type": "nearby_rider", "t_start": 100.0, "t_end": 102.0, "score": 1.0},
        {"recording_id": "F", "type": "nearby_rider", "t_start": 180.0, "t_end": 182.0, "score": 1.0},
        {"recording_id": "F", "type": "kerb_close", "t_start": 250.0, "t_end": 251.0, "score": 0.5},
        {"recording_id": "F", "type": "own_off_track", "t_start": 280.0, "t_end": 282.0, "score": 1.8},
    ]
    riders = {t: 1 if 15 <= t < 30 else 3 if 95 <= t < 110 else 0 for t in range(300)}
    day.action["F"] = [(t + 0.5, riders[t], 1.0 if 248 <= t < 253 else 0.0) for t in range(300)]
    monkeypatch.setattr(shorts, "_recording_gps", lambda clock: ([SpeedSample(t, 110.0) for t in range(300)], []))
    kept = {round(m.segments[0].t_start): m for m in shorts.day_moments(day, crashes=False)}
    assert sorted(kept) == [17, 97, 246, 275]  # 180 s: nothing in view; pieces 8 s+
    assert kept[97].score > kept[17].score
    assert shorts.clip_drop_reason(day, "F", 175.0, 185.0) == "no rider or kerb in view"
    day.action.clear()  # scanned before: no check
    assert len(shorts.day_moments(day, crashes=False)) == 5


def test_recordings_without_a_date_each_get_their_own_lane(monkeypatch):
    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock

    clocks = {"A": CameraClock("A", None, 300.0, []), "B": CameraClock("B", None, 300.0, [])}
    day = shorts.DayData("unknown", clocks, {"A": "cockpit", "B": "rear"}, {"A": 0, "B": 0}, [], [])
    day.facings = {"A": "front", "B": "rear"}
    # The same recording times on both: without lanes they paired up as an
    # overtake, or one crowded the other out.
    day.events = [
        {"recording_id": "A", "type": "approaching_rider", "t_start": 50.0, "t_end": 53.0, "score": 1.0},
        {"recording_id": "B", "type": "receding_rider", "t_start": 53.5, "t_end": 56.0, "score": 1.0},
        {"recording_id": "B", "type": "nearby_rider", "t_start": 50.0, "t_end": 52.0, "score": 1.0},
    ]
    for rid in clocks:
        day.motion[rid] = [(t + 0.5, 15.0) for t in range(300)]
        day.riding[rid] = [(t + 0.5, 1.0) for t in range(300)]
    monkeypatch.setattr(shorts, "_recording_gps", lambda clock: ([], []))
    moments = shorts.day_moments(day, crashes=False)
    assert "overtake" not in {m.theme for m in moments}
    assert {m.segments[0].recording_id for m in moments} == {"A", "B"}
    assert shorts._file_time(day.offset("B") + 3725.0) == "010205"


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


def test_finish_line_start_stays_out_of_the_whole_day_match(monkeypatch):
    import numpy as np

    import videobrowser.racebox_align as ra
    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock

    # A: placed by the finish line in the scan (RaceBox time, 31 s off the
    # camera clock); B: same camera, scene motion leaves it unplaced.
    clocks = {"A": CameraClock("A", 1031.0, 300.0, []), "B": CameraClock("B", 2000.0, 300.0, [])}
    day = shorts.DayData("2026-08-01", clocks, {"A": "cockpit", "B": "cockpit"}, {"A": 0, "B": 0}, [], [])
    day.motion = {rid: [(t + 0.5, 5.0) for t in range(300)] for rid in clocks}
    day.line_placed = {"A"}
    monkeypatch.setattr(shorts, "_recording_gps", lambda clock: ([], []))
    monkeypatch.setattr(ra, "motion_match", lambda *a: ra.MotionMatch(None, "x"))
    seen = []

    def fake_day(recs, track):
        seen.extend(start for start, _ in recs)
        return ra.MotionMatch(1990.0, "r=0.70")

    monkeypatch.setattr(ra, "day_match", fake_day)
    utc = np.arange(0.0, 4000.0, 0.1)
    shorts.place_without_gps(day, ra.RaceboxTrack(utc, np.full(len(utc), 100.0), np.zeros(len(utc))))
    assert clocks["A"].start_utc == 1031.0 and clocks["A"].source == "racebox-line"
    assert seen == [2000.0]


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


def test_moments_rank_by_how_near_and_how_many_riders(monkeypatch):
    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock

    day = shorts.DayData("2026-08-01", {"F": CameraClock("F", 1000.0, 300.0, [])}, {"F": "cockpit"}, {"F": 0}, [], [])
    # Same event score; one far rider at 50 s, two near riders at 150 s.
    day.events = [
        {"recording_id": "F", "type": "nearby_rider", "t_start": 50.0, "t_end": 52.0, "score": 0.9},
        {"recording_id": "F", "type": "nearby_rider", "t_start": 150.0, "t_end": 152.0, "score": 0.5},
    ]
    day.motion["F"] = [(t + 0.5, 15.0) for t in range(300)]
    day.riding["F"] = [(t + 0.5, 1.0) for t in range(300)]
    day.action["F"] = [
        (t + 0.5, 1 if t < 100 else 2, 0.0, 0.04 if t < 100 else 0.2) for t in range(300)
    ]
    monkeypatch.setattr(shorts, "_recording_gps", lambda clock: ([], []))
    moments = sorted(shorts.day_moments(day, crashes=False), key=lambda m: -m.score)
    assert [m.segments[0].t_start > 100 for m in moments] == [True, False]
    assert any(w.startswith("action ") for w in moments[0].why)


def test_moments_with_a_red_signal_lamp_are_left_out(monkeypatch):
    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock
    from videobrowser.telemetry import SpeedSample

    day = shorts.DayData("2026-08-01", {"F": CameraClock("F", 1000.0, 300.0, [])}, {"F": "cockpit"}, {"F": 0}, [], [])
    day.events = [
        {"recording_id": "F", "type": "nearby_rider", "t_start": 50.0, "t_end": 52.0, "score": 1.0},
        {"recording_id": "F", "type": "nearby_rider", "t_start": 200.0, "t_end": 202.0, "score": 1.0},
    ]
    # The session ends at 195 s: red lamps blink at the marshal posts.
    day.action["F"] = [(t + 0.5, 1, 0.0, 0.2, 1 if 195 <= t < 210 else 0) for t in range(300)]
    monkeypatch.setattr(shorts, "_recording_gps", lambda clock: ([SpeedSample(t, 110.0) for t in range(300)], []))
    left_out = []
    kept = shorts.day_moments(day, crashes=False, left_out=left_out)
    assert [round(m.segments[0].t_start) for m in kept] == [47]
    assert left_out[0].why[-1] == "left out: red signal lamp (session over)"
    assert shorts.clip_drop_reason(day, "F", 195.0, 215.0) == "red signal lamp (session over)"


def test_bridges_say_when_a_recording_is_on_track():
    import numpy as np

    import videobrowser.racebox_align as ra
    import videobrowser.shorts as shorts
    from videobrowser.sync import CameraClock
    from videobrowser.themes import Moment, Segment

    # A: placed on RaceBox, bridges at 100-400 s of it; B: not placed, the
    # scan saw bridges at 50 and 300 s (one missed between).
    clocks = {"A": CameraClock("A", 1000.0, 900.0, []), "B": CameraClock("B", 9000.0, 900.0, [])}
    day = shorts.DayData("2026-08-01", clocks, {"A": "cockpit", "B": "cockpit"}, {"A": 0, "B": 0}, [], [])
    utc = np.arange(0.0, 20000.0, 1.0)
    landmarks = 1000.0 + np.array([100.0, 140.0, 180.0, 220.0, 260.0, 300.0, 340.0, 380.0, 400.0])
    day.track = ra.RaceboxTrack(utc, np.full(len(utc), 100.0), np.zeros(len(utc)), landmarks - 1.0, landmarks)
    day.line_placed = {"A"}
    day.bridges = {"B": [50.0, 300.0]}

    def gate(rid, t):
        return shorts._bridge_gate(Moment("action", t, t + 4, 0.0, [Segment(rid, t, t + 4)]), day)

    assert gate("A", 50.0).startswith("before the first bridge")
    assert gate("A", 250.0) is None
    assert gate("A", 480.0) is None  # the in-lap after the last
    assert gate("A", 600.0).startswith("no bridge for")
    assert gate("B", 200.0) is None  # one missed bridge is tolerated
    assert gate("B", 700.0).startswith("no bridge for")
    assert gate("C", 10.0) is None  # nothing to tell by
