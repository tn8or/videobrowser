import pytest

from videobrowser.cli import build_parser


def test_old_clip_flags_on_scan_are_accepted_and_ignored():
    parse = build_parser().parse_args
    args = parse(["scan", "in", "-o", "out", "--no-overlay", "--no-landscape", "--no-vertical", "--no-clips"])
    assert args.old_no_overlay and args.old_no_landscape
    assert not parse(["scan", "in", "-o", "out"]).old_no_overlay


def test_export_commands_take_overlay_date_and_force():
    parse = build_parser().parse_args
    for cmd in ("horizontals", "verticals"):
        args = parse([cmd, "-o", "out", "--date", "2026-08-01", "--no-racebox", "--force"])
        assert args.no_overlay and args.force and args.date == ["2026-08-01"]
        assert not parse([cmd, "-o", "out"]).no_overlay
    assert parse(["horizontals", "-o", "out"]).stats is None


def test_scan_only_analyses(tmp_path, monkeypatch):
    from pathlib import Path

    import videobrowser.cli as cli
    import videobrowser.exports as exports
    import videobrowser.racebox_stage as rbs
    import videobrowser.shorts as shorts
    from videobrowser.gopro import Chapter, Recording
    from videobrowser.probe import VideoInfo

    info = VideoInfo(Path("R.mp4"), 60.0, 1920, 1080, 30.0, 0, "h264", False, None, 1785571200.0)
    rec = Recording("R", "GX", "R")
    rec.chapters.append(Chapter(info.path, "GX", 1, "R", 60.0, 0.0, info))
    monkeypatch.setattr(cli, "iter_video_paths", lambda *a, **k: [Path("x.mp4")])
    monkeypatch.setattr(cli, "group_recordings", lambda *a, **k: [rec])
    monkeypatch.setattr(rbs, "fetch_sessions", lambda d, **kw: None)
    scanned, cut = [], []
    monkeypatch.setattr(cli, "scan_recording", lambda r, *a: scanned.append(r.recording_id) or [])
    for name in ("export_horizontals", "export_verticals"):
        monkeypatch.setattr(exports, name, lambda *a, **k: cut.append(name))
    monkeypatch.setattr(shorts, "build_shorts", lambda *a, **k: cut.append("shorts"))
    out = tmp_path / "out"
    assert cli.scan_cmd(build_parser().parse_args(["scan", "in", "-o", str(out), "--no-overlay"])) == 0
    assert scanned == ["R"] and cut == []
    assert (out / "events.json").exists()


def test_min_speed_flags():
    parse = build_parser().parse_args
    assert parse(["scan", "in", "-o", "out"]).min_speed == 10.0
    assert parse(["scan", "in", "-o", "out", "--min-speed", "0"]).min_speed == 0.0
    args = parse(["shorts", "-o", "out"])
    assert args.min_speed == 40.0


def test_scan_fetches_racebox_sessions_unless_no_fetch(tmp_path, monkeypatch):
    from pathlib import Path

    import videobrowser.cli as cli
    import videobrowser.racebox_stage as rbs

    fetched = []
    monkeypatch.setattr(rbs, "fetch_sessions", lambda d, **kw: fetched.append(d))
    monkeypatch.setattr(cli, "iter_video_paths", lambda *a, **k: [Path("x.mp4")])
    monkeypatch.setattr(cli, "group_recordings", lambda *a, **k: [])
    out = tmp_path / "out"
    cli.scan_cmd(build_parser().parse_args(["scan", "in", "-o", str(out)]))
    assert fetched == [out / "sessions"]
    fetched.clear()
    cli.scan_cmd(build_parser().parse_args(["scan", "in", "-o", str(out), "--no-fetch"]))
    assert fetched == []


def test_scan_date_keeps_only_that_day(tmp_path, monkeypatch):
    from pathlib import Path

    import videobrowser.cli as cli
    import videobrowser.racebox_stage as rbs

    recs = ["A", "B"]
    monkeypatch.setattr(cli, "iter_video_paths", lambda *a, **k: [Path("x.mp4")])
    monkeypatch.setattr(cli, "group_recordings", lambda *a, **k: list(recs))
    monkeypatch.setattr(cli, "sort_by_start", lambda r: r)
    monkeypatch.setattr(cli, "drop_duplicates", lambda r: r)
    monkeypatch.setattr(cli, "recording_day", lambda r: {"A": "2026-08-01", "B": "2026-08-02"}[r])
    monkeypatch.setattr(cli, "RiderTracker", lambda **k: None)
    monkeypatch.setattr(rbs, "fetch_sessions", lambda d, **kw: None)
    scanned = []
    monkeypatch.setattr(cli, "scan_recording", lambda r, *a: scanned.append(r) or [])
    out = tmp_path / "out"
    args = build_parser().parse_args(["scan", "in", "-o", str(out), "--date", "2026-08-02"])
    assert cli.scan_cmd(args) == 0 and scanned == ["B"]
    args = build_parser().parse_args(["scan", "in", "-o", str(out), "--date", "2026-09-01"])
    assert cli.scan_cmd(args) == 1


def test_facing_command_stores_the_facing_without_a_scan(tmp_path, monkeypatch):
    from pathlib import Path

    import videobrowser.cli as cli
    import videobrowser.exports as exports
    import videobrowser.pipeline as pipeline
    from videobrowser.db import ScanDB

    out = tmp_path / "out"
    db = ScanDB.open(out / "scan.sqlite")
    db.save_recording_meta("GX", "rear", 0, 1.0, "2026-08-14", 60.0, [{"path": "x.mp4", "offset": 0.0, "duration": 60.0}])
    db.close()
    monkeypatch.setattr(exports, "recording_from_meta", lambda row: Path(row["recording_id"]))
    seen = []
    monkeypatch.setattr(
        pipeline, "facing_for_recording", lambda rec, size, profile: seen.append(profile.name) or "front"
    )
    assert cli.main(["facing", "-o", str(out)]) == 0
    assert seen == ["rear"]
    db = ScanDB.open(out / "scan.sqlite")
    assert db.recording_metas()[0]["facing"] == "front"
    db.close()


def test_facing_set_by_hand_wins_and_is_kept(tmp_path, monkeypatch):
    import videobrowser.cli as cli
    import videobrowser.pipeline as pipeline
    from videobrowser.db import ScanDB

    out = tmp_path / "out"
    db = ScanDB.open(out / "scan.sqlite")
    db.save_recording_meta("2026-08-14_GX-0014", "rear", 0, 1.0, "2026-08-14", 60.0, [])
    db.close()
    monkeypatch.setattr(pipeline, "facing_for_recording", lambda *a: pytest.fail("measured despite --set"))
    assert cli.main(["facing", "-o", str(out), "--set", "GX=front"]) == 0
    db = ScanDB.open(out / "scan.sqlite")
    assert db.recording_metas()[0]["facing"] == "front"
    assert db.facing_override("2026-08-15_GX-0020") == "front"
    assert db.facing_override("2026-08-15_GH-0020") is None
    db.close()
    assert cli.main(["facing", "-o", str(out), "--set", "GX=up"]) == 2


def test_facing_clear_measures_again(tmp_path, monkeypatch):
    from pathlib import Path

    import videobrowser.cli as cli
    import videobrowser.exports as exports
    import videobrowser.pipeline as pipeline
    from videobrowser.db import ScanDB

    out = tmp_path / "out"
    db = ScanDB.open(out / "scan.sqlite")
    db.save_recording_meta("2026-08-14_GX-0014", "rear", 0, 1.0, "2026-08-14", 60.0, [])
    db.set_facing_override("GX", "front")
    db.close()
    monkeypatch.setattr(exports, "recording_from_meta", lambda row: Path(row["recording_id"]))
    monkeypatch.setattr(pipeline, "facing_for_recording", lambda *a: "rear")
    assert cli.main(["facing", "-o", str(out), "--clear"]) == 0
    db = ScanDB.open(out / "scan.sqlite")
    assert db.facing_override("2026-08-14_GX-0014") is None
    assert db.recording_metas()[0]["facing"] == "rear"
    db.close()
