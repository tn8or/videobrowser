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
