from videobrowser.cli import build_parser


def test_no_landscape_flag():
    args = build_parser().parse_args(["scan", "in", "-o", "out", "--no-landscape"])
    assert args.no_landscape and not args.no_vertical
    args = build_parser().parse_args(["scan", "in", "-o", "out"])
    assert not args.no_landscape


def test_no_overlay_flag_and_racebox_alias():
    parse = build_parser().parse_args
    assert parse(["scan", "in", "-o", "out", "--no-overlay"]).no_overlay
    assert parse(["scan", "in", "-o", "out", "--no-racebox"]).no_overlay
    assert not parse(["scan", "in", "-o", "out"]).no_overlay


def test_scan_builds_each_days_shorts_as_soon_as_the_day_is_done(tmp_path, monkeypatch):
    from pathlib import Path

    import videobrowser.cli as cli
    import videobrowser.shorts as shorts
    from videobrowser.gopro import Chapter, Recording
    from videobrowser.probe import VideoInfo, local_date_label

    def rec(rid, utc):
        info = VideoInfo(Path(f"{rid}.mp4"), 60.0, 1920, 1080, 30.0, 0, "h264", False, None, utc)
        r = Recording(rid, "GX", rid)
        r.chapters.append(Chapter(info.path, "GX", 1, rid, 60.0, 0.0, info))
        return r

    day2, day1a, day1b = rec("D2", 1785657600.0), rec("D1a", 1785571200.0), rec("D1b", 1785574800.0)
    log = []
    monkeypatch.setattr(cli, "iter_video_paths", lambda *a, **k: [Path("x.mp4")])
    monkeypatch.setattr(cli, "group_recordings", lambda *a, **k: [day2, day1b, day1a])

    def fake_scan(recording, config, db, tracker):
        utc = recording.chapters[0].info.creation_utc
        db.save_recording_meta(recording.recording_id, "cockpit", 0, utc, local_date_label(utc), 60.0, [])
        db.mark_recording(recording.recording_id, "cockpit", 60.0, processed=True)
        log.append(("scan", recording.recording_id))
        return []

    monkeypatch.setattr(cli, "scan_recording", fake_scan)
    monkeypatch.setattr(shorts, "build_shorts", lambda out, sessions_dir=None, dates=None: log.append(("shorts", dates)))
    args = build_parser().parse_args(["scan", "in", "-o", str(tmp_path / "out"), "--no-overlay"])
    assert cli.scan_cmd(args) == 0
    assert log == [
        ("scan", "D1a"), ("scan", "D1b"), ("shorts", ["2026-08-01"]),
        ("scan", "D2"), ("shorts", ["2026-08-02"]),
    ]

    # A re-run with nothing new rebuilds no shorts.
    log.clear()
    assert cli.scan_cmd(args) == 0
    assert [entry for entry in log if entry[0] == "shorts"] == []


def test_failed_shorts_are_retried_by_the_next_scan(tmp_path, monkeypatch):
    from pathlib import Path

    import videobrowser.cli as cli
    import videobrowser.shorts as shorts
    from videobrowser.gopro import Chapter, Recording
    from videobrowser.probe import VideoInfo, local_date_label

    info = VideoInfo(Path("R.mp4"), 60.0, 1920, 1080, 30.0, 0, "h264", False, None, 1785571200.0)
    rec = Recording("R", "GX", "R")
    rec.chapters.append(Chapter(info.path, "GX", 1, "R", 60.0, 0.0, info))
    monkeypatch.setattr(cli, "iter_video_paths", lambda *a, **k: [Path("x.mp4")])
    monkeypatch.setattr(cli, "group_recordings", lambda *a, **k: [rec])

    def fake_scan(recording, config, db, tracker):
        if not db.is_processed("R"):
            utc = info.creation_utc
            db.save_recording_meta("R", "cockpit", 0, utc, local_date_label(utc), 60.0, [])
            db.mark_recording("R", "cockpit", 60.0, processed=True)
        return []

    calls = []

    def flaky(out, sessions_dir=None, dates=None):
        calls.append(dates)
        if len(calls) == 1:
            raise RuntimeError("ffmpeg died")

    monkeypatch.setattr(cli, "scan_recording", fake_scan)
    monkeypatch.setattr(shorts, "build_shorts", flaky)
    args = build_parser().parse_args(["scan", "in", "-o", str(tmp_path / "out"), "--no-overlay"])
    assert cli.scan_cmd(args) == 1
    assert cli.scan_cmd(args) == 0  # nothing new, but the failed day is retried
    assert cli.scan_cmd(args) == 0  # and then left alone
    assert calls == [["2026-08-01"], ["2026-08-01"]]
