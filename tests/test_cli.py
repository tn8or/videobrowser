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
    monkeypatch.setattr(shorts, "build_shorts", lambda out, sessions_dir=None, dates=None, **kw: log.append(("shorts", dates)))
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
    fail_on = set()

    def build(out, sessions_dir=None, dates=None, **kw):
        calls.append(dates)
        if len(calls) in fail_on:
            raise RuntimeError("ffmpeg died")

    monkeypatch.setattr(cli, "scan_recording", fake_scan)
    monkeypatch.setattr(shorts, "build_shorts", build)
    args = build_parser().parse_args(["scan", "in", "-o", str(tmp_path / "out"), "--no-overlay"])

    def run():
        calls.clear()
        return cli.scan_cmd(args)

    fail_on.update({1})
    assert run() == 1 and calls == [["2026-08-01"]]  # tried once per scan
    fail_on.clear()
    assert run() == 0 and calls == [["2026-08-01"]]  # nothing new, failed day retried
    assert run() == 0 and calls == []  # and then left alone


def test_shorts_that_keep_failing_are_tried_once_per_scan(tmp_path, monkeypatch):
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

    def broken(out, sessions_dir=None, dates=None, **kw):
        calls.append(dates)
        raise RuntimeError("ffmpeg died")

    monkeypatch.setattr(cli, "scan_recording", fake_scan)
    monkeypatch.setattr(shorts, "build_shorts", broken)
    args = build_parser().parse_args(["scan", "in", "-o", str(tmp_path / "out"), "--no-overlay"])
    for _ in range(2):
        calls.clear()
        assert cli.scan_cmd(args) == 1
        assert calls == [["2026-08-01"]]


def test_min_speed_and_labels_flags():
    parse = build_parser().parse_args
    assert parse(["scan", "in", "-o", "out"]).min_speed == 10.0
    assert parse(["scan", "in", "-o", "out", "--min-speed", "0"]).min_speed == 0.0
    args = parse(["shorts", "-o", "out"])
    assert args.min_speed == 40.0 and not args.labels
    assert parse(["shorts", "-o", "out", "--labels"]).labels
