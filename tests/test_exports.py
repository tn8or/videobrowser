from pathlib import Path

import pytest

from videobrowser.db import ScanDB
from videobrowser.gopro import Chapter, Recording
from videobrowser.probe import VideoInfo
from videobrowser.riders import TrackPoint


def _scan(out: Path, source: str) -> None:
    db = ScanDB.open(out / "scan.sqlite")
    db.save_recording_meta("R", "cockpit", 0, 1785571200.0, "2026-08-01", 60.0,
                           [{"path": "R.mp4", "offset": 0.0, "duration": 60.0}], source)
    db.replace_events("R", [("R", "approaching_rider", 12.0, 14.0, 0.9, "7", None, "{}")])
    db.replace_clips("R", [(5.0, 20.0, "approaching_rider", 0.9, None, None)])
    db.save_tracks("R", [TrackPoint(13.0, 10, 20, 110, 220, 0.9, 0, 7, 640, 360),
                         TrackPoint(40.0, 10, 20, 110, 220, 0.9, 0, 7, 640, 360)])
    db.close()


def _recording(row):
    info = VideoInfo(Path("R.mp4"), 60.0, 1920, 1080, 30.0, 0, "h264", False, None, 1785571200.0)
    rec = Recording("R", "", "R")
    rec.chapters.append(Chapter(info.path, "", 1, "R", 60.0, 0.0, info))
    return rec


@pytest.mark.parametrize("source,overlay", [("gps", True), ("racebox-motion", True), ("clock", True), ("gps", False)])
def test_verticals_follow_scan_tracks_and_overlay_only_when_sure(tmp_path, monkeypatch, source, overlay):
    import videobrowser.exports as exports
    import videobrowser.racebox_stage as rbs
    import videobrowser.vertical as vertical

    out = tmp_path / "out"
    _scan(out, source)
    monkeypatch.setattr(exports, "recording_from_meta", _recording)
    monkeypatch.setattr(exports, "turns_for_recording", lambda rec: [])
    monkeypatch.setattr(exports, "_racebox_context", lambda *a: object())
    monkeypatch.setattr(rbs, "recording_start_utc", lambda rec, ctx: 1785571200.0)
    calls = []

    def fake_export(recording, window, dest, rotation, samples, overlay_factory=None):
        calls.append((window, samples, overlay_factory))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"v")
        return dest

    monkeypatch.setattr(vertical, "export_vertical", fake_export)
    assert exports.export_verticals(out, overlay=overlay) == 1
    window, samples, factory = calls[0]
    assert [e.track_id for e in window.events] == ["7"]  # the scan's event, for the crop
    assert samples and samples[0].cx != 0.5  # follows the stored rider box
    assert (factory is not None) is (overlay and source != "clock")
    db = ScanDB.open(out / "scan.sqlite")
    assert db.all_clips()[0]["vertical_path"].endswith("R_00005_approaching_rider.mp4")
    db.close()
    # A second run keeps the existing file.
    calls.clear()
    assert exports.export_verticals(out, overlay=overlay) == 0 and calls == []


def test_horizontals_cut_windows_and_drop_files_of_old_windows(tmp_path, monkeypatch):
    import videobrowser.clips as clips
    import videobrowser.exports as exports

    out = tmp_path / "out"
    _scan(out, "clock")
    monkeypatch.setattr(exports, "recording_from_meta", _recording)
    stale = out / "clips" / "2026-08-01" / "R_00099_nearby_rider.mp4"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    cut = []

    def fake_clip(recording, window, dest, rotation=0):
        cut.append(dest.name)
        dest.write_bytes(b"c")
        return dest

    monkeypatch.setattr(clips, "export_clip", fake_clip)
    assert exports.export_horizontals(out) == 1
    assert cut == ["R_00005_approaching_rider.mp4"] and not stale.exists()
    assert exports.export_horizontals(out) == 0
    assert exports.export_horizontals(out, force=True) == 1
