import struct
from datetime import datetime, timezone

import numpy as np

from videobrowser.telemetry import (
    LeanSample,
    _angles_from_vectors,
    _parse_gpsu,
    _walk_grav_accl,
    gopro_clock_offset,
    gps_fixes,
    gps_start_utc,
    lean_at,
    leaned,
)


def _klv(key: bytes, type_code: int, payload: bytes) -> bytes:
    size = 4 if type_code == ord("f") else 1
    if type_code == 0:
        size = 1
        repeat = len(payload)
        header = key + bytes([0, 1]) + repeat.to_bytes(2, "big")
        pad = (4 - (len(payload) % 4)) % 4
        return header + payload + (b"\x00" * pad)
    repeat = len(payload) // size
    header = key + bytes([type_code, size]) + repeat.to_bytes(2, "big")
    pad = (4 - (len(payload) % 4)) % 4
    return header + payload + (b"\x00" * pad)


def test_walk_grav():
    upright = struct.pack(">fff", 0.0, 1.0, 0.0) * 8
    tilted = struct.pack(">fff", 0.5, 0.86, 0.0)
    blob = _klv(b"GRAV", ord("f"), upright + tilted)
    grav, accl, _ = _walk_grav_accl(blob)
    assert len(grav) == 9
    assert accl == []
    angles = _angles_from_vectors(grav)
    assert angles[0] < 1.0
    assert angles[-1] > 20


def test_lean_gate():
    samples = [LeanSample(0.0, 5.0), LeanSample(2.0, 35.0)]
    assert lean_at(samples, 1.0) == 20.0
    assert not leaned(samples, 0.0, 20)
    assert leaned(samples, 2.0, 20)
    assert leaned([], 1.0, 20)


def test_parse_gpsu():
    assert _parse_gpsu("260815094244.950") == datetime(
        2026, 8, 15, 9, 42, 44, 950000, tzinfo=timezone.utc
    )
    assert _parse_gpsu("260815094246") == datetime(
        2026, 8, 15, 9, 42, 46, tzinfo=timezone.utc
    )
    assert _parse_gpsu("not-a-time") is None


def _gpsu_klv(raw: str) -> bytes:
    payload = raw.encode("ascii")
    return _klv(b"GPSU", ord("c"), payload)


def test_gps_start_skips_placeholder_years_and_backdates():
    # 16 placeholder fixes, then a real lock — mirrors GH010013.
    placeholder = _gpsu_klv("151024094229.000")
    real = _gpsu_klv("260815094244.950")
    blob = placeholder * 16 + real
    start = gps_start_utc(blob, duration=1078.72)
    assert start is not None
    # 16 samples over 1078.72s => ~1.039s/sample back from first valid.
    expected = datetime(2026, 8, 15, 9, 42, 44, 950000, tzinfo=timezone.utc).timestamp()
    expected -= 16 * (1078.72 / 17)
    assert abs(start - expected) < 0.01


def test_gps_start_uses_first_sample_when_valid_at_t0():
    blob = _gpsu_klv("260814124350.689")
    start = gps_start_utc(blob, duration=600.0)
    assert start == datetime(
        2026, 8, 14, 12, 43, 50, 689000, tzinfo=timezone.utc
    ).timestamp()


def _track():
    t = np.linspace(1_000.0, 1_080.0, 800)
    lat = 55.99 + 0.00004 * (t - t[0]) + 0.0008 * np.sin((t - t[0]) / 3.0)
    lon = 13.11 + 0.00003 * (t - t[0]) + 0.0005 * np.cos((t - t[0]) / 4.0)
    return t, lat, lon


def test_clock_offset_pulls_fast_gopro_stamps_back():
    t, lat, lon = _track()
    # GoPro GPSU is 2.3s ahead of the position it labels.
    offset = gopro_clock_offset(t + 2.3, lat, lon, t, lat, lon)
    assert abs(offset - (-2.3)) < 0.15


def test_clock_offset_accepts_racebox_decimal_minutes():
    t, lat, lon = _track()
    # RaceBox writes degrees×60, and this export's longitude sign is flipped.
    offset = gopro_clock_offset(t + 1.7, lat, lon, t, lat * 60.0, -lon * 60.0)
    assert abs(offset - (-1.7)) < 0.15


def test_clock_offset_is_zero_without_overlap():
    t, lat, lon = _track()
    assert gopro_clock_offset(t + 10_000.0, lat, lon, t, lat, lon) == 0.0


def test_gps_fixes_reads_scaled_position_and_gpsu():
    def klv(key, type_code, size, payload):
        repeat = len(payload) // size
        header = key + bytes([type_code, size]) + repeat.to_bytes(2, "big")
        pad = (4 - (len(payload) % 4)) % 4
        return header + payload + (b"\x00" * pad)

    scal = struct.pack(">hhhhh", 10, 10, 1, 1, 1)
    gps5 = struct.pack(">iiiii", 555, 132, 0, 0, 0)
    blob = (
        klv(b"GPSU", ord("c"), 1, b"260815094315.000\x00")
        + klv(b"SCAL", ord("s"), 2, scal)
        + klv(b"GPS5", ord("l"), 4, gps5)
    )
    fixes = gps_fixes(blob)
    assert len(fixes) == 1
    assert fixes[0, 1] == 55.5
    assert fixes[0, 2] == 13.2
    assert abs(fixes[0, 0] - datetime(2026, 8, 15, 9, 43, 15, tzinfo=timezone.utc).timestamp()) < 0.01
