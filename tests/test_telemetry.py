import struct

from videobrowser.telemetry import LeanSample, _angles_from_vectors, _walk_grav_accl, lean_at, leaned


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
