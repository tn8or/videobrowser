from __future__ import annotations

import struct
import subprocess
from dataclasses import dataclass

import numpy as np

from videobrowser.probe import VideoInfo


@dataclass(frozen=True)
class LeanSample:
    t: float
    lean_deg: float


def _klv_iter(data: bytes, start: int = 0, end: int | None = None):
    offset = start
    limit = len(data) if end is None else end
    while offset + 8 <= limit:
        key = data[offset : offset + 4]
        type_ = data[offset + 4]
        size = data[offset + 5]
        repeat = int.from_bytes(data[offset + 6 : offset + 8], "big")
        payload_len = size * repeat
        payload_start = offset + 8
        payload_end = payload_start + payload_len
        if payload_end > limit:
            break
        yield key, type_, size, repeat, data[payload_start:payload_end]
        pad = (4 - (payload_len % 4)) % 4
        offset = payload_end + pad


def _parse_floats(payload: bytes) -> list[float]:
    n = len(payload) // 4
    return list(struct.unpack(">" + "f" * n, payload[: n * 4]))


def _parse_int16(payload: bytes) -> list[int]:
    n = len(payload) // 2
    return list(struct.unpack(">" + "h" * n, payload[: n * 2]))


def extract_gpmd_bytes(path, stream_index: int) -> bytes | None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-map",
        f"0:{stream_index}",
        "-codec",
        "copy",
        "-copy_unknown",
        "-f",
        "data",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return None
    return proc.stdout


def _walk_grav_accl(data: bytes) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]], list[int]]:
    grav: list[tuple[float, ...]] = []
    accl: list[tuple[float, ...]] = []
    stamps: list[int] = []

    def walk(blob: bytes) -> None:
        for key, type_, size, repeat, payload in _klv_iter(blob):
            if type_ == 0:
                walk(payload)
                continue
            if key == b"STMP" and payload:
                if size == 8:
                    stamps.append(int.from_bytes(payload[:8], "big"))
                elif size == 4:
                    stamps.append(int.from_bytes(payload[:4], "big"))
            elif key == b"GRAV" and payload:
                vals = _parse_floats(payload) if type_ == ord("f") else []
                for i in range(0, len(vals) - 2, 3):
                    grav.append((vals[i], vals[i + 1], vals[i + 2]))
            elif key == b"ACCL" and payload:
                if type_ == ord("f"):
                    vals = _parse_floats(payload)
                elif chr(type_) in "sS":
                    raw = _parse_int16(payload)
                    vals = [v / 1000.0 for v in raw]
                else:
                    continue
                for i in range(0, len(vals) - 2, 3):
                    accl.append((vals[i], vals[i + 1], vals[i + 2]))

    walk(data)
    return grav, accl, stamps


def _angles_from_vectors(vectors: list[tuple[float, ...]]) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-6, None)
    unit = arr / norms
    baseline = np.median(unit[: max(1, min(len(unit), 50))], axis=0)
    bnorm = np.linalg.norm(baseline)
    if bnorm < 1e-6:
        return np.zeros(len(unit))
    baseline = baseline / bnorm
    dots = np.clip(unit @ baseline, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def lean_series(info: VideoInfo) -> list[LeanSample]:
    if not info.has_gpmd or info.gpmd_index is None:
        return []
    blob = extract_gpmd_bytes(info.path, info.gpmd_index)
    if not blob:
        return []
    grav, accl, stamps = _walk_grav_accl(blob)
    vectors = grav or accl
    if len(vectors) < 8:
        return []
    angles = _angles_from_vectors(vectors)
    n = len(angles)
    if stamps and len(stamps) >= 2:
        t0 = stamps[0]
        times = []
        for i in range(n):
            idx = min(i, len(stamps) - 1)
            times.append((stamps[idx] - t0) / 1_000_000.0)
        duration = times[-1] if times[-1] > 1e-3 else info.duration
        if duration <= 0:
            duration = info.duration or 1.0
        # STMP is often per-DEVC, not per sample; fall back to even spacing if crazy.
        if times[-1] <= 0 or times[-1] > info.duration * 4:
            times = list(np.linspace(0.0, info.duration or 1.0, n))
    else:
        times = list(np.linspace(0.0, info.duration or 1.0, n))
    return [LeanSample(t=float(t), lean_deg=float(a)) for t, a in zip(times, angles)]


def lean_at(samples: list[LeanSample], t: float) -> float | None:
    if not samples:
        return None
    if t <= samples[0].t:
        return samples[0].lean_deg
    if t >= samples[-1].t:
        return samples[-1].lean_deg
    lo, hi = 0, len(samples) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if samples[mid].t <= t:
            lo = mid
        else:
            hi = mid
    a, b = samples[lo], samples[hi]
    if b.t <= a.t:
        return a.lean_deg
    frac = (t - a.t) / (b.t - a.t)
    return a.lean_deg + frac * (b.lean_deg - a.lean_deg)


def leaned(samples: list[LeanSample], t: float, gate_deg: float) -> bool:
    if gate_deg <= 0:
        return True
    value = lean_at(samples, t)
    if value is None:
        return True
    return value >= gate_deg
