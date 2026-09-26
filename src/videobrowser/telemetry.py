from __future__ import annotations

import struct
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from videobrowser.probe import VideoInfo

# Reject GPSU samples from before the camera had a real fix (GoPro often emits
# placeholder years like 2015 while acquiring satellites).
_MIN_GPS_YEAR = 2020


@dataclass(frozen=True)
class LeanSample:
    t: float
    lean_deg: float


@dataclass(frozen=True)
class SpeedSample:
    t: float
    speed_kmh: float


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


def _parse_int32(payload: bytes, signed: bool = True) -> list[int]:
    n = len(payload) // 4
    fmt = "i" if signed else "I"
    return list(struct.unpack(">" + fmt * n, payload[: n * 4]))


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


def _walk_gps5(data: bytes) -> list[float]:
    """Return per-sample 2D ground speed in km/h from GPS5 payloads.

    GPS5 samples are ``[lat, lon, alt, speed_2d, speed_3d]`` int32 values
    divided by the sibling ``SCAL`` divisors. Only the 2D speed (index 3),
    converted from m/s to km/h, is returned.
    """
    speeds: list[float] = []

    def walk(blob: bytes) -> None:
        items = list(_klv_iter(blob))
        scal: list[float] | None = None
        for key, type_, size, repeat, payload in items:
            if key == b"SCAL" and payload:
                if type_ == ord("s"):
                    scal = [float(v) for v in _parse_int16(payload)]
                elif type_ == ord("S"):
                    n = len(payload) // 2
                    scal = [float(v) for v in struct.unpack(">" + "H" * n, payload[: n * 2])]
                elif type_ in (ord("l"), ord("L")):
                    scal = [float(v) for v in _parse_int32(payload, signed=type_ == ord("l"))]
                elif type_ == ord("f"):
                    scal = _parse_floats(payload)
        for key, type_, size, repeat, payload in items:
            if type_ == 0:
                walk(payload)
                continue
            if key == b"GPS5" and payload:
                vals = _parse_int32(payload, signed=True)
                div = scal[3] if scal and len(scal) > 3 and scal[3] else 1.0
                for i in range(0, len(vals) - 4, 5):
                    speeds.append((vals[i + 3] / div) * 3.6)

    walk(data)
    return speeds


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


def lean_series(info: VideoInfo, blob: bytes | None = None) -> list[LeanSample]:
    if not info.has_gpmd or info.gpmd_index is None:
        return []
    if blob is None:
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


def speed_series(info: VideoInfo, blob: bytes | None = None) -> list[SpeedSample]:
    """Extract GoPro GPS ground speed (km/h) sampled evenly over the clip.

    Pass a pre-extracted GPMD *blob* to avoid re-running ffmpeg when the caller
    already fetched it (e.g. alongside :func:`lean_series`).
    """
    if not info.has_gpmd or info.gpmd_index is None:
        return []
    if blob is None:
        blob = extract_gpmd_bytes(info.path, info.gpmd_index)
    if not blob:
        return []
    speeds = _walk_gps5(blob)
    n = len(speeds)
    if n < 2:
        return []
    duration = info.duration if info.duration and info.duration > 0 else float(n)
    times = np.linspace(0.0, duration, n)
    return [SpeedSample(t=float(t), speed_kmh=float(s)) for t, s in zip(times, speeds)]


def _parse_gpsu(raw: str) -> datetime | None:
    """Parse a GPMD ``GPSU`` string (``YYMMDDHHMMSS.sss``) as UTC."""
    text = raw.strip()
    for fmt in ("%y%m%d%H%M%S.%f", "%y%m%d%H%M%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _walk_gpsu(data: bytes) -> list[str]:
    samples: list[str] = []

    def walk(blob: bytes) -> None:
        for key, type_, _size, _repeat, payload in _klv_iter(blob):
            if key == b"GPSU" and payload:
                samples.append(
                    payload.split(b"\x00", 1)[0].decode("ascii", "replace").strip()
                )
            elif type_ == 0:
                walk(payload)

    walk(data)
    return samples


def gps_start_utc(blob: bytes, duration: float | None = None) -> float | None:
    """UTC epoch of video t=0 inferred from GPMD ``GPSU`` samples.

    GoPro ``creation_time`` tags are often the camera wall clock mislabeled as
    ``Z`` (and the clock itself can be wrong). GPS time is authoritative when a
    fix exists. Early placeholder GPSU rows (wrong year) are skipped; the first
    valid fix is back-extrapolated to t=0 using the GPSU sample spacing.
    """
    samples = _walk_gpsu(blob)
    if not samples:
        return None
    valid: list[tuple[int, datetime]] = []
    for index, raw in enumerate(samples):
        parsed = _parse_gpsu(raw)
        if parsed is not None and parsed.year >= _MIN_GPS_YEAR:
            valid.append((index, parsed))
    if not valid:
        return None
    index0, stamp0 = valid[0]
    if index0 == 0:
        return stamp0.timestamp()
    if duration is not None and duration > 0 and len(samples) > 1:
        step = duration / len(samples)
    else:
        step = 1.0  # GoPro GPSU is typically ~1 Hz
    return stamp0.timestamp() - index0 * step


def creation_utc_prefer_gps(
    info: VideoInfo, blob: bytes | None = None
) -> float | None:
    """Prefer GPMD GPS time for the recording start; fall back to file tags."""
    if info.has_gpmd and info.gpmd_index is not None:
        if blob is None:
            blob = extract_gpmd_bytes(info.path, info.gpmd_index)
        if blob:
            gps = gps_start_utc(blob, info.duration)
            if gps is not None:
                return gps
    return info.creation_utc


def _parse_scal(type_: int, payload: bytes) -> list[float] | None:
    if type_ == ord("s"):
        return [float(v) for v in _parse_int16(payload)]
    if type_ == ord("S"):
        n = len(payload) // 2
        return [float(v) for v in struct.unpack(">" + "H" * n, payload[: n * 2])]
    if type_ in (ord("l"), ord("L")):
        return [float(v) for v in _parse_int32(payload, signed=type_ == ord("l"))]
    if type_ == ord("f"):
        return _parse_floats(payload)
    return None


def gps_fixes(blob: bytes) -> np.ndarray:
    """Valid GPS fixes as an ``(N, 3)`` array of ``(utc, lat_deg, lon_deg)``.

    Times come from each packet's ``GPSU``. On these cameras that stamp runs
    ahead of the fix it labels; use :func:`gopro_clock_offset` before lining
    the track up with another GPS log.
    """
    packets: list[tuple[datetime, list[tuple[float, float]]]] = []
    current_stamp: datetime | None = None
    current_pts: list[tuple[float, float]] = []

    def flush() -> None:
        nonlocal current_pts
        if (
            current_stamp is not None
            and current_stamp.year >= _MIN_GPS_YEAR
            and current_pts
        ):
            packets.append((current_stamp, current_pts))
        current_pts = []

    def walk(data: bytes) -> None:
        nonlocal current_stamp
        items = list(_klv_iter(data))
        scal: list[float] | None = None
        for key, type_, _size, _repeat, payload in items:
            if key == b"SCAL" and payload:
                scal = _parse_scal(type_, payload)
        for key, type_, _size, _repeat, payload in items:
            if type_ == 0:
                walk(payload)
                continue
            if key == b"GPSU" and payload:
                flush()
                raw = payload.split(b"\x00", 1)[0].decode("ascii", "replace").strip()
                current_stamp = _parse_gpsu(raw)
            elif key == b"GPS5" and payload and scal and len(scal) >= 2:
                vals = _parse_int32(payload, signed=True)
                lat_div = scal[0] or 1.0
                lon_div = scal[1] or 1.0
                n = len(vals) // 5
                for j in range(n):
                    lat = vals[5 * j] / lat_div
                    lon = vals[5 * j + 1] / lon_div
                    if abs(lat) < 1.0 or abs(lon) < 0.01:
                        continue
                    current_pts.append((lat, lon))

    walk(blob)
    flush()
    if not packets:
        return np.zeros((0, 3), dtype=np.float64)
    rows: list[tuple[float, float, float]] = []
    for i, (stamp, pts) in enumerate(packets):
        gap = 1.0
        if i + 1 < len(packets):
            gap = (packets[i + 1][0] - stamp).total_seconds()
        if gap <= 0.0 or gap > 3.0:
            gap = 1.0
        n = len(pts)
        base = stamp.timestamp()
        for j, (lat, lon) in enumerate(pts):
            rows.append((base + (j / n) * gap, lat, lon))
    return np.asarray(rows, dtype=np.float64)


def gopro_clock_offset(
    gopro_utc: np.ndarray,
    gopro_lat: np.ndarray,
    gopro_lon: np.ndarray,
    rb_utc: np.ndarray,
    rb_lat: np.ndarray,
    rb_lon: np.ndarray,
) -> float:
    """Seconds to add to GoPro GPS time so positions match a RaceBox log.

    The result is negative when the GoPro stamp is ahead of the track.
    Returns ``0.0`` when the tracks do not overlap or never agree.
    """
    if gopro_utc.size < 40 or rb_utc.size < 40:
        return 0.0
    lat = np.asarray(rb_lat, dtype=np.float64)
    lon = np.asarray(rb_lon, dtype=np.float64)
    # RaceBox stores decimal minutes (degrees × 60), not decimal degrees.
    if float(np.nanmedian(np.abs(lat))) > 90.0:
        lat = lat / 60.0
        lon = lon / 60.0
    order = np.argsort(rb_utc, kind="mergesort")
    utc = np.asarray(rb_utc, dtype=np.float64)[order]
    lat = lat[order]
    lon = lon[order]
    # Split on session gaps so interpolation cannot bridge two outings.
    cuts = np.flatnonzero(np.diff(utc) > 2.0) + 1
    segments = np.split(np.arange(utc.size), cuts)
    g_utc = np.asarray(gopro_utc, dtype=np.float64)
    g_lat = np.asarray(gopro_lat, dtype=np.float64)
    g_lon = np.asarray(gopro_lon, dtype=np.float64)
    if g_utc.size > 2500:
        step = int(np.ceil(g_utc.size / 2500))
        g_utc, g_lat, g_lon = g_utc[::step], g_lat[::step], g_lon[::step]

    def median_error(dt: float, sign: float) -> float | None:
        parts: list[np.ndarray] = []
        query = g_utc + dt
        for idx in segments:
            if idx.size < 5:
                continue
            seg_t = utc[idx]
            mask = (query >= seg_t[0]) & (query <= seg_t[-1])
            if int(mask.sum()) < 5:
                continue
            lat_i = np.interp(query[mask], seg_t, lat[idx])
            lon_i = np.interp(query[mask], seg_t, lon[idx] * sign)
            dlat = (lat_i - g_lat[mask]) * 111_320.0
            dlon = (lon_i - g_lon[mask]) * 111_320.0 * np.cos(np.deg2rad(g_lat[mask]))
            parts.append(np.hypot(dlat, dlon))
        if not parts:
            return None
        err = np.concatenate(parts)
        if err.size < 40:
            return None
        return float(np.median(err))

    best: tuple[float, float] | None = None
    for sign in (1.0, -1.0):
        coarse: tuple[float, float] | None = None
        for dt in np.linspace(-15.0, 15.0, 61):
            err = median_error(float(dt), sign)
            if err is None:
                continue
            if coarse is None or err < coarse[0]:
                coarse = (err, float(dt))
        if coarse is None:
            continue
        for dt in np.linspace(coarse[1] - 0.5, coarse[1] + 0.5, 21):
            err = median_error(float(dt), sign)
            if err is None:
                continue
            if best is None or err < best[0]:
                best = (err, float(dt))
    if best is None or best[0] > 40.0:
        return 0.0
    return best[1]


def _interp(samples, t: float, attr: str) -> float | None:
    if not samples:
        return None
    if t <= samples[0].t:
        return getattr(samples[0], attr)
    if t >= samples[-1].t:
        return getattr(samples[-1], attr)
    lo, hi = 0, len(samples) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if samples[mid].t <= t:
            lo = mid
        else:
            hi = mid
    a, b = samples[lo], samples[hi]
    if b.t <= a.t:
        return getattr(a, attr)
    frac = (t - a.t) / (b.t - a.t)
    return getattr(a, attr) + frac * (getattr(b, attr) - getattr(a, attr))


def speed_at(samples: list[SpeedSample], t: float) -> float | None:
    return _interp(samples, t, "speed_kmh")


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
