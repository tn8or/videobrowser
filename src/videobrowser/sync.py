"""Put every camera of a day on one clock.

Each recording starts from its GoPro GPS time (``GPSU``). That stamp runs a
second or two ahead of the picture, by a different amount on every file, so
two cameras on the same bike can disagree by a few seconds. Two refinements
line them up:

1. **GPS track**: cameras on one bike (or the rider's helmet) drive the same
   line, so the time shift that lays one camera's GPS track over the other's
   is the error between their clocks (``telemetry.gopro_clock_offset``). When a
   RaceBox session overlaps, every camera is laid on that log instead, which
   also fixes the absolute time.
2. **Audio**: front and rear cameras hear the engine differently, but throttle
   and gear changes still move both loudness envelopes at the same instant.
   The envelopes are cross-correlated within a second of the GPS estimate,
   and the result is used only when the match is clear.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from videobrowser.probe import ProbeError, probe
from videobrowser.telemetry import extract_gpmd_bytes, gopro_clock_offset, gps_fixes

# Envelope sample rate for audio alignment, Hz.
ENV_RATE = 50
# Search this far either side of the GPS estimate, seconds.
AUDIO_SEARCH_S = 1.0
# Seconds of shared audio compared (taken from the middle of the overlap).
AUDIO_SPAN_S = 90.0
# Accept an audio match only above this normalised correlation ...
AUDIO_MIN_PEAK = 0.3
# ... and when it beats the best lag outside +/-0.15 s by this ratio.
AUDIO_MIN_MARGIN = 1.2


@dataclass(frozen=True)
class ChapterRef:
    path: Path
    offset: float
    duration: float


@dataclass
class CameraClock:
    recording_id: str
    start_utc: float | None
    duration: float
    chapters: list[ChapterRef]
    # How start_utc was settled: "gps", "racebox", "track", plus "+audio".
    source: str = "gps"
    _fixes: np.ndarray | None = field(default=None, repr=False)

    @property
    def end_utc(self) -> float | None:
        return None if self.start_utc is None else self.start_utc + self.duration

    def utc(self, t: float) -> float:
        return float(self.start_utc or 0.0) + t


def _chapter_blob(chapter: ChapterRef) -> bytes | None:
    try:
        info = probe(chapter.path)
    except (ProbeError, OSError):
        return None
    if not info.has_gpmd or info.gpmd_index is None:
        return None
    return extract_gpmd_bytes(chapter.path, info.gpmd_index)


def camera_fixes(clock: CameraClock) -> np.ndarray:
    """GPS fixes of all chapters as ``(utc, lat, lon)`` rows (raw GPSU time)."""
    if clock._fixes is None:
        parts = []
        for chapter in clock.chapters:
            blob = _chapter_blob(chapter)
            if blob:
                fixes = gps_fixes(blob)
                if len(fixes):
                    parts.append(fixes)
        clock._fixes = np.concatenate(parts) if parts else np.zeros((0, 3))
    return clock._fixes


def overlap(a: CameraClock, b: CameraClock) -> tuple[float, float] | None:
    if a.start_utc is None or b.start_utc is None:
        return None
    lo = max(a.start_utc, b.start_utc)
    hi = min(a.end_utc or lo, b.end_utc or lo)
    return (lo, hi) if hi - lo >= 10.0 else None


def clusters(clocks: list[CameraClock]) -> list[list[CameraClock]]:
    """Groups of recordings whose time ranges overlap (one outing each)."""
    timed = sorted((c for c in clocks if c.start_utc is not None), key=lambda c: c.start_utc)
    groups: list[list[CameraClock]] = []
    group_end = -np.inf
    for clock in timed:
        if groups and clock.start_utc < group_end:
            groups[-1].append(clock)
            group_end = max(group_end, clock.end_utc)
        else:
            groups.append([clock])
            group_end = clock.end_utc
    return groups


def track_offset(other: np.ndarray, reference: np.ndarray) -> float:
    """Seconds to add to *other*'s clock so its GPS track lies on *reference*'s."""
    if len(other) < 40 or len(reference) < 40:
        return 0.0
    return gopro_clock_offset(
        other[:, 0], other[:, 1], other[:, 2], reference[:, 0], reference[:, 1], reference[:, 2]
    )


# --- audio ----------------------------------------------------------------


def _decode_audio(path: Path, start: float, duration: float, rate: int = 4000) -> np.ndarray:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, start):.3f}", "-i", str(path), "-t", f"{duration:.3f}",
        "-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0


def audio_window(clock: CameraClock, t0: float, duration: float, rate: int = 4000) -> np.ndarray:
    """Mono audio for recording time ``[t0, t0 + duration)``, across chapters."""
    parts: list[np.ndarray] = []
    for chapter in clock.chapters:
        lo = max(t0, chapter.offset)
        hi = min(t0 + duration, chapter.offset + chapter.duration)
        if hi - lo <= 0.05:
            continue
        parts.append(_decode_audio(chapter.path, lo - chapter.offset, hi - lo, rate))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def envelope(samples: np.ndarray, rate: int = 4000) -> np.ndarray:
    """Loudness changes at ``ENV_RATE`` Hz: log RMS minus its 1 s running mean."""
    hop = rate // ENV_RATE
    n = len(samples) // hop
    if n < ENV_RATE * 4:
        return np.zeros(0)
    frames = samples[: n * hop].reshape(n, hop)
    level = np.log(np.sqrt(np.mean(frames * frames, axis=1)) + 1e-4)
    kernel = np.ones(ENV_RATE) / ENV_RATE
    trend = np.convolve(level, kernel, mode="same")
    return level - trend


def best_lag(reference: np.ndarray, other: np.ndarray, max_lag_s: float = AUDIO_SEARCH_S) -> tuple[float, float, float]:
    """Lag L (s) maximising corr(reference(t + L), other(t)).

    Returns ``(lag_s, peak, runner_up)`` where the runner-up is the best
    correlation at least 0.15 s away from the peak.
    """
    max_lag = int(round(max_lag_s * ENV_RATE))
    n = min(len(reference), len(other))
    if n <= 2 * max_lag + ENV_RATE * 4:
        return 0.0, 0.0, 0.0
    core = other[max_lag : n - max_lag]
    core = (core - core.mean()) / (core.std() + 1e-9)
    scores = []
    for lag in range(-max_lag, max_lag + 1):
        ref = reference[max_lag + lag : n - max_lag + lag]
        ref = (ref - ref.mean()) / (ref.std() + 1e-9)
        scores.append(float(np.mean(ref * core)))
    arr = np.asarray(scores)
    i = int(np.argmax(arr))
    away = int(round(0.15 * ENV_RATE))
    mask = np.ones(len(arr), dtype=bool)
    mask[max(0, i - away) : i + away + 1] = False
    runner = float(arr[mask].max()) if mask.any() else 0.0
    return (i - max_lag) / ENV_RATE, float(arr[i]), runner


def audio_offset(reference: CameraClock, other: CameraClock) -> float | None:
    """Correction (s) to add to *other*'s start from engine sound, or None."""
    span = overlap(reference, other)
    if span is None:
        return None
    lo, hi = span
    length = min(AUDIO_SPAN_S, hi - lo - 2 * AUDIO_SEARCH_S)
    if length < 20.0:
        return None
    mid = 0.5 * (lo + hi)
    u0 = mid - 0.5 * length
    pad = AUDIO_SEARCH_S
    ref_env = envelope(audio_window(reference, u0 - pad - reference.start_utc, length + 2 * pad))
    oth_env = envelope(audio_window(other, u0 - pad - other.start_utc, length + 2 * pad))
    lag, peak, runner = best_lag(ref_env, oth_env)
    if peak < AUDIO_MIN_PEAK or peak < AUDIO_MIN_MARGIN * max(runner, 1e-6):
        return None
    return lag


# --- day alignment --------------------------------------------------------


def align(
    clocks: list[CameraClock],
    racebox_offset=None,
    use_audio: bool = True,
) -> None:
    """Refine ``start_utc`` of every clock in place.

    *racebox_offset(clock) -> float | None* lays one camera on the RaceBox
    log (absolute time); without it cameras are aligned to each other only.
    """
    rb: dict[str, float] = {}
    if racebox_offset is not None:
        for clock in clocks:
            if clock.start_utc is None:
                continue
            try:
                offset = racebox_offset(clock)
            except Exception as exc:
                print(f"  sync {clock.recording_id}: racebox failed ({exc})")
                offset = None
            if offset:
                rb[clock.recording_id] = offset

    for group in clusters(clocks):
        if len(group) == 1:
            clock = group[0]
            if clock.recording_id in rb:
                clock.start_utc += rb[clock.recording_id]
                clock.source = "racebox"
            continue
        # Reference: a RaceBox-aligned camera, else the one with most GPS.
        reference = max(
            group,
            key=lambda c: (c.recording_id in rb, len(camera_fixes(c)), c.duration),
        )
        ref_shift = rb.get(reference.recording_id, 0.0)
        ref_fixes = camera_fixes(reference).copy()
        if len(ref_fixes):
            ref_fixes[:, 0] += ref_shift
        if ref_shift:
            reference.start_utc += ref_shift
            reference.source = "racebox"
        for clock in group:
            if clock is reference:
                continue
            if clock.recording_id in rb:
                clock.start_utc += rb[clock.recording_id]
                clock.source = "racebox"
            else:
                shift = track_offset(camera_fixes(clock), ref_fixes)
                if shift:
                    clock.start_utc += shift
                    clock.source = "track"
            if use_audio:
                lag = audio_offset(reference, clock)
                if lag is not None:
                    clock.start_utc += lag
                    clock.source += "+audio"
            print(
                f"  sync {clock.recording_id} -> {reference.recording_id}: {clock.source}"
            )
