"""Put recordings without GoPro GPS on the RaceBox clock.

Files re-wrapped without the GoPro telemetry track (e.g. Final Cut ``.mov``)
have no GPS time, speed or heading. A RaceBox log of the same day has all
three; what is missing is where the recording sits on it. The scan measures
how much the picture moves every second (``motion.py``): low in the pits,
high on track. That pattern, pit stops and out-laps above all, lines up with
the RaceBox speed trace at one start time. The camera's own clock
(``creation_time``) only bounds the search: GoPro keeps it within seconds,
but writes local wall time labelled UTC, and an editor that re-wraps the
file may or may not convert it, so both readings are tried. Sessions on a
track day repeat at a fixed rhythm, so the windows stay narrow; when two
start times match about equally the recording is left alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from videobrowser.telemetry import SpeedSample, TurnSample, yaw_rates

# Search this far either side of each camera-clock reading, seconds.
SEARCH_S = 180.0
# At least this much overlap between recording and log, seconds.
MIN_OVERLAP_S = 120
# Accept a start time only at this correlation ...
MIN_PEAK = 0.5
# ... and when it beats the best start time this far away (another session)
# by this ratio. Nearby shifts score almost as high and are no rival.
RIVAL_S = 300.0
MIN_MARGIN = 1.15
# Scene motion is median-filtered over this many seconds, then capped at this
# percentile (camera in hand in the paddock).
MOTION_MEDIAN_S = 61
# The match is then refined within this many seconds with this short median.
FINE_SEARCH_S = 20
FINE_MEDIAN_S = 5
MOTION_CLIP_PCT = 95.0
# Speed above this adds nothing to the pit/track pattern motion can see.
SPEED_CLIP_KMH = 80.0


@dataclass
class RaceboxTrack:
    """RaceBox speed and turn rate for a stretch of time, on UTC."""

    utc: np.ndarray
    speed_kmh: np.ndarray
    yaw_rate: np.ndarray

    def _window(self, start_utc: float, duration: float) -> np.ndarray:
        return (self.utc >= start_utc) & (self.utc <= start_utc + duration)

    def speeds(self, start_utc: float, duration: float) -> list[SpeedSample]:
        """Speed samples on recording time for a recording starting at *start_utc*."""
        m = self._window(start_utc, duration)
        return [SpeedSample(float(u - start_utc), float(v)) for u, v in zip(self.utc[m], self.speed_kmh[m])]

    def turns(self, start_utc: float, duration: float) -> list[TurnSample]:
        m = self._window(start_utc, duration)
        return [TurnSample(float(u - start_utc), float(y)) for u, y in zip(self.utc[m], self.yaw_rate[m])]


def track_from_sessions(sessions: list) -> RaceboxTrack | None:
    """One time-ordered track from parsed ``.vbo`` sessions."""
    parts = []
    for session in sessions:
        data = getattr(session, "data", None) or {}
        utc = data.get("time")
        lat, lon = data.get("lat"), data.get("lng")
        speed = data.get("velocity")
        if speed is None:
            speed = data.get("velocity_kmh")
        if utc is None or lat is None or lon is None or speed is None or len(utc) < 10:
            continue
        utc = np.asarray(utc, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        speed = np.asarray(speed, dtype=np.float64)
        parts.append((utc, speed, yaw_rates(utc, lat, lon, speed)))
    if not parts:
        return None
    utc, speed, yaw = (np.concatenate([p[i] for p in parts]) for i in range(3))
    # Sessions may overlap or step back: keep one strictly increasing series.
    order = np.argsort(utc, kind="stable")
    utc, speed, yaw = utc[order], speed[order], yaw[order]
    keep = np.r_[True, np.diff(utc) > 0] & np.isfinite(utc) & np.isfinite(speed)
    if keep.sum() < 10:
        return None
    return RaceboxTrack(utc[keep], speed[keep], yaw[keep])


def load_track(sessions_dir: Path | None) -> RaceboxTrack | None:
    """Track of every ``.vbo`` session in *sessions_dir*, or None."""
    if sessions_dir is None or not sessions_dir.exists() or not any(sessions_dir.glob("*.vbo")):
        return None
    try:
        from videobrowser.racebox_stage import load_sessions

        return track_from_sessions(load_sessions(sessions_dir))
    except Exception as exc:
        print(f"racebox: sessions unavailable for speed/turns ({exc})")
        return None


def clock_guesses(creation_utc: float | None) -> list[float]:
    """Both readings of a camera clock start already read as local wall time
    (``probe`` does): as such, and as if the tag had been true UTC."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    if creation_utc is None:
        return []
    offset = datetime.fromtimestamp(creation_utc, ZoneInfo("Europe/Copenhagen")).utcoffset()
    shift = offset.total_seconds() if offset else 0.0
    return [creation_utc] if abs(shift) < 1.0 else [creation_utc, creation_utc + shift]


@dataclass
class MotionMatch:
    """Outcome of laying a recording on the RaceBox log; *start_utc* None when unclear."""

    start_utc: float | None
    reason: str
    peak: float = float("nan")
    runner: float = float("nan")


def motion_start_utc(
    motion: list[tuple[float, float]],
    track: RaceboxTrack,
    guesses: list[float],
    search_s: float = SEARCH_S,
) -> float | None:
    """UTC of recording t=0 that lays *motion* (per second) on the RaceBox speed.

    None when the match is not clear. Time outside the RaceBox log counts as
    standing still (the logger runs whenever the bike is out).
    """
    return motion_match(motion, track, guesses, search_s).start_utc


def motion_match(
    motion: list[tuple[float, float]],
    track: RaceboxTrack,
    guesses: list[float],
    search_s: float = SEARCH_S,
) -> MotionMatch:
    """Like ``motion_start_utc``, with why a recording was not placed."""
    if len(motion) < MIN_OVERLAP_S:
        return MotionMatch(None, f"only {len(motion)} s of scene motion (need {MIN_OVERLAP_S})")
    if len(track.utc) < 10:
        return MotionMatch(None, "RaceBox log is empty")
    if not guesses:
        return MotionMatch(None, "no camera clock to search around")
    n = int(max(t for t, _ in motion)) + 1
    x = np.full(n, np.nan)
    for t, mag in motion:
        x[int(t)] = mag
    known = ~np.isnan(x)
    if known.sum() < MIN_OVERLAP_S or float(np.nanstd(x)) < 1e-6:
        return MotionMatch(None, "scene motion too short or flat")
    # A camera handled in the paddock moves far more than one on the bike,
    # in bursts of seconds to a minute; sessions last many minutes. A rolling
    # median removes most bursts and a cap stops the rest from outweighing
    # whole sessions.
    raw = x
    x, known = _prepare(raw, MOTION_MEDIAN_S)

    # Only the part of the log the search can reach (the sessions dir holds
    # every day ever logged).
    utc, kmh = track.utc, track.speed_kmh
    keep = (utc >= min(guesses) - search_s - 10.0) & (utc <= max(guesses) + search_s + n + 10.0)
    utc, kmh = utc[keep], kmh[keep]
    if len(utc) < 10:
        return MotionMatch(None, "no RaceBox log near the camera clock " + _clock_text(guesses))
    if utc[-1] - utc[0] > 3 * 86400.0:
        return MotionMatch(None, "RaceBox log spans more than 3 days")

    # RaceBox speed on a 1 s grid, padded by the recording length on both
    # sides so a recording may run past either end of the log.
    g0 = float(np.floor(min(utc[0], min(guesses) - search_s))) - 1.0
    grid = g0 + np.arange(int(np.ceil(max(utc[-1], max(guesses) + search_s) - g0)) + n + 2)
    speed = np.interp(grid, utc, np.minimum(kmh, SPEED_CLIP_KMH), left=0.0, right=0.0)
    # Between sessions (no sample within 5 s) the bike is in.
    i = np.clip(np.searchsorted(utc, grid), 1, len(utc) - 1)
    nearest = np.minimum(np.abs(grid - utc[i - 1]), np.abs(utc[i] - grid))
    speed[nearest > 5.0] = 0.0

    # Pearson correlation of x against the speed window at each candidate
    # start (within the search window of a clock reading): a day's worth of
    # recordings is too long to correlate against every start.
    ks = sorted(
        {
            k
            for guess in guesses
            for k in range(int(guess - search_s - g0), int(guess + search_s - g0) + 1)
            if 0 <= k <= len(grid) - n
        }
    )
    if not ks:
        return MotionMatch(None, "RaceBox log too short")
    r = _pearson(speed, x, known, ks)
    starts = grid[ks]
    best = int(np.argmax(r))
    peak = float(r[best])
    away = np.abs(starts - starts[best]) >= RIVAL_S
    runner = float(r[away].max()) if away.any() else -1.0
    if peak < MIN_PEAK:
        return MotionMatch(None, f"weak match r={peak:.2f} (need {MIN_PEAK})", peak, runner)
    if peak < MIN_MARGIN * max(runner, 1e-6):
        return MotionMatch(
            None, f"ambiguous r={peak:.2f} vs {runner:.2f} elsewhere", peak, runner
        )
    # The long median blurs session edges by seconds: refine nearby with a
    # short one.
    fine_x, fine_known = _prepare(raw, FINE_MEDIAN_S)
    k0 = ks[best]
    near = [k for k in range(k0 - FINE_SEARCH_S, k0 + FINE_SEARCH_S + 1) if 0 <= k <= len(grid) - n]
    fine = _pearson(speed, fine_x, fine_known, near)
    if float(fine.max()) > 0:
        k0 = near[int(np.argmax(fine))]
    # Motion sample i covers recording second [i, i + 1).
    start = float(grid[k0]) - 0.5
    return MotionMatch(start, f"r={peak:.2f} (next best {runner:.2f})", peak, runner)


def _prepare(raw: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Median-filtered, capped, standardised motion (0 where unknown), and
    where it is known."""
    x = _rolling_median(raw, width)
    known = ~np.isnan(x)
    x = np.minimum(x, np.nanpercentile(x, MOTION_CLIP_PCT))
    return np.where(known, (x - np.nanmean(x)) / max(float(np.nanstd(x)), 1e-6), 0.0), known


def _pearson(speed: np.ndarray, x: np.ndarray, known: np.ndarray, ks: list[int]) -> np.ndarray:
    """Correlation of *x* with the speed window starting at each index in *ks*,
    over the seconds with motion only (gaps between recordings say nothing)."""
    n = len(x)
    mask = known.astype(np.float64)
    count = float(mask.sum())
    r = np.full(len(ks), -1.0)
    for j, k in enumerate(ks):
        window = speed[k : k + n]
        mean = float(np.dot(window, mask)) / count
        std = np.sqrt(max(float(np.dot(window * window, mask)) / count - mean * mean, 0.0))
        if std > 1.0:
            r[j] = float(np.dot(window, x)) / (count * std)
    return r


def _rolling_median(x: np.ndarray, width: int) -> np.ndarray:
    """Centred rolling median ignoring NaN; NaN where the window has no data."""
    half = width // 2
    padded = np.r_[np.full(half, np.nan), x, np.full(half, np.nan)]
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    out = np.full(len(x), np.nan)
    has = (~np.isnan(windows)).any(axis=1)
    with np.errstate(all="ignore"):
        out[has] = np.nanmedian(windows[has], axis=1)
    return out


def day_match(
    recordings: list[tuple[float, list[tuple[float, float]]]],
    track: RaceboxTrack,
    search_s: float = SEARCH_S,
) -> MotionMatch:
    """One camera's whole day laid on the RaceBox log at once.

    *recordings* are (camera-clock start, per-second motion) pairs from the
    same camera, so they share one clock error. Hours of footage with pit
    stops between sessions match far more clearly than one recording, and a
    recording made only in the paddock is placed along with the rest. The
    result's start is that of the earliest recording; add the same shift
    (``start_utc - earliest clock start``) to the others.
    """
    if not recordings:
        return MotionMatch(None, "no recordings")
    first = min(start for start, _ in recordings)
    combined = [
        (t + start - first, mag) for start, motion in recordings for t, mag in motion
    ]
    return motion_match(combined, track, clock_guesses(first), search_s)


def _clock_text(guesses: list[float]) -> str:
    from datetime import datetime, timezone

    if not guesses:
        return "(none)"
    return " / ".join(
        datetime.fromtimestamp(g, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ") for g in guesses
    )
