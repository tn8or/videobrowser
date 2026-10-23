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
    if len(motion) < MIN_OVERLAP_S or len(track.utc) < 10:
        return None
    n = int(max(t for t, _ in motion)) + 1
    x = np.full(n, np.nan)
    for t, mag in motion:
        x[int(t)] = mag
    known = ~np.isnan(x)
    if known.sum() < MIN_OVERLAP_S or float(np.nanstd(x)) < 1e-6:
        return None
    x = np.where(known, (x - np.nanmean(x)) / np.nanstd(x), 0.0)

    # Only the part of the log the search can reach (the sessions dir holds
    # every day ever logged).
    utc, kmh = track.utc, track.speed_kmh
    if guesses:
        keep = (utc >= min(guesses) - search_s - 10.0) & (utc <= max(guesses) + search_s + n + 10.0)
        utc, kmh = utc[keep], kmh[keep]
    if len(utc) < 10 or utc[-1] - utc[0] > 3 * 86400.0:
        return None

    # RaceBox speed on a 1 s grid, padded by the recording length on both
    # sides so a recording may run past either end of the log.
    g0 = float(np.floor(utc[0])) - n
    grid = g0 + np.arange(int(np.ceil(utc[-1] - g0)) + n + 1)
    speed = np.interp(grid, utc, np.minimum(kmh, SPEED_CLIP_KMH), left=0.0, right=0.0)
    # Between sessions (no sample within 5 s) the bike is in.
    i = np.clip(np.searchsorted(utc, grid), 1, len(utc) - 1)
    nearest = np.minimum(np.abs(grid - utc[i - 1]), np.abs(utc[i] - grid))
    speed[nearest > 5.0] = 0.0
    if len(speed) < n + 2:
        return None

    # Pearson correlation of x against every window of the speed grid.
    num = np.correlate(speed, x, mode="valid")
    csum = np.r_[0.0, np.cumsum(speed)]
    csq = np.r_[0.0, np.cumsum(speed * speed)]
    win = csum[n:] - csum[:-n]
    win_sq = csq[n:] - csq[:-n]
    std = np.sqrt(np.maximum(win_sq / n - (win / n) ** 2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(std > 1.0, num / (n * std), -1.0)

    starts = grid[: len(r)]
    if guesses:
        near = np.zeros(len(r), dtype=bool)
        for guess in guesses:
            near |= np.abs(starts - guess) <= search_s
        r = np.where(near, r, -1.0)
    best = int(np.argmax(r))
    peak = float(r[best])
    away = np.abs(starts - starts[best]) >= RIVAL_S
    runner = float(r[away].max()) if away.any() else -1.0
    if peak < MIN_PEAK or peak < MIN_MARGIN * max(runner, 1e-6):
        return None
    # Motion sample i covers recording second [i, i + 1).
    return float(starts[best]) - 0.5
