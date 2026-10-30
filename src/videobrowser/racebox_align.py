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

from dataclasses import dataclass, field
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

# Finish-line passes in the picture (``finish_line``) against RaceBox's
# crossings (``line_place``). The first crossing is searched this far (s)
# either side of where scene motion put it (half a lap off happens) ...
LINE_COARSE_S = 40.0
# ... the others this far, once the first gives a start ...
LINE_FINE_S = 0.5
# ... and a pass pairs with a crossing within this many seconds.
LINE_TOL_S = 0.1
# The line leaves the picture this long before the camera is over it (a few
# metres of road below the frame edge).
LINE_LEAD_S = 0.06
# After the finish line the track has no painted line for this many metres;
# grid lines and the start line lie 13-50 m before it, each like it.
LINE_CLEAR_M = 30.0
# Only crossings at speed: slow, the line is in view long and the pass vague.
LINE_MIN_KMH = 60.0
# Starts the first crossing's passes give that are checked against the rest.
LINE_MAX_STARTS = 25
# A recording with one crossing only is placed by it when the start lands
# this close (s) to the scene-motion start.
LINE_NEAR_S = 10.0


@dataclass
class RaceboxTrack:
    """RaceBox speed and turn rate for a stretch of time, on UTC."""

    utc: np.ndarray
    speed_kmh: np.ndarray
    yaw_rate: np.ndarray
    # UTC of each finish-line crossing, from the sessions' start line.
    crossings: np.ndarray = field(default_factory=lambda: np.empty(0))

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
    crossings: list[float] = []
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
        line = getattr(session, "start_line", None)
        if line is not None:
            crossings.extend(line_crossings(utc, lat, lon, line))
    if not parts:
        return None
    utc, speed, yaw = (np.concatenate([p[i] for p in parts]) for i in range(3))
    # Sessions may overlap or step back: keep one strictly increasing series.
    order = np.argsort(utc, kind="stable")
    utc, speed, yaw = utc[order], speed[order], yaw[order]
    keep = np.r_[True, np.diff(utc) > 0] & np.isfinite(utc) & np.isfinite(speed)
    if keep.sum() < 10:
        return None
    return RaceboxTrack(utc[keep], speed[keep], yaw[keep], np.unique(np.round(crossings, 3)))


def line_crossings(
    utc: np.ndarray, lat: np.ndarray, lon: np.ndarray, line: tuple[float, float, float, float]
) -> list[float]:
    """UTC of each crossing of the start/finish *line* (lat1, lon1, lat2,
    lon2), interpolated between the samples either side of it."""
    lat1, lon1, lat2, lon2 = line
    # Flat metres around the line: the track is a few km across.
    k = np.cos(np.radians((lat1 + lat2) / 2.0))
    ax, ay = lon1 * k, lat1
    dx, dy = (lon2 - lon1) * k, lat2 - lat1
    px, py = lon * k, lat
    # Which side of the line each sample is on, and where along it.
    side = (px - ax) * dy - (py - ay) * dx
    along = ((px - ax) * dx + (py - ay) * dy) / max(dx * dx + dy * dy, 1e-30)
    out = []
    # A sample right on the line counts to the side it came from.
    ahead = side >= 0
    for i in np.flatnonzero(ahead[:-1] != ahead[1:]):
        f = side[i] / (side[i] - side[i + 1])
        if 0.0 <= along[i] + f * (along[i + 1] - along[i]) <= 1.0:
            out.append(float(utc[i] + f * (utc[i + 1] - utc[i])))
    return out


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


@dataclass
class LineMatch:
    """Outcome of placing a recording by its finish-line passes; *start_utc*
    None when unclear. *passes* are (recording time, crossing UTC) pairs, and
    *spread* (s) is how far they scatter around the start."""

    start_utc: float | None
    reason: str
    passes: list[tuple[float, float]] = field(default_factory=list)
    spread: float = float("nan")


def finish_passes(passes: list[float], speed_ms: float) -> list[float]:
    """The passes that can be the finish line: no other line follows within
    LINE_CLEAR_M at *speed_ms* (the grid and start lines lie before it)."""
    clear = LINE_CLEAR_M / max(speed_ms, 1.0)
    p = sorted(passes)
    return [t for i, t in enumerate(p) if i + 1 == len(p) or p[i + 1] - t > clear]


def line_place(start0: float, duration: float, track: RaceboxTrack, find_passes) -> LineMatch:
    """UTC of recording t=0 from the finish line, near *start0* (scene motion).

    *find_passes(t0, t1)* returns the line passes (recording time, s) the
    picture shows between t0 and t1. The crossing at the highest speed is
    searched LINE_COARSE_S either side; each finish-line pass there gives a
    start, which is checked at every other crossing in a LINE_FINE_S window.
    Lap times differ by tenths, so a wrong start meets the line elsewhere by
    chance at most. The start most crossings agree with wins.
    """
    c = np.asarray(track.crossings, dtype=np.float64)
    c = c[(c >= start0 - LINE_COARSE_S) & (c <= start0 + duration + LINE_COARSE_S)]
    kmh = np.interp(c, track.utc, track.speed_kmh) if len(c) else c
    keep = kmh >= LINE_MIN_KMH
    c, ms = c[keep], kmh[keep] / 3.6
    if not len(c):
        return LineMatch(None, "no RaceBox finish-line crossing at speed in the recording")
    windows: dict[tuple[float, float], list[float]] = {}

    def passes_in(t0: float, t1: float) -> list[float]:
        key = (round(max(0.0, t0), 2), round(min(duration, t1), 2))
        if key[1] - key[0] < 0.2:
            return []
        if key not in windows:
            windows[key] = [p.t if hasattr(p, "t") else float(p) for p in find_passes(*key)]
        return windows[key]

    def check(start: float) -> list[tuple[float, float]]:
        pairs = []
        for cj, vj in zip(c, ms):
            t = cj - start - LINE_LEAD_S
            if not 0.0 <= t <= duration:
                continue
            near = [p for p in finish_passes(passes_in(t - LINE_FINE_S, t + LINE_FINE_S), vj) if abs(p - t) <= LINE_TOL_S]
            if near:
                pairs.append((min(near, key=lambda p: abs(p - t)), float(cj)))
        return pairs

    results: list[tuple[float, list[tuple[float, float]]]] = []
    tried = 0
    for j in np.argsort(-ms)[:2]:
        t = c[j] - start0
        if not -LINE_COARSE_S <= t <= duration + LINE_COARSE_S:
            continue
        tried += 1
        found = finish_passes(passes_in(t - LINE_COARSE_S, t + LINE_COARSE_S), ms[j])
        starts = sorted((c[j] - p - LINE_LEAD_S for p in found), key=lambda s0: abs(s0 - start0))
        for s0 in starts[:LINE_MAX_STARTS]:
            if any(abs(s0 - r) <= LINE_TOL_S for r, _ in results):
                continue
            results.append((s0, check(s0)))
        if any(len(pairs) >= 2 for _, pairs in results):
            break
    if not tried:
        return LineMatch(None, "no RaceBox finish-line crossing inside the recording")
    if not results:
        return LineMatch(None, "no finish line in the picture near the first crossing")
    results.sort(key=lambda r: len(r[1]), reverse=True)
    best, pairs = results[0]
    rival = len(results[1][1]) if len(results) > 1 else 0
    in_view = sum(1 for cj in c if 0.0 <= cj - best <= duration)
    if not pairs:
        # A pass right at the end of the recording, its crossing just past it.
        return LineMatch(None, f"no finish line in the picture at any of {in_view} crossings")
    offsets = np.array([cj - p - LINE_LEAD_S for p, cj in pairs])
    start = float(np.median(offsets))
    spread = float(np.max(np.abs(offsets - start)))
    reason = (
        f"{len(pairs)} of {in_view} crossings on the line, spread {spread:.2f} s, "
        f"next best {rival}, {start - start0:+.2f} s from scene motion"
    )
    if len(pairs) >= 2 and len(pairs) > rival:
        return LineMatch(start, reason, pairs, spread)
    # Each start pairs with the crossing it came from: one pair proves
    # nothing unless the recording has one crossing only and no other
    # finish-line pass gives a start as near.
    near_rivals = [r for r, _ in results[1:] if abs(r - start0) <= LINE_NEAR_S]
    if len(pairs) == 1 and in_view == 1 and abs(start - start0) <= LINE_NEAR_S and not near_rivals:
        return LineMatch(start, reason + " (one crossing, near)", pairs, spread)
    return LineMatch(None, reason + (" (ambiguous)" if len(pairs) and len(pairs) <= rival else ""), pairs, spread)


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
