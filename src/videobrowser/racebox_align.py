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

# Venues with landmarks the camera sees on every lap: (name, start-line
# latitude, longitude (degrees, sign ignored), metres along the track from
# RaceBox's start line to where each bridge's underside leaves the top of
# the picture). Knutstorp: the "Porsche Center Malmö" bridge over the
# straight (56-61 m measured on three recordings) and the "Förenade Bil"
# bridge (664-668 m). Anderstorp: the "PORSCHE" bridge, 0.6 s past the
# painted line at 135 km/h (23 m; RaceBox's line assumed on the painted
# one). Other venues get none.
VENUE_LANDMARKS = (
    ("Knutstorp", 55.98689, 13.11448, (56.0, 666.0)),
    ("Anderstorp", 57.26147, 13.59942, (23.0,)),
)
# A start line this close (m) to a venue's is that venue's.
VENUE_RADIUS_M = 500.0

# Landmark passes in the picture against RaceBox's (``landmark_match``): a
# pass and RaceBox's time pair up within this many seconds (the picture
# sees the landmark a few metres early or late, by the camera's tilt) ...
LANDMARK_TOL_S = 0.3
# ... at least this many pairs, and this many more than any start over
# LANDMARK_RIVAL_S away with two or more (a lap off).
MIN_LANDMARK_PAIRS = 2
LANDMARK_MARGIN = 1
LANDMARK_RIVAL_S = 1.0

# Bridge passes in the picture (``bridge``) against RaceBox's
# (``bridge_place``): searched LINE_COARSE_S either side of this many
# landmark passes, spread over the recording ...
BRIDGE_WINDOWS = 3
# ... then the start is refined by the painted finish line, searched this
# far (s) either side of each crossing: where the bridge leaves the picture
# depends on the camera's height and pitch, 0.25 s apart between two cameras
# at Anderstorp. Lines past this do not count either: at Anderstorp
# something line-like follows the finish line under the bridge, and the
# finish line is the last line before clear track. (The checkered line lies
# 0.58 s before the finish line, outside.)
BRIDGE_REFINE_S = 0.35
# Crossings that must agree (within LINE_TOL_S) to move the bridge start:
# one line pass alone has moved it the wrong way.
BRIDGE_REFINE_MIN = 2

# A recording scene motion could not place is searched whole for the bridge
# (``bridge_place_anywhere``) and laid on the landmark passes within this
# many seconds of its camera clock (local time read as UTC is hours off) ...
ANYWHERE_S = 36 * 3600.0
# ... pairing within this many seconds (a constant lag of the mount goes into
# the start; what is left is a frame or so) ...
ANYWHERE_TOL_S = 0.15
# ... in recordings this long (s) or more, about three laps ...
ANYWHERE_MIN_S = 250.0
# ... when the picture shows this share of sky (``bridge.sky_share``): a
# camera pitched down to the road sees no bridge, only false edges ...
ANYWHERE_MIN_SKY = 0.02
# ... with this many pairs and this many times the next best start: chance
# pairs up to five passes somewhere in a day, and laps of nearly the same
# time pair up half of them a lap off.
ANYWHERE_MIN_PAIRS = 6
ANYWHERE_RATIO = 1.5

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
    # UTC of each pass under the venue's landmark (``VENUE_LANDMARKS``),
    # empty at venues without one.
    landmarks: np.ndarray = field(default_factory=lambda: np.empty(0))

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
    landmarks: list[float] = []
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
            crossed = line_crossings(utc, lat, lon, line)
            crossings.extend(crossed)
            for metres in venue_landmark_m(line):
                landmarks.extend(along_track(utc, lat, lon, crossed, metres))
    if not parts:
        return None
    utc, speed, yaw = (np.concatenate([p[i] for p in parts]) for i in range(3))
    # Sessions may overlap or step back: keep one strictly increasing series.
    order = np.argsort(utc, kind="stable")
    utc, speed, yaw = utc[order], speed[order], yaw[order]
    keep = np.r_[True, np.diff(utc) > 0] & np.isfinite(utc) & np.isfinite(speed)
    if keep.sum() < 10:
        return None
    return RaceboxTrack(
        utc[keep], speed[keep], yaw[keep], np.unique(np.round(crossings, 3)), np.unique(np.round(landmarks, 3))
    )


def _degrees(x):
    """RaceBox stores decimal minutes (degrees x 60); GoPro and maps degrees."""
    x = np.asarray(x, dtype=np.float64)
    return x / 60.0 if float(np.nanmedian(np.abs(x))) > 90.0 else x


def venue_landmark_m(line: tuple[float, float, float, float]) -> tuple[float, ...]:
    """Metres from the start line along the track to each of the venue's
    landmarks, by where the start line lies; none at other venues."""
    lat = float(_degrees([line[0], line[2]]).mean())
    lon = abs(float(_degrees([line[1], line[3]]).mean()))
    for _, vlat, vlon, metres in VENUE_LANDMARKS:
        north = (lat - vlat) * 111_320.0
        east = (lon - vlon) * 111_320.0 * np.cos(np.radians(vlat))
        if np.hypot(north, east) <= VENUE_RADIUS_M:
            return metres
    return ()


def along_track(
    utc: np.ndarray, lat: np.ndarray, lon: np.ndarray, crossings: list[float], metres: float
) -> list[float]:
    """UTC at which the bike is *metres* along its track past each crossing."""
    lat, lon = _degrees(lat), _degrees(lon)
    step = np.hypot(
        np.diff(lat) * 111_320.0, np.diff(lon) * 111_320.0 * np.cos(np.radians(float(np.median(lat))))
    )
    dist = np.r_[0.0, np.cumsum(step)]
    out = []
    for c in crossings:
        target = float(np.interp(c, utc, dist)) + metres
        if target <= dist[-1]:
            out.append(float(np.interp(target, dist, utc)))
    return out


def line_crossings(
    utc: np.ndarray, lat: np.ndarray, lon: np.ndarray, line: tuple[float, float, float, float]
) -> list[float]:
    """UTC of each crossing of the start/finish *line* (lat1, lon1, lat2,
    lon2), interpolated between the samples either side of it."""
    lat1, lon1, lat2, lon2 = line
    # Flat metres around the line: the track is a few km across.
    k = np.cos(np.radians(float(_degrees([lat1, lat2]).mean())))
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
    # Pairs the next best start (over LANDMARK_RIVAL_S away) has.
    rival: int = 0


def landmark_match(
    passes: list[float],
    landmarks: np.ndarray,
    guesses: list[float],
    search_s: float = SEARCH_S,
    tol_s: float = LANDMARK_TOL_S,
) -> LineMatch:
    """UTC of recording t=0 that lays the landmark *passes* (recording time)
    on RaceBox's *landmarks* (UTC), searched within *search_s* of each guess.

    Every pass-landmark pair is a candidate start; the one most passes agree
    with wins. Lap times differ by tenths, so a start a lap off pairs up only
    by chance.
    """
    p = np.sort(np.asarray(passes, dtype=np.float64))
    lm = np.sort(np.asarray(landmarks, dtype=np.float64))
    if not len(p):
        return LineMatch(None, "no landmark pass in the picture")
    if not guesses:
        return LineMatch(None, "no start to search around")
    lo, hi = min(guesses) - search_s, max(guesses) + search_s
    lm = lm[(lm >= lo + p[0] - 1.0) & (lm <= hi + p[-1] + 1.0)]
    if not len(lm):
        return LineMatch(None, "no RaceBox landmark pass near the recording")
    starts = (lm[None, :] - p[:, None]).ravel()
    starts = starts[np.any(np.abs(starts[:, None] - np.asarray(guesses)[None, :]) <= search_s, axis=1)]
    if not len(starts):
        return LineMatch(None, "no landmark pass lines up with RaceBox")
    at = p[None, :] + starts[:, None]
    i = np.clip(np.searchsorted(lm, at), 1, max(1, len(lm) - 1))
    resid = np.abs(at - lm[np.minimum(i, len(lm) - 1)])
    if len(lm) > 1:
        resid = np.minimum(resid, np.abs(at - lm[i - 1]))
    paired = resid <= tol_s
    counts = paired.sum(axis=1)
    err = np.where(paired, resid, 0.0).sum(axis=1) / np.maximum(counts, 1)
    best = int(np.lexsort((err, -counts))[0])
    start, n = float(starts[best]), int(counts[best])
    away = np.abs(starts - start) > LANDMARK_RIVAL_S
    rival = int(counts[away].max()) if away.any() else 0
    pairs = []
    for t in p[paired[best]]:
        j = int(np.argmin(np.abs(lm - (t + start))))
        pairs.append((float(t), float(lm[j])))
    offsets = np.array([u - t for t, u in pairs])
    start = float(np.median(offsets))
    spread = float(np.max(np.abs(offsets - start)))
    reason = f"{n} of {len(p)} landmark passes on RaceBox's, spread {spread:.2f} s, next best {rival}"
    if n < MIN_LANDMARK_PAIRS:
        return LineMatch(None, reason + f" (need {MIN_LANDMARK_PAIRS})", pairs, spread, rival)
    # One pass meets some landmark at almost any start: a rival of one is noise.
    if rival >= 2 and n < rival + LANDMARK_MARGIN:
        return LineMatch(None, reason + " (ambiguous)", pairs, spread, rival)
    return LineMatch(start, reason, pairs, spread, rival)


def finish_passes(passes: list[float], speed_ms: float) -> list[float]:
    """The passes that can be the finish line: no other line follows within
    LINE_CLEAR_M at *speed_ms* (the grid and start lines lie before it)."""
    clear = LINE_CLEAR_M / max(speed_ms, 1.0)
    p = sorted(passes)
    return [t for i, t in enumerate(p) if i + 1 == len(p) or p[i + 1] - t > clear]


def _merged(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def bridge_place(
    start0: float, duration: float, track: RaceboxTrack, find_bridges, find_passes=None
) -> LineMatch:
    """UTC of recording t=0 from the venue's bridge, near *start0* (scene
    motion), refined by the painted finish line.

    *find_bridges(t0, t1)* returns the bridge passes (recording time, s) the
    picture shows between t0 and t1; they are searched LINE_COARSE_S either
    side of up to BRIDGE_WINDOWS of RaceBox's landmark passes and laid on
    them by ``landmark_match`` (two passes or more). Then *find_passes* (``finish_line``) looks
    for the finish line BRIDGE_REFINE_S either side of each crossing; the
    passes that agree move the start to the frame. Without them the bridge
    start stands.
    """
    lm = np.asarray(track.landmarks, dtype=np.float64)
    lm = lm[(lm >= start0 - LINE_COARSE_S) & (lm <= start0 + duration + LINE_COARSE_S)]
    if not len(lm):
        return LineMatch(None, "no RaceBox landmark pass in the recording")
    picks = lm[np.unique(np.round(np.linspace(0, len(lm) - 1, min(BRIDGE_WINDOWS, len(lm)))).astype(int))]
    spans = _merged(
        [
            (max(0.0, u - start0 - LINE_COARSE_S), min(duration, u - start0 + LINE_COARSE_S))
            for u in picks
        ]
    )
    passes = sorted({round(float(p), 3) for lo, hi in spans if hi - lo >= 0.2 for p in find_bridges(lo, hi)})
    # One bridge pass agrees with some landmark near almost any start: a
    # recording with one is left to scene motion.
    match = landmark_match(passes, lm, [start0], LINE_COARSE_S)
    if match.start_utc is None:
        return match
    reason = f"bridge: {match.reason}, {match.start_utc - start0:+.2f} s from scene motion"
    return _refine_by_line(match, reason, duration, track, find_passes)


def bridge_place_anywhere(
    clock_utc: float, duration: float, track: RaceboxTrack, find_bridges, find_passes=None, sky=None
) -> LineMatch:
    """UTC of recording t=0 from the venue's bridge, with no start to go by
    but the camera clock's day: every bridge pass in the recording is laid on
    every landmark pass within ANYWHERE_S of *clock_utc* (camera clocks are
    hours off), refined by the finish line as ``bridge_place``.

    Not searched (it decodes the whole recording) when RaceBox has fewer
    than ANYWHERE_MIN_PAIRS landmark passes in any stretch as long as the
    recording, or when *sky()* (share of sky in the picture) is below
    ANYWHERE_MIN_SKY.

    Over a whole day, chance and laps of nearly the same time pair up a few
    passes anywhere, so the start needs ANYWHERE_MIN_PAIRS and
    ANYWHERE_RATIO times the next best.
    """
    lm = np.asarray(track.landmarks, dtype=np.float64)
    lm = lm[(lm >= clock_utc - ANYWHERE_S) & (lm <= clock_utc + duration + ANYWHERE_S)]
    if not len(lm):
        return LineMatch(None, "no RaceBox landmark pass within a day of the camera clock")
    if duration < ANYWHERE_MIN_S:
        return LineMatch(None, f"too short to search whole ({duration:.0f} s)")
    most = int(np.max(np.searchsorted(lm, lm + duration, side="right") - np.arange(len(lm))))
    if most < ANYWHERE_MIN_PAIRS:
        return LineMatch(
            None, f"RaceBox has at most {most} landmark passes in {duration:.0f} s near the camera clock"
        )
    if sky is not None:
        share = float(sky())
        if share < ANYWHERE_MIN_SKY:
            return LineMatch(None, f"camera pitched too low to see the bridge ({share:.1%} sky)")
    passes = sorted({round(float(p), 3) for p in find_bridges(0.0, duration)})
    match = landmark_match(passes, lm, [clock_utc], ANYWHERE_S, ANYWHERE_TOL_S)
    if not match.passes:
        return match
    n, rival = len(match.passes), match.rival
    reason = f"bridge, whole recording: {match.reason.split(' (')[0]}"
    if n < ANYWHERE_MIN_PAIRS or n < ANYWHERE_RATIO * rival:
        return LineMatch(None, reason + f" (need {ANYWHERE_MIN_PAIRS} and {ANYWHERE_RATIO:g}x the next best)")
    start = float(np.median([u - t for t, u in match.passes]))
    match = LineMatch(start, match.reason, match.passes, match.spread)
    return _refine_by_line(match, reason + f", camera clock {start - clock_utc:+.1f} s", duration, track, find_passes)


def _refine_by_line(match: LineMatch, reason: str, duration: float, track: RaceboxTrack, find_passes) -> LineMatch:
    """*match* (a bridge start) moved to the frame by the finish line, where
    BRIDGE_REFINE_MIN crossings or more agree; else as it is."""
    start = match.start_utc
    if find_passes is None or not len(track.crossings):
        return LineMatch(start, reason, match.passes, match.spread)
    c = np.asarray(track.crossings, dtype=np.float64)
    c = c[(c >= start) & (c <= start + duration)]
    offsets = []
    for cj in c:
        v = float(np.interp(cj, track.utc, track.speed_kmh)) / 3.6
        if v * 3.6 < LINE_MIN_KMH:
            continue
        t = cj - start - LINE_LEAD_S
        window = find_passes(max(0.0, t - BRIDGE_REFINE_S), min(duration, t + BRIDGE_REFINE_S))
        found = [p.t if hasattr(p, "t") else float(p) for p in window]
        near = finish_passes([p for p in found if abs(p - t) <= BRIDGE_REFINE_S], v)
        if near:
            offsets.append(cj - min(near, key=lambda p: abs(p - t)) - LINE_LEAD_S)
    if not offsets:
        reason += "; no finish line in the picture to refine by"
        return LineMatch(start, reason, match.passes, match.spread)
    mid = float(np.median(offsets))
    agree = [o for o in offsets if abs(o - mid) <= LINE_TOL_S]
    if len(agree) < BRIDGE_REFINE_MIN or len(agree) * 2 < len(offsets):
        reason += f"; finish line on {len(agree)} of {len(offsets)} crossings agreeing, not enough to refine by"
        return LineMatch(start, reason, match.passes, match.spread)
    line_start = float(np.median(agree))
    spread = float(np.max(np.abs(np.asarray(agree) - line_start)))
    reason += f"; finish line {len(agree)} of {len(c)} crossings, {line_start - start:+.2f} s, spread {spread:.2f} s"
    return LineMatch(line_start, reason, match.passes, spread)


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
