"""Near-finished vertical shorts per day, built from a finished scan.

``videobrowser shorts -o <out>`` reads ``scan.sqlite``, puts every camera of a
day on one clock (``sync.py``), groups events into themed moments
(``themes.py``) and writes:

- ``shorts/YYYY-MM-DD/action-01/`` ...: the day's best moments, several
  shorts of up to ``--short-seconds`` each, best moments first
- ``shorts/YYYY-MM-DD/overhalinger-01/`` etc.: one series per theme
- ``shorts/YYYY-MM-DD/index.html``: the shorts plus every clip, by theme
- ``shorts/YYYY-MM-DD/moments.json``: what was found, for checking by hand

A short is a folder of its source clips, numbered in order, to put together
in an editor (joining them with ffmpeg clicks in the sound). ``klip.txt``
lists which part of each clip the moment covers. Each piece is the vertical
follow-cam clip that covers it (crop and HUD included), copied whole. A piece
no vertical clip covers (a rear-camera hand-over that did not make the clip
list, an own crash) is cut from the source with a 9:16 crop instead, which
looks into the corner like the scan's clips do.
Moments where the bike stands still or rides below ``--min-speed`` (GoPro
GPS; paddock, pit lane) are left out, the same rule ``verticals`` and
``horizontals`` use (``clip_drop_reason``). Without GPS the picture decides:
whether the camera rides on a moving bike (``motion.MountWatch``). Moments with a car, van or truck in view most of the
time (``--max-vehicle-share``) are in the paddock and left out.
"""

from __future__ import annotations

import html
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from videobrowser.db import ScanDB
from videobrowser.events import ClipWindow
from videobrowser.motion import DEFAULT_THRESHOLD, TRACK_MOTION
from videobrowser.probe import ProbeError, local_date_label, probe
from videobrowser.sync import CameraClock, ChapterRef, align, parallel
from videobrowser.telemetry import DEFAULT_MIN_TRACK_KMH, STOPPED_KMH
from videobrowser.themes import (
    LABELS,
    THEME_ORDER,
    CamEvent,
    Moment,
    Segment,
    crash_moment,
    detect_crashes,
    facing,
    find_moments,
)
from videobrowser.vehicles import DEFAULT_MAX_SHARE
from videobrowser.vehicles import share as vehicle_share

# Label shown at the start of each piece, seconds.
LABEL_S = 1.8
FILE_NAMES = {
    "action": "action",
    "overtake": "overhalinger",
    "overtaken": "overhalet",
    "catching_up": "indhentninger",
    "duel": "dueller",
    "kneepuck": "knae",
    "crash": "styrt",
    "crash_other": "mulige-styrt",
    "off_track": "grus",
    "own_off_track": "afkoersler",
}
# Themes need this many moments for a short of their own (a crash or an
# off-track excursion needs one).
MIN_THEME_MOMENTS = {"crash": 1, "crash_other": 1, "off_track": 1, "own_off_track": 1}
# Own crashes end stopped, so they skip the pace check. Other riders "down"
# while we stand still are almost always people beside bikes in the paddock.
SPEED_EXEMPT_THEMES = frozenset({"crash"})
# Running off the track slows the bike by itself: these are judged on the
# seconds before (``_approach``).
APPROACH_THEMES = {"own_off_track": 6.0}
# Share of a moment's frames riding on a moving bike for "track", when the
# picture decides (``motion.MountWatch``).
RIDING_MIN_SHARE = 0.5
_TZ = ZoneInfo("Europe/Copenhagen")


@dataclass
class ShortPlan:
    name: str
    theme: str
    moments: list[Moment]

    @property
    def duration(self) -> float:
        return sum(m.duration for m in self.moments)


@dataclass(frozen=True)
class ClipRow:
    recording_id: str
    t_start: float
    t_end: float
    types: str
    vertical_path: Path | None
    clip_path: Path | None


# --- planning --------------------------------------------------------------


def _pack(moments: list[Moment], max_seconds: float) -> list[list[Moment]]:
    """Greedy fill, in the given order, into lists of at most *max_seconds*."""
    shorts: list[list[Moment]] = []
    current: list[Moment] = []
    length = 0.0
    for m in moments:
        if current and length + m.duration > max_seconds:
            shorts.append(current)
            current, length = [], 0.0
        current.append(m)
        length += m.duration
    if current:
        shorts.append(current)
    return shorts


def plan_shorts(
    moments: list[Moment], max_seconds: float = 50.0, max_action_shorts: int = 6
) -> list[ShortPlan]:
    plans: list[ShortPlan] = []
    best_first = sorted(moments, key=lambda m: -m.score)
    packed = _pack(best_first, max_seconds)
    for i, group in enumerate(packed[:max_action_shorts], start=1):
        group.sort(key=lambda m: m.utc_start)
        plans.append(ShortPlan(f"{FILE_NAMES['action']}-{i:02d}", "action", group))
    for theme in THEME_ORDER:
        themed = sorted(
            (m for m in moments if m.theme == theme), key=lambda m: m.utc_start
        )
        if len(themed) < MIN_THEME_MOMENTS.get(theme, 2):
            continue
        for i, group in enumerate(_pack(themed, max_seconds), start=1):
            plans.append(ShortPlan(f"{FILE_NAMES[theme]}-{i:02d}", theme, group))
    return plans


# --- loading the scan ------------------------------------------------------

_NAME_RE = re.compile(r"^(?P<rid>.+)_(?P<start>\d{5})_(?P<types>[a-z_+]+)\.mp4$")


def _clips_from_folder(out_dir: Path, rid: str, date_label: str) -> list[ClipRow]:
    """Clip rows rebuilt from file names (scans from before the clips table)."""
    rows: list[ClipRow] = []
    vdir = out_dir / "vertical" / date_label
    cdir = out_dir / "clips" / date_label
    for path in sorted(vdir.glob(f"{rid}_*.mp4")) if vdir.exists() else []:
        match = _NAME_RE.match(path.name)
        if not match or match.group("rid") != rid:
            continue
        try:
            duration = probe(path).duration
        except ProbeError:
            continue
        start = float(match.group("start"))
        landscape = cdir / path.name
        rows.append(
            ClipRow(rid, start, start + duration, match.group("types"), path,
                    landscape if landscape.exists() else None)
        )
    return rows


def backfill_meta(db: ScanDB, inputs: list[Path], out_dir: Path) -> None:
    """Fill ``recording_meta`` for recordings scanned before it existed."""
    from videobrowser.gopro import group_recordings, iter_video_paths
    from videobrowser.telemetry import creation_utc_prefer_gps

    known = {row["recording_id"] for row in db.recording_metas()}
    profiles = {
        row["recording_id"]: row["profile"]
        for row in db.conn.execute("SELECT recording_id, profile FROM recordings")
    }
    paths = iter_video_paths(inputs, exclude_roots=[out_dir])
    for recording in group_recordings(paths, input_roots=inputs):
        rid = recording.recording_id
        if rid in known or rid not in profiles:
            continue
        info0 = recording.chapters[0].info
        start = creation_utc_prefer_gps(info0)
        db.save_recording_meta(
            rid,
            profiles[rid],
            info0.rotation,
            start,
            local_date_label(start),
            recording.duration,
            [
                {"path": str(ch.path), "offset": ch.offset, "duration": ch.duration}
                for ch in recording.chapters
            ],
        )
        print(f"  meta {rid}: backfilled from {recording.chapters[0].path.name}")


@dataclass
class DayData:
    date_label: str
    clocks: dict[str, CameraClock]
    profiles: dict[str, str]
    rotations: dict[str, int]
    events: list
    clips: list[ClipRow]
    # recording id -> (speed samples, turn samples), filled on first use.
    gps: dict = field(default_factory=dict)
    # recording id -> per-second scene motion from the scan.
    motion: dict = field(default_factory=dict)
    # recording id -> per-second share of frames with a car in view.
    vehicles: dict = field(default_factory=dict)
    # RaceBox log of the day, when there is one (``place_without_gps``).
    track: object = None
    # recording id -> camera-clock readings, for recordings without GPS the
    # RaceBox match could not place.
    unplaced: dict = field(default_factory=dict)
    # recordings whose speed in ``gps`` comes from RaceBox, not the camera.
    from_racebox: set = field(default_factory=set)
    # recording id -> per-second share of frames riding on a moving bike;
    # no entry when the scan saw no part of a bike or predates it.
    riding: dict = field(default_factory=dict)
    # recording id -> "front"/"rear" the scan read from scene flow.
    facings: dict = field(default_factory=dict)

    def facing(self, rid: str) -> str:
        """Which way the camera looks: scene flow when the scan had it, else
        the mount guess."""
        return self.facings.get(rid) or facing(self.profiles.get(rid, ""))


def load_days(db: ScanDB, out_dir: Path) -> dict[str, DayData]:
    days: dict[str, DayData] = {}
    for row in db.recording_metas():
        chapters = [
            ChapterRef(Path(c["path"]), float(c["offset"]), float(c["duration"]))
            for c in json.loads(row["chapters"])
        ]
        day = days.setdefault(row["date_label"], DayData(row["date_label"], {}, {}, {}, [], []))
        rid = row["recording_id"]
        day.clocks[rid] = CameraClock(rid, row["start_utc"], float(row["duration"]), chapters)
        day.profiles[rid] = row["profile"]
        day.rotations[rid] = int(row["rotation"])
        if "facing" in row.keys() and row["facing"]:
            day.facings[rid] = row["facing"]
    owner = {rid: day for day in days.values() for rid in day.clocks}
    for rid, samples in db.motion().items():
        if rid in owner:
            owner[rid].motion[rid] = samples
    for rid, samples in db.riding().items():
        if rid in owner:
            owner[rid].riding[rid] = samples
    for rid, samples in db.vehicles().items():
        if rid in owner:
            owner[rid].vehicles[rid] = samples
    for row in db.all_events():
        if row["recording_id"] in owner:
            owner[row["recording_id"]].events.append(row)
    with_rows: set[str] = set()
    for row in db.all_clips():
        day = owner.get(row["recording_id"])
        if day is None:
            continue
        with_rows.add(row["recording_id"])
        day.clips.append(
            ClipRow(
                row["recording_id"], row["t_start"], row["t_end"], row["types"],
                Path(row["vertical_path"]) if row["vertical_path"] else None,
                Path(row["clip_path"]) if row["clip_path"] else None,
            )
        )
    for rid, day in owner.items():
        if rid not in with_rows:
            day.clips.extend(_clips_from_folder(out_dir, rid, day.date_label))
    return days


def _moving_after(motion: list | None, t: float) -> bool:
    """True when the scene keeps streaming past at track pace after *t*: the
    bike did not stop, whatever the GPS says."""
    if not motion:
        return False
    after = [m for s, m in motion if t + 3.0 <= s <= t + 10.0]
    return len(after) >= 3 and float(np.median(after)) >= TRACK_MOTION


def _crash_moments(day: DayData) -> list[Moment]:
    from videobrowser.telemetry import _walk_grav_accl, extract_gpmd_bytes, speed_series

    found: list[Moment] = []
    for rid, clock in day.clocks.items():
        if clock.start_utc is None:
            continue
        for chapter in clock.chapters:
            try:
                info = probe(chapter.path)
            except (ProbeError, OSError):
                continue
            if not info.has_gpmd or info.gpmd_index is None:
                continue
            blob = extract_gpmd_bytes(chapter.path, info.gpmd_index)
            if not blob:
                continue
            speed = speed_series(info, blob)
            _grav, accl, _stamps = _walk_grav_accl(blob)
            if not speed or not accl:
                continue
            accl_t = np.linspace(0.0, info.duration, len(accl))
            for t in detect_crashes(
                np.array([s.t for s in speed]),
                np.array([s.speed_kmh for s in speed]),
                accl_t,
                np.asarray(accl),
            ):
                t_rec = chapter.offset + t
                if _moving_after(day.motion.get(rid), t_rec):
                    print(f"  crash? {rid} t={t_rec:.1f}s: no, the picture keeps moving")
                    continue
                print(f"  crash? {rid} t={t_rec:.1f}s")
                found.append(crash_moment(rid, t_rec, clock.start_utc))
    # Several cameras feel the same crash: keep one moment per incident.
    found.sort(key=lambda m: m.utc_start)
    merged: list[Moment] = []
    for m in found:
        if merged and m.utc_start - merged[-1].utc_start < 15.0:
            front = day.facing(m.segments[0].recording_id) == "front"
            if front and day.facing(merged[-1].segments[0].recording_id) != "front":
                merged[-1] = m
            continue
        merged.append(m)
    return merged


# Margin around a moment when checking an unplaced recording against the log.
UNPLACED_SLACK_S = 60.0


def _recording_gps(clock: CameraClock) -> tuple[list, list]:
    """GoPro GPS speed and turn-rate samples of a recording, on recording time."""
    from videobrowser.telemetry import (
        SpeedSample,
        TurnSample,
        extract_gpmd_bytes,
        speed_series,
        turn_series,
    )

    speeds: list[SpeedSample] = []
    turns: list[TurnSample] = []
    for chapter in clock.chapters:
        try:
            info = probe(chapter.path)
        except (ProbeError, OSError):
            continue
        if not info.has_gpmd or info.gpmd_index is None:
            continue
        blob = extract_gpmd_bytes(chapter.path, info.gpmd_index)
        if not blob:
            continue
        for sample in speed_series(info, blob):
            speeds.append(SpeedSample(sample.t + chapter.offset, sample.speed_kmh))
        for sample in turn_series(info, blob):
            turns.append(TurnSample(sample.t + chapter.offset, sample.yaw_rate))
    return speeds, turns


def _day_gps(day: DayData, rid: str) -> tuple[list, list]:
    if rid not in day.gps:
        day.gps[rid] = _recording_gps(day.clocks[rid])
    return day.gps[rid]


def _moment_speed(m: Moment, day: DayData) -> float | None:
    """Median GPS speed over the moment's pieces, None without GPS there."""
    vals: list[float] = []
    for seg in m.segments:
        speeds, _turns = _day_gps(day, seg.recording_id)
        vals.extend(s.speed_kmh for s in speeds if seg.t_start <= s.t <= seg.t_end)
    return float(np.median(vals)) if vals else None


def _moment_motion(m: Moment, day: DayData) -> float | None:
    """Median scene motion over the moment's pieces, None if not stored."""
    vals: list[float] = []
    for seg in m.segments:
        vals.extend(
            v for t, v in day.motion.get(seg.recording_id, []) if seg.t_start <= t <= seg.t_end
        )
    return float(np.median(vals)) if vals else None


def _moment_riding(m: Moment, day: DayData) -> float | None:
    """Share of the moment's frames riding on a moving bike, None if the
    scan could not tell for any of its pieces."""
    total = weight = 0.0
    for seg in m.segments:
        samples = day.riding.get(seg.recording_id)
        if not samples:
            continue
        vals = [v for t, v in samples if seg.t_start <= t <= seg.t_end]
        if vals:
            total += float(np.mean(vals)) * seg.duration
            weight += seg.duration
    return total / weight if weight > 0 else None


def _has_gopro_gps(m: Moment, day: DayData) -> bool:
    """True when a piece of the moment has the camera's own GPS speed."""
    return any(
        s.recording_id not in day.from_racebox and _day_gps(day, s.recording_id)[0]
        for s in m.segments
    )


def _moment_pace(m: Moment, day: DayData, min_speed: float) -> str | None:
    """"stopped", "slow" (paddock, pit lane) or "track"; None when unknown.

    GoPro GPS speed decides. Without it (e.g. files re-wrapped without the
    GoPro telemetry track) the picture does: whether the camera rides on a
    moving bike (``motion.MountWatch``). A camera in the hand moves the
    picture as much as the track streaming past, and RaceBox time laid on
    by scene motion can be off, so RaceBox speed and plain scene motion
    only stand in when the scan could not tell.
    """
    riding = _moment_riding(m, day)
    if riding is not None and not _has_gopro_gps(m, day):
        if riding >= RIDING_MIN_SHARE:
            return "track"
        motion = _moment_motion(m, day)
        return "stopped" if motion is not None and motion < DEFAULT_THRESHOLD else "slow"
    speed = _moment_speed(m, day)
    if speed is not None:
        if speed < min(STOPPED_KMH, min_speed):
            return "stopped"
        return "slow" if speed < min_speed else "track"
    unplaced = _unplaced_pace(m, day)
    if unplaced is not None:
        motion = _moment_motion(m, day)
        if motion is not None and motion < DEFAULT_THRESHOLD:
            return "stopped"
        return unplaced
    motion = _moment_motion(m, day)
    if motion is None:
        return None
    if motion < DEFAULT_THRESHOLD:
        return "stopped"
    return "slow" if motion < TRACK_MOTION else "track"


def place_without_gps(day: DayData, track) -> None:
    """Lay recordings without GoPro GPS on the RaceBox log by scene motion.

    A clearly placed recording takes its start time, speed and turn rate
    from RaceBox. The rest of a camera's day is then matched as a whole (one
    clock error per camera), which gives those recordings RaceBox speed. A
    recording still unplaced keeps its camera clock and is checked against
    the log at both readings of that clock (``_moment_pace``).
    """
    from videobrowser.racebox_align import clock_guesses, day_match, motion_match

    if track is None:
        return
    day.track = track
    missing = [rid for rid in day.clocks if rid not in day.gps]
    for rid, gps in zip(missing, parallel(lambda rid: _recording_gps(day.clocks[rid]), missing)):
        day.gps[rid] = gps
    cameras: dict[str, list[str]] = {}
    for rid, clock in day.clocks.items():
        if rid in day.motion and clock.start_utc is not None and not _day_gps(day, rid)[0]:
            cameras.setdefault(day.profiles[rid], []).append(rid)

    def place(rid: str, found: float, how: str) -> None:
        clock = day.clocks[rid]
        print(f"  {rid}: no GPS, laid on RaceBox by scene motion ({found - clock.start_utc:+.1f} s, {how})")
        clock.start_utc = found
        clock.source = "racebox-motion"
        day.from_racebox.add(rid)
        day.gps[rid] = (track.speeds(found, clock.duration), track.turns(found, clock.duration))

    # Camera-clock starts, before any recording is moved onto RaceBox time:
    # the whole-day match needs them all on the same (camera) clock.
    camera_start = {rid: day.clocks[rid].start_utc for rids in cameras.values() for rid in rids}
    for profile, rids in sorted(cameras.items()):
        unplaced = []
        for rid in rids:
            clock = day.clocks[rid]
            match = motion_match(day.motion[rid], track, clock_guesses(clock.start_utc))
            if match.start_utc is None:
                print(f"  {rid}: no GPS, no clear RaceBox match: {match.reason}")
                unplaced.append(rid)
                continue
            place(rid, match.start_utc, match.reason)
        if not unplaced:
            continue
        # The camera's whole day at once: clear where single recordings are
        # not, but only to within a minute or so at the session edges. Good
        # enough to tell paddock from track; too rough for start time (camera
        # sync) or turn rate, which stay as they are.
        first = min(camera_start[rid] for rid in rids)
        match = day_match([(camera_start[rid], day.motion[rid]) for rid in rids], track)
        if match.start_utc is None:
            print(f"  {profile}: whole day not matched to RaceBox either ({match.reason})")
            for rid in unplaced:
                day.unplaced[rid] = clock_guesses(day.clocks[rid].start_utc)
            continue
        shift = match.start_utc - first
        print(f"  {profile}: whole day on RaceBox ({shift:+.0f} s, {match.reason}); speed only")
        for rid in unplaced:
            clock = day.clocks[rid]
            day.gps[rid] = (track.speeds(camera_start[rid] + shift, clock.duration), [])
            day.from_racebox.add(rid)


def _unplaced_pace(m: Moment, day: DayData) -> str | None:
    """Pace of a moment on recordings the RaceBox match could not place.

    The camera clock is right to a few seconds under one of its two
    readings. When the log shows the bike in (or standing) under every
    reading, the moment is in the paddock. Otherwise it may be on track, but
    that is not sure, so it counts as slow and is left out too.
    """
    seg = next((s for s in m.segments if s.recording_id in day.unplaced), None)
    if seg is None or day.track is None:
        return None
    clock = day.clocks[seg.recording_id]
    for guess in day.unplaced[seg.recording_id]:
        shift = guess - (clock.start_utc or guess)
        lo = (m.utc_start + shift) - UNPLACED_SLACK_S
        hi = (m.utc_end + shift) + UNPLACED_SLACK_S
        speeds = [s.speed_kmh for s in day.track.speeds(lo, hi - lo)]
        if speeds and max(speeds) >= STOPPED_KMH:
            return "slow"
    return "stopped"


def _moment_vehicles(m: Moment, day: DayData) -> float | None:
    """Share of the moment's time with a car in view, None if not stored."""
    total = weight = 0.0
    for seg in m.segments:
        found = vehicle_share(day.vehicles.get(seg.recording_id, []), [(seg.t_start, seg.t_end)])
        if found is not None:
            total += found * seg.duration
            weight += seg.duration
    return total / weight if weight > 0 else None


def day_moments(
    day: DayData,
    crashes: bool = True,
    min_speed: float = DEFAULT_MIN_TRACK_KMH,
    max_vehicle_share: float = DEFAULT_MAX_SHARE,
) -> list[Moment]:
    events = [
        CamEvent(
            recording_id=row["recording_id"],
            facing=day.facing(row["recording_id"]),
            type=row["type"],
            t_start=float(row["t_start"]),
            t_end=float(row["t_end"]),
            score=float(row["score"]),
            utc_offset=float(day.clocks[row["recording_id"]].start_utc or 0.0),
        )
        for row in day.events
    ]
    moments = find_moments(events)
    if crashes:
        moments.extend(_crash_moments(day))
    if max_vehicle_share > 0:
        before = len(moments)
        moments = [
            m
            for m in moments
            if m.theme in SPEED_EXEMPT_THEMES
            or (_moment_vehicles(m, day) or 0.0) < max_vehicle_share
        ]
        if len(moments) < before:
            print(f"  {before - len(moments)} moment(s) with a car in view left out (paddock)")
    if min_speed <= 0:
        return moments
    kept: list[Moment] = []
    stopped = slow = 0
    for m in moments:
        pace = None if m.theme in SPEED_EXEMPT_THEMES else _moment_pace(_approach(m), day, min_speed)
        if pace == "stopped":
            stopped += 1
        elif pace == "slow":
            slow += 1
        else:
            kept.append(m)
    if stopped or slow:
        print(f"  left out: {stopped} moment(s) standing still, {slow} below {min_speed:g} km/h (paddock)")
    return kept


def _approach(m: Moment) -> Moment:
    """The part of *m* whose speed tells track from paddock: the run-up for
    ``APPROACH_THEMES``, else the whole moment."""
    lead = APPROACH_THEMES.get(m.theme)
    if lead is None:
        return m
    seg = m.segments[0]
    lo = max(0.0, seg.t_start - 1.0)
    start = m.utc_start - (seg.t_start - lo)
    return Moment(m.theme, start, start + lead, m.score, [Segment(seg.recording_id, lo, lo + lead)])


def clip_drop_reason(
    day: DayData,
    rid: str,
    t_start: float,
    t_end: float,
    min_speed: float = DEFAULT_MIN_TRACK_KMH,
    max_vehicle_share: float = DEFAULT_MAX_SHARE,
) -> str | None:
    """Why a scan clip window is left out of the exports, None to keep it.

    The same tests ``day_moments`` puts a moment of other riders through: a
    car in view, the bike standing still or riding below *min_speed*.
    """
    start = day.clocks[rid].start_utc or 0.0
    m = Moment("action", start + t_start, start + t_end, 0.0, [Segment(rid, t_start, t_end)])
    if max_vehicle_share > 0 and (_moment_vehicles(m, day) or 0.0) >= max_vehicle_share:
        return "car in view"
    if min_speed <= 0:
        return None
    pace = _moment_pace(m, day, min_speed)
    if pace == "stopped":
        return "standing still"
    if pace == "slow":
        return f"below {min_speed:g} km/h"
    return None


# --- rendering -------------------------------------------------------------


def _clock_label(utc: float) -> str:
    return datetime.fromtimestamp(utc, tz=timezone.utc).astimezone(_TZ).strftime("%H:%M")


def _file_time(utc: float) -> str:
    return datetime.fromtimestamp(utc, tz=timezone.utc).astimezone(_TZ).strftime("%H%M%S")


def _source_for(segment: Segment, clips: list[ClipRow]) -> tuple[Path, float, float] | None:
    """(vertical file, start inside it, duration) covering most of *segment*."""
    best: tuple[float, ClipRow] | None = None
    for clip in clips:
        if clip.recording_id != segment.recording_id or clip.vertical_path is None:
            continue
        if not clip.vertical_path.exists():
            continue
        inter = min(segment.t_end, clip.t_end) - max(segment.t_start, clip.t_start)
        if inter > 0 and (best is None or inter > best[0]):
            best = (inter, clip)
    if best is None or best[0] < 0.6 * segment.duration:
        return None
    clip = best[1]
    lo = max(segment.t_start, clip.t_start)
    hi = min(segment.t_end, clip.t_end)
    return clip.vertical_path, lo - clip.t_start, hi - lo


# Every cut used to probe its recording's files again, over the network share.
_probe_once = lru_cache(maxsize=None)(probe)


def _from_source(day: DayData, segment: Segment, dest: Path) -> Path:
    """9:16 cut straight from the GoPro files (no HUD), looking into corners."""
    from videobrowser.gopro import Chapter, Recording
    from videobrowser.vertical import export_vertical, turn_focus

    clock = day.clocks[segment.recording_id]
    recording = Recording(segment.recording_id, "", segment.recording_id)
    for i, ref in enumerate(clock.chapters, start=1):
        info = _probe_once(ref.path)
        recording.chapters.append(
            Chapter(ref.path, "", i, segment.recording_id, ref.duration, ref.offset, info)
        )
    end = min(segment.t_end, clock.duration)
    window = ClipWindow(segment.t_start, end, segment.recording_id, [], 0.0, [])
    _speeds, turns = _day_gps(day, segment.recording_id)
    focus = turn_focus(turns, segment.t_start, end, day.facing(segment.recording_id))
    return export_vertical(recording, window, dest, day.rotations[segment.recording_id], focus)


def collect_short(plan: ShortPlan, day: DayData, dest: Path, jobs: int = 1) -> Path | None:
    """Folder with the source clips of *plan*, in order, for editing by hand.

    Each moment's camera pieces are copied whole from the scan's vertical
    clips (no cuts, so no audio clicks at the joins); a piece no clip covers
    is cut from the GoPro files, *jobs* of those cuts at a time. ``klip.txt``
    lists each file with the part the plan would use.
    """
    dest.mkdir(parents=True, exist_ok=True)
    lines = [f"{plan.name}: {len(plan.moments)} moments, {plan.duration:.0f} s", ""]
    copied: dict[Path, str] = {}
    cuts: list[tuple[Segment, str]] = []
    for i, moment in enumerate(plan.moments, start=1):
        for j, segment in enumerate(moment.segments):
            cam = day.facing(segment.recording_id)
            stem = f"{i:02d}{'abcdefgh'[j] if len(moment.segments) > 1 else ''}"
            stem += f"_{_file_time(moment.utc_start)}_{moment.theme}_{cam}"
            found = _source_for(segment, day.clips)
            try:
                if found is None:
                    name = f"{stem}.mp4"
                    cuts.append((segment, name))
                    use = "whole file"
                else:
                    src, start, duration = found
                    name = copied.get(src) or f"{stem}_{src.name}"
                    if src not in copied:
                        shutil.copy2(src, dest / name)
                        copied[src] = name
                    use = f"{start:.1f}-{start + duration:.1f} s"
            except Exception as exc:
                # All or nothing: an incomplete folder must not replace a whole one.
                raise RuntimeError(f"{segment.recording_id}@{segment.t_start:.0f}s: {exc}") from exc
            lines.append(f"{name}  ({moment.label}, {_clock_label(moment.utc_start)}, use {use})")
    if len(lines) == 2:
        return None

    def cut(item: tuple[Segment, str]) -> None:
        segment, name = item
        try:
            _from_source(day, segment, dest / name)
        except Exception as exc:
            raise RuntimeError(f"{segment.recording_id}@{segment.t_start:.0f}s: {exc}") from exc

    # Each cut is an ffmpeg decode + encode that mostly waits on NVDEC/NVENC
    # or the network, so several run side by side.
    if jobs > 1 and len(cuts) > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for _ in pool.map(cut, cuts):
                pass
    else:
        for item in cuts:
            cut(item)
    (dest / "klip.txt").write_text("\n".join(lines) + "\n")
    return dest


# --- index page ------------------------------------------------------------


def _moment_json(m: Moment, day: DayData) -> dict:
    return {
        "theme": m.theme,
        "label": m.label,
        "time": _clock_label(m.utc_start),
        "utc_start": round(m.utc_start, 2),
        "score": round(m.score, 3),
        "segments": [
            {
                "recording_id": s.recording_id,
                "camera": day.facing(s.recording_id),
                "t_start": round(s.t_start, 2),
                "t_end": round(s.t_end, 2),
            }
            for s in m.segments
        ],
    }


def write_index(
    day_dir: Path, day: DayData, plans: list[ShortPlan], written: dict[str, Path]
) -> Path:
    def rel(path: Path) -> str:
        return html.escape(os.path.relpath(path, day_dir))

    def video(path: Path, caption: str) -> str:
        return (
            f'<figure><video src="{rel(path)}" controls preload="metadata"></video>'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )

    parts = [
        "<!doctype html><html lang=da><meta charset=utf-8>",
        "<meta name=viewport content='width=device-width,initial-scale=1'>",
        f"<title>Shorts {html.escape(day.date_label)}</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee}"
        "h1,h2{font-weight:600}.grid{display:flex;flex-wrap:wrap;gap:12px}"
        "figure{margin:0;width:200px}video{width:200px;aspect-ratio:9/16;background:#000;border-radius:8px}"
        ".wide video{aspect-ratio:16/9;width:320px}.wide{width:320px}"
        "figcaption{font-size:13px;opacity:.8;margin-top:4px}</style>",
        f"<h1>{html.escape(day.date_label)}</h1>",
    ]
    for plan in plans:
        folder = written.get(plan.name)
        if folder is None:
            continue
        title = f"{plan.name} · {len(plan.moments)} øjeblikke · {plan.duration:.0f}s"
        parts.append(f'<h2><a href="{rel(folder)}/">{html.escape(title)}</a></h2><div class=grid>')
        for clip in sorted(folder.glob("*.mp4")):
            parts.append(video(clip, clip.name))
        parts.append("</div>")
    by_type: dict[str, list[ClipRow]] = {}
    for clip in sorted(day.clips, key=lambda c: (c.recording_id, c.t_start)):
        for kind in clip.types.split("+"):
            by_type.setdefault(kind, []).append(clip)
    names = {
        "approaching_rider": "Rytter nærmer sig",
        "receding_rider": "Rytter fjerner sig",
        "nearby_rider": "Rytter tæt på",
        "rider_pass": "Ryttere bytter plads",
        "kneepuck_visible": "Knæ i asfalten",
        "rider_down": "Muligt styrt",
        "off_track": "Ude i gruset",
        "own_off_track": "Afkørsel",
    }
    for kind, clips in by_type.items():
        parts.append(f"<h2>Alle klip: {html.escape(names.get(kind, kind))}</h2><div class=grid>")
        for clip in clips:
            clock = day.clocks.get(clip.recording_id)
            when = _clock_label(clock.utc(clip.t_start)) if clock and clock.start_utc else "?"
            cam = day.facing(clip.recording_id)
            caption = f"{when} · {cam} · {clip.recording_id}"
            if clip.vertical_path and clip.vertical_path.exists():
                parts.append(video(clip.vertical_path, caption))
            elif clip.clip_path and clip.clip_path.exists():
                parts.append(f"<div class=wide>{video(clip.clip_path, caption)}</div>")
        parts.append("</div>")
    dest = day_dir / "index.html"
    dest.write_text("\n".join(parts))
    return dest


# --- entry point -----------------------------------------------------------


def _racebox_offsets(sessions_dir: Path | None):
    if sessions_dir is None or not sessions_dir.exists() or not any(sessions_dir.glob("*.vbo")):
        return None
    try:
        from videobrowser.racebox_stage import _gopro_session_offset, load_sessions
        from videobrowser.sync import _chapter_blob

        sessions = load_sessions(sessions_dir)
    except Exception as exc:
        print(f"shorts: RaceBox sessions unavailable ({exc})")
        return None
    if not sessions:
        return None

    def offset(clock: CameraClock) -> float | None:
        blob = _chapter_blob(clock.chapters[0]) if clock.chapters else None
        return _gopro_session_offset(blob, sessions) if blob else None

    return offset


def build_shorts(
    out_dir: Path,
    inputs: list[Path] | None = None,
    sessions_dir: Path | None = None,
    dates: list[str] | None = None,
    max_seconds: float = 50.0,
    max_action_shorts: int = 6,
    audio_sync: bool = True,
    crashes: bool = True,
    render: bool = True,
    min_speed: float = DEFAULT_MIN_TRACK_KMH,
    jobs: int = 1,
    max_vehicle_share: float = DEFAULT_MAX_SHARE,
) -> dict[str, list[Path]]:
    db = ScanDB.open(out_dir / "scan.sqlite")
    try:
        if inputs:
            backfill_meta(db, inputs, out_dir)
        days = load_days(db, out_dir)
    finally:
        db.close()
    if not days:
        print("shorts: nothing scanned yet (or scanned before this version; pass the input folders)")
        return {}
    rb_offset = _racebox_offsets(sessions_dir)
    from videobrowser.racebox_align import load_track

    track = load_track(sessions_dir)
    results: dict[str, list[Path]] = {}
    picked = [(label, day) for label, day in sorted(days.items()) if not dates or label in dates]
    for d, (label, day) in enumerate(picked, 1):
        print(
            f"[day {d} of {len(picked)}] shorts {label}: "
            f"{len(day.clocks)} camera recording(s), {len(day.events)} events"
        )
        place_without_gps(day, track)
        align(list(day.clocks.values()), racebox_offset=rb_offset, use_audio=audio_sync)
        moments = day_moments(
            day, crashes=crashes, min_speed=min_speed, max_vehicle_share=max_vehicle_share
        )
        plans = plan_shorts(moments, max_seconds, max_action_shorts)
        day_dir = out_dir / "shorts" / label
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "moments.json").write_text(
            json.dumps(
                {
                    "cameras": {
                        rid: {
                            "camera": day.facing(rid),
                            "profile": day.profiles[rid],
                            "start_utc": c.start_utc,
                            "sync": c.source,
                        }
                        for rid, c in day.clocks.items()
                    },
                    "moments": [_moment_json(m, day) for m in sorted(moments, key=lambda m: m.utc_start)],
                    "shorts": [
                        {"name": p.name, "theme": p.theme, "seconds": round(p.duration, 1),
                         "moments": [_clock_label(m.utc_start) for m in p.moments]}
                        for p in plans
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        counts = {t: sum(1 for m in moments if m.theme == t) for t in LABELS}
        print("  moments: " + ", ".join(f"{t}={n}" for t, n in counts.items() if n))
        written: dict[str, Path] = {}
        if render:
            for n, plan in enumerate(plans, 1):
                # Fill a folder beside the old one and swap it in only once
                # complete, so a failed or interrupted run keeps the previous one.
                final = day_dir / plan.name
                partial = day_dir / f".{plan.name}.partial"
                shutil.rmtree(partial, ignore_errors=True)
                try:
                    path = collect_short(plan, day, partial, jobs=jobs)
                except Exception as exc:
                    print(f"  [short {n} of {len(plans)}] {plan.name} failed: {exc}")
                    shutil.rmtree(partial, ignore_errors=True)
                    continue
                if path is None:
                    shutil.rmtree(partial, ignore_errors=True)
                    continue
                # A folder cannot be renamed over a full one: move the old one
                # aside first, so it is never gone before the new one is in.
                aside = day_dir / f".{plan.name}.old"
                shutil.rmtree(aside, ignore_errors=True)
                if final.exists():
                    os.replace(final, aside)
                os.replace(partial, final)
                shutil.rmtree(aside, ignore_errors=True)
                written[plan.name] = final
                print(f"  [short {n} of {len(plans)}] wrote {final.relative_to(out_dir)}/ ({plan.duration:.0f}s)")
            # Shorts this plan no longer produces (fewer moments than last
            # run), and finished .mp4 shorts from before the folders.
            names = {plan.name for plan in plans}
            for old in day_dir.glob("*.mp4"):
                old.unlink()
            for old in day_dir.iterdir():
                if old.is_dir() and old.name not in names and (old / "klip.txt").exists():
                    shutil.rmtree(old)
        write_index(day_dir, day, plans, written)
        results[label] = list(written.values())
    return results
