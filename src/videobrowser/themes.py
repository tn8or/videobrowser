"""Turn per-camera events into themed moments on the day's shared clock.

A scan finds the same event types on every camera (``nearby_rider``,
``approaching_rider``, ``receding_rider``, ``kneepuck_visible``). What they
mean depends on which way the camera faces: a rider growing in a front camera
is one being caught, in a rear camera one catching up. With all cameras on one
clock (``sync.py``) the hand-over from one camera to the other tells an
overtake apart from a rider merely closing in:

- **overtake**: front ``approaching`` ends, then rear ``receding`` starts
- **overtaken**: rear ``approaching`` ends, then front ``receding`` starts
- **catching_up**: a sustained front ``approaching`` that is not an overtake
- **duel**: ``rider_pass`` (two other riders swap places in view, see
  ``riders.score_passes``)
- **kneepuck**: ``kneepuck_visible``
- **crash**: own crash from GoPro telemetry (hard jolt, then stopped on track)
- **crash_other**: ``rider_down`` (another rider off the bike, see incidents.py)
- **off_track**: ``off_track`` (a rider out on grass or gravel)
- **own_off_track**: ``own_off_track`` (our own bike over the edge of the track)
- **action**: any other clip-worthy event, for the mixed "dagens action" shorts
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REAR_PROFILES = frozenset({"rear", "rear_right", "rear_rotated"})

# Theme -> label shown on screen.
LABELS = {
    "overtake": "OVERHALING",
    "overtaken": "BLIVER OVERHALET",
    "catching_up": "INDHENTER",
    "duel": "DUEL",
    "kneepuck": "KNÆ I ASFALTEN",
    "crash": "STYRT",
    "crash_other": "MULIGT STYRT",
    "off_track": "UDE I GRUSET",
    "own_off_track": "AFKØRSEL",
    "action": "ACTION",
}
# Order themed shorts are written in.
THEME_ORDER = (
    "overtake", "overtaken", "catching_up", "duel", "kneepuck", "crash", "crash_other", "off_track",
    "own_off_track",
)

# Rear camera must pick the rider up within this window around the front
# camera losing them (seconds, relative to the end of the first event).
HANDOVER_BEFORE_S = 1.5
HANDOVER_AFTER_S = 5.0
# A front "approaching" this long counts as catching up on its own.
MIN_CATCH_UP_S = 1.5
# Longest piece of one camera in a moment.
MAX_SEGMENT_S = 7.0


def facing(profile: str) -> str:
    return "rear" if profile in REAR_PROFILES else "front"


@dataclass(frozen=True)
class CamEvent:
    """One scan event placed on the shared clock."""

    recording_id: str
    facing: str
    type: str
    t_start: float  # recording time
    t_end: float
    score: float
    utc_offset: float  # recording start_utc

    @property
    def utc_start(self) -> float:
        return self.utc_offset + self.t_start

    @property
    def utc_end(self) -> float:
        return self.utc_offset + self.t_end

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclass(frozen=True)
class Segment:
    recording_id: str
    t_start: float  # recording time
    t_end: float

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclass
class Moment:
    theme: str
    utc_start: float
    utc_end: float
    score: float
    segments: list[Segment] = field(default_factory=list)
    # Why the moment was kept (or left out): what was in view, the pace and
    # where it came from. Written to klip.txt and moments.json.
    why: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    @property
    def label(self) -> str:
        return LABELS[self.theme]


def _segment(ev: CamEvent, lo: float, hi: float) -> Segment:
    lo = max(0.0, lo)
    if hi - lo > MAX_SEGMENT_S:
        lo = hi - MAX_SEGMENT_S
    return Segment(ev.recording_id, lo, hi)


def _handover(first: CamEvent, candidates: list[CamEvent]) -> CamEvent | None:
    lo = first.utc_end - HANDOVER_BEFORE_S
    hi = first.utc_end + HANDOVER_AFTER_S
    hits = [c for c in candidates if lo <= c.utc_start <= hi]
    return min(hits, key=lambda c: abs(c.utc_start - first.utc_end)) if hits else None


def _pass_moment(theme: str, first: CamEvent, second: CamEvent) -> Moment:
    seg_a = _segment(first, first.t_start - 1.0, first.t_end + 0.5)
    seg_b = Segment(second.recording_id, max(0.0, second.t_start - 0.3), second.t_start + 3.0)
    return Moment(
        theme=theme,
        utc_start=first.utc_offset + seg_a.t_start,
        utc_end=second.utc_offset + seg_b.t_end,
        score=first.score + second.score,
        segments=[seg_a, seg_b],
    )


def _single(theme: str, ev: CamEvent, before: float, after: float, score: float | None = None) -> Moment:
    seg = _segment(ev, ev.t_start - before, ev.t_end + after)
    return Moment(
        theme=theme,
        utc_start=ev.utc_offset + seg.t_start,
        utc_end=ev.utc_offset + seg.t_end,
        score=ev.score if score is None else score,
        segments=[seg],
    )


def find_moments(events: list[CamEvent]) -> list[Moment]:
    """Themed moments, best first within each theme, overlaps removed."""
    by = lambda f, t: [e for e in events if e.facing == f and e.type == t]  # noqa: E731
    front_in, front_out = by("front", "approaching_rider"), by("front", "receding_rider")
    rear_in, rear_out = by("rear", "approaching_rider"), by("rear", "receding_rider")

    moments: list[Moment] = []
    used: set[int] = set()
    for first in sorted(front_in, key=lambda e: e.utc_end):
        second = _handover(first, rear_out)
        if second is not None:
            moments.append(_pass_moment("overtake", first, second))
            used.update((id(first), id(second)))
    for first in sorted(rear_in, key=lambda e: e.utc_end):
        second = _handover(first, front_out)
        if second is not None:
            moments.append(_pass_moment("overtaken", first, second))
            used.update((id(first), id(second)))
    for ev in front_in:
        if id(ev) not in used and ev.duration >= MIN_CATCH_UP_S:
            moments.append(_single("catching_up", ev, 1.0, 1.0, ev.score * ev.duration))
            used.add(id(ev))
    for ev in events:
        if ev.type == "rider_pass":
            moments.append(_single("duel", ev, 2.0, 2.0))
            used.add(id(ev))
    for ev in events:
        if ev.type == "kneepuck_visible":
            moments.append(_single("kneepuck", ev, 1.5, 1.5))
            used.add(id(ev))
    for ev in events:
        if ev.type == "rider_down":
            moments.append(_single("crash_other", ev, 4.0, 3.0, 5.0 * ev.score))
            used.add(id(ev))
        elif ev.type == "off_track":
            moments.append(_single("off_track", ev, 2.0, 2.0, 2.0 * ev.score))
            used.add(id(ev))
        elif ev.type == "own_off_track":
            # The run-up belongs to it, longer than MAX_SEGMENT_S allows.
            seg = Segment(ev.recording_id, max(0.0, ev.t_start - 5.0), ev.t_end + 3.0)
            moments.append(
                Moment("own_off_track", ev.utc_offset + seg.t_start, ev.utc_offset + seg.t_end,
                       4.0 * ev.score, [seg])
            )
            used.add(id(ev))
    for ev in events:
        if id(ev) in used or ev.type == "kneepuck_visible":
            continue
        peak = 0.5 * (ev.t_start + ev.t_end)
        seg = _segment(ev, peak - 3.0, peak + 3.0)
        moments.append(
            Moment("action", ev.utc_offset + seg.t_start, ev.utc_offset + seg.t_end, ev.score, [seg])
        )
    return dedupe(moments)


def dedupe(moments: list[Moment], max_overlap: float = 0.5) -> list[Moment]:
    """Drop a moment that mostly repeats a better one already kept.

    Same-theme overlaps are always duplicates. "action" also yields to any
    themed moment it overlaps, so the mixed shorts do not show it twice.
    """
    rank = {theme: i for i, theme in enumerate(THEME_ORDER)}
    ordered = sorted(moments, key=lambda m: (m.theme == "action", rank.get(m.theme, 99), -m.score))
    kept: list[Moment] = []
    for m in ordered:
        clash = False
        for k in kept:
            if k.theme != m.theme and m.theme != "action":
                continue
            inter = min(m.utc_end, k.utc_end) - max(m.utc_start, k.utc_start)
            if inter > max_overlap * min(m.utc_end - m.utc_start, k.utc_end - k.utc_start):
                clash = True
                break
        if not clash:
            kept.append(m)
    return kept


# --- own crash from telemetry ----------------------------------------------

# Jolt, as a multiple of the median accelerometer magnitude (~1 g).
CRASH_MIN_JOLT = 3.0
# Moving at least this fast in the few seconds before ...
CRASH_MIN_SPEED_KMH = 30.0
# ... and stopped (median below this) a few seconds after.
CRASH_STOPPED_KMH = 8.0


def detect_crashes(
    speed_t: np.ndarray,
    speed_kmh: np.ndarray,
    accl_t: np.ndarray,
    accl_xyz: np.ndarray,
) -> list[float]:
    """Recording times where the bike took a hard jolt and then stood still.

    Hard braking peaks well under 2 g and still leaves the bike rolling; a
    kerb strike is a jolt but the speed carries on. Both conditions together
    are rare outside a crash.
    """
    if len(speed_t) < 10 or len(accl_t) < 50:
        return []
    mag = np.linalg.norm(np.asarray(accl_xyz, dtype=np.float64), axis=1)
    base = float(np.median(mag))
    if base <= 0:
        return []
    jolt = mag / base
    hits: list[float] = []
    for i in np.flatnonzero(jolt >= CRASH_MIN_JOLT):
        t = float(accl_t[i])
        if hits and t - hits[-1] < 15.0:
            continue
        before = speed_kmh[(speed_t >= t - 4.0) & (speed_t <= t)]
        after = speed_kmh[(speed_t >= t + 3.0) & (speed_t <= t + 10.0)]
        if len(before) == 0 or len(after) < 3:
            continue
        # Median, not max: a single GPS spike while parked is not "moving".
        if float(np.median(before)) >= CRASH_MIN_SPEED_KMH and float(np.median(after)) <= CRASH_STOPPED_KMH:
            hits.append(t)
    return hits


def crash_moment(recording_id: str, t: float, utc_offset: float) -> Moment:
    seg = Segment(recording_id, max(0.0, t - 6.0), t + 4.0)
    return Moment("crash", utc_offset + seg.t_start, utc_offset + seg.t_end, 10.0, [seg])
