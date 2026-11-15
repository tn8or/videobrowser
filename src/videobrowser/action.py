"""Is there anything to see? Other riders and kerbs in view, per second.

A short needs action in the picture: at least one other rider, more is
better, or a kerb right beside the bike. A rider is a bike with a person on
it: a parked bike, or the camera's own bike filmed by hand, is not. The scan stores both per second
(``action`` table): the most other riders' bike boxes in one frame, and the
share of frames with a red-and-white kerb close by. A kerb close by on its own
is a ``kerb_close`` event, so riding over or along one becomes a moment too.
"""

from __future__ import annotations

import cv2
import numpy as np

from videobrowser.events import Event
from videobrowser.riders import TrackPoint, is_ego_track

PERSON, MOTORCYCLE = 0, 3
# Bike boxes smaller than this (fraction of frame height) are too far off to
# count; the vertical crop follows riders down to the same size.
MIN_RIDER_HEIGHT_FRAC = 0.03
# A bike track has a rider when a person box sits on it in this share of the
# frames it is seen in: the person's centre over the bike, and most of the
# person box overlapping it or above it.
MIN_MOUNTED_SHARE = 0.3
MIN_PERSON_OVERLAP = 0.2
# Two boxes overlapping this much (intersection over union) are one bike.
SAME_BIKE_IOU = 0.5
# A bike box this share of the frame, in this many frames of a second, means
# the camera looks at its own cockpit.
OWN_VIEW_AREA_FRAC = 0.5
OWN_VIEW_FRAMES = 1
OWN_VIEW_CLASSES = frozenset({2, 3, 5, 7})  # car, motorcycle, bus, truck

# Kerb pixels are red or white next to the other colour, in the lower part
# of the frame (the ground near the bike), compared at this width (px).
KERB_WIDTH = 160
KERB_TOP = 0.4
# Share of that part of the frame that must be kerb for it to be close.
KERB_CLOSE_SHARE = 0.04
# A kerb_close event needs the kerb close in this many consecutive frames.
KERB_MIN_FRAMES = 4


def _person_on(bike: TrackPoint, person: TrackPoint) -> bool:
    cx = 0.5 * (person.x1 + person.x2)
    if not bike.x1 <= cx <= bike.x2 or person.y1 > bike.y1 + 0.5 * bike.height:
        return False
    overlap_w = min(bike.x2, person.x2) - max(bike.x1, person.x1)
    overlap_h = min(bike.y2, person.y2) - max(bike.y1, person.y1)
    if overlap_w <= 0 or overlap_h <= 0:
        return False
    return overlap_w * overlap_h >= MIN_PERSON_OVERLAP * max(person.area, 1e-9)


def mounted_tracks(tracks: dict[int, list[TrackPoint]]) -> set[int]:
    """Ids of the bike tracks with a person on them: riders, not parked bikes."""
    people: dict[float, list[TrackPoint]] = {}
    for points in tracks.values():
        for p in points:
            if p.cls == PERSON:
                people.setdefault(p.t, []).append(p)
    mounted: set[int] = set()
    for track_id, points in tracks.items():
        bikes = [p for p in points if p.cls == MOTORCYCLE]
        if not bikes:
            continue
        on = sum(any(_person_on(b, q) for q in people.get(b.t, [])) for b in bikes)
        if on >= MIN_MOUNTED_SHARE * len(bikes):
            mounted.add(track_id)
    return mounted


def _counts(p: TrackPoint) -> bool:
    return p.cls == MOTORCYCLE and p.height_frac >= MIN_RIDER_HEIGHT_FRAC


def _iou(a: TrackPoint, b: TrackPoint) -> float:
    w = min(a.x2, b.x2) - max(a.x1, b.x1)
    h = min(a.y2, b.y2) - max(a.y1, b.y1)
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / max(a.area + b.area - inter, 1e-9)


def rider_points(tracks: dict[int, list[TrackPoint]]) -> list[TrackPoint]:
    """The boxes ``riders_per_second`` counts: other riders' bikes big enough
    to matter. Stored beside the event windows' tracks, so a short cut from
    the source can follow a rider outside any window."""
    mounted = mounted_tracks(tracks)
    return [
        p
        for track_id, points in tracks.items()
        if track_id in mounted and not is_ego_track(points)
        for p in points
        if _counts(p)
    ]


def riders_per_second(tracks: dict[int, list[TrackPoint]]) -> dict[int, int]:
    """Most other riders (bikes with a person on) seen in one frame, per whole second.

    The tracker sometimes puts two tracks on one bike (two boxes in nearly the
    same place); boxes overlapping by ``SAME_BIKE_IOU`` count once."""
    mounted = mounted_tracks(tracks)
    per_frame: dict[float, list[TrackPoint]] = {}
    for track_id, points in tracks.items():
        if track_id not in mounted or is_ego_track(points):
            continue
        for p in points:
            if _counts(p):
                per_frame.setdefault(p.t, []).append(p)
    out: dict[int, int] = {}
    for t, boxes in per_frame.items():
        distinct: list[TrackPoint] = []
        for box in sorted(boxes, key=lambda b: -b.area):
            if not any(_iou(box, d) >= SAME_BIKE_IOU for d in distinct):
                distinct.append(box)
        out[int(t)] = max(out.get(int(t), 0), len(distinct))
    return out


def own_view_frame(points: list[TrackPoint]) -> bool:
    """True when one frame has a box of a bike or vehicle class filling
    ``OWN_VIEW_AREA_FRAC`` of the picture. Seen from right up against the
    cockpit, the tracker calls the bike a motorcycle one moment and a car the
    next, so every vehicle class counts (the rider boxes drop cars later)."""
    return any(p.cls in OWN_VIEW_CLASSES and p.area_frac >= OWN_VIEW_AREA_FRAC for p in points)


def own_view_per_second(samples: list[tuple[float, bool]]) -> set[int]:
    """Whole seconds where the camera looks at its own bike: ``own_view_frame``
    held in ``OWN_VIEW_FRAMES`` frames or more (the camera was knocked or
    turned down at the cockpit)."""
    seen: dict[int, int] = {}
    for t, flag in samples:
        if flag:
            seen[int(t)] = seen.get(int(t), 0) + 1
    return {k for k, n in seen.items() if n >= OWN_VIEW_FRAMES}


def nearest_rider_per_second(tracks: dict[int, list[TrackPoint]]) -> dict[int, float]:
    """Height (fraction of the frame) of the nearest other rider, per whole second."""
    mounted = mounted_tracks(tracks)
    out: dict[int, float] = {}
    for track_id, points in tracks.items():
        if track_id not in mounted or is_ego_track(points):
            continue
        for p in points:
            if _counts(p):
                out[int(p.t)] = max(out.get(int(p.t), 0.0), p.height_frac)
    return out


def kerb_share(image: np.ndarray) -> float:
    """Share of the lower part of the frame that is red-and-white kerb."""
    h, w = image.shape[:2]
    small = cv2.resize(image, (KERB_WIDTH, max(8, round(KERB_WIDTH * h / max(1, w)))),
                       interpolation=cv2.INTER_AREA)
    small = small[int(small.shape[0] * KERB_TOP):]
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((hue <= 8) | (hue >= 172)) & (sat >= 90) & (val >= 70)).astype(np.uint8)
    white = ((sat <= 40) & (val >= 180)).astype(np.uint8)
    # Kerb stripes alternate: red counts where white is within a few pixels,
    # and the other way round. A red bike or a white line alone does not.
    kernel = np.ones((5, 5), np.uint8)
    near_white = cv2.dilate(white, kernel) > 0
    near_red = cv2.dilate(red, kernel) > 0
    kerb = ((red > 0) & near_white) | ((white > 0) & near_red)
    return float(np.mean(kerb))


def kerb_events(samples: list[tuple[float, float]], recording_id: str) -> list[Event]:
    """``kerb_close`` where a kerb is close for ``KERB_MIN_FRAMES`` frames in a row."""
    events: list[Event] = []
    run: list[tuple[float, float]] = []

    def close() -> None:
        if len(run) >= KERB_MIN_FRAMES:
            score = min(1.0, float(np.mean([s for _, s in run])) / (4 * KERB_CLOSE_SHARE))
            events.append(Event("kerb_close", run[0][0], run[-1][0], 0.3 + 0.4 * score, recording_id))

    for t, share in sorted(samples):
        if share >= KERB_CLOSE_SHARE:
            run.append((t, share))
            continue
        close()
        run = []
    close()
    return events


def per_second(
    riders: dict[int, int],
    kerbs: list[tuple[float, float]],
    nearest: dict[int, float] | None = None,
    red_lamp: set[int] | None = None,
    own_view: set[int] | None = None,
) -> list[tuple[float, int, float, float, int, int]]:
    """(t, most riders in one frame, share of frames with a kerb close,
    height of the nearest rider, 1 with a red signal lamp blinking, 1 with the
    camera looking at its own cockpit) per second."""
    close: dict[int, list[float]] = {}
    for t, share in kerbs:
        close.setdefault(int(t), []).append(1.0 if share >= KERB_CLOSE_SHARE else 0.0)
    nearest = nearest or {}
    red_lamp = red_lamp or set()
    own_view = own_view or set()
    seconds = sorted(set(riders) | set(close) | red_lamp | own_view)
    return [
        (
            k + 0.5,
            riders.get(k, 0),
            float(np.mean(close[k])) if k in close else 0.0,
            nearest.get(k, 0.0),
            1 if k in red_lamp else 0,
            1 if k in own_view else 0,
        )
        for k in seconds
    ]


def in_view(samples: list[tuple], t0: float, t1: float) -> tuple[int, float] | None:
    """(most riders in one frame, longest kerb share of a second) over [t0, t1],
    None without samples there."""
    rows = [(row[1], row[2]) for row in samples if t0 - 0.5 <= row[0] <= t1 + 0.5]
    if not rows:
        return None
    return max(r for r, _ in rows), max(k for _, k in rows)


# How much there is to see, per second: each rider in view (up to
# MAX_RIDERS) counts by how near it is, full from NEAR_HEIGHT of the frame
# height up and at least FAR_WEIGHT; a kerb close by counts KERB_WEIGHT. Scans
# from before rider heights were stored count each rider as UNKNOWN_NEAR.
MAX_RIDERS = 4
NEAR_HEIGHT = 0.15
FAR_WEIGHT = 0.2
KERB_WEIGHT = 0.5
UNKNOWN_NEAR = 0.5


def action_density(samples: list[tuple], t0: float, t1: float) -> float | None:
    """Mean of the per-second action value over [t0, t1], None without samples."""
    vals: list[float] = []
    for row in samples:
        if not t0 - 0.5 <= row[0] <= t1 + 0.5:
            continue
        riders, kerb = row[1], row[2]
        height = row[3] if len(row) > 3 else 0.0
        if riders and height > 0:
            near = min(1.0, max(FAR_WEIGHT, height / NEAR_HEIGHT))
        else:
            near = UNKNOWN_NEAR
        vals.append(min(riders, MAX_RIDERS) * near + KERB_WEIGHT * kerb)
    return float(np.mean(vals)) if vals else None


def own_view_seconds(samples: list[tuple]) -> set[int]:
    """Whole seconds in which the camera looks at its own cockpit."""
    return {int(row[0]) for row in samples if len(row) > 5 and row[5]}


def red_lamp_in(samples: list[tuple], t0: float, t1: float) -> bool:
    """True when a red signal lamp blinks anywhere over [t0, t1]."""
    return any(len(row) > 4 and row[4] for row in samples if t0 - 0.5 <= row[0] <= t1 + 0.5)
