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


def riders_per_second(tracks: dict[int, list[TrackPoint]]) -> dict[int, int]:
    """Most other riders (bikes with a person on) seen in one frame, per whole second."""
    mounted = mounted_tracks(tracks)
    per_frame: dict[float, set[int]] = {}
    for track_id, points in tracks.items():
        if track_id not in mounted or is_ego_track(points):
            continue
        for p in points:
            if p.cls == MOTORCYCLE and p.height_frac >= MIN_RIDER_HEIGHT_FRAC:
                per_frame.setdefault(p.t, set()).add(track_id)
    out: dict[int, int] = {}
    for t, ids in per_frame.items():
        out[int(t)] = max(out.get(int(t), 0), len(ids))
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
    riders: dict[int, int], kerbs: list[tuple[float, float]]
) -> list[tuple[float, int, float]]:
    """(t, most riders in one frame, share of frames with a kerb close) per second."""
    close: dict[int, list[float]] = {}
    for t, share in kerbs:
        close.setdefault(int(t), []).append(1.0 if share >= KERB_CLOSE_SHARE else 0.0)
    seconds = sorted(set(riders) | set(close))
    return [
        (k + 0.5, riders.get(k, 0), float(np.mean(close[k])) if k in close else 0.0)
        for k in seconds
    ]


def in_view(samples: list[tuple[float, int, float]], t0: float, t1: float) -> tuple[int, float] | None:
    """(most riders in one frame, longest kerb share of a second) over [t0, t1],
    None without samples there."""
    rows = [(r, k) for t, r, k in samples if t0 - 0.5 <= t <= t1 + 0.5]
    if not rows:
        return None
    return max(r for r, _ in rows), max(k for _, k in rows)
