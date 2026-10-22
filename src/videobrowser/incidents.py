"""Possible crashes of other riders, from tracker boxes and the ground below them.

YOLO has no "crash" class, so this looks for what a crash does to the boxes
the scan already tracks (5 fps):

- **rider off the bike**: a motorcycle box that carried a person box for a
  second or more loses it, while a person box shows up beside it
- **bike down**: a motorcycle box turns low and wide (height -35%, aspect
  x1.8) and stays that way
- **off track**: the ground just below a rider's box turns from grey
  asphalt to green grass or pale gravel and stays there

``rider_down`` fires for "off the bike", or for "bike down" together with
"off track" (a bike leaned over in a corner is low and wide too). A lone
``off_track`` is its own, milder event: a trip through the gravel trap is
worth seeing even without a fall. All of these are candidates for a human to
check, not verdicts.
"""

from __future__ import annotations

import cv2
import numpy as np

from videobrowser.events import Event
from videobrowser.riders import TrackPoint

PERSON, MOTORCYCLE = 0, 3
# Boxes smaller than this (fraction of frame height) are too far to judge.
MIN_HEIGHT_FRAC = 0.06
# Ground share that counts as grass/gravel, and how long it must last.
OFFTRACK_FRAC = 0.5
OFFTRACK_POINTS = 3
# A person box covering this much of itself with a bike box rides that bike.
MOUNTED_OVERLAP = 0.5


def offtrack_fraction(image: np.ndarray, point: TrackPoint) -> float:
    """Share of grass or gravel pixels in a strip just below *point*'s box.

    The strip is where the tyres meet the ground. Asphalt is grey (low
    saturation); grass is green, gravel pale yellow-brown. Kerbs (red/white)
    count as neither.
    """
    h, w = image.shape[:2]
    box_h = point.y2 - point.y1
    x1 = int(max(0, point.x1))
    x2 = int(min(w, point.x2))
    y1 = int(max(0, point.y2))
    y2 = int(min(h, point.y2 + max(4.0, 0.15 * box_h)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    hsv = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grass = (hue >= 30) & (hue <= 90) & (sat >= 60) & (val >= 40)
    gravel = (hue >= 10) & (hue < 30) & (sat >= 45) & (sat <= 170) & (val >= 110)
    return float(np.mean(grass | gravel))


def _overlap(person: TrackPoint, bike: TrackPoint) -> float:
    ix = min(person.x2, bike.x2) - max(person.x1, bike.x1)
    iy = min(person.y2, bike.y2) - max(person.y1, bike.y1)
    if ix <= 0 or iy <= 0:
        return 0.0
    return ix * iy / max(1.0, person.area)


def _cx(p: TrackPoint) -> float:
    return 0.5 * (p.x1 + p.x2)


def _aspect(p: TrackPoint) -> float:
    return (p.x2 - p.x1) / max(1.0, p.y2 - p.y1)


def _off_bike_times(bike: list[TrackPoint], persons_at: dict[float, list[TrackPoint]]) -> list[float]:
    mounted: list[bool] = []
    beside: list[bool] = []
    for p in bike:
        people = persons_at.get(p.t, [])
        on = any(_overlap(q, p) >= MOUNTED_OVERLAP for q in people)
        width = p.x2 - p.x1
        near = any(
            _overlap(q, p) < 0.1 and abs(_cx(q) - _cx(p)) <= 3.0 * width and q.height_frac >= 0.03
            for q in people
        )
        mounted.append(on)
        beside.append(not on and near)
    hits: list[float] = []
    for i in range(5, len(bike) - 2):
        if sum(mounted[i - 5 : i]) >= 4 and all(beside[i : i + 3]):
            if not hits or bike[i].t - hits[-1] > 5.0:
                hits.append(bike[i].t)
    return hits


def _bike_down_times(bike: list[TrackPoint]) -> list[float]:
    hits: list[float] = []
    for i in range(5, len(bike) - 8):
        before = bike[i - 5 : i]
        after = bike[i : i + 8]
        a0 = float(np.median([_aspect(p) for p in before]))
        h0 = float(np.median([p.height for p in before]))
        if all(_aspect(p) >= 1.5 * a0 for p in after):
            a1 = float(np.median([_aspect(p) for p in after]))
            h1 = float(np.median([p.height for p in after]))
            if a1 >= 1.8 * a0 and h1 <= 0.65 * h0:
                if not hits or bike[i].t - hits[-1] > 5.0:
                    hits.append(bike[i].t)
    return hits


def _offtrack_spans(points: list[TrackPoint]) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    run: list[TrackPoint] = []
    was_on_track = False
    for p in points:
        if p.offtrack >= OFFTRACK_FRAC:
            run.append(p)
            continue
        if was_on_track and len(run) >= OFFTRACK_POINTS:
            spans.append((run[0].t, run[-1].t))
        run = []
        was_on_track = was_on_track or p.offtrack < 0.2
    if was_on_track and len(run) >= OFFTRACK_POINTS:
        spans.append((run[0].t, run[-1].t))
    return spans


def score_incidents(tracks: dict[int, list[TrackPoint]], recording_id: str) -> list[Event]:
    persons_at: dict[float, list[TrackPoint]] = {}
    for points in tracks.values():
        for p in points:
            if p.cls == PERSON:
                persons_at.setdefault(p.t, []).append(p)

    events: list[Event] = []
    offtrack: list[tuple[float, float, str]] = []
    for track_id, raw in tracks.items():
        points = sorted((p for p in raw if p.height_frac >= MIN_HEIGHT_FRAC), key=lambda p: p.t)
        if len(points) < 6:
            continue
        for lo, hi in _offtrack_spans(points):
            offtrack.append((lo, hi, str(track_id)))
        if points[0].cls != MOTORCYCLE:
            continue
        for t in _off_bike_times(points, persons_at):
            events.append(
                Event("rider_down", t - 1.0, t + 1.0, 2.0, recording_id, str(track_id), {"why": "off_bike"})
            )
        for t in _bike_down_times(points):
            if any(lo - 2.0 <= t <= hi + 2.0 for lo, hi, _ in offtrack):
                events.append(
                    Event("rider_down", t - 1.0, t + 1.0, 1.5, recording_id, str(track_id), {"why": "bike_down"})
                )
    for lo, hi, track_id in offtrack:
        if any(e.t_start - 2.0 <= lo <= e.t_end + 2.0 for e in events if e.type == "rider_down"):
            continue
        events.append(Event("off_track", lo, hi, 1.0, recording_id, track_id))
    return events
