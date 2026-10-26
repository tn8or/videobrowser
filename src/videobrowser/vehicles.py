"""Cars, vans and trucks in view: a sign of the paddock.

There are no cars on track, but the paddock is full of vans and trailers. The
scan asks YOLO for car, bus and truck boxes beside the riders, keeps them out
of the rider tracks, and stores per second the share of frames with a vehicle
in view. Events and shorts moments with a vehicle in view most of the time
are left out.

A parked car beside the track or far off in the background must not count,
so only a vehicle box of some size counts, and only when it is in view for
most of the moment: riding past a marshal's car takes a second or two.
"""

from __future__ import annotations

from videobrowser.riders import TrackPoint

CAR, BUS, TRUCK = 2, 5, 7
VEHICLE_CLASSES = frozenset({CAR, BUS, TRUCK})
# Smaller boxes (fraction of frame height) are cars in the background.
MIN_HEIGHT_FRAC = 0.08
# A moment with a vehicle in view in this share of its frames is in the paddock.
DEFAULT_MAX_SHARE = 0.4
# Scan events can be a fraction of a second; the check looks this far around.
EVENT_PAD_S = 2.0


def is_vehicle(point: TrackPoint) -> bool:
    return point.cls in VEHICLE_CLASSES


def vehicle_in_view(points: list[TrackPoint]) -> bool:
    """True when any of one frame's boxes is a vehicle near enough to count."""
    return any(is_vehicle(p) and p.height_frac >= MIN_HEIGHT_FRAC for p in points)


def per_second(samples: list[tuple[float, bool]]) -> list[tuple[float, float]]:
    """Share of frames with a vehicle in view per whole second, for storage."""
    buckets: dict[int, list[bool]] = {}
    for t, seen in samples:
        buckets.setdefault(int(t), []).append(seen)
    return [(k + 0.5, sum(v) / len(v)) for k, v in sorted(buckets.items())]


def share(samples: list[tuple[float, float]], spans: list[tuple[float, float]]) -> float | None:
    """Mean vehicle share over *spans* (``(t_start, t_end)``), None without samples there.

    *samples* are ``(t, share)``: per frame (share 0 or 1) or per second.
    """
    vals = [v for t, v in samples for lo, hi in spans if lo <= t <= hi]
    if not vals:
        return None
    return sum(vals) / len(vals)


def in_paddock(
    samples: list[tuple[float, float]],
    spans: list[tuple[float, float]],
    max_share: float = DEFAULT_MAX_SHARE,
) -> bool:
    """True when a vehicle is in view for at least *max_share* of *spans*."""
    if max_share <= 0:
        return False
    found = share(samples, spans)
    return found is not None and found >= max_share
