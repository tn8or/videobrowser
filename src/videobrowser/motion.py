from __future__ import annotations

import cv2
import numpy as np

from videobrowser.events import Event
from videobrowser.profiles import Rect

# Paddock / parked: block-median absdiff ~0.3–2. On-track at 5 fps: ~12–20.
DEFAULT_THRESHOLD = 4.0
# Below this (and above DEFAULT_THRESHOLD) the bike is most likely rolling
# slowly (paddock, pit lane); used only when there is no GPS speed.
TRACK_MOTION = 10.0


def scene_motion(
    prev: np.ndarray,
    curr: np.ndarray,
    ego_masks: tuple[Rect, ...] = (),
    blocks: int = 8,
) -> float:
    """Median per-block absdiff, ignoring ego-masked regions.

    A person walking through one corner of a parked shot stays low; asphalt
    streaming past on track lights up most blocks.
    """
    if prev.shape != curr.shape:
        return 0.0
    a = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) if prev.ndim == 3 else prev
    b = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) if curr.ndim == 3 else curr
    h, w = a.shape
    valid = np.ones((h, w), dtype=bool)
    for rect in ego_masks:
        x1, y1, x2, y2 = rect.to_pixels(w, h)
        valid[y1:y2, x1:x2] = False
    bh = max(1, h // blocks)
    bw = max(1, w // blocks)
    cells: list[float] = []
    for by in range(blocks):
        for bx in range(blocks):
            y1, x1 = by * bh, bx * bw
            y2, x2 = min(h, y1 + bh), min(w, x1 + bw)
            patch_valid = valid[y1:y2, x1:x2]
            if float(patch_valid.mean()) < 0.3:
                continue
            diff = cv2.absdiff(a[y1:y2, x1:x2], b[y1:y2, x1:x2])
            cells.append(float(diff[patch_valid].mean()))
    if not cells:
        return 0.0
    return float(np.median(cells))


def event_is_moving(
    event: Event,
    samples: list[tuple[float, float]],
    threshold: float = DEFAULT_THRESHOLD,
) -> bool:
    if threshold <= 0:
        return True
    if not samples:
        return True
    vals = [mag for t, mag in samples if event.t_start - 0.15 <= t <= event.t_end + 0.15]
    if len(vals) < 2:
        nearest = min(samples, key=lambda item: abs(item[0] - event.t_peak))
        return nearest[1] >= threshold
    return float(np.median(vals)) >= threshold


def per_second(samples: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Median motion per whole second, for storage."""
    buckets: dict[int, list[float]] = {}
    for t, mag in samples:
        buckets.setdefault(int(t), []).append(mag)
    return [(k + 0.5, float(np.median(v))) for k, v in sorted(buckets.items())]


# Fewest tracked points for one frame pair to say anything about facing.
_MIN_FLOW_POINTS = 12
# Frame pairs that must vote, and the share that must agree, for a facing.
MIN_FACING_VOTES = 20
FACING_MAJORITY = 0.7


def flow_divergence(
    prev: np.ndarray,
    curr: np.ndarray,
    ego_masks: tuple[Rect, ...] = (),
) -> float | None:
    """Median outward flow of tracked points, relative to their distance from
    the frame centre: positive when the scene expands (camera moving the way
    it looks), negative when it contracts (camera looking back).

    The common shift (pan, tilt, a vanishing point off centre) is taken out
    first and roll turns points around the centre, so neither changes the
    sign; the mount's rotation does not matter either. None when
    too few points track both ways.
    """
    if prev.shape != curr.shape:
        return None
    a = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) if prev.ndim == 3 else prev
    b = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) if curr.ndim == 3 else curr
    h, w = a.shape
    mask = np.full((h, w), 255, dtype=np.uint8)
    for rect in ego_masks:
        x1, y1, x2, y2 = rect.to_pixels(w, h)
        mask[y1:y2, x1:x2] = 0
    p0 = cv2.goodFeaturesToTrack(a, maxCorners=300, qualityLevel=0.01, minDistance=7, mask=mask)
    if p0 is None or len(p0) < _MIN_FLOW_POINTS:
        return None
    lk = dict(winSize=(21, 21), maxLevel=3)
    p1, ok1, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, **lk)
    back, ok2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, **lk)
    # Forward-backward check drops points lost in motion blur.
    good = (ok1.ravel() == 1) & (ok2.ravel() == 1) & (np.linalg.norm(back - p0, axis=2).ravel() < 1.0)
    start, end = p0.reshape(-1, 2)[good], p1.reshape(-1, 2)[good]
    radial = start - np.array([w / 2.0, h / 2.0])
    dist2 = (radial**2).sum(axis=1)
    keep = dist2 > (0.05 * min(w, h)) ** 2
    if keep.sum() < _MIN_FLOW_POINTS:
        return None
    # Without the common shift, pans, tilts and a vanishing point away from
    # the centre leave only the expansion (or contraction).
    flow = end - start
    flow = flow - np.median(flow, axis=0)
    outward = (flow[keep] * radial[keep]).sum(axis=1) / dist2[keep]
    return float(np.median(outward))


def facing_from_flow(divergences: list[float], min_abs: float = 0.002) -> str | None:
    """"front" or "rear" when the frame pairs taken on the move clearly agree."""
    front = sum(1 for d in divergences if d > min_abs)
    rear = sum(1 for d in divergences if d < -min_abs)
    if front + rear < MIN_FACING_VOTES:
        return None
    if front >= FACING_MAJORITY * (front + rear):
        return "front"
    if rear >= FACING_MAJORITY * (front + rear):
        return "rear"
    return None
