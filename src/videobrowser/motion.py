from __future__ import annotations

import cv2
import numpy as np

from videobrowser.events import Event
from videobrowser.profiles import Rect

# Paddock / parked: block-median absdiff ~0.3–2. On-track at 5 fps: ~12–20.
DEFAULT_THRESHOLD = 4.0


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
