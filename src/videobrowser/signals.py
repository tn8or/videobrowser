"""Red signal lights at the marshal posts: the session is over.

When a session ends, the marshal posts flash a red lamp and everyone rides
slowly into the pit lane, so nothing after it is worth a short. A lit lamp is
a small, bright, saturated red (to orange, where it glares) spot above the
track surface, not on a rider; it blinks, so it is only taken as a signal
when it comes, goes and comes back within a couple of seconds. A red sign or
a red bike drifting through the picture does not do that.
"""

from __future__ import annotations

import cv2
import numpy as np

from videobrowser.riders import TrackPoint

# Frames are searched at this width (px), above LAMP_BOTTOM of the height.
LAMP_WIDTH = 320
LAMP_BOTTOM = 0.6
# A lamp spot: red to orange, saturated and bright, with a glaring core.
LAMP_SAT = 150
LAMP_VAL = 180
LAMP_CORE_VAL = 235
# Its size in pixels at LAMP_WIDTH, and how compact it is.
LAMP_MIN_AREA = 6
LAMP_MAX_AREA = 400
LAMP_MAX_ASPECT = 2.5
LAMP_MIN_FILL = 0.4
# Blinking: lit, dark and lit again within BLINK_WINDOW_S, each lit run at
# most BLINK_MAX_ON_S long.
BLINK_WINDOW_S = 3.0
BLINK_MAX_ON_S = 1.2


def red_lamp(image: np.ndarray, boxes: list[TrackPoint] = ()) -> bool:
    """True when a lit red lamp shows above the track, outside rider boxes."""
    h, w = image.shape[:2]
    small_h = max(8, round(LAMP_WIDTH * h / max(1, w)))
    small = cv2.resize(image, (LAMP_WIDTH, small_h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((hue <= 15) | (hue >= 170)) & (sat >= LAMP_SAT) & (val >= LAMP_VAL)
    red[int(small_h * LAMP_BOTTOM):] = False
    for b in boxes:  # box coordinates are in the analysed frame's pixels
        sx, sy = LAMP_WIDTH / max(1, b.frame_w), small_h / max(1, b.frame_h)
        x1, x2 = int(b.x1 * sx), int(np.ceil(b.x2 * sx))
        y1, y2 = int(b.y1 * sy), int(np.ceil(b.y2 * sy))
        red[max(0, y1):y2, max(0, x1):x2] = False
    if not red.any():
        return False
    core = val >= LAMP_CORE_VAL
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cv2.dilate(red.astype(np.uint8), np.ones((3, 3), np.uint8)), connectivity=8
    )
    for i in range(1, count):
        x, y, bw, bh, area = stats[i]
        if not LAMP_MIN_AREA <= area <= LAMP_MAX_AREA:
            continue
        if max(bw, bh) > LAMP_MAX_ASPECT * max(1, min(bw, bh)):
            continue
        if area < LAMP_MIN_FILL * bw * bh:
            continue
        if core[labels == i].any():
            return True
    return False


def blinking_seconds(samples: list[tuple[float, bool]]) -> set[int]:
    """Whole seconds where a red lamp blinks: lit, dark, lit again within
    ``BLINK_WINDOW_S``, the lit runs short."""
    runs: list[tuple[float, float]] = []  # (first, last) lit frame of each run
    for t, lit in sorted(samples):
        if not lit:
            continue
        if runs and t - runs[-1][1] <= 0.25:
            runs[-1] = (runs[-1][0], t)
        else:
            runs.append((t, t))
    found: set[int] = set()
    for a, b in zip(runs, runs[1:]):
        if b[1] - a[0] > BLINK_WINDOW_S:
            continue
        if a[1] - a[0] > BLINK_MAX_ON_S or b[1] - b[0] > BLINK_MAX_ON_S:
            continue
        found.update(range(int(a[0]), int(b[1]) + 1))
    return found
