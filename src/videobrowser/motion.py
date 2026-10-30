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
MIN_FACING_VOTES = 15
FACING_MAJORITY = 0.62


def flow_divergence(
    prev: np.ndarray,
    curr: np.ndarray,
    ego_masks: tuple[Rect, ...] = (),
) -> float | None:
    """How much the scene grows between two frames: positive when it expands
    (camera moving the way it looks), negative when it contracts (camera
    looking back).

    It is the divergence (trace minus 2) of an affine motion fitted to the
    tracked points with RANSAC, so a pan, tilt, roll or a vanishing point
    away from the centre does not change it, nor do ego masks that leave
    points on one side of the frame only. None when too few points track
    both ways or the fit fails.
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
    if len(start) < _MIN_FLOW_POINTS:
        return None
    model, inliers = cv2.estimateAffine2D(start, end, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if model is None or inliers is None or int(inliers.sum()) < _MIN_FLOW_POINTS:
        return None
    return float(model[0, 0] + model[1, 1] - 2.0)


# Smaller divergences (standing still, crawling through the pits) do not vote.
FACING_MIN_DIVERGENCE = 0.002


def facing_from_flow(divergences: list[float]) -> str | None:
    """"front" or "rear" when the frame pairs taken on the move clearly agree."""
    front = sum(1 for d in divergences if d > FACING_MIN_DIVERGENCE)
    rear = sum(1 for d in divergences if d < -FACING_MIN_DIVERGENCE)
    if front + rear < MIN_FACING_VOTES:
        return None
    if front >= FACING_MAJORITY * (front + rear):
        return "front"
    if rear >= FACING_MAJORITY * (front + rear):
        return "rear"
    return None


# --- camera on a moving bike ---------------------------------------------------

# Frames are compared at this width (px), lightly blurred so engine vibration
# of a pixel or two does not count as movement. Much smaller (it was 96 px)
# and the asphalt and verges lose the texture that shows them streaming past.
MOUNT_WIDTH = 320
MOUNT_GRID = 8
MOUNT_BLUR = 1.0
# Only blocks with this much texture (grey std) tell anything: sky and
# smooth asphalt look the same whether they move or not. Change is measured
# against a block's own texture, so low-contrast scenes (haze, shade, soft
# compression) stream as well as sharp ones: a block changing less than
# FIXED_REL of its texture stands still, one changing STREAM_REL or more
# streams past.
TEXTURE_STD = 4.0
FIXED_REL = 0.3
STREAM_REL = 0.6
# A frame pair is on the move when this share of the textured blocks streams ...
MOVING_SHARE = 0.3
# ... and the bike is in view when this share of its own blocks stands still.
EGO_HELD = 0.5
# Blocks that stand still in this share of the moving frames are the bike's
# own (fairing, dash, tail): at least MIN_EGO_BLOCKS of them, seen over at
# least MIN_MOVING_FRAMES.
EGO_FIXED_SHARE = 0.6
MIN_EGO_BLOCKS = 2
# The bike's own parts are in the lower part of the picture (from this
# share of its height down). Higher up, the far scenery and a sharp skyline
# hold nearly still too, also in a camera held in the hand.
EGO_TOP = 0.5
MIN_MOVING_FRAMES = 25


class MountWatch:
    """Tells a camera riding on a moving bike from one in the hand.

    Scene motion alone cannot: a camera waved about in the paddock moves the
    picture as much as the track streaming past. What it lacks is a part of
    the picture that stays put while the rest moves: a camera on the bike
    always sees the same bit of fairing, dash or tail. Those blocks are found
    per recording (still, with texture, in most moving frames); a frame then
    rides when the scene streams past and those blocks hold still.
    """

    def __init__(self) -> None:
        self._prev: np.ndarray | None = None
        self._t: list[float] = []
        self._fixed: list[np.ndarray] = []
        self._stream: list[float] = []

    def update(self, t: float, image: np.ndarray) -> None:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = grey.shape
        small = cv2.resize(grey, (MOUNT_WIDTH, max(MOUNT_GRID, round(MOUNT_WIDTH * h / max(1, w)))),
                           interpolation=cv2.INTER_AREA).astype(np.float32)
        small = cv2.GaussianBlur(small, (0, 0), MOUNT_BLUR)
        prev, self._prev = self._prev, small
        if prev is None or prev.shape != small.shape:
            return
        sh, sw = small.shape
        bh, bw = sh // MOUNT_GRID, sw // MOUNT_GRID
        crop = (slice(0, bh * MOUNT_GRID), slice(0, bw * MOUNT_GRID))
        diff = np.abs(small - prev)[crop].reshape(MOUNT_GRID, bh, MOUNT_GRID, bw).mean(axis=(1, 3))
        texture = np.maximum(
            small[crop].reshape(MOUNT_GRID, bh, MOUNT_GRID, bw).std(axis=(1, 3)),
            prev[crop].reshape(MOUNT_GRID, bh, MOUNT_GRID, bw).std(axis=(1, 3)),
        )
        textured = texture >= TEXTURE_STD
        self._t.append(t)
        self._fixed.append((textured & (diff < FIXED_REL * texture)).ravel())
        streaming = textured & (diff >= STREAM_REL * texture)
        self._stream.append(float(streaming.sum() / textured.sum()) if textured.any() else 0.0)

    def ego_blocks(self) -> np.ndarray | None:
        """Blocks of the camera's own bike, None when the recording shows none."""
        moving = np.array(self._stream) >= MOVING_SHARE
        if moving.sum() < MIN_MOVING_FRAMES:
            return None
        share = np.array(self._fixed)[moving].mean(axis=0)
        ego = share >= EGO_FIXED_SHARE
        ego[: int(MOUNT_GRID * EGO_TOP) * MOUNT_GRID] = False
        return ego if ego.sum() >= MIN_EGO_BLOCKS else None

    def summary(self) -> str:
        """How close the recording came to showing a bike on the move, for the log."""
        if not self._stream:
            return "no frames"
        stream = np.array(self._stream)
        moving = stream >= MOVING_SHARE
        held = 0
        if moving.any():
            share = np.array(self._fixed)[moving].mean(axis=0)
            share[: int(MOUNT_GRID * EGO_TOP) * MOUNT_GRID] = 0.0
            held = int((share >= EGO_FIXED_SHARE).sum())
        return (
            f"streaming median {np.median(stream):.2f}, p90 {np.percentile(stream, 90):.2f} "
            f"(moving at {MOVING_SHARE:g}); {int(moving.sum())}/{len(stream)} frames moving "
            f"(need {MIN_MOVING_FRAMES}); {held} bike blocks holding still (need {MIN_EGO_BLOCKS})"
        )

    def per_second(self) -> list[tuple[float, float]] | None:
        """Share of frames per second riding on a moving bike, None when the
        recording shows no part of a bike (helmet camera, camera in hand)."""
        ego = self.ego_blocks()
        if ego is None:
            return None
        buckets: dict[int, list[float]] = {}
        for t, fixed, stream in zip(self._t, self._fixed, self._stream):
            riding = stream >= MOVING_SHARE and fixed[ego].mean() >= EGO_HELD
            buckets.setdefault(int(t), []).append(1.0 if riding else 0.0)
        return [(k + 0.5, float(np.mean(v))) for k, v in sorted(buckets.items())]


def riding_share(samples: list[tuple[float, float]], t0: float, t1: float) -> float | None:
    """Mean riding share over [t0, t1], None without samples there."""
    vals = [v for t, v in samples if t0 - 0.5 <= t <= t1 + 0.5]
    return float(np.mean(vals)) if vals else None
