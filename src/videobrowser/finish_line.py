"""The painted finish line in the picture: when the camera rides over it.

Footage without GoPro GPS is laid on the RaceBox log by scene motion
(``racebox_align``), which lands within seconds at best and sometimes half a
lap off. The finish line is painted across the track (white, full width),
and RaceBox logs when the bike crosses it, so each time the line slides out
under the camera is a time stamp on both clocks. A handful of them, with the
lap times between them as a fingerprint, places a recording to a frame.

The scan looks for the line in every frame it decodes anyway (a few per
second, ``LineWatch``); that finds where it is, not when exactly. Each find
is then decoded again at the full frame rate (``refine_passes``) and the
pass is the frame it is closest in, just before it slides out of view. Lines
that keep their place in the picture are the bike's own (windscreen, fairing
edge) and are left out.

A line across the track runs across the picture, give or take the camera's
roll: a thin bright band, wider than most of the road, with road above
and below. Lane edges and kerbs run along the track, so they are tall in the
picture and drop out; asphalt grain is bright but short; guardrails and far
edges run nearly flat too, but high up or in the grass.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# The ground close ahead: from this share of the picture height down. Higher
# up, guardrails and far edges run nearly flat across the picture too; the
# line is found on its last few metres, which is when it passes anyway.
LINE_TOP = 0.55
# Frames are searched at this width (px).
LINE_WIDTH = 640
# A line is at most this tall (px at LINE_WIDTH), motion blur included; a
# taller bright patch (wall, sky, white paint along the track) is not one.
LINE_MAX_PX = 15
# Brighter than the ground just above and below it by this much (grey).
LINE_CONTRAST = 30
# Roll of the camera (degrees) the line may show at.
LINE_MAX_ANGLE = 20.0
LINE_ANGLE_STEP = 2.0
# One unbroken run (gaps up to LINE_GAP_PX) over this share of the width.
LINE_MIN_RUN = 0.3
LINE_GAP_PX = 7
# The line is painted on the road, not in the grass: a guardrail seen far off
# runs nearly flat across the picture too, but with grass around it. Grass
# is this hue range (OpenCV, 0-180) at this saturation or more.
GRASS_HUE = (30, 90)
GRASS_SAT = 60
# Nor on the bike: a glove, the dash or a fairing edge is dark on one side.
# Road is at least this bright (HSV value), and the line has road both above
# and below it.
ROAD_MIN_V = 50

# Lines found per frame, this far apart (share of the height).
MAX_LINES = 4
LINE_APART = 0.03
# A line that holds its row (share of the height) to within this over
# STILL_LAG_S is fixed to the camera: a windscreen or fairing edge.
STILL_ROW = 0.015
STILL_LAG_S = 0.2

# Sightings this close (s) are one pass.
PASS_GAP_S = 0.6
# The refine decode runs from this far before the last sighting to this far
# after it (s): the line leaves the picture before the next scan frame.
REFINE_BEFORE_S = 0.2
REFINE_AFTER_S = 0.4
# Passes refined per recording, strongest sightings first; a paddock full of
# painted bays would otherwise decode for ever.
MAX_REFINES = 80


@dataclass
class LineHit:
    """A line across the picture: how far it runs (share of the width), where
    (share of the height, at the centre) and its roll in degrees."""

    run: float
    row: float
    angle: float


@dataclass
class LinePass:
    """The camera riding over a line: *t* the recording time of the frame it
    is closest in, *row* where it was then (share of the height)."""

    t: float
    row: float
    run: float


def _longest_run(mask: np.ndarray, gap: int) -> int:
    """Longest run of True with gaps of up to *gap* closed."""
    if not mask.any():
        return 0
    idx = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return int((ends - starts + 1).max())


def _within(mask: np.ndarray, px: int, above: bool) -> np.ndarray:
    """True where *mask* holds somewhere in the *px* rows above (or below)."""
    c = np.cumsum(np.r_[np.zeros((1, mask.shape[1]), np.int32), mask.astype(np.int32)], axis=0)
    n = mask.shape[0]
    y = np.arange(n)
    if above:
        lo, hi = np.clip(y - px, 0, n), y
    else:
        lo, hi = np.clip(y + 1, 0, n), np.clip(y + 1 + px, 0, n)
    return (c[hi] - c[lo]) > 0


_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (1, LINE_MAX_PX))
_ANGLES = np.arange(-LINE_MAX_ANGLE, LINE_MAX_ANGLE + 1e-6, LINE_ANGLE_STEP)


def find_lines(image: np.ndarray) -> list[LineHit]:
    """Lines across the ground in *image*, longest first, at most MAX_LINES
    of them and LINE_APART apart."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = grey.shape
    if w != LINE_WIDTH:
        grey = cv2.resize(grey, (LINE_WIDTH, max(1, round(h * LINE_WIDTH / w))), interpolation=cv2.INTER_AREA)
        h, w = grey.shape
    top = int(h * LINE_TOP)
    ground = grey[top:]
    # Bright bands thinner than LINE_MAX_PX: what an opening with a tall
    # kernel takes away.
    bright = cv2.morphologyEx(ground, cv2.MORPH_TOPHAT, _KERNEL) >= LINE_CONTRAST
    if bright.mean() < 0.002:
        return []
    if image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if hsv.shape[1] != w:
            hsv = cv2.resize(hsv, (w, h), interpolation=cv2.INTER_NEAREST)
        hsv = hsv[top:]
        grass = (hsv[..., 0] >= GRASS_HUE[0]) & (hsv[..., 0] <= GRASS_HUE[1]) & (hsv[..., 1] >= GRASS_SAT)
        road = ~grass & (hsv[..., 2] >= ROAD_MIN_V) & ~bright
        bright = bright & _within(road, LINE_MAX_PX, above=True) & _within(road, LINE_MAX_PX, above=False)
    bright = bright.astype(np.uint8)
    gh = ground.shape[0]
    cx = w / 2.0
    found: list[LineHit] = []
    for angle in _ANGLES:
        slope = float(np.tan(np.radians(angle)))
        # Shear so a line at *angle* lies along one row; a row or two of
        # slack for the rounding.
        m = np.float32([[1, 0, 0], [-slope, 1, slope * cx]])
        flat = cv2.warpAffine(bright, m, (w, gh), flags=cv2.INTER_NEAREST, borderValue=0)
        flat = cv2.dilate(flat, np.ones((3, 1), np.uint8))
        rows = flat.sum(axis=1)
        for r in np.argsort(rows)[::-1][:MAX_LINES]:
            if rows[r] < LINE_MIN_RUN * w:
                break
            run = _longest_run(flat[r] > 0, LINE_GAP_PX) / w
            if run >= LINE_MIN_RUN:
                found.append(LineHit(run, (top + r) / h, float(angle)))
    out: list[LineHit] = []
    for hit in sorted(found, key=lambda x: x.run, reverse=True):
        if all(abs(hit.row - o.row) >= LINE_APART for o in out):
            out.append(hit)
        if len(out) == MAX_LINES:
            break
    return out


def find_line(image: np.ndarray) -> LineHit | None:
    """The longest line across the ground in *image*, None without one."""
    lines = find_lines(image)
    return lines[0] if lines else None


def _still(hit: LineHit, others: list[LineHit]) -> bool:
    """*hit* where one of *others* (a moment before or after) was: part of
    the bike or the camera mount, not the road coming closer."""
    return any(abs(hit.row - o.row) <= STILL_ROW and abs(hit.angle - o.angle) <= LINE_ANGLE_STEP for o in others)


class LineWatch:
    """Line sightings over a recording, fed the frames the scan decodes."""

    def __init__(self) -> None:
        self.hits: list[tuple[float, LineHit]] = []
        self._prev: list[LineHit] = []

    def update(self, t: float, image: np.ndarray) -> None:
        lines = find_lines(image)
        moving = [hit for hit in lines if not _still(hit, self._prev)]
        self._prev = lines
        if moving:
            self.hits.append((t, moving[0]))

    def candidates(self) -> list[float]:
        """Recording time of the last sighting of each pass, the strongest
        MAX_REFINES of them, in time order."""
        groups: list[list[tuple[float, LineHit]]] = []
        for t, hit in self.hits:
            if groups and t - groups[-1][-1][0] <= PASS_GAP_S:
                groups[-1].append((t, hit))
            else:
                groups.append([(t, hit)])
        groups.sort(key=lambda g: max(h.run for _, h in g), reverse=True)
        return sorted(g[-1][0] for g in groups[:MAX_REFINES])


def pass_from_frames(times: list[float], images: list[np.ndarray]) -> LinePass | None:
    """The pass in consecutive frames: the line coming down the picture, at
    the frame it is closest (lowest) in.

    Lines that hold their place STILL_LAG_S before or after are the bike's
    own (windscreen, fairing edge) and do not count. The line must be seen
    coming down in at least two frames.
    """
    lines = [find_lines(image) for image in images]
    moving: list[tuple[float, LineHit]] = []
    for i, (t, hits) in enumerate(zip(times, lines)):
        others = [
            hit
            for j, tj in enumerate(times)
            if j != i and abs(abs(tj - t) - STILL_LAG_S) <= 0.5 * (times[1] - times[0] if len(times) > 1 else 0.04)
            for hit in lines[j]
        ]
        moving.extend((t, hit) for hit in hits if not _still(hit, others))
    if len(moving) < 2:
        return None
    t, hit = max(moving, key=lambda m: (round(m[1].row, 2), m[0]))
    if not any(tt < t and h.row < hit.row - STILL_ROW for tt, h in moving):
        return None
    return LinePass(t, hit.row, hit.run)


def refine_passes(recording, candidates: list[float], max_side: int, rotation: int = 0) -> list[LinePass]:
    """Each candidate decoded again at the full frame rate, as passes."""
    from videobrowser.decode import decode_frames

    passes: list[LinePass] = []
    for t in candidates:
        start = max(0.0, t - REFINE_BEFORE_S)
        chapter = next(
            (c for c in recording.chapters if c.offset <= start < c.offset + c.duration),
            None,
        )
        if chapter is None:
            continue
        fps = chapter.info.fps or 25.0
        count = max(2, int(round((REFINE_BEFORE_S + REFINE_AFTER_S) * fps)))
        local = start - chapter.offset
        try:
            images = decode_frames(
                chapter.path, local, count, max_side, chapter.info.width, chapter.info.height, rotation
            )
        except Exception:
            continue
        times = [chapter.offset + local + i / fps for i in range(len(images))]
        found = pass_from_frames(times, images)
        if found is not None:
            # The line leaves the picture between this frame and the next.
            found.t += 0.5 / fps
        if found is not None and (not passes or found.t - passes[-1].t > PASS_GAP_S):
            passes.append(found)
    return passes
