"""The painted finish line in the picture: when the camera rides over it.

Footage without GoPro GPS is laid on the RaceBox log by scene motion
(``racebox_align``), which lands within seconds at best and sometimes half a
lap off. The finish line is painted across the track (white, full width),
and RaceBox logs when the bike crosses it, so each time the line slides out
under the camera is a time stamp on both clocks.

At track speed the line is in view for a handful of frames only, and the
bottom of the picture is motion blur, so a sparse scan steps over it. The
search therefore decodes every frame, but only in windows around where
RaceBox says the bike crossed (``PassFinder``; ``racebox_align.line_place``
picks the windows).

A line across the track runs across the picture, give or take the camera's
roll: a thin bright band, wider than most of the road, with road above and
below. Lane edges and kerbs run along the track, so they are tall in the
picture and drop out; asphalt grain is bright but short; guardrails and far
edges run nearly flat too, but in the grass. Lines that keep their place in
the picture are the bike's own (windscreen, fairing edge) and are left out.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# The ground ahead: from this share of the picture height down. At track
# speed the line blurs away before the bottom of the picture: it is last
# seen at 55-63 % of the height from a camera low on the bike.
LINE_TOP = 0.4
# Frames are searched at this width (px).
LINE_WIDTH = 640
# A line is at most this tall (px at LINE_WIDTH), motion blur included; a
# taller bright patch (wall, sky, white paint along the track) is not one.
LINE_MAX_PX = 15
# Brighter than the ground just above and below it by this much (grey).
# Lower and the grain of the asphalt streaming past forms lines too.
LINE_CONTRAST = 50
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
# A line followed from frame to frame: seen again within this long (s) and
# at most this much higher up (it comes down the picture as it nears).
TRACK_GAP_S = 0.13
TRACK_BACK = 0.02


@dataclass
class LineHit:
    """A line across the picture: how far it runs (share of the width), where
    (share of the height, at the centre) and its roll in degrees."""

    run: float
    row: float
    angle: float


@dataclass
class LinePass:
    """The camera riding over a line: *t* the recording time it leaves the
    picture, *row* where it was last seen (share of the height)."""

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


def passes_from_hits(times: list[float], hits: list[list[LineHit]], fps: float) -> list[LinePass]:
    """Every line that comes down the picture and leaves it, in time order.

    *hits* are the lines of each frame at *times*. Lines that hold their
    place STILL_LAG_S before or after are the bike's own and do not count. A
    line is followed while it comes down; it passes half a frame after it was
    last seen, once it has been seen in two frames or more.
    """
    lag = max(1, round(STILL_LAG_S * fps))
    tracks: list[list[tuple[float, LineHit]]] = []
    done: list[list[tuple[float, LineHit]]] = []
    for i, (t, frame) in enumerate(zip(times, hits)):
        others = [h for j in (i - lag, i + lag) if 0 <= j < len(hits) for h in hits[j]]
        moving = sorted((h for h in frame if not _still(h, others)), key=lambda h: h.row, reverse=True)
        open_ = [tr for tr in tracks if t - tr[-1][0] <= TRACK_GAP_S]
        done.extend(tr for tr in tracks if t - tr[-1][0] > TRACK_GAP_S)
        tracks = []
        # Lowest line first onto the lowest track it can have come from.
        for hit in moving:
            fits = [tr for tr in open_ if tr[-1][1].row <= hit.row + TRACK_BACK]
            if fits:
                tr = max(fits, key=lambda tr: tr[-1][1].row)
                open_.remove(tr)
                tr.append((t, hit))
                tracks.append(tr)
            else:
                tracks.append([(t, hit)])
        tracks.extend(open_)
    done.extend(tracks)
    out = []
    for tr in done:
        first, (t1, last) = tr[0][1], tr[-1]
        if len(tr) >= 2 and last.row - first.row >= STILL_ROW:
            out.append(LinePass(t1 + 0.5 / fps, last.row, max(h.run for _, h in tr)))
    return sorted(out, key=lambda p: p.t)


class PassFinder:
    """Line passes in stretches of a recording, every frame decoded: called
    with (t0, t1) in recording time, returns the passes in it."""

    def __init__(self, recording, max_side: int = 640, rotation: int = 0) -> None:
        self.recording = recording
        self.max_side = max_side
        self.rotation = rotation
        # Seconds of video decoded, for the log.
        self.decoded_s = 0.0

    def __call__(self, t0: float, t1: float) -> list[LinePass]:
        from videobrowser.decode import FrameDecoder

        out: list[LinePass] = []
        for chapter in self.recording.chapters:
            lo = max(t0, chapter.offset)
            hi = min(t1, chapter.offset + chapter.duration)
            if hi - lo < 0.1:
                continue
            fps = chapter.info.fps or 25.0
            times: list[float] = []
            hits: list[list[LineHit]] = []
            with FrameDecoder(
                chapter.path,
                chapter.info,
                fps=fps,
                max_side=self.max_side,
                rotation=self.rotation,
                max_seconds=hi - lo,
                start=lo - chapter.offset,
            ) as decoder:
                for frame in decoder.frames():
                    times.append(chapter.offset + frame.t)
                    hits.append(find_lines(frame.image))
            self.decoded_s += hi - lo
            out.extend(passes_from_hits(times, hits, fps))
        return out
