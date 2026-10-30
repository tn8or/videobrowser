"""A bridge over the track in the picture: when the camera rides under it.

The painted finish line is in view for a few frames only, the bike's shadow
can hide it, and other lines across the track look just like it. A bridge
over the track (Knutstorp's "Porsche Center Malmö") is large and overhead:
every mount sees it, nothing else on the lap looks like it, and a modest
frame rate finds it. RaceBox gives the time the bike is under it
(``racebox_align``: distance along the track from the start line), so each
pass is a time stamp on both clocks.

Seen from the bike, the bridge's underside is a long, nearly flat edge with
the bridge dark above and the sky bright below. It rises up the picture as
the bike nears and leaves over the top edge. Road features come down the
picture instead, and the bike's own edges hold their place.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from videobrowser.finish_line import PassFinder, _longest_run

# Frames are searched at this width (px) ...
BRIDGE_WIDTH = 640
# ... above this share of the height (the bridge is over the horizon).
BRIDGE_BOTTOM = 0.6
# Darker above than below by this much (grey, over two rows).
BRIDGE_CONTRAST = 40
# One unbroken run (gaps up to BRIDGE_GAP_PX) over this share of the width;
# a side mount sees the bridge past the bike, over a third of the width.
BRIDGE_MIN_RUN = 0.2
BRIDGE_GAP_PX = 7
# Roll of the camera (degrees) the edge may show at, and the rows of slack
# for an underside that bows (wide lens). Under the bridge the bike is
# upright; a fence or pit wall runs off to one side at a steeper angle. A
# side mount sees a bridge from off its centre line, the far end lower.
BRIDGE_MAX_ANGLE = 12.0
BRIDGE_SIDE_ANGLE = 20.0
BRIDGE_ANGLE_STEP = 3.0
BRIDGE_BOW_PX = 5
# Edges found per frame, this far apart (share of the height).
MAX_EDGES = 4
EDGE_APART = 0.03

# Searched at this frame rate: the underside crosses the top third of the
# picture in a quarter of a second at 170 km/h.
BRIDGE_FPS = 12.5
# An edge followed from frame to frame: seen again within this long (s) and
# at most this much lower (it rises as the bridge nears) ...
TRACK_GAP_S = 0.3
TRACK_BACK = 0.02
# ... is the bridge once it has risen this much (share of the height) in
# its last BRIDGE_RISE_S (at track speed; awnings and roofs in the paddock
# rise slowly) and was last seen this near the top.
BRIDGE_RISE = 0.1
BRIDGE_RISE_S = 0.4
BRIDGE_EXIT_ROW = 0.15
# An edge that holds its row (share of the height) to within this over
# STILL_LAG_S is fixed to the camera: a windscreen or fairing edge.
STILL_ROW = 0.015
STILL_LAG_S = 0.2
# Sky: bright, not reddish and smooth (HSV value, grey gradient). Its share
# of the picture is measured at this many moments spread over a recording.
SKY_MIN_V = 110
SKY_MAX_GRADIENT = 40
SKY_SAMPLES = 9

# Passes this close (s) are one bridge, timed by the first: its edges leave
# one after another, and a fence or wall just past it leaves later.
BRIDGE_SAME_S = 0.5


@dataclass
class EdgeHit:
    """An edge across the top of the picture, dark above and bright below:
    how far it runs (share of the width), where (share of the height, at the
    centre) and its roll in degrees."""

    run: float
    row: float
    angle: float


def find_edges(image: np.ndarray, max_angle: float = BRIDGE_MAX_ANGLE) -> list[EdgeHit]:
    """Edges across the top of *image* with dark above, at most *max_angle*
    off level, longest first, at most MAX_EDGES of them and EDGE_APART
    apart."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = grey.shape
    if w != BRIDGE_WIDTH:
        grey = cv2.resize(grey, (BRIDGE_WIDTH, max(1, round(h * BRIDGE_WIDTH / w))), interpolation=cv2.INTER_AREA)
        h, w = grey.shape
    g = cv2.GaussianBlur(grey, (0, 0), 1.0).astype(np.int16)
    bottom = int(h * BRIDGE_BOTTOM)
    edge = np.zeros((bottom, w), np.uint8)
    edge[1:] = (g[2 : bottom + 1] - g[: bottom - 1]) >= BRIDGE_CONTRAST
    if edge.mean() < 0.001:
        return []
    cx = w / 2.0
    found: list[EdgeHit] = []
    for angle in np.arange(-max_angle, max_angle + 1e-6, BRIDGE_ANGLE_STEP):
        slope = float(np.tan(np.radians(angle)))
        m = np.float32([[1, 0, 0], [-slope, 1, slope * cx]])
        flat = cv2.warpAffine(edge, m, (w, bottom), flags=cv2.INTER_NEAREST, borderValue=0)
        flat = cv2.dilate(flat, np.ones((BRIDGE_BOW_PX, 1), np.uint8))
        rows = flat.sum(axis=1)
        for r in np.argsort(rows)[::-1][:MAX_EDGES]:
            if rows[r] < BRIDGE_MIN_RUN * w:
                break
            run = _longest_run(flat[r] > 0, BRIDGE_GAP_PX) / w
            if run >= BRIDGE_MIN_RUN:
                found.append(EdgeHit(run, r / h, float(angle)))
    out: list[EdgeHit] = []
    for hit in sorted(found, key=lambda x: x.run, reverse=True):
        if all(abs(hit.row - o.row) >= EDGE_APART for o in out):
            out.append(hit)
        if len(out) == MAX_EDGES:
            break
    return out


def passes_from_edges(times: list[float], hits: list[list[EdgeHit]], fps: float) -> list[float]:
    """Times the bridge leaves the top of the picture, in order.

    *hits* are the edges of each frame at *times*. Edges that hold their
    place STILL_LAG_S before and after are the bike's own and do not count.
    An edge is followed while it rises; it is the bridge once it has risen
    BRIDGE_RISE in its last BRIDGE_RISE_S and was last seen near the top, and
    it passes half a frame after that.
    """
    lag = max(1, round(STILL_LAG_S * fps))
    tracks: list[list[tuple[float, EdgeHit]]] = []
    done: list[list[tuple[float, EdgeHit]]] = []

    def still(hit: EdgeHit, around: list[list[EdgeHit]]) -> bool:
        # There both before and after: a moving edge passes a row once.
        return bool(around) and all(
            any(abs(hit.row - o.row) <= STILL_ROW and abs(hit.angle - o.angle) <= BRIDGE_ANGLE_STEP for o in others)
            for others in around
        )

    def rose(tr: list[tuple[float, EdgeHit]]) -> float:
        t1, last = tr[-1]
        return max(h.row for t, h in tr if t >= t1 - BRIDGE_RISE_S) - last.row

    def ended(tr: list[tuple[float, EdgeHit]], t: float) -> bool:
        # Lost for long, or gone a frame once at the top: it left.
        last_t, last = tr[-1]
        return t - last_t > TRACK_GAP_S or (last.row <= BRIDGE_EXIT_ROW and t - last_t > 1.5 / fps)

    for i, (t, frame) in enumerate(zip(times, hits)):
        around = [hits[j] for j in (i - lag, i + lag) if 0 <= j < len(hits)]
        frame = [h for h in frame if not still(h, around)]
        open_ = [tr for tr in tracks if not ended(tr, t)]
        done.extend(tr for tr in tracks if ended(tr, t))
        tracks = []
        # Highest edge first onto the highest track it can have come from.
        for hit in sorted(frame, key=lambda h: h.row):
            fits = [tr for tr in open_ if hit.row <= tr[-1][1].row + TRACK_BACK]
            if fits:
                tr = min(fits, key=lambda tr: tr[-1][1].row)
                open_.remove(tr)
                tr.append((t, hit))
                tracks.append(tr)
            else:
                tracks.append([(t, hit)])
        tracks.extend(open_)
    done.extend(tracks)
    found = sorted(
        tr[-1][0] + 0.5 / fps
        for tr in done
        if len(tr) >= 2
        and tr[-1][1].row <= BRIDGE_EXIT_ROW
        and rose(tr) >= BRIDGE_RISE
    )
    out: list[float] = []
    for t in found:
        if not out or t - out[-1] > BRIDGE_SAME_S:
            out.append(t)
    return out


def sky_share(image: np.ndarray) -> float:
    """Share of *image* that is sky: bright, blue or grey, and smooth."""
    if image.shape[1] != BRIDGE_WIDTH:
        h, w = image.shape[:2]
        image = cv2.resize(image, (BRIDGE_WIDTH, max(1, round(h * BRIDGE_WIDTH / w))), interpolation=cv2.INTER_AREA)
    value = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 2]
    g = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), 1.5).astype(np.float32)
    gradient = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1))
    blue_or_grey = image[..., 0].astype(np.int16) >= image[..., 2].astype(np.int16)
    return float(((value >= SKY_MIN_V) & blue_or_grey & (gradient < SKY_MAX_GRADIENT)).mean())


class BridgeFinder(PassFinder):
    """Bridge passes in stretches of a recording, at BRIDGE_FPS: called with
    (t0, t1) in recording time, returns the times (s) in it. *side*: the
    camera is mounted on the side of the bike (BRIDGE_SIDE_ANGLE)."""

    rate = BRIDGE_FPS

    def __init__(self, recording, max_side: int = 640, rotation: int = 0, side: bool = False) -> None:
        super().__init__(recording, max_side, rotation)
        self.max_angle = BRIDGE_SIDE_ANGLE if side else BRIDGE_MAX_ANGLE
        # Every pass, once the whole recording has been searched; else None.
        self.whole: list[float] | None = None

    def __call__(self, t0: float, t1: float) -> list:
        out = super().__call__(t0, t1)
        if t0 <= 0.0 and t1 >= self.recording.duration - 0.1:
            self.whole = sorted(out)
        return out

    def find(self, image: np.ndarray) -> list[EdgeHit]:
        return find_edges(image, self.max_angle)

    def sky_share(self) -> float:
        """Median share of sky in the picture over SKY_SAMPLES moments
        between 10 % and 90 % of the recording."""
        from videobrowser.decode import decode_one_frame
        from videobrowser.gopro import chapter_at

        shares = []
        for t in np.linspace(0.1, 0.9, SKY_SAMPLES) * self.recording.duration:
            chapter = chapter_at(self.recording, float(t))
            if chapter is None:
                continue
            image = decode_one_frame(
                chapter.path, float(t) - chapter.offset, self.max_side, chapter.info.width, chapter.info.height, self.rotation
            )
            if image is not None:
                shares.append(sky_share(image))
        return float(np.median(shares)) if shares else 1.0

    def passes(self, times: list[float], hits: list, fps: float) -> list[float]:
        return passes_from_edges(times, hits, fps)


# --- on track ----------------------------------------------------------------

# The pit lane does not pass under the bridges, so riding under one means a
# lap on track. On track from the first pass; off track once no bridge has
# come for this long (s): every flying lap of 2025-2026 at Knutstorp took
# under 100 s.
GATE_GAP_S = 100.0


def on_track_spans(passes: list[float], gap_s: float = GATE_GAP_S) -> list[tuple[float, float]]:
    """Stretches (recording time) on track by the bridge *passes*: from a
    pass until *gap_s* after the last one that follows within *gap_s*."""
    spans: list[tuple[float, float]] = []
    for t in sorted(passes):
        if spans and t - spans[-1][1] <= 0.0:
            spans[-1] = (spans[-1][0], t + gap_s)
        else:
            spans.append((t, t + gap_s))
    return spans
