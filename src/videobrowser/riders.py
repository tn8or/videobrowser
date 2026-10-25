from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from videobrowser.events import Event


@dataclass
class TrackPoint:
    t: float
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int
    track_id: int
    frame_w: int
    frame_h: int
    # Share of grass/gravel pixels just below the box (see incidents.py).
    offtrack: float = 0.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area_frac(self) -> float:
        den = max(1, self.frame_w * self.frame_h)
        return self.area / den

    @property
    def height_frac(self) -> float:
        return self.height / max(1, self.frame_h)


@dataclass
class RiderParams:
    min_box_height_frac: float = 0.05
    nearby_height_frac: float = 0.11
    nearby_area_frac: float = 0.01
    min_track_points: int = 4
    min_track_seconds: float = 0.6
    log_slope_per_s: float = 0.32
    window_points: int = 8


@dataclass
class RiderTracker:
    model_path: str
    device: str
    conf: float
    imgsz: int = 640
    classes: tuple[int, ...] = (0, 3)  # person, motorcycle
    tensorrt: bool = False
    _model: object | None = field(default=None, init=False, repr=False)
    _started: bool = field(default=False, init=False)
    engine: bool = field(default=False, init=False)

    def load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        path = self.model_path
        if self.device == "coreml" and not path.endswith((".mlpackage", ".mlmodel")):
            model = YOLO(path)
            exported = model.export(format="coreml", imgsz=self.imgsz, nms=True)
            self._model = YOLO(exported)
            return
        if self.tensorrt and str(self.device).startswith("cuda"):
            engine = self._engine(path)
            if engine is not None:
                self._model = YOLO(engine, task="detect")
                self.engine = True
                return
        self._model = YOLO(path)

    def _engine(self, path: str) -> str | None:
        """TensorRT engine for *path* at this size in FP16, built once and kept
        next to the model; None (plain PyTorch) when the build fails."""
        from ultralytics import YOLO

        if path.endswith(".engine"):
            return path
        cached = Path(path).with_name(f"{Path(path).stem}-{self.imgsz}-fp16.engine")
        if cached.exists():
            return str(cached)
        print(f"tensorrt: building {cached.name} (a few minutes, once)")
        try:
            built = YOLO(path).export(
                format="engine", imgsz=self.imgsz, device=self.device, **self._precision()
            )
            Path(built).replace(cached)
        except Exception as exc:
            print(f"tensorrt: build failed ({exc}); using PyTorch")
            return None
        return str(cached)

    def reset(self) -> None:
        self._started = False
        model = self._model
        predictor = getattr(model, "predictor", None) if model is not None else None
        if predictor is not None:
            predictor.trackers = []

    def detect(self, image) -> list[TrackPoint]:
        self.load()
        results = self._model.predict(
            image,
            conf=self.conf,
            classes=list(self.classes),
            imgsz=self.imgsz,
            device=self._ultralytics_device(),
            **self._precision(),
            verbose=False,
        )
        return self._points_from_result(results[0], image, ids=None)

    def update(self, image) -> list[TrackPoint]:
        self.load()
        persist = self._started
        self._started = True
        results = self._model.track(
            image,
            persist=persist,
            tracker="bytetrack.yaml",
            conf=self.conf,
            classes=list(self.classes),
            imgsz=self.imgsz,
            device=self._ultralytics_device(),
            **self._precision(),
            verbose=False,
        )
        result = results[0]
        ids = None
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy()
        return self._points_from_result(result, image, ids)

    def update_batch(self, images: list) -> list[list[TrackPoint]]:
        """``update`` for several consecutive frames at once.

        YOLO runs on the whole batch in one call, which removes most of the
        per-call overhead (one frame per call left the GPU mostly waiting);
        the tracker still steps through the frames in order. A TensorRT
        engine is built for batch 1, so it takes them one by one.
        """
        self.load()
        if self.engine or len(images) <= 1:
            return [self.update(image) for image in images]
        persist = self._started
        self._started = True
        results = self._model.track(
            images,
            persist=persist,
            tracker="bytetrack.yaml",
            conf=self.conf,
            classes=list(self.classes),
            imgsz=self.imgsz,
            device=self._ultralytics_device(),
            **self._precision(),
            verbose=False,
        )
        out = []
        for result, image in zip(results, images):
            ids = None
            if result.boxes is not None and result.boxes.id is not None:
                ids = result.boxes.id.cpu().numpy()
            out.append(self._points_from_result(result, image, ids))
        return out

    def _points_from_result(self, result, image, ids) -> list[TrackPoint]:
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return []
        h, w = image.shape[:2]
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        clss = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(xyxy))
        if ids is None:
            ids = np.arange(len(xyxy))
        points: list[TrackPoint] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            points.append(
                TrackPoint(
                    t=0.0,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    conf=float(confs[i]),
                    cls=int(clss[i]),
                    track_id=int(ids[i]),
                    frame_w=w,
                    frame_h=h,
                )
            )
        return points

    def _precision(self) -> dict:
        """FP16 on NVIDIA, which roughly halves the time per frame.

        Newer ultralytics takes ``quantize=16`` and warns on every frame for
        the older ``half=True``.
        """
        if self.engine or not str(self.device).startswith("cuda"):
            return {}
        try:
            from ultralytics.cfg import DEFAULT_CFG_DICT
        except ImportError:
            DEFAULT_CFG_DICT = {}
        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16}
        return {"half": True}

    def _ultralytics_device(self) -> str | int:
        if self.device == "coreml":
            return "cpu"
        if self.device == "mps":
            return "mps"
        if self.device == "cpu":
            return "cpu"
        return self.device


def _log_slope(points: list[TrackPoint]) -> float | None:
    if len(points) < 3:
        return None
    t = np.array([p.t for p in points], dtype=np.float64)
    if t[-1] - t[0] < 0.25:
        return None
    area = np.array([max(p.area, 1.0) for p in points], dtype=np.float64)
    slope = np.polyfit(t - t[0], np.log(area), 1)[0]
    return float(slope)


def _runs(mask: list[bool], points: list[TrackPoint]) -> list[list[TrackPoint]]:
    runs: list[list[TrackPoint]] = []
    current: list[TrackPoint] = []
    for flag, point in zip(mask, points):
        if flag:
            current.append(point)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


# A box this close to the left, right or bottom edge (fraction of the frame)
# touches it.
_EDGE_FRAC = 0.02
# Parts of the camera's own bike (exhaust, fairing, tyre) that YOLO takes for
# a motorcycle: a box on the frame edge that hardly moves for this long.
EGO_MIN_SECONDS = 2.0
EGO_EDGE_SHARE = 0.6
EGO_MAX_DRIFT = 0.04
EGO_MAX_SIZE_CHANGE = 0.1


def is_ego_track(points: list[TrackPoint]) -> bool:
    """True for a track that is part of the camera's own bike, not a rider:
    it sits on the left, right or bottom edge and barely moves or changes
    size in the frame."""
    if len(points) < 5 or points[-1].t - points[0].t < EGO_MIN_SECONDS:
        return False

    def on_edge(p: TrackPoint) -> bool:
        w, h = max(1, p.frame_w), max(1, p.frame_h)
        return p.x1 <= _EDGE_FRAC * w or p.x2 >= (1 - _EDGE_FRAC) * w or p.y2 >= (1 - _EDGE_FRAC) * h

    if sum(on_edge(p) for p in points) < EGO_EDGE_SHARE * len(points):
        return False
    cx = [0.5 * (p.x1 + p.x2) / max(1, p.frame_w) for p in points]
    cy = [0.5 * (p.y1 + p.y2) / max(1, p.frame_h) for p in points]
    heights = np.array([p.height for p in points], dtype=np.float64)
    # A rider closing in or dropping back changes size; the bike's own parts do not.
    steady = float(np.std(heights)) < EGO_MAX_SIZE_CHANGE * max(1e-6, float(np.mean(heights)))
    return steady and float(np.std(cx)) < EGO_MAX_DRIFT and float(np.std(cy)) < EGO_MAX_DRIFT


def score_tracks(
    tracks: dict[int, list[TrackPoint]],
    recording_id: str,
    params: RiderParams | None = None,
) -> list[Event]:
    params = params or RiderParams()
    events: list[Event] = []
    for track_id, raw in tracks.items():
        if is_ego_track(raw):
            continue
        points = [p for p in raw if p.height_frac >= params.min_box_height_frac]
        if len(points) < params.min_track_points:
            continue
        span = points[-1].t - points[0].t
        if span < params.min_track_seconds:
            continue
        nearby_mask = [
            p.height_frac >= params.nearby_height_frac or p.area_frac >= params.nearby_area_frac
            for p in points
        ]
        for run in _runs(nearby_mask, points):
            if run[-1].t - run[0].t < 0.3:
                continue
            peak = max(p.height_frac for p in run)
            events.append(
                Event(
                    type="nearby_rider",
                    t_start=run[0].t,
                    t_end=run[-1].t,
                    score=float(peak),
                    recording_id=recording_id,
                    track_id=str(track_id),
                )
            )
        window = params.window_points
        for i in range(0, max(1, len(points) - window + 1)):
            chunk = points[i : i + window]
            if len(chunk) < max(4, window // 2):
                continue
            slope = _log_slope(chunk)
            if slope is None:
                continue
            if slope >= params.log_slope_per_s:
                events.append(
                    Event(
                        type="approaching_rider",
                        t_start=chunk[0].t,
                        t_end=chunk[-1].t,
                        score=float(slope),
                        recording_id=recording_id,
                        track_id=str(track_id),
                        extra={"log_slope": slope},
                    )
                )
            elif slope <= -params.log_slope_per_s:
                events.append(
                    Event(
                        type="receding_rider",
                        t_start=chunk[0].t,
                        t_end=chunk[-1].t,
                        score=float(-slope),
                        recording_id=recording_id,
                        track_id=str(track_id),
                        extra={"log_slope": slope},
                    )
                )
    return events


# --- other riders passing each other -----------------------------------------

MOTORCYCLE = 3
# Box heights (a stand-in for distance) must differ by this ratio before and
# after a swap, so two riders side by side wobbling around 1:1 do not count.
SWAP_MARGIN = 1.15
# Each order must hold for this many shared frames (5 fps: 0.6 s).
SWAP_HOLD_POINTS = 3
# The swap itself, from the last frame in the old order to the first in the
# new one, takes at most this long.
SWAP_MAX_S = 3.0
# A tracker that hands one rider's id to the other makes both boxes jump; a
# real pass moves them smoothly. Largest step between frames, as a share of
# the frame width (centre) and in log height.
SWAP_MAX_STEP_X = 0.12
SWAP_MAX_STEP_LOGH = 0.35
# Left/right order compared this long before and after the swap.
SWAP_SIDE_S = 2.0


def _is_bike(points: list[TrackPoint]) -> bool:
    return sum(p.cls == MOTORCYCLE for p in points) * 2 > len(points)


def _smooth(values: np.ndarray) -> np.ndarray:
    """Median of three, so one bad box does not flip the order."""
    if len(values) < 3:
        return values
    padded = np.concatenate(([values[0]], values, [values[-1]]))
    return np.median(np.stack([padded[:-2], padded[1:-1], padded[2:]]), axis=0)


def _smooth_steps(points: list[TrackPoint]) -> bool:
    for a, b in zip(points, points[1:]):
        cx_a = 0.5 * (a.x1 + a.x2) / max(1, a.frame_w)
        cx_b = 0.5 * (b.x1 + b.x2) / max(1, b.frame_w)
        dh = abs(np.log(max(b.height, 1.0)) - np.log(max(a.height, 1.0)))
        if abs(cx_b - cx_a) > SWAP_MAX_STEP_X or dh > SWAP_MAX_STEP_LOGH:
            return False
    return True


def score_passes(
    tracks: dict[int, list[TrackPoint]],
    recording_id: str,
    params: RiderParams | None = None,
) -> list[Event]:
    """``rider_pass``: two other riders in view swap places, one passing the other.

    Box height stands in for distance, so the order that swaps is who is
    nearer the camera. That holds whichever way the camera faces and does not
    change when the bike leans; the left/right order does (a lean rotates the
    picture), so a left/right swap at the same time only adds to the score.
    Only motorcycle boxes count: a rider's person box rides on their bike box
    and would pair up with it.
    """
    params = params or RiderParams()
    keep = SWAP_HOLD_POINTS
    margin = float(np.log(SWAP_MARGIN))
    bikes: dict[int, dict[float, TrackPoint]] = {}
    for track_id, raw in tracks.items():
        if is_ego_track(raw) or not _is_bike(raw):
            continue
        points = [p for p in raw if p.height_frac >= params.min_box_height_frac]
        if len(points) >= 2 * keep:
            bikes[track_id] = {round(p.t, 3): p for p in points}

    events: list[Event] = []
    ids = sorted(bikes)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            shared = sorted(bikes[a].keys() & bikes[b].keys())
            if len(shared) < 2 * keep:
                continue
            pa = [bikes[a][t] for t in shared]
            pb = [bikes[b][t] for t in shared]
            ratio = _smooth(
                np.array([np.log(max(x.height, 1.0) / max(y.height, 1.0)) for x, y in zip(pa, pb)])
            )
            side = np.where(ratio >= margin, 1, np.where(ratio <= -margin, -1, 0))
            last_end = -1.0
            for j in range(keep, len(shared)):
                if side[j] == 0 or shared[j] <= last_end:
                    continue
                # j is the first frame of a new order; find the old one before it.
                k = j - 1
                while k >= 0 and side[k] == 0:
                    k -= 1
                if k < keep - 1 or side[k] != -side[j]:
                    continue
                if shared[j] - shared[k] > SWAP_MAX_S:
                    continue
                before = side[k - keep + 1 : k + 1]
                after = side[j : j + keep]
                if len(after) < keep or not (np.all(before == side[k]) and np.all(after == side[j])):
                    continue
                lo, hi = k - keep + 1, j + keep
                if not (_smooth_steps(pa[lo:hi]) and _smooth_steps(pb[lo:hi])):
                    continue
                # Left/right order a little before and after the swap.
                first = next(n for n, t in enumerate(shared) if t >= shared[k] - SWAP_SIDE_S)
                last = max(n for n, t in enumerate(shared) if t <= shared[j] + SWAP_SIDE_S)
                dx = [0.5 * (pa[n].x1 + pa[n].x2 - pb[n].x1 - pb[n].x2) for n in (first, last)]
                sideways = dx[0] * dx[1] < 0
                events.append(
                    Event(
                        type="rider_pass",
                        t_start=shared[k],
                        t_end=shared[j],
                        score=1.5 if sideways else 1.0,
                        recording_id=recording_id,
                        track_id=f"{a}+{b}",
                        extra={"sideways": sideways},
                    )
                )
                last_end = shared[j + keep - 1]
    return events
