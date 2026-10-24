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
