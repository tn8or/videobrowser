from __future__ import annotations

from dataclasses import dataclass, field

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
    _model: object | None = field(default=None, init=False, repr=False)
    _started: bool = field(default=False, init=False)

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
        self._model = YOLO(path)

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


def score_tracks(
    tracks: dict[int, list[TrackPoint]],
    recording_id: str,
    params: RiderParams | None = None,
) -> list[Event]:
    params = params or RiderParams()
    events: list[Event] = []
    for track_id, raw in tracks.items():
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
