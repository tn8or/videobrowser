from videobrowser.riders import TrackPoint, is_ego_track, score_passes, score_tracks


def _pt(t: float, height_frac: float, track_id: int = 1) -> TrackPoint:
    fh, fw = 360, 640
    h = height_frac * fh
    w = h * 0.55
    return TrackPoint(
        t=t,
        x1=200,
        y1=fh - h,
        x2=200 + w,
        y2=fh,
        conf=0.8,
        cls=3,
        track_id=track_id,
        frame_w=fw,
        frame_h=fh,
    )


def test_nearby_and_approaching():
    points = [_pt(i * 0.2, 0.08 + i * 0.03) for i in range(12)]
    events = score_tracks({1: points}, "GH-0008")
    types = {e.type for e in events}
    assert "nearby_rider" in types
    assert "approaching_rider" in types


def test_receding():
    points = [_pt(i * 0.2, 0.28 - i * 0.02) for i in range(12)]
    events = score_tracks({1: points}, "GX-0011")
    assert any(e.type == "receding_rider" for e in events)


def test_own_exhaust_on_the_frame_edge_is_not_a_rider():
    exhaust = [_pt(i * 0.2, 0.5) for i in range(20)]  # same box on the bottom edge for 4 s
    assert is_ego_track(exhaust)
    assert score_tracks({1: exhaust}, "GX-0014") == []
    closing = [_pt(i * 0.2, 0.08 + i * 0.03) for i in range(20)]  # also on the edge, but growing
    assert not is_ego_track(closing)


def test_fp16_uses_quantize_when_ultralytics_has_it(monkeypatch):
    import sys
    import types

    from videobrowser.riders import RiderTracker

    cfg = types.ModuleType("ultralytics.cfg")
    monkeypatch.setitem(sys.modules, "ultralytics", types.ModuleType("ultralytics"))
    monkeypatch.setitem(sys.modules, "ultralytics.cfg", cfg)
    tracker = RiderTracker(model_path="x.pt", device="cuda", conf=0.25)
    cfg.DEFAULT_CFG_DICT = {"quantize": None, "half": False}
    assert tracker._precision() == {"quantize": 16}
    cfg.DEFAULT_CFG_DICT = {"half": False}
    assert tracker._precision() == {"half": True}
    assert RiderTracker(model_path="x.pt", device="mps", conf=0.25)._precision() == {}


def test_tensorrt_engine_is_built_once_and_cached(tmp_path, monkeypatch):
    import sys
    import types

    from videobrowser.riders import RiderTracker

    built = []

    class FakeYOLO:
        def __init__(self, path, task=None):
            self.path = path

        def export(self, **kwargs):
            built.append(kwargs)
            out = tmp_path / "yolo11n.engine"
            out.write_text("engine")
            return str(out)

    ul = types.ModuleType("ultralytics")
    ul.YOLO = FakeYOLO
    cfg = types.ModuleType("ultralytics.cfg")
    cfg.DEFAULT_CFG_DICT = {"quantize": None}
    monkeypatch.setitem(sys.modules, "ultralytics", ul)
    monkeypatch.setitem(sys.modules, "ultralytics.cfg", cfg)

    model = str(tmp_path / "yolo11n.pt")
    tracker = RiderTracker(model_path=model, device="cuda", conf=0.25, tensorrt=True)
    tracker.load()
    assert tracker.engine and tracker._model.path.endswith("yolo11n-640-fp16.engine")
    assert built == [{"format": "engine", "imgsz": 640, "device": "cuda", "quantize": 16}]
    assert tracker._precision() == {}  # the engine fixes its own precision

    again = RiderTracker(model_path=model, device="cuda", conf=0.25, tensorrt=True)
    again.load()
    assert len(built) == 1 and again.engine


def test_tensorrt_build_failure_falls_back_to_pytorch(tmp_path, monkeypatch):
    import sys
    import types

    from videobrowser.riders import RiderTracker

    class FakeYOLO:
        def __init__(self, path, task=None):
            self.path = path

        def export(self, **kwargs):
            raise RuntimeError("no tensorrt")

    ul = types.ModuleType("ultralytics")
    ul.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", ul)
    monkeypatch.setitem(sys.modules, "ultralytics.cfg", types.ModuleType("ultralytics.cfg"))
    tracker = RiderTracker(model_path=str(tmp_path / "m.pt"), device="cuda", conf=0.25, tensorrt=True)
    tracker.load()
    assert not tracker.engine and tracker._model.path.endswith("m.pt")


def test_update_batch_runs_yolo_once_per_batch():
    import numpy as np

    from videobrowser.riders import RiderTracker

    class Boxes:
        def __init__(self, n):
            self.xyxy = _Arr(np.array([[1.0, 2.0, 3.0, 4.0]] * n))
            self.conf = _Arr(np.ones(n))
            self.cls = _Arr(np.zeros(n))
            self.id = _Arr(np.arange(n) + 7) if n else None

        def __len__(self):
            return len(self.xyxy.a)

    class _Arr:
        def __init__(self, a):
            self.a = a

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    class Result:
        def __init__(self, n):
            self.boxes = Boxes(n)

    calls = []

    class Model:
        def track(self, images, **kwargs):
            calls.append((len(images), kwargs["persist"]))
            return [Result(i % 2) for i in range(len(images))]

    tracker = RiderTracker(model_path="x.pt", device="cpu", conf=0.25)
    tracker._model = Model()
    images = [np.zeros((36, 64, 3), np.uint8)] * 4
    first = tracker.update_batch(images)
    second = tracker.update_batch(images)
    assert calls == [(4, False), (4, True)]
    assert [len(p) for p in first] == [0, 1, 0, 1]
    assert first[1][0].track_id == 7 and len(second) == 4


def _bike(t: float, height_frac: float, cx: float, track_id: int) -> TrackPoint:
    fh, fw = 360, 640
    h = height_frac * fh
    w = h * 0.8
    return TrackPoint(
        t=t, x1=cx - w / 2, y1=150, x2=cx + w / 2, y2=150 + h, conf=0.8, cls=3,
        track_id=track_id, frame_w=fw, frame_h=fh,
    )


def _pass(n: int = 25, ids: tuple[int, int] = (1, 2)):
    """Rider 1 starts nearer and drops back; rider 2 comes past on the other side."""
    a = [_bike(i * 0.2, 0.20 - i * 0.004, 250 + i * 6, ids[0]) for i in range(n)]
    b = [_bike(i * 0.2, 0.12 + i * 0.004, 390 - i * 6, ids[1]) for i in range(n)]
    return {ids[0]: a, ids[1]: b}


def test_two_riders_swapping_places_is_a_pass():
    events = score_passes(_pass(), "GX-0020")
    assert len(events) == 1
    ev = events[0]
    assert ev.type == "rider_pass" and ev.track_id == "1+2"
    assert ev.extra["sideways"] and ev.score > 1.0
    assert 1.0 < ev.t_start < ev.t_end < 4.0


def test_riders_holding_their_order_are_not_a_pass():
    a = [_bike(i * 0.2, 0.20, 250, 1) for i in range(25)]
    b = [_bike(i * 0.2, 0.12 + i * 0.002, 390, 2) for i in range(25)]
    assert score_passes({1: a, 2: b}, "GX-0020") == []


def test_tracker_swapping_ids_is_not_a_pass():
    near = [(0.20, 250), (0.10, 420)]
    a, b = [], []
    for i in range(20):
        (ha, xa), (hb, xb) = near if i < 10 else near[::-1]
        a.append(_bike(i * 0.2, ha, xa, 1))
        b.append(_bike(i * 0.2, hb, xb, 2))
    assert score_passes({1: a, 2: b}, "GX-0020") == []


def test_person_boxes_do_not_pair_with_their_bike():
    tracks = _pass()
    for p in tracks[2]:
        p.cls = 0
    assert score_passes(tracks, "GX-0020") == []
