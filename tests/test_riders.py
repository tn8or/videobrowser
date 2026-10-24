from videobrowser.riders import TrackPoint, is_ego_track, score_tracks


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
