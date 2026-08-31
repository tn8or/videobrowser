from videobrowser.riders import TrackPoint, score_tracks


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
