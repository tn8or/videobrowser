from videobrowser import learn
from videobrowser.shorts import _motion_features, _rider_features
from videobrowser.telemetry import LeanSample, SpeedSample, TurnSample
from videobrowser.themes import Moment, Segment


class Day:
    def __init__(self, speeds=(), turns=(), leans=(), action=()):
        self.gps = {"A": (list(speeds), list(turns))}
        self._leans = {"A": list(leans)}
        self.action = {"A": list(action)}
        self.clocks = {}


def _m():
    return Moment("action", 0, 10, 1.0, [Segment("A", 0.0, 10.0)])


def test_motion_features_braking_corner_and_lean():
    # 150 km/h down to 60 km/h over 3 s (about 0.85 g), then a corner.
    speeds = [SpeedSample(t, v) for t, v in [(0, 150), (1, 120), (2, 90), (3, 60), (4, 60), (5, 65), (20, 200)]]
    turns = [TurnSample(t, r) for t, r in [(1, 2.0), (4, -30.0), (5, 25.0), (6, 5.0)]]
    leans = [LeanSample(t, a) for t, a in [(4, -48.0), (5, 41.0), (30, 60.0)]]
    out = _motion_features(_m(), Day(speeds, turns, leans))
    assert out["speed_max"] == 150 and out["speed_min"] == 60 and out["speed_change"] == -85
    assert 0.8 < out["brake_g"] < 0.9 and out["accel_g"] > 0
    assert out["turn_rate_max"] == 30 and out["cornering_share"] == 0.5
    assert out["lean_max"] == 48  # the 60° sample is outside the moment
    assert _motion_features(_m(), Day()) == {}


def test_rider_features_share_group_and_closing():
    # (t, riders, kerb, nearest height, red lamp): a rider growing in the frame.
    rows = [(t + 0.5, 2 if t > 6 else 1 if t > 2 else 0, 0.0, 0.02 * t if t > 2 else 0.0, 0) for t in range(10)]
    out = _rider_features(_m(), Day(action=rows))
    assert out["rider_share"] == 0.7 and out["group_share"] == 0.3
    assert abs(out["closing"] - 0.02) < 1e-6
    assert _rider_features(_m(), Day()) == {}


def test_new_features_reach_the_model_with_plain_names():
    vec = learn.vectorize({"brake_g": 0.9, "lean_max": 50.0, "closing": 0.01})
    assert vec["brake_g"] == 0.9 and vec["lean_max"] == 50.0 and vec["missing:group_share"] == 1
    assert learn._plain("lean_max") == "deepest lean" and learn._plain("missing:brake_g") == "no hard braking stored"
