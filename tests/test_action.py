import numpy as np

from videobrowser.action import (
    KERB_CLOSE_SHARE,
    in_view,
    kerb_events,
    kerb_share,
    per_second,
    riders_per_second,
)
from videobrowser.riders import TrackPoint

FW, FH = 640, 360
RED, WHITE, GREY = (30, 30, 200), (240, 240, 240), (110, 110, 110)


def _frame():
    return np.full((FH, FW, 3), GREY, dtype=np.uint8)


def test_red_and_white_kerb_close_by_counts():
    image = _frame()
    for i, x in enumerate(range(0, FW, 40)):
        image[260:340, x : x + 40] = RED if i % 2 else WHITE
    assert kerb_share(image) >= KERB_CLOSE_SHARE


def test_a_red_bike_or_a_white_line_alone_is_no_kerb():
    image = _frame()
    image[200:330, 250:380] = RED
    image[300:310, :] = WHITE
    image[300:310, 250:380] = RED  # where they touch, only a sliver
    assert kerb_share(image) < KERB_CLOSE_SHARE


def test_kerb_close_for_a_few_frames_is_an_event():
    samples = [(i * 0.2, 0.2 if 10 <= i < 16 else 0.0) for i in range(30)]
    samples += [(10 + i * 0.2, 0.2 if i < 2 else 0.0) for i in range(10)]  # too short
    events = kerb_events(samples, "R")
    assert [(e.type, round(e.t_start, 1)) for e in events] == [("kerb_close", 2.0)]


def _bike(t, track_id, h=60, x=300):
    return TrackPoint(t, x, 200 - h, x + 30, 200, 0.9, 3, track_id, FW, FH)


def _rider(t, track_id, h=60, x=300):
    """The person on a bike drawn by ``_bike`` with the same h and x."""
    return TrackPoint(t, x + 2, 200 - h - h // 2, x + 28, 200 - h // 3, 0.9, 0, track_id, FW, FH)


def test_riders_per_second_counts_other_bikes_in_one_frame():
    tracks = {
        1: [_bike(t / 5, 1) for t in range(10)],
        2: [_bike(t / 5, 2, x=100) for t in range(5, 10)],
        3: [_bike(t / 5, 3, h=5) for t in range(10)],  # far off
        11: [_rider(t / 5, 11) for t in range(10)],
        12: [_rider(t / 5, 12, x=100) for t in range(5, 10)],
        13: [_rider(t / 5, 13, h=5) for t in range(10)],
    }
    assert riders_per_second(tracks) == {0: 1, 1: 2}
    samples = per_second(riders_per_second(tracks), [(t / 5, 0.0) for t in range(15)])
    assert in_view(samples, 0.0, 0.6) == (1, 0.0)
    assert in_view(samples, 1.0, 1.8) == (2, 0.0)
    assert in_view(samples, 2.2, 2.8) == (0, 0.0)
    assert in_view(samples, 50.0, 60.0) is None


def test_placeholder_creation_dates_mean_no_date():
    from videobrowser.probe import _creation_utc

    assert _creation_utc({"creation_time": "1904-01-01T00:00:00.000000Z"}) is None
    assert _creation_utc({"creation_time": "1970-01-01T00:00:00.000000Z"}) is None
    assert _creation_utc({"creation_time": "2025-08-01T10:13:00.000000Z"}) is not None


def test_a_bike_without_a_person_on_it_is_no_rider():
    tracks = {
        1: [_bike(t / 5, 1) for t in range(10)],  # parked, or our own bike by hand
        2: [_bike(t / 5, 2, x=100) for t in range(10)],
        # Someone standing beside bike 2, not on it.
        12: [TrackPoint(t / 5, 140, 90, 170, 200, 0.9, 0, 12, FW, FH) for t in range(10)],
    }
    assert riders_per_second(tracks) == {}


def test_action_density_counts_near_riders_and_kerbs_most():
    from videobrowser.action import action_density, nearest_rider_per_second

    tracks = {
        1: [_bike(t / 5, 1, h=60) for t in range(10)],
        11: [_rider(t / 5, 11, h=60) for t in range(10)],
    }
    assert nearest_rider_per_second(tracks) == {0: 60 / FH, 1: 60 / FH}
    near = [(0.5, 1, 0.0, 0.2)]
    far = [(0.5, 1, 0.0, 0.04)]
    two = [(0.5, 2, 0.0, 0.2)]
    kerb = [(0.5, 0, 1.0, 0.0)]
    old_scan = [(0.5, 1, 0.0)]
    values = [action_density(s, 0.0, 1.0) for s in (near, far, two, kerb, old_scan)]
    assert values[2] > values[0] > values[3] > values[1] > 0
    assert values[4] == 0.5
    assert action_density(near, 10.0, 12.0) is None


def test_action_rows_keep_rider_height_and_read_old_rows(tmp_path):
    from videobrowser.db import ScanDB

    db = ScanDB.open(tmp_path / "scan.sqlite")
    db.save_action("A", per_second({0: 1}, [(0.2, 0.0)], {0: 0.123}, {1}))
    db.save_action("B", [(0.5, 2, 0.0)])
    assert db.action() == {
        "A": [(0.5, 1, 0.0, 0.123, 0), (1.5, 0, 0.0, 0.0, 1)],
        "B": [(0.5, 2, 0.0, 0.0, 0)],
    }
    db.close()
