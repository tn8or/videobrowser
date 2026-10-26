from videobrowser.riders import TrackPoint
from videobrowser.vehicles import EVENT_PAD_S, in_paddock, per_second, share, vehicle_in_view


def _box(cls, height_frac, frame_h=360):
    h = height_frac * frame_h
    return TrackPoint(0.0, 100, 100, 100 + 2 * h, 100 + h, 0.8, cls, 1, 640, frame_h)


def test_only_a_near_vehicle_counts():
    assert vehicle_in_view([_box(0, 0.3), _box(2, 0.2)])  # person beside a car
    assert vehicle_in_view([_box(7, 0.1)])  # truck
    assert not vehicle_in_view([_box(2, 0.03)])  # car far off in the background
    assert not vehicle_in_view([_box(3, 0.4), _box(0, 0.3)])  # a rider
    assert not vehicle_in_view([])


def test_per_second_share_of_frames():
    samples = [(10.0, True), (10.2, True), (10.4, False), (10.6, False), (11.0, False)]
    assert per_second(samples) == [(10.5, 0.5), (11.5, 0.0)]


def test_a_car_seen_most_of_the_time_is_the_paddock():
    # Paddock: a van in view almost the whole time.
    paddock = [(t / 5, t % 5 != 0) for t in range(100)]
    assert in_paddock(paddock, [(5.0, 10.0)])
    # Riding past a marshal's car: in view for one second of ten.
    past = [(t / 5, 20 <= t < 25) for t in range(50)]
    assert not in_paddock(past, [(0.0, 10.0)])
    # No samples (no detector, or scanned before cars) keeps it.
    assert share([], [(0.0, 10.0)]) is None
    assert not in_paddock([], [(0.0, 10.0)])
    assert not in_paddock(paddock, [(5.0, 10.0)], max_share=0)
    assert EVENT_PAD_S > 0
