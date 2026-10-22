import numpy as np

from videobrowser.incidents import MOTORCYCLE, PERSON, offtrack_fraction, score_incidents
from videobrowser.riders import TrackPoint

FW, FH = 640, 360


def _box(t, cls, track_id, cx, w, h, bottom=300, offtrack=0.0):
    return TrackPoint(t, cx - w / 2, bottom - h, cx + w / 2, bottom, 0.9, cls, track_id, FW, FH, offtrack)


def _ride(n, start=0):
    """Rider on a bike, front view: tall narrow bike with the person on it."""
    bike = [_box((start + i) * 0.2, MOTORCYCLE, 1, 320, 40, 80) for i in range(n)]
    rider = [_box((start + i) * 0.2, PERSON, 2, 320, 30, 70, bottom=290) for i in range(n)]
    return bike, rider


def test_rider_separating_from_bike_is_rider_down():
    bike, rider = _ride(8)
    # Rider slides off to the side; the bike keeps its box.
    for i in range(8, 14):
        t = i * 0.2
        bike.append(_box(t, MOTORCYCLE, 1, 320, 40, 80))
        rider.append(_box(t, PERSON, 2, 320 + 20 * (i - 4), 50, 30))
    events = score_incidents({1: bike, 2: rider}, "F")
    assert [e.type for e in events] == ["rider_down"]
    assert abs(events[0].t_peak - 1.6) < 0.3


def test_leaning_bike_alone_is_not_a_crash():
    bike, rider = _ride(6)
    for i in range(6, 20):
        t = i * 0.2
        bike.append(_box(t, MOTORCYCLE, 1, 320, 90, 45))  # low and wide, like a deep lean
        rider.append(_box(t, PERSON, 2, 320, 70, 40, bottom=295))
    assert score_incidents({1: bike, 2: rider}, "F") == []


def test_bike_down_on_gravel_is_rider_down_and_gravel_alone_is_off_track():
    bike = [_box(i * 0.2, MOTORCYCLE, 1, 320, 40, 80) for i in range(6)]
    bike += [_box(i * 0.2, MOTORCYCLE, 1, 320, 90, 40, offtrack=0.8) for i in range(6, 20)]
    assert [e.type for e in score_incidents({1: bike}, "F")] == ["rider_down"]

    wide = [_box(i * 0.2, MOTORCYCLE, 1, 320, 40, 80) for i in range(6)]
    wide += [_box(i * 0.2, MOTORCYCLE, 1, 320, 40, 80, offtrack=0.9) for i in range(6, 12)]
    wide += [_box(i * 0.2, MOTORCYCLE, 1, 320, 40, 80) for i in range(12, 16)]
    assert [e.type for e in score_incidents({1: wide}, "F")] == ["off_track"]


def test_offtrack_fraction_tells_asphalt_from_grass_and_gravel():
    import cv2

    def frame(bgr):
        img = np.full((FH, FW, 3), 90, dtype=np.uint8)  # grey asphalt
        img[300:340, :] = bgr
        return img

    box = _box(0, MOTORCYCLE, 1, 320, 40, 80, bottom=300)
    assert offtrack_fraction(frame((90, 90, 90)), box) == 0.0
    green = cv2.cvtColor(np.uint8([[[60, 160, 120]]]), cv2.COLOR_HSV2BGR)[0, 0]
    gravel = cv2.cvtColor(np.uint8([[[20, 90, 190]]]), cv2.COLOR_HSV2BGR)[0, 0]
    assert offtrack_fraction(frame(green), box) > 0.9
    assert offtrack_fraction(frame(gravel), box) > 0.9
