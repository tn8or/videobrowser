import numpy as np

from videobrowser.riders import TrackPoint
from videobrowser.signals import blinking_seconds, red_lamp

FW, FH = 640, 360


def _frame(lamp_at=None):
    image = np.full((FH, FW, 3), (110, 110, 110), dtype=np.uint8)
    image[:120] = (235, 200, 160)  # sky
    if lamp_at is not None:
        x, y = lamp_at
        image[y : y + 14, x : x + 12] = (20, 40, 245)  # red lamp ...
        image[y + 5 : y + 9, x + 4 : x + 8] = (210, 240, 255)  # ... glaring in the middle
    return image


def test_a_lit_red_lamp_above_the_track_is_found():
    assert red_lamp(_frame((500, 60)))
    assert not red_lamp(_frame())


def test_red_low_in_the_frame_or_on_a_rider_is_no_lamp():
    assert not red_lamp(_frame((300, 300)))  # the track surface, kerbs
    rider = TrackPoint(0.0, 480, 40, 540, 200, 0.9, 3, 1, FW, FH)
    assert not red_lamp(_frame((500, 60)), [rider])


def test_only_a_blinking_lamp_counts():
    blink = [(i * 0.2, (i // 3) % 2 == 0) for i in range(30)]  # 0.6 s on, 0.6 s off
    assert blinking_seconds(blink) == set(range(0, 6))
    steady = [(i * 0.2, 5 <= i < 25) for i in range(30)]  # a red sign drifting past
    assert blinking_seconds(steady) == set()
