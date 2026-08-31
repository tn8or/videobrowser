import numpy as np

from videobrowser.events import Event
from videobrowser.motion import event_is_moving, scene_motion
from videobrowser.profiles import Rect


def test_identical_frames_are_still():
    image = np.full((360, 640, 3), 80, dtype=np.uint8)
    image[:, 400:] = 20
    assert scene_motion(image, image) < 0.5


def test_streaming_asphalt_is_moving():
    rng = np.random.default_rng(0)
    ego = Rect(0.5, 0.0, 0.5, 1.0)
    prev = np.zeros((360, 640, 3), dtype=np.uint8)
    prev[:, :320] = rng.integers(40, 160, size=(360, 320, 3), dtype=np.uint8)
    prev[:, 320:] = 25
    curr = np.zeros((360, 640, 3), dtype=np.uint8)
    curr[:, :320] = rng.integers(40, 160, size=(360, 320, 3), dtype=np.uint8)
    curr[:, 320:] = 25
    assert scene_motion(prev, curr, (ego,)) > 10


def test_one_person_walking_does_not_look_like_track():
    prev = np.full((360, 640, 3), 90, dtype=np.uint8)
    prev[:, 400:] = 20
    curr = prev.copy()
    curr[80:260, 40:140] = 200
    mag = scene_motion(prev, curr, (Rect(0.5, 0.0, 0.5, 1.0),))
    assert mag < 4.0


def test_event_dropped_when_parked():
    event = Event("nearby_rider", 2.0, 20.0, 0.9, "GX-0009")
    samples = [(t, 1.5) for t in [i * 0.2 for i in range(5, 100)]]
    assert not event_is_moving(event, samples)
    moving = [(t, 14.0) for t, _ in samples]
    assert event_is_moving(event, moving)
    assert event_is_moving(event, samples, threshold=0)
