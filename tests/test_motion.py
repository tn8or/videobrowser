import cv2
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


def test_flow_tells_a_camera_looking_ahead_from_one_looking_back():
    import cv2

    from videobrowser.motion import MIN_FACING_VOTES, facing_from_flow, flow_divergence

    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur((rng.random((720, 1280)) * 255).astype(np.uint8), (5, 5), 1.5)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

    def view(scale, angle=0.0, dx=0.0):
        m = cv2.getRotationMatrix2D((740.0, 280.0), angle, scale)  # vanishing point off centre
        m[0, 2] += dx
        return cv2.resize(cv2.warpAffine(base, m, (1280, 720)), (640, 360))

    ahead = flow_divergence(view(1.0), view(1.04, 3.0, 30.0))  # roll and yaw while moving on
    behind = flow_divergence(view(1.04, 3.0, 30.0), view(1.0))
    assert ahead > 0.01 and behind < -0.01
    # An ego mask over the left 28% leaves points on one side only.
    mask = (Rect(0.0, 0.0, 0.28, 1.0),)
    assert flow_divergence(view(1.0), view(1.04, 3.0, 30.0), mask) > 0.01
    assert flow_divergence(view(1.04, 3.0, 30.0), view(1.0), mask) < -0.01
    assert facing_from_flow([ahead] * MIN_FACING_VOTES) == "front"
    assert facing_from_flow([behind] * MIN_FACING_VOTES + [ahead] * 3) == "rear"
    assert facing_from_flow([ahead] * 5) is None
    assert facing_from_flow([ahead, behind] * MIN_FACING_VOTES) is None


def _mount_frames(n, mounted, seed=0):
    """Frames of a camera on a moving bike (a fixed, textured dash at the
    bottom, the rest streaming) or in the hand (everything moving)."""
    rng = np.random.default_rng(seed)
    dash = rng.integers(0, 255, (60, 320), dtype=np.uint8)
    frames = []
    for i in range(n):
        img = rng.integers(0, 255, (180, 320), dtype=np.uint8)
        if mounted:
            img[120:] = dash
        frames.append(np.dstack([img] * 3))
    return frames


def test_mount_watch_tells_a_camera_on_the_bike_from_one_in_the_hand():
    from videobrowser.motion import MountWatch

    on_bike = MountWatch()
    for i, frame in enumerate(_mount_frames(60, mounted=True)):
        on_bike.update(i * 0.2, frame)
    riding = on_bike.per_second()
    assert riding is not None and min(v for _, v in riding[1:]) == 1.0

    in_hand = MountWatch()
    for i, frame in enumerate(_mount_frames(60, mounted=False)):
        in_hand.update(i * 0.2, frame)
    assert in_hand.per_second() is None


def test_mount_watch_parked_bike_is_not_riding():
    from videobrowser.motion import MountWatch

    watch = MountWatch()
    frames = _mount_frames(60, mounted=True)
    still = frames[0]
    for i in range(90):
        # Riding for 12 s, then parked: the whole picture holds still.
        watch.update(i * 0.2, frames[i] if i < 60 else still)
    riding = dict(watch.per_second())
    assert riding[5.5] == 1.0 and riding[15.5] == 0.0


def test_mount_watch_sees_a_soft_low_contrast_scene_streaming_past():
    """Haze, shade and compression leave little texture; at 96 px none of it
    streamed, so no recording ever showed the bike riding."""
    from videobrowser.motion import MountWatch

    rng = np.random.default_rng(1)
    world = cv2.GaussianBlur(rng.normal(0, 1, (720, 4000)).astype(np.float32), (0, 0), 3)
    world = np.clip(110 + world / world.std() * 8, 0, 255).astype(np.uint8)
    dash = world[:, :640][-120:].copy()
    on_bike = MountWatch()
    for i in range(60):
        img = world[:360, i * 40 : i * 40 + 640].copy()  # the scene streams past ...
        img[240:] = dash[:120]  # ... over a fairing that holds still
        on_bike.update(i * 0.2, np.dstack([img] * 3))
    riding = on_bike.per_second()
    assert riding is not None and min(v for _, v in riding[1:]) >= 0.8
    assert "bike blocks holding still" in on_bike.summary()
