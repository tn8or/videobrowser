import numpy as np

from videobrowser.decode import rotate_bgr, scaled_size
from videobrowser.probe import normalize_rotation
from videobrowser.profiles import (
    SIDE,
    CameraMap,
    apply_ego_mask,
    guess_profile,
    guess_profile_from_samples,
    infer_rotation,
    resolve_profile,
)


def test_scaled_size_16_9():
    assert scaled_size(3840, 2160, 640) == (640, 360)


def test_scaled_size_rotated():
    w, h = scaled_size(3840, 2160, 640, rotation=90)
    assert w == 360
    assert h == 640


def test_guess_cockpit():
    image = np.full((360, 640, 3), 140, dtype=np.uint8)
    image[280:, :] = 20
    image[:80, :] = (210, 150, 40)
    assert guess_profile(image) == "cockpit"


def test_guess_cockpit_despite_dark_pillar():
    image = np.full((360, 640, 3), 140, dtype=np.uint8)
    image[270:, :] = 25
    image[:, 520:] = 40
    image[:90, :400] = (210, 150, 40)
    assert guess_profile(image) == "cockpit"


def test_guess_side():
    image = np.full((360, 640, 3), 130, dtype=np.uint8)
    image[:, 460:] = 25
    assert guess_profile(image) == "side"


def test_guess_rear():
    image = np.full((360, 640, 3), 130, dtype=np.uint8)
    image[:, :180] = 50
    image[:90, 200:] = (220, 160, 40)
    assert guess_profile(image) == "rear"


def test_guess_ground():
    image = np.full((360, 640, 3), 110, dtype=np.uint8)
    image[:100, :] = (220, 155, 35)
    assert guess_profile(image) == "ground"


def test_infer_rotation_sky_on_right():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:, :400] = (90, 90, 90)
    image[:, 400:] = (220, 155, 35)
    image[:90, :] = (90, 90, 90)  # stored top quarter is track, not sky
    assert infer_rotation(image) == 270


def test_infer_rotation_upside_down():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:200, :] = (90, 90, 90)
    image[200:, :] = (220, 155, 35)
    assert infer_rotation(image) == 180


def test_normalize_display_matrix():
    assert normalize_rotation(-180) == 180
    assert normalize_rotation(90) == 90
    assert normalize_rotation(-90) == 270


def test_classify_after_180_metadata():
    stored = np.full((360, 640, 3), 140, dtype=np.uint8)
    stored[:80, :] = 20
    stored[280:, :] = (210, 150, 40)
    assert infer_rotation(stored) == 180
    assert guess_profile(stored) == "cockpit"
    assert guess_profile(rotate_bgr(stored, 180)) == "cockpit"
    assert guess_profile(stored, metadata_rotation=180) == "cockpit"


def test_auto_uses_pixels_not_recording_id():
    image = np.full((360, 640, 3), 130, dtype=np.uint8)
    image[:, 460:] = 25
    profile = resolve_profile(
        "auto",
        CameraMap({}, {}),
        "GH-0008",
        "GH020008",
        0,
        image,
    )
    assert profile.name == "side"


def test_explicit_map_still_overrides():
    cmap = CameraMap(recordings={"GX-0009": "side"}, stems={})
    profile = resolve_profile("auto", cmap, "GX-0009", "GX010009", 0, None)
    assert profile.name == "side"
    image = np.full((360, 640, 3), 100, dtype=np.uint8)
    masked = apply_ego_mask(image, SIDE)
    assert masked[:, 500:].sum() == 0
    assert masked[:, 50:100].sum() > 0


def test_vote_from_samples():
    side = np.full((360, 640, 3), 130, dtype=np.uint8)
    side[:, 460:] = 25
    profile = guess_profile_from_samples([side, side, side])
    assert profile.name == "side"
    assert profile.ego_side == "right"
    assert "puck" in profile.pipelines
