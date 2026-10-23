import numpy as np

from videobrowser.decode import rotate_bgr, scaled_size
from videobrowser.probe import local_date_label, normalize_rotation
from videobrowser.profiles import (
    SIDE,
    CameraMap,
    apply_ego_mask,
    guess_profile,
    guess_profile_from_samples,
    infer_rotation,
    infer_rotation_from_samples,
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
    assert infer_rotation(image) == 270


def test_infer_rotation_upside_down():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:200, :] = (90, 90, 90)
    image[200:, :] = (220, 155, 35)
    assert infer_rotation(image) == 180


def test_local_date_label_copenhagen():
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    cph = ZoneInfo("Europe/Copenhagen")
    morning = datetime(2026, 8, 1, 9, 15, tzinfo=cph)
    assert local_date_label(morning.astimezone(timezone.utc).timestamp()) == "2026-08-01"
    assert local_date_label(None) == "unknown"


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


def _sky_right() -> np.ndarray:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:, :400] = (90, 90, 90)
    image[:, 400:] = (220, 155, 35)
    return image


def test_lone_sideways_frame_does_not_rotate_recording():
    # Frames without sky abstain and classify differently; before, the single
    # "sky on the right" frame won the name tie and turned the clip 270.
    side = np.full((360, 640, 3), 130, dtype=np.uint8)
    side[:, 460:] = 25
    dash = np.full((360, 640, 3), 140, dtype=np.uint8)
    dash[270:, :] = 20
    profile = guess_profile_from_samples([_sky_right(), side, dash])
    assert profile.rotation == 0
    assert infer_rotation_from_samples([_sky_right(), side, dash, side, dash]) == 0


def test_sideways_mount_needs_majority():
    flat = np.full((360, 640, 3), 110, dtype=np.uint8)
    assert infer_rotation_from_samples([_sky_right()] * 3 + [flat] * 2) == 270
    assert infer_rotation_from_samples([_sky_right()] * 2 + [flat] * 3) == 0


def test_upside_down_wins_over_abstaining_frames():
    upside = np.zeros((360, 640, 3), dtype=np.uint8)
    upside[:200, :] = (90, 90, 90)
    upside[200:, :] = (220, 155, 35)
    flat = np.full((360, 640, 3), 110, dtype=np.uint8)
    assert infer_rotation_from_samples([upside, flat, flat]) == 180


def _tilted(sky_below: bool, lean_deg: float) -> np.ndarray:
    """Track and sky split by a horizon tilted *lean_deg* off level."""
    h, w = 360, 640
    ys, xs = np.mgrid[0:h, 0:w]
    slope = np.tan(np.radians(lean_deg))
    below = (ys - h / 2) > slope * (xs - w / 2)
    image = np.full((h, w, 3), 90, dtype=np.uint8)
    image[below if sky_below else ~below] = (220, 155, 35)
    return image


def test_upside_down_frames_leaned_into_corners():
    # Rear camera mounted upside down: in corners the horizon sits ~40 degrees
    # off level, which edge bands read as "sky on the right" and kept at 0.
    corner_left, corner_right = _tilted(True, 40), _tilted(True, -40)
    flat = np.full((360, 640, 3), 110, dtype=np.uint8)
    assert infer_rotation_from_samples([corner_left, corner_right, flat]) == 180
    assert infer_rotation_from_samples([corner_left, _tilted(True, 10), flat, flat, flat]) == 180


def test_upright_frames_leaned_into_corners_stay_upright():
    frames = [_tilted(False, 40), _tilted(False, -40), _tilted(False, 15)]
    assert infer_rotation_from_samples(frames) == 0


def test_glare_patch_does_not_turn_upright_frame():
    image = _tilted(False, 0)
    image[200:280, 560:] = (240, 240, 240)  # white fairing / bridge on the right
    assert infer_rotation(image) == 0
