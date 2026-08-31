from pathlib import Path

import pytest

from videobrowser.decode import decode_one_frame
from videobrowser.gopro import group_recordings, iter_video_paths
from videobrowser.profiles import guess_profile_from_samples


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[1] / "inputclips").exists(),
    reason="inputclips not present",
)
def test_rotation_and_mounts_from_pixels():
    root = Path(__file__).resolve().parents[1] / "inputclips"
    recordings = group_recordings(iter_video_paths([root]))
    by_id = {r.recording_id: r for r in recordings}

    expected_total_rot = {
        "GH-0004": 180,
        "GX-1429": 180,
        "GX-1431": 180,
    }
    expected_name = {
        "GX-0009": "side",
        "GH-0004": "ground",
        "GH-0008": "cockpit",
    }

    for rec_id, rec in by_id.items():
        chapter = rec.chapters[0]
        meta = chapter.info.rotation
        frames = []
        for frac in (0.22, 0.5, 0.78):
            t = min(max(rec.duration * frac, 4.0), max(rec.duration - 2.0, 0.0))
            image = decode_one_frame(
                chapter.path,
                t,
                640,
                chapter.info.width,
                chapter.info.height,
                rotation=meta,
            )
            if image is not None:
                frames.append(image)
        if not frames:
            continue
        profile = guess_profile_from_samples(frames)
        total = (meta + profile.rotation) % 360
        if rec_id in expected_total_rot:
            assert total == expected_total_rot[rec_id], (
                f"{rec_id} rot {total} (meta={meta} visual={profile.rotation})"
            )
        if rec_id in expected_name:
            assert profile.name == expected_name[rec_id], (
                f"{rec_id} classified as {profile.name} ego={profile.ego_side}"
            )
