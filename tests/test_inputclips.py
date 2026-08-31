from pathlib import Path

import pytest

from videobrowser.gopro import group_recordings, iter_video_paths


def test_group_inputclips_if_present():
    root = Path(__file__).resolve().parents[1] / "inputclips"
    if not root.exists():
        pytest.skip("inputclips not present")
    paths = iter_video_paths([root])
    if not paths:
        pytest.skip("no videos in inputclips")
    recs = group_recordings(paths)
    ids = {r.recording_id for r in recs}
    assert "GH-0008" in ids
    assert "GX-0009" in ids
    assert "GX-0011" in ids
    # Filenames are not mounts; extra angles are just more recordings.
    assert len(recs) >= 3
