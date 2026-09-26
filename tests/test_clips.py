import subprocess
from pathlib import Path

import numpy as np
import pytest

from videobrowser.clips import _reencode_pipelines, export_clip
from videobrowser.decode import rotation_filter
from videobrowser.events import ClipWindow
from videobrowser.gopro import Chapter, Recording
from videobrowser.probe import probe

_GOPRO_180 = Path(__file__).resolve().parents[1] / "inputclips" / "2026-08-14" / "GH010013.MP4"


def _make_src(path: Path) -> None:
    # -timecode adds a tmcd "codec none" track. MP4 cannot store it, so -map 0 fails.
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x240:d=2:r=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-timecode",
            "00:00:00:00",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )


def _make_split_src(path: Path) -> None:
    """Red on top, blue on bottom — so a 180 bake is visible in the pixels."""
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=320x120:d=2:r=25",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x120:d=2:r=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-filter_complex",
            "[0:v][1:v]vstack=inputs=2[v]",
            "-map",
            "[v]",
            "-map",
            "2:a",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )


def _recording(src: Path) -> Recording:
    info = probe(src)
    return Recording(
        recording_id="GX-0001",
        prefix="GX",
        recording="0001",
        chapters=[
            Chapter(
                path=src,
                prefix="GX",
                chapter=1,
                recording="0001",
                duration=info.duration,
                offset=0.0,
                info=info,
            )
        ],
    )


def _frame_rgb(path: Path, *, noautorotate: bool, t: float = 0.3) -> np.ndarray:
    info = probe(path)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if noautorotate:
        cmd.append("-noautorotate")
    cmd += [
        "-ss",
        f"{t:.3f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(info.height, info.width, 3)


def test_export_clip_skips_timecode_track(tmp_path: Path):
    src = tmp_path / "src.mov"
    _make_src(src)
    info = probe(src)
    recording = Recording(
        recording_id="GX-0001",
        prefix="GX",
        recording="0001",
        chapters=[
            Chapter(
                path=src,
                prefix="GX",
                chapter=1,
                recording="0001",
                duration=info.duration,
                offset=0.0,
                info=info,
            )
        ],
    )
    dest = tmp_path / "out.mp4"
    export_clip(
        recording,
        ClipWindow(0.2, 1.2, "GX-0001", ["nearby_rider"], 1.0, []),
        dest,
    )
    assert dest.exists()
    assert dest.stat().st_size > 1000
    out = probe(dest)
    assert out.duration > 0.5
    assert out.codec


def test_reencode_pipelines_skips_videotoolbox_for_90_270():
    hw_90 = _reencode_pipelines(90)
    hw_270 = _reencode_pipelines(270)
    hw_180 = _reencode_pipelines(180)
    blob_90 = " ".join(" ".join(a + b) for a, b in hw_90)
    blob_270 = " ".join(" ".join(a + b) for a, b in hw_270)
    blob_180 = " ".join(" ".join(a + b) for a, b in hw_180)
    assert "videotoolbox" not in blob_90
    assert "videotoolbox" not in blob_270
    assert "videotoolbox" in blob_180
    assert "h264_nvenc" in blob_90
    assert "h264_nvenc" in blob_180
    assert rotation_filter(90) == "transpose=1"
    assert rotation_filter(270) == "transpose=2"


def test_export_clip_bakes_90_rotation(tmp_path: Path):
    src = tmp_path / "src.mp4"
    _make_src(src)
    info = probe(src)
    recording = Recording(
        recording_id="GX-0001",
        prefix="GX",
        recording="0001",
        chapters=[
            Chapter(
                path=src,
                prefix="GX",
                chapter=1,
                recording="0001",
                duration=info.duration,
                offset=0.0,
                info=info,
            )
        ],
    )
    dest = tmp_path / "out90.mp4"
    export_clip(
        recording,
        ClipWindow(0.2, 1.2, "GX-0001", ["nearby_rider"], 1.0, []),
        dest,
        rotation=90,
    )
    out = probe(dest)
    assert (out.width, out.height) == (info.height, info.width)
    assert out.rotation == 0


def test_export_clip_bakes_180_rotation(tmp_path: Path):
    src = tmp_path / "split.mp4"
    _make_split_src(src)
    dest = tmp_path / "out180.mp4"
    export_clip(
        _recording(src),
        ClipWindow(0.2, 1.2, "GX-0001", ["nearby_rider"], 1.0, []),
        dest,
        rotation=180,
    )
    out = probe(dest)
    assert (out.width, out.height) == (320, 240)
    assert out.rotation == 0
    frame = _frame_rgb(dest, noautorotate=True)
    # Blue (was bottom) must now be on top.
    assert frame[10, 160].tolist() == pytest.approx([0, 0, 255], abs=40)
    assert frame[-10, 160].tolist() == pytest.approx([255, 0, 0], abs=40)
    auto = _frame_rgb(dest, noautorotate=False)
    np.testing.assert_array_equal(frame, auto)


@pytest.mark.skipif(not _GOPRO_180.exists(), reason="GH010013 fixture not present")
def test_export_clears_gopro_display_matrix(tmp_path: Path):
    info = probe(_GOPRO_180)
    assert info.rotation == 180
    recording = Recording(
        recording_id="GH-0013",
        prefix="GH",
        recording="0013",
        chapters=[
            Chapter(
                path=_GOPRO_180,
                prefix="GH",
                chapter=1,
                recording="0013",
                duration=info.duration,
                offset=0.0,
                info=info,
            )
        ],
    )
    dest = tmp_path / "baked.mp4"
    export_clip(
        recording,
        ClipWindow(1.0, 2.2, "GH-0013", ["nearby_rider"], 1.0, []),
        dest,
        rotation=180,
    )
    out = probe(dest)
    assert out.rotation == 0
    stored = _frame_rgb(dest, noautorotate=True, t=0.2)
    auto = _frame_rgb(dest, noautorotate=False, t=0.2)
    np.testing.assert_array_equal(stored, auto)
    # Sky/track: after a correct 180 bake the top band is brighter than the bottom.
    assert stored[: stored.shape[0] // 5].mean() > stored[-stored.shape[0] // 5 :].mean()
