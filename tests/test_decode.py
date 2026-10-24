import subprocess
from pathlib import Path

import numpy as np

from videobrowser import decode
from videobrowser.decode import FrameDecoder, decode_frames, filter_chain, scaled_size
from videobrowser.probe import probe


def test_decode_tiny_video(tmp_path: Path):
    src = tmp_path / "tiny.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=1280x720:d=1:r=25",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ],
        check=True,
    )
    info = probe(src)
    assert info.width == 1280
    assert info.height == 720
    frames = []
    with FrameDecoder(src, info, fps=5, max_side=640, max_seconds=1.0) as decoder:
        assert decoder.width == 640
        assert decoder.height == 360
        frames.extend(decoder.frames())
    assert len(frames) >= 4
    assert frames[0].image.shape == (360, 640, 3)
    assert scaled_size(1280, 720, 640) == (640, 360)


def _gradient_video(path: Path, seconds: int = 3) -> None:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=s=1280x720:d={seconds}:r=30",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


def test_filter_chain_scales_before_rotating():
    assert filter_chain(640, 360, 0, fps=5) == "fps=5,scale=640:360"
    # Upright 360x640 from a landscape source turned 90: scale to 640x360 first.
    assert filter_chain(360, 640, 90, fps=5) == "fps=5,scale=640:360,transpose=1"
    gpu = filter_chain(640, 360, 180, fps=5, gpu=True)
    assert gpu.startswith("fps=5,scale_cuda=640:360:format=nv12,hwdownload,format=nv12,")
    assert gpu.endswith("transpose=1,transpose=1")


def test_rotated_decode_is_upright_and_matches_seek(tmp_path: Path):
    src = tmp_path / "grad.mp4"
    _gradient_video(src)
    info = probe(src)
    with FrameDecoder(src, info, fps=5, max_side=320, rotation=90) as decoder:
        frames = list(decoder.frames())
    assert frames[0].image.shape == (320, 180, 3)
    seek = decode_frames(src, 0.0, 1, 320, info.width, info.height, rotation=90)
    assert seek[0].shape == (320, 180, 3)
    assert np.abs(seek[0].astype(int) - frames[0].image.astype(int)).mean() < 8


def test_stopping_early_does_not_hang(tmp_path: Path):
    src = tmp_path / "grad.mp4"
    _gradient_video(src, seconds=10)
    info = probe(src)
    decoder = FrameDecoder(src, info, fps=30, max_side=320)
    for i, _ in enumerate(decoder.frames()):
        if i == 2:
            break
    decoder.close()
    assert decoder._proc is None


def test_modes_that_never_work_are_skipped(monkeypatch):
    monkeypatch.setattr(decode, "_MODE_LOG", {})
    names = [m[0] for m in decode._modes()]
    assert names == ["videotoolbox", "cuda", "cuda-copy"]
    for _ in range(decode._GIVE_UP_AFTER):
        decode._note("videotoolbox", False)
    decode._note("cuda", False)
    decode._note("cuda", True)
    decode._note("cuda", False)
    assert [m[0] for m in decode._modes()] == ["cuda", "cuda-copy"]
