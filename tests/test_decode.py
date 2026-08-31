import subprocess
from pathlib import Path

from videobrowser.decode import FrameDecoder, scaled_size
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
