import subprocess
from pathlib import Path

from videobrowser.clips import export_clip
from videobrowser.events import ClipWindow
from videobrowser.gopro import Chapter, Recording
from videobrowser.probe import probe


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
