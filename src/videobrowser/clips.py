from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from videobrowser.decode import rotation_filter
from videobrowser.events import ClipWindow
from videobrowser.gopro import Chapter, Recording, chapter_at


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"command failed: {cmd[:4]}")


_DISPLAY_ROTATION_SUPPORTED: bool | None = None


def ffmpeg_supports_display_rotation() -> bool:
    """True when ffmpeg accepts ``-display_rotation`` (FFmpeg >= ~6.1)."""
    global _DISPLAY_ROTATION_SUPPORTED
    if _DISPLAY_ROTATION_SUPPORTED is None:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "full"],
            capture_output=True,
            text=True,
        )
        blob = (proc.stdout or "") + (proc.stderr or "")
        _DISPLAY_ROTATION_SUPPORTED = "-display_rotation" in blob
    return _DISPLAY_ROTATION_SUPPORTED


def display_rotation_args(degrees: int = 0) -> list[str]:
    """Input args that clear/override Display Matrix when the build supports it."""
    if ffmpeg_supports_display_rotation():
        return ["-display_rotation", str(degrees)]
    return []


# GoPro rotation metadata is unreliable, so orientation is detected visually and
# baked into the pixels here. Each pipeline pairs a decoder with an encoder;
# hardware is tried first (VideoToolbox on macOS, NVENC on Windows/Linux),
# then a pure software fallback.
_REENCODE_PIPELINES: tuple[tuple[list[str], list[str]], ...] = (
    (["-hwaccel", "videotoolbox"], ["-c:v", "h264_videotoolbox", "-b:v", "60M"]),
    (
        ["-hwaccel", "cuda"],
        [
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc", "vbr",
            "-b:v", "60M",
            "-pix_fmt", "yuv420p",
        ],
    ),
    ([], ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p"]),
)


def _reencode_pipelines(rotation: int) -> tuple[tuple[list[str], list[str]], ...]:
    """Pick encode pipelines for a baked rotation.

    90/270 produce portrait frames. VideoToolbox often reports success but
    writes the unrotated landscape buffer into the 9:16 size (stretched,
    still sideways). NVENC does not have that bug, so it stays in the stack.
    180 stays 16:9 and VideoToolbox does apply the filter; leftover Display
    Matrix is stripped separately in ``_reencode``.
    """
    if rotation % 180:
        return tuple(
            p for p in _REENCODE_PIPELINES if "videotoolbox" not in " ".join(p[0] + p[1])
        )
    return _REENCODE_PIPELINES


def _reencode(input_args: list[str], rot: str, dest: Path, rotation: int = 180) -> None:
    """Re-encode with a baked rotation filter, trying hardware then software."""
    last: RuntimeError | None = None
    for decode_args, encode_args in _reencode_pipelines(rotation):
        cmd = (
            ["ffmpeg", "-hide_banner", "-y", "-loglevel", "error", *decode_args]
            # Input option: treat the source as unrotated so the encoder does
            # not copy GoPro Display Matrix side data onto the baked file.
            # ``rotate=0`` metadata alone does not clear that matrix; leftover
            # -180 makes the overlay decoder (and VLC) flip the footage again
            # while the HUD is drawn upright. Needs FFmpeg >= ~6.1; omitted on
            # older builds (e.g. Debian bookworm 5.1).
            + display_rotation_args(0)
            + input_args
            + ["-vf", rot, *encode_args, "-c:a", "copy"]
            + ["-metadata:s:v:0", "rotate=0", "-movflags", "+faststart", str(dest)]
        )
        try:
            _run(cmd)
            return
        except RuntimeError as exc:
            last = exc
            if dest.exists():
                dest.unlink()
    raise last if last is not None else RuntimeError("no re-encode pipeline available")


def _cut(
    src: Path, local_start: float, duration: float, dest: Path, rotation: int = 0
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    input_args = [
        "-noautorotate",
        "-ss",
        f"{max(0.0, local_start):.3f}",
        "-i",
        str(src),
        "-t",
        f"{max(0.2, duration):.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
    ]
    rot = rotation_filter(rotation)
    try:
        if rot:
            _reencode(input_args, rot, dest, rotation=rotation)
        else:
            _run(
                ["ffmpeg", "-hide_banner", "-y", "-loglevel", "error"]
                + input_args
                + [
                    "-c",
                    "copy",
                    "-metadata:s:v:0",
                    "rotate=0",
                    "-avoid_negative_ts",
                    "make_zero",
                    "-movflags",
                    "+faststart",
                    str(dest),
                ]
            )
    except RuntimeError:
        if dest.exists():
            dest.unlink()
        raise


def _local_span(chapter: Chapter, global_start: float, global_end: float) -> tuple[float, float]:
    local_start = max(0.0, global_start - chapter.offset)
    local_end = min(chapter.duration, global_end - chapter.offset)
    return local_start, max(local_start + 0.05, local_end)


def overlapping_chapters(
    recording: Recording, t_start: float, t_end: float
) -> list[tuple[Chapter, float, float]]:
    parts: list[tuple[Chapter, float, float]] = []
    for chapter in recording.chapters:
        ch_start = chapter.offset
        ch_end = chapter.offset + chapter.duration
        if t_end <= ch_start or t_start >= ch_end:
            continue
        local_start, local_end = _local_span(chapter, t_start, t_end)
        parts.append((chapter, local_start, local_end - local_start))
    if not parts:
        chapter = chapter_at(recording, t_start)
        if chapter is None:
            raise RuntimeError(f"no chapter covers t={t_start} in {recording.recording_id}")
        local_start, local_end = _local_span(chapter, t_start, t_end)
        parts.append((chapter, local_start, local_end - local_start))
    return parts


def export_clip(
    recording: Recording, window: ClipWindow, dest: Path, rotation: int = 0
) -> Path:
    parts = overlapping_chapters(recording, window.t_start, window.t_end)
    if len(parts) == 1:
        chapter, start, duration = parts[0]
        _cut(chapter.path, start, duration, dest, rotation=rotation)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pieces: list[Path] = []
        for i, (chapter, start, duration) in enumerate(parts):
            piece = tmp_path / f"p{i:02d}{chapter.path.suffix}"
            # Cut losslessly; rotation is baked once during the concat below.
            _cut(chapter.path, start, duration, piece, rotation=0)
            pieces.append(piece)
        listing = tmp_path / "concat.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in pieces))
        input_args = [
            "-noautorotate",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
        ]
        rot = rotation_filter(rotation)
        try:
            if rot:
                _reencode(input_args, rot, dest, rotation=rotation)
            else:
                _run(
                    ["ffmpeg", "-hide_banner", "-y", "-loglevel", "error"]
                    + input_args
                    + [
                        "-c",
                        "copy",
                        "-metadata:s:v:0",
                        "rotate=0",
                        "-movflags",
                        "+faststart",
                        str(dest),
                    ]
                )
        except RuntimeError:
            if dest.exists():
                dest.unlink()
            raise
    return dest


def grab_frame(
    src: Path, local_t: float, dest: Path, max_side: int = 960, rotation: int = 0
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    vf_parts: list[str] = []
    rot = rotation_filter(rotation)
    if rot:
        vf_parts.append(rot)
    vf_parts.append(f"scale='min({max_side},iw)':-2")
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-loglevel",
            "error",
            "-noautorotate",
            "-ss",
            f"{max(0.0, local_t):.3f}",
            "-i",
            str(src),
            "-frames:v",
            "1",
            "-vf",
            ",".join(vf_parts),
            str(dest),
        ]
    )
    return dest
