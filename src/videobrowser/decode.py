from __future__ import annotations

import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from videobrowser.probe import VideoInfo

# Tried in order for analysis decode. Unavailable backends fail fast.
_HWACCEL_TRY: tuple[list[str], ...] = (
    ["-hwaccel", "videotoolbox"],
    ["-hwaccel", "cuda"],
)


def _hwaccel_name(args: list[str]) -> str:
    if len(args) >= 2 and args[0] == "-hwaccel":
        return args[1]
    return "software"


def scaled_size(width: int, height: int, max_side: int, rotation: int = 0) -> tuple[int, int]:
    if rotation % 180:
        width, height = height, width
    if width <= 0 or height <= 0:
        return max_side, max_side
    if width >= height:
        out_w = max_side
        out_h = max(2, int(round(height * max_side / width)))
    else:
        out_h = max_side
        out_w = max(2, int(round(width * max_side / height)))
    return out_w - out_w % 2, out_h - out_h % 2


def rotation_filter(rotation: int) -> str | None:
    rotation = rotation % 360
    if rotation == 90:
        return "transpose=1"
    if rotation == 180:
        return "transpose=1,transpose=1"
    if rotation == 270:
        return "transpose=2"
    return None


def rotate_bgr(image: np.ndarray, rotation: int) -> np.ndarray:
    rotation = rotation % 360
    if rotation == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def decode_one_frame(
    path: Path,
    t: float,
    max_side: int,
    src_width: int,
    src_height: int,
    rotation: int = 0,
) -> np.ndarray | None:
    """Seek to t seconds and return a scaled BGR frame."""
    width, height = scaled_size(src_width, src_height, max_side, rotation)
    vf_parts: list[str] = []
    rot = rotation_filter(rotation)
    if rot:
        vf_parts.append(rot)
    vf_parts.append(f"scale={width}:{height}")
    vf = ",".join(vf_parts)
    expected = width * height * 3

    def _run(hw: list[str]) -> bytes:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *hw]
        cmd += [
            "-noautorotate",
            "-ss",
            f"{max(0.0, t):.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            vf,
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        return subprocess.run(cmd, capture_output=True).stdout

    raw = b""
    for hw in (*_HWACCEL_TRY, []):
        raw = _run(hw)
        if len(raw) >= expected:
            break
    if len(raw) < expected:
        return None
    return np.frombuffer(raw[:expected], dtype=np.uint8).copy().reshape(height, width, 3)


@dataclass
class Frame:
    t: float
    image: np.ndarray  # BGR, already scaled


class FrameDecoder:
    """Hardware-decode when possible, immediately scale, never keep 4K RGB."""

    def __init__(
        self,
        path: Path,
        info: VideoInfo,
        fps: float,
        max_side: int,
        rotation: int = 0,
        max_seconds: float | None = None,
    ) -> None:
        self.path = path
        self.info = info
        self.fps = fps
        self.rotation = rotation % 360
        self.max_seconds = max_seconds
        self.width, self.height = scaled_size(info.width, info.height, max_side, self.rotation)
        self._proc: subprocess.Popen[bytes] | None = None
        self.hwaccel = False
        self.backend = "software"

    def __enter__(self) -> FrameDecoder:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._proc is None:
            return
        if self._proc.stdout:
            self._proc.stdout.close()
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()
        self._proc = None

    def _vf(self) -> str:
        parts: list[str] = []
        rot = rotation_filter(self.rotation)
        if rot:
            parts.append(rot)
        parts.append(f"fps={self.fps:g}")
        parts.append(f"scale={self.width}:{self.height}")
        return ",".join(parts)

    def _cmd(self, hw: list[str]) -> list[str]:
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            *hw,
            "-noautorotate",
            "-i",
            str(self.path),
            "-vf",
            self._vf(),
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]

    def _iter(self, hw: list[str]) -> Iterator[Frame]:
        try:
            self._proc = subprocess.Popen(
                self._cmd(hw),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg not found on PATH") from exc
        assert self._proc.stdout is not None
        frame_bytes = self.width * self.height * 3
        t = 0.0
        dt = 1.0 / self.fps if self.fps else 0.2
        stdout = self._proc.stdout
        yielded = False
        try:
            while True:
                if self.max_seconds is not None and t >= self.max_seconds:
                    break
                buf = stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                image = np.frombuffer(buf, dtype=np.uint8).copy().reshape(
                    self.height, self.width, 3
                )
                yielded = True
                yield Frame(t=t, image=image)
                t += dt
        finally:
            self.close()
        if not yielded and hw:
            raise RuntimeError(f"{_hwaccel_name(hw)} decode produced no frames")

    def frames(self) -> Iterator[Frame]:
        last: RuntimeError | None = None
        for hw in _HWACCEL_TRY:
            self.backend = _hwaccel_name(hw)
            self.hwaccel = True
            try:
                yield from self._iter(hw)
                return
            except RuntimeError as exc:
                last = exc
        self.backend = "software"
        self.hwaccel = False
        try:
            yield from self._iter([])
        except RuntimeError:
            if last is not None:
                raise last
            raise
