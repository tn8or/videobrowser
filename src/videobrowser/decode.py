from __future__ import annotations

import queue
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from videobrowser.probe import VideoInfo

# Tried in order for analysis decode. Unavailable backends fail fast.
# (name, ffmpeg input args, scale on the GPU). "cuda" keeps the decoded frames
# in GPU memory and only downloads the few that pass the fps filter, already
# scaled; without that every 4K frame is copied to RAM and scaled on the CPU.
_DECODE_MODES: tuple[tuple[str, list[str], bool], ...] = (
    ("videotoolbox", ["-hwaccel", "videotoolbox"], False),
    ("cuda", ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"], True),
    ("cuda-copy", ["-hwaccel", "cuda"], False),
)

# Frames the decoder reads ahead, so ffmpeg keeps decoding while the
# detector works on the current frame.
_PREFETCH = 16

# Per decode mode: (failures, successes) in this process. A mode that keeps
# failing and has never worked (VideoToolbox on Windows, cuda without an
# NVIDIA card) is skipped, so each short seek does not pay for its start-up.
_MODE_LOG: dict[str, list[int]] = {}
_MODE_LOCK = threading.Lock()  # the profile/facing seeks run in parallel
_GIVE_UP_AFTER = 2


def _note(name: str, ok: bool) -> None:
    with _MODE_LOCK:
        log = _MODE_LOG.setdefault(name, [0, 0])
        log[1 if ok else 0] += 1


def _modes() -> list[tuple[str, list[str], bool]]:
    """Hardware modes worth trying now, in order."""
    out = []
    for mode in _DECODE_MODES:
        with _MODE_LOCK:
            failures, successes = _MODE_LOG.get(mode[0], (0, 0))
        if successes or failures < _GIVE_UP_AFTER:
            out.append(mode)
    return out


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


def filter_chain(
    width: int, height: int, rotation: int = 0, fps: float | None = None, gpu: bool = False
) -> str:
    """ffmpeg -vf for analysis frames of *width* x *height* (upright size).

    Frames are dropped to *fps* and scaled before they are rotated, so the
    rotation only touches small frames.
    """
    pre_w, pre_h = (height, width) if rotation % 180 else (width, height)
    parts: list[str] = []
    if fps:
        parts.append(f"fps={fps:g}")
    if gpu:
        parts += [f"scale_cuda={pre_w}:{pre_h}:format=nv12", "hwdownload", "format=nv12"]
    else:
        parts.append(f"scale={pre_w}:{pre_h}")
    rot = rotation_filter(rotation)
    if rot:
        parts.append(rot)
    return ",".join(parts)


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
    frames = decode_frames(path, t, 1, max_side, src_width, src_height, rotation)
    return frames[0] if frames else None


def decode_frames(
    path: Path,
    t: float,
    count: int,
    max_side: int,
    src_width: int,
    src_height: int,
    rotation: int = 0,
) -> list[np.ndarray]:
    """*count* consecutive scaled BGR frames from t seconds on (fewer near the end)."""
    width, height = scaled_size(src_width, src_height, max_side, rotation)
    expected = width * height * 3

    def _run(hw: list[str], gpu: bool) -> bytes:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", *hw]
        cmd += [
            "-noautorotate",
            "-ss",
            f"{max(0.0, t):.3f}",
            "-i",
            str(path),
            "-frames:v",
            str(count),
            "-vf",
            filter_chain(width, height, rotation, gpu=gpu),
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        return subprocess.run(cmd, capture_output=True).stdout

    for name, hw, gpu in _modes():
        raw = _run(hw, gpu)
        _note(name, bool(raw))
        # Fewer than *count* from a working mode means the file ends there.
        if len(raw) >= expected:
            return _split(raw, width, height)
    return _split(_run([], False), width, height)


def _split(raw: bytes, width: int, height: int) -> list[np.ndarray]:
    expected = width * height * 3
    n = len(raw) // expected
    return [
        np.frombuffer(raw[i * expected : (i + 1) * expected], dtype=np.uint8).copy().reshape(height, width, 3)
        for i in range(n)
    ]


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
        # Kill before closing the pipe: closing it while the prefetch thread
        # is blocked in read() would wait on that read forever.
        if self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()
        if self._proc.stdout:
            self._proc.stdout.close()
        self._proc = None

    def _cmd(self, hw: list[str], gpu: bool) -> list[str]:
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
            filter_chain(self.width, self.height, self.rotation, self.fps, gpu),
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]

    def _iter(self, name: str, hw: list[str], gpu: bool) -> Iterator[Frame]:
        try:
            self._proc = subprocess.Popen(
                self._cmd(hw, gpu),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg not found on PATH") from exc
        assert self._proc.stdout is not None
        frame_bytes = self.width * self.height * 3
        stdout = self._proc.stdout
        # A reader thread keeps ffmpeg's pipe drained while the caller runs
        # the detector, so decode and inference overlap.
        buffered: queue.Queue[bytes | None] = queue.Queue(maxsize=_PREFETCH)
        stop = threading.Event()

        def _put(item: bytes | None) -> bool:
            while not stop.is_set():
                try:
                    buffered.put(item, timeout=0.2)
                    return True
                except queue.Full:
                    continue
            return False

        def _read() -> None:
            try:
                while True:
                    buf = stdout.read(frame_bytes)
                    if len(buf) < frame_bytes or not _put(buf):
                        break
            except (OSError, ValueError):
                pass
            finally:
                _put(None)

        reader = threading.Thread(target=_read, daemon=True)
        reader.start()
        t = 0.0
        dt = 1.0 / self.fps if self.fps else 0.2
        yielded = False
        try:
            while True:
                if self.max_seconds is not None and t >= self.max_seconds:
                    break
                buf = buffered.get()
                if buf is None:
                    break
                image = np.frombuffer(buf, dtype=np.uint8).reshape(self.height, self.width, 3)
                yielded = True
                yield Frame(t=t, image=image.copy())
                t += dt
        finally:
            stop.set()
            self.close()
            reader.join(timeout=5.0)
        if not yielded and hw:
            raise RuntimeError(f"{name} decode produced no frames")

    def frames(self) -> Iterator[Frame]:
        last: RuntimeError | None = None
        for name, hw, gpu in _modes():
            self.backend = name
            self.hwaccel = True
            try:
                yield from self._iter(name, hw, gpu)
                _note(name, True)
                return
            except RuntimeError as exc:
                _note(name, False)
                last = exc
        self.backend = "software"
        self.hwaccel = False
        try:
            yield from self._iter("software", [], False)
        except RuntimeError:
            if last is not None:
                raise last
            raise
