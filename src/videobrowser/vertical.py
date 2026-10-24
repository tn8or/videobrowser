"""1080x1920 vertical copies of highlight clips that follow the action.

Each clip window gets a 9:16 crop that pans sideways to keep the rider the
scan flagged in frame. The crop path comes from the tracker boxes collected
during the scan (5 fps by default): the event's own tracks first, then any
other rider in view. With nobody in view the crop looks into the corner the
way a rider would, steered by the turn rate from the GoPro GPS track (left in
a left-hander on a front camera, mirrored on a rear one). The clip opens
centred (or on the subject when it is already in view) and glides over to the
action. The
path is interpolated to the clip frame rate, smoothed and speed-limited so the
virtual camera glides instead of jumping, then applied by ffmpeg's ``crop``
filter via ``sendcmd``.
A compact HUD laid out for phone screens is composited on top.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from videobrowser.clips import _reencode_pipelines, display_rotation_args, export_clip
from videobrowser.decode import rotation_filter
from videobrowser.events import ClipWindow
from videobrowser.gopro import Recording
from videobrowser.probe import ProbeError, probe
from videobrowser.riders import TrackPoint
from videobrowser.telemetry import TurnSample

OUT_W = 1080
OUT_H = 1920

# Boxes smaller than this (fraction of frame height) are too far away to follow.
MIN_FOLLOW_HEIGHT_FRAC = 0.05
# A rider already followed is kept down to this size, so the crop does not
# drop them for the corner as they pull away.
KEEP_FOLLOW_HEIGHT_FRAC = 0.03
# Stay on the current target until another candidate is this much larger.
SWITCH_RATIO = 0.5
# Zero-phase smoothing time constant, seconds.
SMOOTH_TAU = 0.35
# Fastest pan, in source-frame widths per second.
MAX_PAN_PER_S = 0.6
# When the subject first shows up later than this into the clip, the crop
# starts at the fallback (centre) and pans over this long before it appears,
# instead of opening on wherever the subject will be seconds later.
LEAD_IN_S = 1.0
# Turn rate (deg/s) at which the crop looks fully into the corner.
FULL_TURN_DEG_S = 25.0
# Largest look-into-the-corner shift from the centre, fraction of frame width.
MAX_TURN_SHIFT = 0.35
# A front camera looks this far ahead into the coming corner, seconds.
TURN_LOOK_AHEAD_S = 0.7
# A turn sample this close before a rider sample yields to the rider.
TURN_RIDER_GAP_S = 0.3
# After the last sight of a rider the crop holds on them this long before it
# turns into the corner: the tracker loses riders for a few frames at a time.
RIDER_HOLD_S = 1.5
# Vertical output bitrate (the landscape exports use 60M for full-res frames).
_BITRATE = "20M"


@dataclass(frozen=True)
class FocusSample:
    t: float
    cx: float  # horizontal centre of interest, 0..1 of the frame width


def turn_focus(
    turns: list[TurnSample], t_start: float, t_end: float, facing: str = "front"
) -> list[FocusSample]:
    """Crop centres looking into the corner, from the GPS turn rate.

    A rear camera sees the inside of a left-hander on the right of its frame,
    so the shift is mirrored, and it gets no look-ahead (it shows where the
    bike has been).
    """
    if not turns:
        return []
    ts = np.array([s.t for s in turns], dtype=np.float64)
    yaw = np.array([s.yaw_rate for s in turns], dtype=np.float64)
    sign, ahead = (1.0, TURN_LOOK_AHEAD_S) if facing == "front" else (-1.0, 0.0)
    out: list[FocusSample] = []
    for t in ts[(ts >= t_start) & (ts <= t_end)]:
        rate = float(np.interp(t + ahead, ts, yaw))
        frac = min(max(rate / FULL_TURN_DEG_S, -1.0), 1.0)
        out.append(FocusSample(float(t), 0.5 + sign * MAX_TURN_SHIFT * frac))
    return out


def focus_samples(
    window: ClipWindow,
    tracks: dict[int, list[TrackPoint]],
    turns: list[TurnSample] | None = None,
    facing: str = "front",
) -> list[FocusSample]:
    """Where the action is over *window*, one sample per analysed frame.

    Priority per frame: tracks named by the window's events, then any other
    rider big enough to matter, then the corner ahead (*turns*), which only
    takes over once no rider has been seen for ``RIDER_HOLD_S``. Among
    candidates the largest box wins, but the current target is kept while it
    stays reasonably large so the crop does not flick between two riders side
    by side.
    """
    event_ids: set[int] = set()
    for event in window.events:
        try:
            event_ids.add(int(event.track_id))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass

    by_t: dict[float, list[TrackPoint]] = {}
    for points in tracks.values():
        for point in points:
            if not window.t_start <= point.t <= window.t_end:
                continue
            if point.height_frac < KEEP_FOLLOW_HEIGHT_FRAC:
                continue
            by_t.setdefault(point.t, []).append(point)

    samples: list[FocusSample] = []
    current: int | None = None
    for t in sorted(by_t):
        # Small boxes only for the rider already followed.
        points = [
            p for p in by_t[t] if p.height_frac >= MIN_FOLLOW_HEIGHT_FRAC or p.track_id == current
        ]
        if not points:
            continue
        primary = [p for p in points if p.track_id in event_ids]
        candidates = primary or points
        best = max(candidates, key=lambda p: p.area)
        kept = next((p for p in candidates if p.track_id == current), None)
        chosen = kept if kept is not None and kept.area >= SWITCH_RATIO * best.area else best
        current = chosen.track_id
        cx = 0.5 * (chosen.x1 + chosen.x2) / max(1, chosen.frame_w)
        samples.append(FocusSample(t, float(cx)))

    rider_ts = np.array([s.t for s in samples], dtype=np.float64)
    for sample in turn_focus(turns or [], window.t_start, window.t_end, facing):
        if len(rider_ts):
            # Riders win: just before one shows up, and for a while after one
            # was last seen.
            gap = rider_ts - sample.t
            if np.any((gap >= 0) & (gap < TURN_RIDER_GAP_S)) or np.any((gap <= 0) & (gap > -RIDER_HOLD_S)):
                continue
        samples.append(sample)
    samples.sort(key=lambda s: s.t)
    return samples


def crop_size(src_w: int, src_h: int) -> tuple[int, int]:
    """Largest even-sized 9:16 crop that fits in a *src_w* x *src_h* frame."""
    target = OUT_W / OUT_H
    if src_w / max(1, src_h) > target:
        ch = src_h
        cw = int(round(src_h * target))
    else:
        cw = src_w
        ch = int(round(src_w / target))
    return max(2, cw - cw % 2), max(2, ch - ch % 2)


def crop_centers(
    samples: list[FocusSample],
    t_start: float,
    n_frames: int,
    fps: float,
    half_width: float,
    fallback: float = 0.5,
) -> np.ndarray:
    """Per-frame crop centre (0..1 of frame width) for a clip starting at *t_start*."""
    lo, hi = half_width, 1.0 - half_width
    if hi < lo:
        lo = hi = 0.5
    n = max(1, n_frames)
    if not samples:
        return np.full(n, min(max(fallback, lo), hi))

    frame_t = t_start + np.arange(n) / fps
    ts = np.array([s.t for s in samples], dtype=np.float64)
    xs = np.array([s.cx for s in samples], dtype=np.float64)
    order = np.argsort(ts, kind="stable")
    ts, xs = ts[order], xs[order]
    if ts[0] - LEAD_IN_S > t_start:
        # Subject appears later: open on the fallback and pan in just before.
        ts = np.concatenate(([t_start, ts[0] - LEAD_IN_S], ts))
        xs = np.concatenate(([fallback, fallback], xs))
    # Holds the last sample after the subject leaves.
    raw = np.clip(np.interp(frame_t, ts, xs), lo, hi)

    alpha = 1.0 - math.exp(-1.0 / max(1e-6, fps * SMOOTH_TAU))
    smooth = raw.copy()
    for i in range(1, n):
        smooth[i] = smooth[i - 1] + alpha * (smooth[i] - smooth[i - 1])
    for i in range(n - 2, -1, -1):
        smooth[i] = smooth[i + 1] + alpha * (smooth[i] - smooth[i + 1])

    step = MAX_PAN_PER_S / fps
    out = smooth.copy()
    for i in range(1, n):
        out[i] = out[i - 1] + min(max(smooth[i] - out[i - 1], -step), step)
    return np.clip(out, lo, hi)


def crop_offsets(centers: np.ndarray, src_w: int, crop_w: int) -> np.ndarray:
    """Even left-edge pixel offsets for each crop centre."""
    x = np.round(centers * src_w - crop_w / 2.0).astype(np.int64)
    x = np.clip(x, 0, max(0, src_w - crop_w))
    return x - x % 2


def sendcmd_script(offsets: np.ndarray, fps: float, target: str = "crop@vc") -> str:
    """ffmpeg ``sendcmd`` lines that move the crop only when the offset changes."""
    lines: list[str] = []
    prev: int | None = None
    for i, x in enumerate(offsets.tolist()):
        if x != prev:
            lines.append(f"{i / fps:.4f} {target} x {x};")
            prev = x
    return "\n".join(lines) + "\n"


def _filter_path(path: Path) -> str:
    """Escape a path for use as a filter option value."""
    return path.as_posix().replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _encode_args(encode_args: list[str]) -> list[str]:
    return [_BITRATE if a == "60M" else a for a in encode_args]


@dataclass
class _Source:
    input_args: list[str]
    rotation_vf: str | None
    width: int  # upright (post-rotation) size
    height: int
    fps: float
    duration: float


def _render(
    src: _Source,
    dest: Path,
    offsets: np.ndarray,
    crop_w: int,
    crop_h: int,
    n_frames: int,
    overlay_factory: Optional[Callable[[], Any]],
    rotation: int,
    work: Path,
) -> None:
    cmds = work / "crop.cmd"
    cmds.write_text(sendcmd_script(offsets, src.fps))
    crop_y = max(0, (src.height - crop_h) // 2)
    crop_y -= crop_y % 2
    chain = [src.rotation_vf] if src.rotation_vf else []
    chain += [
        f"sendcmd=f='{_filter_path(cmds)}'",
        f"crop@vc={crop_w}:{crop_h}:{int(offsets[0])}:{crop_y}",
        f"scale={OUT_W}:{OUT_H}:flags=lanczos",
        "setsar=1",
    ]
    graph = "[0:v]" + ",".join(chain)
    overlay_args: list[str] = []
    if overlay_factory is not None:
        graph += "[bg];[bg][1:v]overlay=0:0:format=auto[vout]"
        overlay_args = [
            "-f", "rawvideo",
            "-pix_fmt", "bgra",
            "-s", f"{OUT_W}x{OUT_H}",
            "-r", str(Fraction(src.fps).limit_denominator(1001)),
            "-i", "pipe:0",
        ]
    else:
        graph += "[vout]"

    dest.parent.mkdir(parents=True, exist_ok=True)
    last: RuntimeError | None = None
    for decode_args, encode_args in _reencode_pipelines(rotation):
        cmd = (
            ["ffmpeg", "-hide_banner", "-y", "-loglevel", "error", *decode_args]
            + display_rotation_args(0)
            + src.input_args
            + overlay_args
            + ["-filter_complex", graph, "-map", "[vout]", "-map", "0:a:0?"]
            + ["-frames:v", str(n_frames)]
            + _encode_args(encode_args)
            + ["-c:a", "copy", "-movflags", "+faststart", str(dest)]
        )
        try:
            _run_with_overlay(cmd, overlay_factory, n_frames)
            return
        except RuntimeError as exc:
            last = exc
            dest.unlink(missing_ok=True)
    raise last if last is not None else RuntimeError("no encode pipeline available")


def _run_with_overlay(
    cmd: list[str], overlay_factory: Optional[Callable[[], Any]], n_frames: int
) -> None:
    if overlay_factory is None:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "ffmpeg failed")
        return
    source = overlay_factory()
    empty = b"\x00" * (OUT_W * OUT_H * 4)
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err
        )
        try:
            for idx in range(n_frames):
                frame = source.frame_for(idx) if source is not None else None
                try:
                    proc.stdin.write(frame if frame is not None else empty)  # type: ignore[union-attr]
                except BrokenPipeError:
                    break
        finally:
            try:
                proc.stdin.close()  # type: ignore[union-attr]
            except BrokenPipeError:
                pass
        ret = proc.wait()
        if ret != 0:
            err.seek(0)
            msg = err.read().decode("utf-8", "replace").strip()
            raise RuntimeError(msg or f"ffmpeg exited {ret}")


def _single_chapter_source(
    recording: Recording, window: ClipWindow, rotation: int
) -> _Source | None:
    from videobrowser.clips import overlapping_chapters

    parts = overlapping_chapters(recording, window.t_start, window.t_end)
    if len(parts) != 1:
        return None
    chapter, start, duration = parts[0]
    info = chapter.info
    width, height = info.width, info.height
    if rotation % 180:
        width, height = height, width
    return _Source(
        # -ss before -i decodes from the previous keyframe and drops frames up
        # to *start*, so the first output frame lines up with the crop path.
        input_args=[
            "-noautorotate",
            "-ss", f"{start:.3f}",
            "-i", str(chapter.path),
            "-t", f"{max(0.2, duration):.3f}",
        ],
        rotation_vf=rotation_filter(rotation),
        width=width,
        height=height,
        fps=info.fps or 30.0,
        duration=duration,
    )


def export_vertical(
    recording: Recording,
    window: ClipWindow,
    dest: Path,
    rotation: int,
    samples: list[FocusSample],
    fallback_cx: float = 0.5,
    overlay_factory: Optional[Callable[[int, int, float], Any]] = None,
) -> Path:
    """Write a 1080x1920 follow-cam copy of *window* to *dest*.

    *overlay_factory(width, height, fps)* returns an object with
    ``frame_for(idx) -> bytes | None`` (BGRA frames), or None for no overlay.
    """
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        src = _single_chapter_source(recording, window, rotation)
        enc_rotation = rotation
        if src is None:
            # Window spans chapters: join them (rotation baked) and crop that.
            joined = work / "joined.mp4"
            export_clip(recording, window, joined, rotation=rotation)
            try:
                meta = probe(joined)
            except ProbeError as exc:
                raise RuntimeError(str(exc)) from exc
            src = _Source(
                input_args=["-noautorotate", "-i", str(joined)],
                rotation_vf=None,
                width=meta.width,
                height=meta.height,
                fps=meta.fps or 30.0,
                duration=meta.duration,
            )
            enc_rotation = 0

        n_frames = max(1, int(round(src.duration * src.fps)))
        crop_w, crop_h = crop_size(src.width, src.height)
        centers = crop_centers(
            samples,
            window.t_start,
            n_frames,
            src.fps,
            half_width=crop_w / (2.0 * src.width),
            fallback=fallback_cx,
        )
        offsets = crop_offsets(centers, src.width, crop_w)
        factory = None
        first = overlay_factory(OUT_W, OUT_H, src.fps) if overlay_factory else None
        if first is not None:
            pending = [first]
            fps = src.fps

            def factory():
                # Sources are stateful, so each encode attempt gets a fresh one.
                return pending.pop() if pending else overlay_factory(OUT_W, OUT_H, fps)

        _render(src, dest, offsets, crop_w, crop_h, n_frames, factory, enc_rotation, work)
    return dest
