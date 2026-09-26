"""
video_compositor.py
~~~~~~~~~~~~~~~~~~~
Composite RaceBox telemetry overlays onto GoPro (or other) video files by matching
UTC wall-clock timestamps.

Matching strategy
-----------------
* Video start time prefers GPMD ``GPSU`` (true UTC) when present; otherwise it
  falls back to ``com.apple.quicktime.creationdate`` / ``creation_time``.
  GoPro often labels local wall-clock digits with a ``Z`` suffix, so tag times
  are interpreted as Europe/Copenhagen local.
* VBO ``time`` column is already stored as UTC epoch seconds (see vbo_parser.py).
* For every video frame at index *i* the real-world UTC time is::

      t = video_start_utc + i / video_fps

  We look up the nearest VBO sample at that time and render the overlay.
* Frames that fall outside *every* VBO session's time range get a fully
  transparent overlay so the original video shows through unchanged.
* Multiple cameras may record the same session simultaneously – all of them
  are processed independently.
"""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from zoneinfo import ZoneInfo

from .cli import (
    CornerEvent,
    _build_lap_distance,
    _color_for_corner,
    _corner_score,
    _ensure_ffmpeg,
    _format_color_for_lap,
    _get_day_bounds,
    _merge_corner_best,
)
from .lap_timing import Lap, compute_sectors, detect_corners, detect_laps
from .overlay_render import (
    CornerDisplay,
    FrameState,
    LapRow,
    RenderConfig,
    build_map_layer,
    build_static_layer,
    render_frame,
)
from .stats import CornerBest, StatsStore
from .track_map import build_track_map, map_point
from .vbo_parser import VboSession

# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

_MIN_GPS_YEAR = 2020


@dataclass
class VideoMetadata:
    path: Path
    start_utc: datetime  # UTC recording start (GPS preferred, else creation tag)
    duration: float  # seconds
    fps: float
    width: int
    height: int
    codec: str
    pix_fmt: str

    @property
    def end_utc(self) -> datetime:
        return datetime.fromtimestamp(
            self.start_utc.timestamp() + self.duration, tz=timezone.utc
        )

    @property
    def total_frames(self) -> int:
        return int(self.duration * self.fps)


def _parse_creation_tag(raw: str) -> Optional[datetime]:
    """Interpret GoPro creation tags as Europe/Copenhagen local wall time."""
    try:
        cleaned = raw.replace("Z", "").replace("+00:00", "")
        naive = datetime.fromisoformat(cleaned)
        return naive.replace(tzinfo=ZoneInfo("Europe/Copenhagen")).astimezone(
            timezone.utc
        )
    except ValueError:
        return None


def _gpmd_stream_index(streams: list) -> Optional[int]:
    for stream in streams:
        tag = f"{stream.get('codec_tag_string') or ''} {stream.get('codec_name') or ''}".lower()
        handler = str((stream.get("tags") or {}).get("handler_name") or "").lower()
        if "gpmd" in tag or "goprometa" in handler or "gopro met" in handler:
            return int(stream["index"])
    return None


def _extract_gpmd_bytes(path: Path, stream_index: int) -> Optional[bytes]:
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-map",
            f"0:{stream_index}",
            "-codec",
            "copy",
            "-copy_unknown",
            "-f",
            "data",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout:
        return None
    return result.stdout


def _parse_gpsu(raw: str) -> Optional[datetime]:
    text = raw.strip()
    for fmt in ("%y%m%d%H%M%S.%f", "%y%m%d%H%M%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _gps_start_from_gpmd(blob: bytes, duration: float) -> Optional[datetime]:
    """UTC start of the video from GPMD GPSU, or None if no fix."""
    samples: list[str] = []

    def walk(data: bytes, start: int = 0, end: Optional[int] = None) -> None:
        offset = start
        limit = len(data) if end is None else end
        while offset + 8 <= limit:
            key = data[offset : offset + 4]
            type_ = data[offset + 4]
            size = data[offset + 5]
            repeat = int.from_bytes(data[offset + 6 : offset + 8], "big")
            payload_len = size * repeat
            payload_start = offset + 8
            payload_end = payload_start + payload_len
            if payload_end > limit:
                break
            payload = data[payload_start:payload_end]
            if key == b"GPSU" and payload:
                samples.append(
                    payload.split(b"\x00", 1)[0].decode("ascii", "replace").strip()
                )
            elif type_ == 0:
                walk(payload)
            pad = (4 - (payload_len % 4)) % 4
            offset = payload_end + pad

    walk(blob)
    if not samples:
        return None
    valid: list[tuple[int, datetime]] = []
    for index, raw in enumerate(samples):
        parsed = _parse_gpsu(raw)
        if parsed is not None and parsed.year >= _MIN_GPS_YEAR:
            valid.append((index, parsed))
    if not valid:
        return None
    index0, stamp0 = valid[0]
    if index0 == 0:
        return stamp0
    step = duration / len(samples) if duration > 0 and len(samples) > 1 else 1.0
    return datetime.fromtimestamp(stamp0.timestamp() - index0 * step, tz=timezone.utc)


def get_video_metadata(path: Path) -> Optional[VideoMetadata]:
    """Return VideoMetadata via ffprobe, or None if no UTC creation timestamp."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        d = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None

    fmt = d.get("format", {})
    tags = fmt.get("tags", {})
    streams = d.get("streams") or []

    creation_date = tags.get("com.apple.quicktime.creationdate") or tags.get(
        "creation_time"
    )
    start_utc = _parse_creation_tag(str(creation_date)) if creation_date else None

    duration = float(fmt.get("duration", 0.0))
    gpmd_index = _gpmd_stream_index(streams)
    if gpmd_index is not None:
        blob = _extract_gpmd_bytes(path, gpmd_index)
        if blob:
            gps_start = _gps_start_from_gpmd(blob, duration)
            if gps_start is not None:
                start_utc = gps_start

    if start_utc is None:
        return None

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        return None
    vs = video_streams[0]

    fps_str = vs.get("r_frame_rate", "25/1")
    try:
        num, den = fps_str.split("/")
        fps = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        fps = 25.0

    return VideoMetadata(
        path=path,
        start_utc=start_utc,
        duration=duration,
        fps=fps,
        width=int(vs.get("width", 1920)),
        height=int(vs.get("height", 1080)),
        codec=vs.get("codec_name", "h264"),
        pix_fmt=vs.get("pix_fmt", "yuv420p"),
    )


# ---------------------------------------------------------------------------
# Encoder selection
# ---------------------------------------------------------------------------


# Cached result of `ffmpeg -encoders` – queried at most once per process.
class _EncoderCache:
    value: Optional[str] = None


def _get_available_encoders() -> str:
    if _EncoderCache.value is None:
        r = subprocess.run(
            ["ffmpeg", "-encoders"], capture_output=True, text=True, check=False
        )
        _EncoderCache.value = r.stdout or ""
    cached: str = _EncoderCache.value or ""
    return cached


class _HwaccelCache:
    value: Optional[str] = None


def _get_hwaccels() -> str:
    if _HwaccelCache.value is None:
        r = subprocess.run(
            ["ffmpeg", "-hwaccels"], capture_output=True, text=True, check=False
        )
        _HwaccelCache.value = r.stdout or ""
    cached: str = _HwaccelCache.value or ""
    return cached


@dataclass
class GpuCapabilities:
    has_cuda: bool = False
    has_nvenc: bool = False
    nvenc_hevc: bool = False
    nvenc_h264: bool = False
    has_videotoolbox: bool = False


def _detect_gpu_capabilities() -> GpuCapabilities:
    encoders = _get_available_encoders()
    hwaccels = _get_hwaccels()
    caps = GpuCapabilities()
    caps.has_cuda = "cuda" in hwaccels
    caps.has_videotoolbox = "videotoolbox" in hwaccels
    caps.has_nvenc = "nvenc" in encoders
    caps.nvenc_hevc = "hevc_nvenc" in encoders
    caps.nvenc_h264 = "h264_nvenc" in encoders
    return caps


def _fps_fraction(fps: float) -> Tuple[int, int]:
    """Convert float fps to an integer num/den pair."""
    if abs(fps - round(fps)) < 0.01:
        return int(round(fps)), 1
    if abs(fps - 29.97) < 0.01:
        return 30000, 1001
    if abs(fps - 23.976) < 0.01:
        return 24000, 1001
    if abs(fps - 59.94) < 0.01:
        return 60000, 1001
    # Generic rational
    return int(round(fps * 1000)), 1000


def _select_encoder_args(
    codec: str, gpu_caps: Optional[GpuCapabilities] = None
) -> Tuple[List[str], str]:
    """
    Return ``(ffmpeg_args, description)`` for the best available encoder.

    Prefers hardware encoders in priority order:
      macOS:  VideoToolbox (hevc_videotoolbox / h264_videotoolbox)
      Windows/Linux:  NVENC (hevc_nvenc / h264_nvenc) with p5 preset
    Falls back to software encoders (libx265, libx264, prores_ks).
    Since composited output never carries an alpha channel, standard
    YUV pixel formats are used for maximum hardware compatibility.
    """
    if gpu_caps is None:
        gpu_caps = _detect_gpu_capabilities()

    if platform.system() == "Darwin":
        if codec in ("hevc", "h265") and gpu_caps.has_videotoolbox:
            return (
                [
                    "-c:v", "hevc_videotoolbox",
                    "-q:v", "75",
                    "-allow_sw", "1",
                    "-tag:v", "hvc1",
                    "-pix_fmt", "yuv420p",
                ],
                "hevc_videotoolbox (hardware)",
            )
        if codec in ("h264", "avc") and gpu_caps.has_videotoolbox:
            return (
                [
                    "-c:v", "h264_videotoolbox",
                    "-q:v", "75",
                    "-allow_sw", "1",
                    "-pix_fmt", "yuv420p",
                ],
                "h264_videotoolbox (hardware)",
            )

    if gpu_caps.has_nvenc:
        if codec in ("hevc", "h265") and gpu_caps.nvenc_hevc:
            return (
                [
                    "-c:v", "hevc_nvenc",
                    "-preset", "p5",
                    "-rc", "vbr",
                    "-cq", "28",
                    "-pix_fmt", "yuv420p",
                ],
                "hevc_nvenc (NVIDIA)",
            )
        if codec in ("h264", "avc") and gpu_caps.nvenc_h264:
            return (
                [
                    "-c:v", "h264_nvenc",
                    "-preset", "p5",
                    "-rc", "vbr",
                    "-cq", "28",
                    "-pix_fmt", "yuv420p",
                ],
                "h264_nvenc (NVIDIA)",
            )

    if codec in ("hevc", "h265"):
        return [
            "-c:v", "libx265", "-crf", "22", "-preset", "fast",
        ], "libx265 (software)"
    if codec in ("h264", "avc"):
        return [
            "-c:v", "libx264", "-crf", "22", "-preset", "fast",
        ], "libx264 (software)"
    if codec == "prores":
        return ["-c:v", "prores_ks", "-profile:v", "3"], "prores_ks (software)"
    return ["-c:v", "libx264", "-crf", "22", "-preset", "fast"], "libx264 (software)"


# ---------------------------------------------------------------------------
# GPU filter-graph & decode helpers
# ---------------------------------------------------------------------------


def _build_filter_complex(
    out_w: int,
    out_h: int,
    video_width: int,
    video_height: int,
    use_cuda_overlay: bool = False,
) -> str:
    needs_scale = out_w != video_width or out_h != video_height

    if use_cuda_overlay:
        v0 = "[0:v]"
        if needs_scale:
            v0 = f"[0:v]scale={out_w}:{out_h},hwupload_cuda[scaled];[scaled]"
        else:
            v0 = "[0:v]hwupload_cuda[v0c];[v0c]"
        return f"{v0}[1:v]overlay_cuda=0:0[vout]"

    if needs_scale:
        return (
            f"[0:v]scale={out_w}:{out_h}[scaled];"
            f"[scaled][1:v]overlay=0:0:format=auto[vout]"
        )
    return "[0:v][1:v]overlay=0:0:format=auto[vout]"


def _resolve_gpu_decode_args(
    codec: str, gpu_caps: GpuCapabilities, use_cuda_overlay: bool
) -> List[str]:
    if platform.system() == "Darwin" and codec in ("hevc", "h265", "h264", "avc"):
        return ["-hwaccel", "videotoolbox"]

    if gpu_caps.has_cuda and codec in ("hevc", "h265", "h264", "avc"):
        if use_cuda_overlay:
            return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        # Download to system memory so the CPU overlay filter can read the frames.
        return ["-hwaccel", "cuda"]

    return []


# ---------------------------------------------------------------------------
# Per-session stateful renderer
# ---------------------------------------------------------------------------


class _SessionRenderer:
    """
    Builds all pre-computed state for one VBO session and exposes
    ``get_frame(t)`` to render the overlay at an arbitrary UTC timestamp.

    **Must** be called with monotonically non-decreasing *t* values.
    """

    def __init__(
        self,
        session: VboSession,
        config: RenderConfig,
        stats_path: Path,
    ) -> None:
        self.session = session
        self.config = config
        self.times: np.ndarray = session.data["time"]
        self.lat: np.ndarray = session.data["lat"]
        self.lon: np.ndarray = session.data["lng"]

        speed = session.data.get("velocity")
        if speed is None:
            speed = session.data.get("velocity_kmh")
        if speed is None:
            raise ValueError(f"No velocity column in {session.path}")
        self.speed = speed
        lean = session.data.get("lean-angle")
        long_acc = session.data.get("LongAcc")
        self.lean = lean if lean is not None else np.zeros_like(self.times)
        self.long_acc = long_acc if long_acc is not None else np.zeros_like(self.times)

        self.laps = detect_laps(self.times, self.lat, self.lon, session.start_line)
        compute_sectors(self.laps, self.lat, self.lon, self.times)
        # Corner Pre/Entry/Min metrics are disabled in the merged pipeline.
        lap_corners: List = []

        active_mask = np.zeros_like(self.lat, dtype=bool)
        for lap in self.laps:
            if not lap.is_out_lap and not lap.is_in_lap:
                active_mask[lap.start_idx : lap.end_idx + 1] = True

        self.track_map = build_track_map(self.lat, self.lon, active_mask)
        self.map_layer = build_map_layer(config, self.track_map)
        self.static_layer = build_static_layer(config, self.track_map, map_layer=self.map_layer)

        with StatsStore(stats_path) as stats:
            tz = ZoneInfo("Europe/Copenhagen")
            day_start_utc, day_end_utc = _get_day_bounds(session.session_start_utc, tz)

            best_summary = stats.fetch_best_lap(
                session.track_name, session.session_start_utc.timestamp()
            )
            best_overall_time = best_summary.best_lap_time
            best_day_time = stats.fetch_best_day_lap(
                session.track_name, day_start_utc, day_end_utc
            )

            # Lap distances for on-track comparison
            self.lap_distances: Dict[int, Tuple[np.ndarray, float]] = {}
            for lap in self.laps:
                if not lap.is_out_lap and not lap.is_in_lap:
                    self.lap_distances[lap.index] = _build_lap_distance(
                        lap, self.lat, self.lon
                    )

            # Corner events pre-computed from stats
            self.corner_events: List[CornerEvent] = []
            self.corner_best_overall: Dict[int, float] = {}
            self.corner_best_day: Dict[int, float] = {}
            self.corner_best_values_overall: Dict[int, CornerBest] = {}
            self.corner_best_values_day: Dict[int, CornerBest] = {}

            for lc in lap_corners:
                for corner in lc.corners:
                    self.corner_events.append(
                        CornerEvent(
                            end_time=self.times[corner.end_idx],
                            corner_index=corner.corner_index,
                            pre_corner_max_speed=corner.pre_corner_max_speed,
                            entry_speed=corner.entry_speed,
                            min_speed=corner.min_speed,
                            max_lean=corner.max_lean,
                        )
                    )
                    ci = corner.corner_index
                    if ci not in self.corner_best_overall:
                        best = stats.fetch_corner_best(
                            session.track_name, ci, session.session_start_utc.timestamp()
                        )
                        if best.pre_corner_max_speed is not None:
                            self.corner_best_overall[ci] = _corner_score(
                                best.pre_corner_max_speed,
                                best.entry_speed,
                                best.min_speed,
                                best.max_lean,
                            )
                            self.corner_best_values_overall[ci] = best
                        best_day = stats.fetch_corner_best_day(
                            session.track_name, ci, day_start_utc, day_end_utc
                        )
                        if best_day.pre_corner_max_speed is not None:
                            self.corner_best_day[ci] = _corner_score(
                                best_day.pre_corner_max_speed,
                                best_day.entry_speed,
                                best_day.min_speed,
                                best_day.max_lean,
                            )
                            self.corner_best_values_day[ci] = best_day

        self.corner_events.sort(key=lambda c: c.end_time)

        # Session time bounds
        self.t_start = float(self.times[0])
        self.t_end = float(self.times[-1])

        # Mutable playback state (advanced monotonically per get_frame call)
        self.lap_pointer = 0
        self.corner_ptr = 0
        self.completed_laps: List[LapRow] = []
        self.last_completed_lap: Optional[Lap] = None
        self.current_best_overall: Optional[float] = best_overall_time
        self.current_best_day: Optional[float] = best_day_time
        self.best_speed_kmh: Optional[float] = None

    # ------------------------------------------------------------------

    def _advance_state(self, t: float) -> None:
        """Advance lap/corner state pointers up to time *t*."""
        while (
            self.lap_pointer < len(self.laps)
            and self.times[self.laps[self.lap_pointer].end_idx] <= t
        ):
            lap = self.laps[self.lap_pointer]
            if not lap.is_out_lap and not lap.is_in_lap:
                lap_color = _format_color_for_lap(
                    lap.lap_time, self.current_best_day, self.current_best_overall
                )
                self.completed_laps.append(
                    LapRow(
                        lap_number=lap.index,
                        lap_time=lap.lap_time,
                        sector_times=lap.sector_times,
                        color=lap_color,
                    )
                )
                if (
                    self.current_best_overall is None
                    or lap.lap_time <= self.current_best_overall
                ):
                    self.current_best_overall = lap.lap_time
                if (
                    self.current_best_day is None
                    or lap.lap_time <= self.current_best_day
                ):
                    self.current_best_day = lap.lap_time
                self.last_completed_lap = lap
            self.lap_pointer += 1

        while (
            self.corner_ptr < len(self.corner_events)
            and self.corner_events[self.corner_ptr].end_time <= t
        ):
            event = self.corner_events[self.corner_ptr]
            score = _corner_score(
                event.pre_corner_max_speed,
                event.entry_speed,
                event.min_speed,
                event.max_lean,
            )
            best_o = self.corner_best_overall.get(event.corner_index)
            best_d = self.corner_best_day.get(event.corner_index)
            event.color = _color_for_corner(score, best_d, best_o)
            if best_o is None or score >= best_o:
                self.corner_best_overall[event.corner_index] = score
            if best_d is None or score >= best_d:
                self.corner_best_day[event.corner_index] = score
            self.corner_best_values_overall[event.corner_index] = _merge_corner_best(
                self.corner_best_values_overall.get(event.corner_index), event
            )
            self.corner_best_values_day[event.corner_index] = _merge_corner_best(
                self.corner_best_values_day.get(event.corner_index), event
            )
            self.corner_ptr += 1

    def skip_to(self, t: float) -> None:
        """
        Fast-forward internal state to time *t* without rendering.
        Called once before the render loop when the video starts mid-session.
        Correctly handles best_speed_kmh by scanning active-lap samples.
        """
        self._advance_state(t)

        # Compute best_speed_kmh for all active-lap samples up to t
        cut_idx = int(np.searchsorted(self.times, t, side="right"))
        for lap in self.laps:
            if lap.is_out_lap or lap.is_in_lap:
                continue
            lo = lap.start_idx
            hi = min(lap.end_idx + 1, cut_idx)
            if lo >= hi:
                continue
            s = float(np.max(self.speed[lo:hi]))
            if self.best_speed_kmh is None or s > self.best_speed_kmh:
                self.best_speed_kmh = s

    def get_frame(self, t: float) -> Optional[np.ndarray]:
        """
        Return a rendered RGBA numpy array for time *t*, or None when *t*
        falls outside this session's time range.
        """
        if t < self.t_start or t > self.t_end:
            return None

        self._advance_state(t)

        sample_idx = int(np.searchsorted(self.times, t, side="right") - 1)
        sample_idx = max(0, min(sample_idx, len(self.times) - 1))

        # Find current lap
        current_lap: Optional[Lap] = None
        for lap in self.laps:
            if lap.start_idx <= sample_idx <= lap.end_idx:
                current_lap = lap
                break

        lap_number: Optional[int] = None
        lap_time: Optional[float] = None
        on_track_faster: Optional[bool] = None

        if current_lap is None or current_lap.is_out_lap:
            pass
        elif current_lap.is_in_lap:
            if self.last_completed_lap is not None:
                lap_number = self.last_completed_lap.index
                lap_time = self.last_completed_lap.lap_time
        else:
            lap_number = current_lap.index
            lap_time = t - current_lap.start_time
            if (
                self.current_best_overall is not None
                and lap_number in self.lap_distances
            ):
                dist, total = self.lap_distances[lap_number]
                local_idx = min(
                    max(sample_idx - current_lap.start_idx, 0), len(dist) - 1
                )
                fraction = dist[local_idx] / total if total > 0 else 0.0
                predicted = self.current_best_overall * fraction
                on_track_faster = lap_time <= predicted if predicted > 0 else None
            s = float(self.speed[sample_idx])
            if self.best_speed_kmh is None or s > self.best_speed_kmh:
                self.best_speed_kmh = s

        last_laps = list(reversed(self.completed_laps[-4:]))

        last_corner_event: Optional[CornerEvent] = None
        for event in reversed(self.corner_events[: self.corner_ptr]):
            if event.end_time <= t:
                last_corner_event = event
                break

        corner_display: Optional[CornerDisplay] = None
        if last_corner_event is not None and last_corner_event.color is not None:
            bv = self.corner_best_values_overall.get(last_corner_event.corner_index)
            corner_display = CornerDisplay(
                pre_corner_max_speed=last_corner_event.pre_corner_max_speed,
                entry_speed=last_corner_event.entry_speed,
                min_speed=last_corner_event.min_speed,
                max_lean=last_corner_event.max_lean,
                best_pre_corner_max_speed=(
                    bv.pre_corner_max_speed
                    if bv
                    else last_corner_event.pre_corner_max_speed
                ),
                best_entry_speed=(
                    bv.entry_speed if bv else last_corner_event.entry_speed
                ),
                best_min_speed=(bv.min_speed if bv else last_corner_event.min_speed),
                best_max_lean=(bv.max_lean if bv else last_corner_event.max_lean),
                color=last_corner_event.color,
            )

        map_pos = None
        if not (current_lap and (current_lap.is_out_lap or current_lap.is_in_lap)):
            map_pos = map_point(
                float(self.lat[sample_idx]),
                float(self.lon[sample_idx]),
                self.track_map.bounds,
            )

        state = FrameState(
            lap_number=lap_number,
            lap_time=lap_time,
            on_track_faster=on_track_faster,
            last_laps=last_laps,
            speed_kmh=float(self.speed[sample_idx]),
            best_speed_kmh=self.best_speed_kmh,
            corner_display=corner_display,
            lean_angle=float(self.lean[sample_idx]),
            map_pos=map_pos,
            is_in_lap=current_lap is not None and current_lap.is_in_lap,
            wall_time_utc=t,
        )

        return render_frame(
            self.config, self.track_map, state,
            map_layer=self.map_layer, static_layer=self.static_layer,
        )


# ---------------------------------------------------------------------------
# Main compositing functions
# ---------------------------------------------------------------------------


def composite_video(
    video_meta: VideoMetadata,
    output_dir: Path,
    sessions: List[VboSession],
    font_path: str,
    stats_path: Path,
    overlay_width: Optional[int] = None,
    gpu_accel: str = "auto",
) -> bool:
    """
    Composite racebox overlay onto *video_meta*.

    Searches *sessions* for any session whose time range overlaps with the
    video.  If none found, returns False and writes no file.

    Multiple sessions may overlap a single video (e.g. session boundary
    falls mid-clip); all are consulted per frame.

    *gpu_accel* controls GPU usage:
      ``"auto"`` — detect and use best available hardware
      ``"on"``   — require GPU (CUDA overlay if NVIDIA, else fail)
      ``"off"``  — software-only path
    """
    _ensure_ffmpeg()

    gpu_caps = _detect_gpu_capabilities() if gpu_accel != "off" else GpuCapabilities()
    use_cuda_overlay = (
        gpu_accel != "off"
        and gpu_caps.has_cuda
        and video_meta.codec in ("hevc", "h265", "h264", "avc")
        and platform.system() != "Darwin"
    )

    vid_start_ts = video_meta.start_utc.timestamp()
    vid_end_ts = vid_start_ts + video_meta.duration

    overlapping = [
        s
        for s in sessions
        if float(s.data["time"][0]) < vid_end_ts
        and float(s.data["time"][-1]) > vid_start_ts
    ]
    if not overlapping:
        return False

    if overlay_width is not None:
        out_w = overlay_width
        out_h = int(round(overlay_width * video_meta.height / video_meta.width))
    else:
        out_w = video_meta.width
        out_h = video_meta.height

    config = RenderConfig(
        width=out_w,
        height=out_h,
        fps=video_meta.fps,
        font_path=font_path,
    )

    renderers: List[_SessionRenderer] = []
    for session in overlapping:
        try:
            r = _SessionRenderer(session, config, stats_path)
            renderers.append(r)
        except (ValueError, KeyError) as exc:
            print(f"  Warning: cannot build renderer for {session.path}: {exc}")

    if not renderers:
        return False

    # Trim output to the portion actually covered by VBO data so the output
    # file starts/ends exactly where the telemetry overlay exists.
    vbo_t_start = min(r.t_start for r in renderers)
    vbo_t_end = max(r.t_end for r in renderers)
    trim_start_ts = max(vid_start_ts, vbo_t_start)
    trim_end_ts = min(vid_end_ts, vbo_t_end)

    if trim_start_ts >= trim_end_ts:
        return False

    # Fast-forward renderer state to the trim start point.
    for r in renderers:
        if trim_start_ts > r.t_start:
            r.skip_to(trim_start_ts)

    trim_offset = trim_start_ts - vid_start_ts   # seconds to seek into video
    trim_duration = trim_end_ts - trim_start_ts
    trim_frames = int(trim_duration * video_meta.fps) + 1

    output_path = output_dir / video_meta.path.name
    encoder_args, encoder_desc = _select_encoder_args(video_meta.codec, gpu_caps)
    fps_num, fps_den = _fps_fraction(video_meta.fps)

    hw_decode_args = _resolve_gpu_decode_args(
        video_meta.codec, gpu_caps, use_cuda_overlay
    )

    seek_args = ["-ss", f"{trim_offset:.3f}"] if trim_offset > 0.001 else []

    trim_start_utc = datetime.fromtimestamp(trim_start_ts, tz=timezone.utc)
    creation_date_str = trim_start_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    gpu_desc_parts = []
    if hw_decode_args:
        gpu_desc_parts.append(f"hw decode: {hw_decode_args[1]}")
    if use_cuda_overlay:
        gpu_desc_parts.append("overlay_cuda")
    gpu_desc = "  |  " + ", ".join(gpu_desc_parts) if gpu_desc_parts else ""
    print(f"  encoder: {encoder_desc}{gpu_desc}")
    if trim_offset > 0.001:
        print(
            f"  trim: +{trim_offset:.1f}s into video → {trim_duration:.1f}s output"
        )

    filter_complex = _build_filter_complex(
        out_w, out_h, video_meta.width, video_meta.height,
        use_cuda_overlay=use_cuda_overlay,
    )

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        *seek_args,
        *hw_decode_args,
        "-i",
        str(video_meta.path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgra",
        "-s",
        f"{out_w}x{out_h}",
        "-r",
        f"{fps_num}/{fps_den}",
        "-i",
        "pipe:0",
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "0:a?",
        "-map",
        "0:d?",
        "-t",
        f"{trim_duration:.3f}",
        *encoder_args,
        "-c:a",
        "copy",
        "-c:d",
        "copy",
        "-map_metadata",
        "0",
        "-metadata",
        f"com.apple.quicktime.creationdate={creation_date_str}",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    empty_frame_bytes = b"\x00" * (out_h * out_w * 4)
    has_overlay = False

    for frame_idx in range(trim_frames):
        t = trim_start_ts + frame_idx / video_meta.fps

        frame = None
        for renderer in renderers:
            f = renderer.get_frame(t)
            if f is not None:
                frame = f
                has_overlay = True
                break

        if proc.stdin is None:
            break
        if frame is not None:
            proc.stdin.write(frame if isinstance(frame, bytes) else frame.tobytes())
        else:
            proc.stdin.write(empty_frame_bytes)

    if proc.stdin:
        proc.stdin.close()
    proc.wait()

    if not has_overlay:
        output_path.unlink(missing_ok=True)
        return False

    return True


_LAP_CLIP_PAD = 15.0  # seconds of video padding before/after each lap


def _lap_file_label(lap: Lap, all_laps: List[Lap]) -> str:
    """Return a short label for *lap* used in output filenames."""
    if lap.is_out_lap:
        return "outlap"
    if lap.is_in_lap:
        return "inlap"
    timed = [l for l in all_laps if not l.is_out_lap and not l.is_in_lap]
    n = timed.index(lap) + 1
    return f"lap{n}"


def _run_clip_ffmpeg(
    video_meta: VideoMetadata,
    clip_start_ts: float,
    clip_end_ts: float,
    renderer: "_SessionRenderer",
    out_w: int,
    out_h: int,
    output_path: Path,
    gpu_caps: Optional[GpuCapabilities] = None,
    use_cuda_overlay: bool = False,
) -> None:
    """Run ffmpeg to write one clip with the overlay piped from *renderer*."""
    fps_num, fps_den = _fps_fraction(video_meta.fps)
    if gpu_caps is None:
        gpu_caps = _detect_gpu_capabilities()
    encoder_args, _ = _select_encoder_args(video_meta.codec, gpu_caps)

    trim_offset = clip_start_ts - video_meta.start_utc.timestamp()
    trim_duration = clip_end_ts - clip_start_ts
    trim_frames = int(trim_duration * video_meta.fps) + 1

    hw_decode_args = _resolve_gpu_decode_args(
        video_meta.codec, gpu_caps, use_cuda_overlay
    )

    seek_args = ["-ss", f"{trim_offset:.3f}"] if trim_offset > 0.001 else []

    clip_start_utc = datetime.fromtimestamp(clip_start_ts, tz=timezone.utc)
    creation_date_str = clip_start_utc.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    filter_complex = _build_filter_complex(
        out_w, out_h, video_meta.width, video_meta.height,
        use_cuda_overlay=use_cuda_overlay,
    )

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        *seek_args,
        *hw_decode_args,
        "-i", str(video_meta.path),
        "-f", "rawvideo",
        "-pix_fmt", "bgra",
        "-s", f"{out_w}x{out_h}",
        "-r", f"{fps_num}/{fps_den}",
        "-i", "pipe:0",
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a?",
        "-map", "0:d?",
        "-t", f"{trim_duration:.3f}",
        *encoder_args,
        "-c:a", "copy",
        "-c:d", "copy",
        "-map_metadata", "0",
        "-metadata", f"com.apple.quicktime.creationdate={creation_date_str}",
        "-movflags", "+faststart",
        str(output_path),
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    empty_frame_bytes = b"\x00" * (out_h * out_w * 4)

    for frame_idx in range(trim_frames):
        t = clip_start_ts + frame_idx / video_meta.fps
        frame = renderer.get_frame(t)
        if proc.stdin is None:
            break
        try:
            if frame is not None:
                proc.stdin.write(frame if isinstance(frame, bytes) else frame.tobytes())
            else:
                proc.stdin.write(empty_frame_bytes)
        except BrokenPipeError:
            break

    if proc.stdin:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
    ret = proc.wait()
    if ret != 0:
        output_path.unlink(missing_ok=True)
        print(f"ffmpeg exited {ret}", end=" ")


def _composite_session_laps(
    session: "VboSession",
    video_metas: List[VideoMetadata],
    output_dir: Path,
    font_path: str,
    stats_path: Path,
    overlay_width: Optional[int],
    gpu_caps: GpuCapabilities,
    gpu_accel: str = "auto",
) -> int:
    """Produce one output clip per lap for *session*, matched against *video_metas*.
    Returns the number of clips actually written."""
    times = session.data["time"]
    lat = session.data["lat"]
    lon = session.data["lng"]

    try:
        laps = detect_laps(times, lat, lon, session.start_line)
    except ValueError as exc:
        print(f"  no laps: {exc}")
        return 0

    compute_sectors(laps, lat, lon, times)

    if not laps:
        print("  no laps detected — skipped")
        return 0

    vbo_stem = Path(session.path).stem

    clips_written = 0
    laps_without_video = 0
    laps_without_video_details: List[str] = []
    videos_used_names: List[str] = []

    for lap in laps:
        lap_label = _lap_file_label(lap, laps)
        clip_start_ts = float(lap.start_time) - _LAP_CLIP_PAD
        clip_end_ts = float(lap.end_time) + _LAP_CLIP_PAD

        matching = [
            m for m in video_metas
            if m.start_utc.timestamp() < clip_end_ts
            and m.end_utc.timestamp() > clip_start_ts
        ]
        if not matching:
            laps_without_video += 1
            laps_without_video_details.append(lap_label)
            continue

        use_angle = len(matching) > 1

        for angle_idx, video_meta in enumerate(matching):
            if overlay_width is not None:
                out_w = overlay_width
                out_h = int(round(overlay_width * video_meta.height / video_meta.width))
            else:
                out_w = video_meta.width
                out_h = video_meta.height

            angle_suffix = f"-angle{angle_idx + 1}" if use_angle else ""
            out_name = f"{vbo_stem}-{lap_label}-{out_w}{angle_suffix}.mov"
            output_path = output_dir / out_name

            vid_start_ts = video_meta.start_utc.timestamp()
            vid_end_ts = video_meta.end_utc.timestamp()
            actual_start = max(clip_start_ts, vid_start_ts)
            actual_end = min(clip_end_ts, vid_end_ts)

            if actual_start >= actual_end:
                continue

            print(f"  {out_name} … ", end="", flush=True)

            config = RenderConfig(
                width=out_w, height=out_h, fps=video_meta.fps, font_path=font_path
            )
            try:
                renderer = _SessionRenderer(session, config, stats_path)
            except (ValueError, KeyError) as exc:
                print(f"failed ({exc})")
                continue

            if actual_start > renderer.t_start:
                renderer.skip_to(actual_start)

            _, enc_desc = _select_encoder_args(video_meta.codec, gpu_caps)
            print(f"[{enc_desc}] ", end="", flush=True)

            use_cuda = (
                gpu_accel != "off"
                and gpu_caps.has_cuda
                and video_meta.codec in ("hevc", "h265", "h264", "avc")
                and platform.system() != "Darwin"
            )

            _run_clip_ffmpeg(
                video_meta=video_meta,
                clip_start_ts=actual_start,
                clip_end_ts=actual_end,
                renderer=renderer,
                out_w=out_w,
                out_h=out_h,
                output_path=output_path,
                gpu_caps=gpu_caps,
                use_cuda_overlay=use_cuda,
            )
            print("done")
            clips_written += 1
            videos_used_names.append(video_meta.path.name)

    if laps_without_video:
        print(
            f"  {laps_without_video} lap(s) with no matching video: "
            f"{', '.join(laps_without_video_details)}"
        )
    used_unique = sorted(set(videos_used_names))
    print(
        f"  {clips_written} clip(s) written from "
        f"{len(used_unique)}/{len(video_metas)} video file(s)"
    )
    return clips_written, used_unique


def scan_and_composite(
    video_dir: Path,
    output_dir: Path,
    sessions: List[VboSession],
    font_path: str,
    stats_path: Path,
    overlay_width: Optional[int] = None,
    gpu_accel: str = "auto",
) -> None:
    """
    For each VBO session in *sessions*, produce one composited video clip per
    lap using any matching video files found in *video_dir*.

    Each clip spans the lap with ``_LAP_CLIP_PAD`` seconds of video padding
    before and after, clamped to the available footage.  When multiple camera
    angles cover the same lap the output files are suffixed ``-angle1``,
    ``-angle2``, etc.

    Files without a UTC creation timestamp (``com.apple.quicktime.creationdate``)
    are silently skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    video_extensions = {".mov", ".mp4"}
    video_files = sorted(
        f for f in video_dir.iterdir() if f.suffix.lower() in video_extensions
    )
    if not video_files:
        print(f"No video files (.mov/.mp4) found in {video_dir}")
        return

    print(f"\nFound {len(video_files)} video file(s) in {video_dir}")

    video_metas: List[VideoMetadata] = []
    for vf in video_files:
        meta = get_video_metadata(vf)
        if meta is None:
            print(f"  {vf.name}: no UTC timestamp — skipped")
        else:
            video_metas.append(meta)

    if not video_metas:
        print("No usable video files found")
        return

    video_time_range = (
        min(m.start_utc for m in video_metas),
        max(m.end_utc for m in video_metas),
    )
    print(
        f"  video coverage: {video_time_range[0]:%Y-%m-%d %H:%M:%S} → "
        f"{video_time_range[1]:%H:%M:%S} UTC"
    )

    session_errors: List[str] = []
    total_clips = 0
    used_videos: set[str] = set()

    gpu_caps = _detect_gpu_capabilities() if gpu_accel != "off" else GpuCapabilities()
    gpu_parts = []
    if gpu_caps.has_videotoolbox:
        gpu_parts.append("videotoolbox")
    if gpu_caps.has_nvenc:
        gpu_parts.append("nvenc")
    if gpu_caps.has_cuda:
        gpu_parts.append("cuda")
    gpu_info = ", ".join(gpu_parts) if gpu_parts else "none"
    print(f"  GPU acceleration: {gpu_info} (mode: {gpu_accel})")

    for session in sessions:
        print(f"\n{Path(session.path).name}")
        session_start = session.session_start_utc
        print(f"  session time (UTC): {session_start:%H:%M:%S}")
        try:
            clips, videos_used = _composite_session_laps(
                session, video_metas, output_dir, font_path, stats_path, overlay_width,
                gpu_caps=gpu_caps, gpu_accel=gpu_accel,
            )
            total_clips += clips
            used_videos.update(videos_used)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            session_errors.append(f"{Path(session.path).name}: {exc}")

    print(f"\n--- Summary ---")
    print(f"Sessions processed: {len(sessions) - len(session_errors)} / {len(sessions)}")
    print(f"Total clips written: {total_clips}")
    if session_errors:
        print(f"  Errors ({len(session_errors)}):")
        for err in session_errors:
            print(f"    {err}")

    unused = [m.path.name for m in video_metas if m.path.name not in used_videos]
    if unused:
        print(f"\n  {len(unused)} video file(s) produced no output:")
        for name in sorted(unused):
            print(f"    {name}")
