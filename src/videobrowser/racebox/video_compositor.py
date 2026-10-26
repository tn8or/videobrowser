"""
video_compositor.py
~~~~~~~~~~~~~~~~~~~
Per-session overlay rendering and ffmpeg encoder/GPU helpers used by
``videobrowser``'s RaceBox stage.

* VBO ``time`` column is already stored as UTC epoch seconds (see vbo_parser.py).
* ``_SessionRenderer`` looks up the VBO sample nearest a given UTC time and
  renders the overlay frame for it.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from zoneinfo import ZoneInfo

from .cli import (
    CornerEvent,
    _build_lap_distance,
    _color_for_corner,
    _corner_score,
    _format_color_for_lap,
    _get_day_bounds,
    _merge_corner_best,
)
from .lap_timing import Lap, compute_sectors, detect_laps
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
