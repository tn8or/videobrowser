from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from .lap_timing import Lap, detect_corners, detect_laps, compute_sectors
from .overlay_render import (
    RenderConfig,
    FrameState,
    LapRow,
    CornerDisplay,
    build_map_layer,
    build_static_layer,
    render_frame,
    WHITE,
    GREEN,
    PURPLE,
)
from .stats import StatsStore, CornerBest
from .track_map import build_track_map, map_point
from .vbo_parser import parse_vbo


@dataclass
class CornerEvent:
    end_time: float
    corner_index: int
    pre_corner_max_speed: float
    entry_speed: float
    min_speed: float
    max_lean: float
    best_pre_corner_max_speed: float | None = None
    best_entry_speed: float | None = None
    best_min_speed: float | None = None
    best_max_lean: float | None = None
    color: Tuple[int, int, int, int] | None = None


def _format_color_for_lap(
    lap_time: float,
    best_day: float | None,
    best_overall: float | None,
) -> Tuple[int, int, int, int]:
    if best_overall is not None and lap_time <= best_overall:
        return PURPLE
    if best_day is not None and lap_time <= best_day:
        return GREEN
    return WHITE


def _corner_score(pre: float, entry: float, minimum: float, max_lean: float) -> float:
    return pre + entry + minimum + (max_lean * 0.5)


def _color_for_corner(
    score: float, best_day: float | None, best_overall: float | None
) -> Tuple[int, int, int, int]:
    if best_overall is not None and score >= best_overall:
        return PURPLE
    if best_day is not None and score >= best_day:
        return GREEN
    return WHITE


def _merge_corner_best(existing: CornerBest | None, event: CornerEvent) -> CornerBest:
    if existing is None:
        return CornerBest(
            pre_corner_max_speed=event.pre_corner_max_speed,
            entry_speed=event.entry_speed,
            min_speed=event.min_speed,
            max_lean=event.max_lean,
        )
    return CornerBest(
        pre_corner_max_speed=max(
            existing.pre_corner_max_speed or 0.0, event.pre_corner_max_speed
        ),
        entry_speed=max(existing.entry_speed or 0.0, event.entry_speed),
        min_speed=max(existing.min_speed or 0.0, event.min_speed),
        max_lean=max(existing.max_lean or 0.0, event.max_lean),
    )


def _ensure_ffmpeg() -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        raise RuntimeError("ffmpeg is required but not available in PATH") from exc


def _compute_distances(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    # Cumulative distance using equirectangular approximation
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    dist = np.zeros_like(lat_rad)
    for i in range(1, len(lat_rad)):
        x = (lon_rad[i] - lon_rad[i - 1]) * np.cos((lat_rad[i] + lat_rad[i - 1]) / 2)
        y = lat_rad[i] - lat_rad[i - 1]
        dist[i] = dist[i - 1] + (np.sqrt(x * x + y * y) * 6371000.0)
    return dist


def _build_lap_distance(
    lap: Lap, lat: np.ndarray, lon: np.ndarray
) -> Tuple[np.ndarray, float]:
    lap_lat = lat[lap.start_idx : lap.end_idx + 1]
    lap_lon = lon[lap.start_idx : lap.end_idx + 1]
    dist = _compute_distances(lap_lat, lap_lon)
    total = float(dist[-1]) if dist.size else 0.0
    return dist, total


def _get_day_bounds(session_start_utc: datetime, tz: ZoneInfo) -> Tuple[float, float]:
    local = session_start_utc.astimezone(tz)
    day_start_local = datetime(local.year, local.month, local.day, tzinfo=tz)
    day_end_local = day_start_local.replace(
        hour=23, minute=59, second=59, microsecond=999000
    )
    return (
        day_start_local.astimezone(timezone.utc).timestamp(),
        day_end_local.astimezone(timezone.utc).timestamp(),
    )


def ingest_session_stats(session, stats: StatsStore) -> None:
    """Import laps from an already-parsed VBO session into *stats*."""
    times = session.data["time"]
    lat = session.data["lat"]
    lon = session.data["lng"]
    speed = session.data.get("velocity")
    if speed is None:
        speed = session.data.get("velocity_kmh")
    if speed is None:
        raise ValueError("Velocity column not found in VBO data")
    lean = session.data.get("lean-angle")
    long_acc = session.data.get("LongAcc")

    if lean is None:
        lean = np.zeros_like(times)
    if long_acc is None:
        long_acc = np.zeros_like(times)

    laps = detect_laps(times, lat, lon, session.start_line)
    compute_sectors(laps, lat, lon, times)
    lap_corners = detect_corners(laps, times, speed, long_acc, lean)

    if stats.session_exists(session.track_name, session.session_start_utc):
        return
    session_id = stats.create_session(
        session.track_name, session.session_start_utc, session.path
    )
    for lap in laps:
        lap_id = stats.insert_lap(
            session_id,
            lap.index,
            lap.lap_time,
            lap.end_time,
            lap.is_out_lap,
            lap.is_in_lap,
            lap.sector_times,
        )
        lap_corner = next((lc for lc in lap_corners if lc.lap_index == lap.index), None)
        if lap_corner:
            for corner in lap_corner.corners:
                stats.insert_corner(
                    lap_id,
                    corner.corner_index,
                    corner.pre_corner_max_speed,
                    corner.entry_speed,
                    corner.min_speed,
                    corner.max_lean,
                )


def capture_stats(path: Path, stats_path: Path) -> None:
    session = parse_vbo(str(path))
    with StatsStore(stats_path) as stats:
        ingest_session_stats(session, stats)


def render_vbo(
    path: Path,
    output_dir: Path,
    config: RenderConfig,
    stats_path: Path,
    preview_seconds: int | None = None,
) -> None:
    session = parse_vbo(str(path))
    times = session.data["time"]
    lat = session.data["lat"]
    lon = session.data["lng"]
    speed = session.data.get("velocity")
    if speed is None:
        speed = session.data.get("velocity_kmh")
    if speed is None:
        raise ValueError("Velocity column not found in VBO data")
    heading = session.data.get("heading")
    lean = session.data.get("lean-angle")
    long_acc = session.data.get("LongAcc")

    if lean is None:
        lean = np.zeros_like(times)
    if long_acc is None:
        long_acc = np.zeros_like(times)

    laps = detect_laps(times, lat, lon, session.start_line)
    compute_sectors(laps, lat, lon, times)
    lap_corners = detect_corners(laps, times, speed, long_acc, lean)

    active_mask = np.zeros_like(lat, dtype=bool)
    for lap in laps:
        if lap.is_out_lap or lap.is_in_lap:
            continue
        active_mask[lap.start_idx : lap.end_idx + 1] = True

    track_map = build_track_map(lat, lon, active_mask)

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

        # Precompute lap distances for on-track comparison
        lap_distances: Dict[int, Tuple[np.ndarray, float]] = {}
        for lap in laps:
            if lap.is_out_lap or lap.is_in_lap:
                continue
            lap_distances[lap.index] = _build_lap_distance(lap, lat, lon)

        # Corner events and bests
        corner_events: List[CornerEvent] = []
        corner_best_overall: Dict[int, float] = {}
        corner_best_day: Dict[int, float] = {}
        corner_best_values_overall: Dict[int, CornerBest] = {}
        corner_best_values_day: Dict[int, CornerBest] = {}
        for lap in lap_corners:
            for corner in lap.corners:
                corner_events.append(
                    CornerEvent(
                        end_time=times[corner.end_idx],
                        corner_index=corner.corner_index,
                        pre_corner_max_speed=corner.pre_corner_max_speed,
                        entry_speed=corner.entry_speed,
                        min_speed=corner.min_speed,
                        max_lean=corner.max_lean,
                    )
                )
                if corner.corner_index not in corner_best_overall:
                    best = stats.fetch_corner_best(
                        session.track_name,
                        corner.corner_index,
                        session.session_start_utc.timestamp(),
                    )
                    if best.pre_corner_max_speed is not None:
                        corner_best_overall[corner.corner_index] = _corner_score(
                            best.pre_corner_max_speed,
                            best.entry_speed,
                            best.min_speed,
                            best.max_lean,
                        )
                        corner_best_values_overall[corner.corner_index] = best
                    best_day = stats.fetch_corner_best_day(
                        session.track_name, corner.corner_index, day_start_utc, day_end_utc
                    )
                    if best_day.pre_corner_max_speed is not None:
                        corner_best_day[corner.corner_index] = _corner_score(
                            best_day.pre_corner_max_speed,
                            best_day.entry_speed,
                            best_day.min_speed,
                            best_day.max_lean,
                        )
                        corner_best_values_day[corner.corner_index] = best_day

    corner_events.sort(key=lambda c: c.end_time)

    map_layer = build_map_layer(config, track_map)
    static_layer = build_static_layer(config, track_map, map_layer=map_layer)

    # Prepare ffmpeg pipe
    _ensure_ffmpeg()
    output_path = output_dir / f"{path.stem}.mov"

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgra",
        "-s",
        f"{config.width}x{config.height}",
        "-r",
        str(config.fps),
        "-i",
        "-",
        "-c:v",
        "prores_ks",
        "-profile:v",
        "4",
        "-pix_fmt",
        "yuva444p10le",
        str(output_path),
    ]

    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    frame_start = times[0]
    frame_end = times[-1]
    if preview_seconds is not None and preview_seconds > 0:
        mid = (frame_start + frame_end) / 2
        half = preview_seconds / 2
        frame_start = max(frame_start, mid - half)
        frame_end = min(frame_end, mid + half)
    total_frames = int((frame_end - frame_start) * config.fps) + 1

    lap_pointer = 0
    corner_ptr = 0
    completed_laps: List[LapRow] = []
    last_completed_lap: Lap | None = None

    # For in-session updates
    current_best_overall = best_overall_time
    current_best_day = best_day_time
    best_speed_kmh: float | None = None

    for frame_idx in range(total_frames):
        t = frame_start + frame_idx / config.fps
        sample_idx = int(np.searchsorted(times, t, side="right") - 1)
        sample_idx = max(0, min(sample_idx, len(times) - 1))

        # Update completed laps list
        while lap_pointer < len(laps) and times[laps[lap_pointer].end_idx] <= t:
            lap = laps[lap_pointer]
            if not lap.is_out_lap and not lap.is_in_lap:
                lap_color = _format_color_for_lap(
                    lap.lap_time, current_best_day, current_best_overall
                )
                completed_laps.append(
                    LapRow(
                        lap_number=lap.index,
                        lap_time=lap.lap_time,
                        sector_times=lap.sector_times,
                        color=lap_color,
                    )
                )
                if current_best_overall is None or lap.lap_time <= current_best_overall:
                    current_best_overall = lap.lap_time
                if current_best_day is None or lap.lap_time <= current_best_day:
                    current_best_day = lap.lap_time
                last_completed_lap = lap
            lap_pointer += 1

        while (
            corner_ptr < len(corner_events) and corner_events[corner_ptr].end_time <= t
        ):
            event = corner_events[corner_ptr]
            score = _corner_score(
                event.pre_corner_max_speed,
                event.entry_speed,
                event.min_speed,
                event.max_lean,
            )
            best_overall = corner_best_overall.get(event.corner_index)
            best_day = corner_best_day.get(event.corner_index)
            event.color = _color_for_corner(score, best_day, best_overall)
            if best_overall is None or score >= best_overall:
                corner_best_overall[event.corner_index] = score
            if best_day is None or score >= best_day:
                corner_best_day[event.corner_index] = score
            corner_best_values_overall[event.corner_index] = _merge_corner_best(
                corner_best_values_overall.get(event.corner_index), event
            )
            corner_best_values_day[event.corner_index] = _merge_corner_best(
                corner_best_values_day.get(event.corner_index), event
            )
            corner_ptr += 1

        # Determine current lap
        current_lap: Lap | None = None
        for lap in laps:
            if lap.start_idx <= sample_idx <= lap.end_idx:
                current_lap = lap
                break

        lap_number: int | None = None
        lap_time: float | None = None
        on_track_faster: bool | None = None

        if current_lap is None or current_lap.is_out_lap:
            lap_number = None
            lap_time = None
            on_track_faster = None
        elif current_lap.is_in_lap:
            if last_completed_lap is not None:
                lap_number = last_completed_lap.index
                lap_time = last_completed_lap.lap_time
            on_track_faster = None
        else:
            lap_number = current_lap.index
            lap_time = t - current_lap.start_time
            if current_best_overall is not None and lap_number in lap_distances:
                dist, total = lap_distances[lap_number]
                local_idx = sample_idx - current_lap.start_idx
                local_idx = min(max(local_idx, 0), len(dist) - 1)
                fraction = dist[local_idx] / total if total > 0 else 0.0
                predicted = current_best_overall * fraction
                on_track_faster = lap_time <= predicted if predicted > 0 else None
            if best_speed_kmh is None or speed[sample_idx] > best_speed_kmh:
                best_speed_kmh = float(speed[sample_idx])

        last_laps = list(reversed(completed_laps[-4:]))

        # Current corner (last completed)
        last_corner_event = None
        for event in reversed(corner_events[:corner_ptr]):
            if event.end_time <= t:
                last_corner_event = event
                break

        corner_display = None
        if last_corner_event is not None and last_corner_event.color is not None:
            best_vals = corner_best_values_overall.get(last_corner_event.corner_index)
            best_pre = (
                best_vals.pre_corner_max_speed
                if best_vals
                else last_corner_event.pre_corner_max_speed
            )
            best_entry = (
                best_vals.entry_speed if best_vals else last_corner_event.entry_speed
            )
            best_min = best_vals.min_speed if best_vals else last_corner_event.min_speed
            best_lean = best_vals.max_lean if best_vals else last_corner_event.max_lean
            corner_display = CornerDisplay(
                pre_corner_max_speed=last_corner_event.pre_corner_max_speed,
                entry_speed=last_corner_event.entry_speed,
                min_speed=last_corner_event.min_speed,
                max_lean=last_corner_event.max_lean,
                best_pre_corner_max_speed=best_pre,
                best_entry_speed=best_entry,
                best_min_speed=best_min,
                best_max_lean=best_lean,
                color=last_corner_event.color,
            )

        map_pos = None
        if not (current_lap and (current_lap.is_out_lap or current_lap.is_in_lap)):
            map_pos = map_point(
                float(lat[sample_idx]), float(lon[sample_idx]), track_map.bounds
            )

        state = FrameState(
            lap_number=lap_number,
            lap_time=lap_time,
            on_track_faster=on_track_faster,
            last_laps=last_laps,
            speed_kmh=float(speed[sample_idx]),
            best_speed_kmh=best_speed_kmh,
            corner_display=corner_display if getattr(config, "show_corners", False) else None,
            lean_angle=float(lean[sample_idx]),
            map_pos=map_pos,
            is_in_lap=current_lap is not None and current_lap.is_in_lap,
            wall_time_utc=float(t),
        )

        frame = render_frame(config, track_map, state, map_layer=map_layer, static_layer=static_layer)
        if proc.stdin is None:
            raise RuntimeError("ffmpeg pipe not available")
        proc.stdin.write(frame)

    if proc.stdin:
        proc.stdin.close()
    proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RaceBox overlays from VBO files"
    )
    parser.add_argument(
        "--input", nargs="+", required=True, help="One or more .vbo files"
    )
    parser.add_argument(
        "--output-dir", default="./output", help="Output directory for .mov files"
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--font", required=True, help="Path to TTF font")
    parser.add_argument(
        "--stats", default="./stats.sqlite", help="Stats SQLite file path"
    )
    parser.add_argument(
        "--preview-30s",
        action="store_true",
        help="Render 30 seconds from the middle of the file",
    )
    parser.add_argument(
        "--gpu-accel",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "GPU acceleration mode: "
            "'auto' uses best available hardware, "
            "'on' requires GPU, "
            "'off' uses software-only path (default: auto)"
        ),
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = RenderConfig(
        width=args.width, height=args.height, fps=args.fps, font_path=args.font
    )
    stats_path = Path(args.stats)

    preview_seconds = 30 if args.preview_30s else None
    for input_path in args.input:
        render_vbo(
            Path(input_path),
            output_dir,
            config,
            stats_path,
            preview_seconds=preview_seconds,
        )


if __name__ == "__main__":
    main()
