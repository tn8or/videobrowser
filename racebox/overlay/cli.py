from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple
from zoneinfo import ZoneInfo

import numpy as np

from .lap_timing import Lap, detect_corners, detect_laps, compute_sectors
from .overlay_render import (
    WHITE,
    GREEN,
    PURPLE,
)
from .stats import StatsStore, CornerBest
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
