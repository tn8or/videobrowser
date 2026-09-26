from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math

import numpy as np


@dataclass
class Lap:
    index: int
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    lap_time: float
    is_out_lap: bool
    is_in_lap: bool
    sector_times: Tuple[float, float, float]


@dataclass
class CornerMetrics:
    corner_index: int
    pre_corner_max_speed: float
    entry_speed: float
    min_speed: float
    max_lean: float
    start_idx: int
    end_idx: int


@dataclass
class LapCorners:
    lap_index: int
    corners: List[CornerMetrics]


def _segments_intersect(a, b, c, d) -> bool:
    def ccw(p1, p2, p3):
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def detect_laps(
    times: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    start_line: Tuple[float, float, float, float] | None,
) -> List[Lap]:
    if start_line is None:
        raise ValueError("Start line not found in VBO header")

    lat1, lon1, lat2, lon2 = start_line
    crossings: List[int] = []

    for i in range(1, len(lat)):
        a = (lon[i - 1], lat[i - 1])
        b = (lon[i], lat[i])
        c = (lon1, lat1)
        d = (lon2, lat2)
        if _segments_intersect(a, b, c, d):
            crossings.append(i)

    laps: List[Lap] = []
    if not crossings:
        return laps

    # Out-lap: data before first crossing
    if crossings[0] > 0:
        laps.append(
            Lap(
                index=0,
                start_idx=0,
                end_idx=crossings[0],
                start_time=times[0],
                end_time=times[crossings[0]],
                lap_time=times[crossings[0]] - times[0],
                is_out_lap=True,
                is_in_lap=False,
                sector_times=(0.0, 0.0, 0.0),
            )
        )

    for lap_idx in range(len(crossings)):
        start_idx = crossings[lap_idx]
        end_idx = (
            crossings[lap_idx + 1] if lap_idx + 1 < len(crossings) else len(times) - 1
        )
        is_in = lap_idx == len(crossings) - 1 and end_idx == len(times) - 1
        lap_time = times[end_idx] - times[start_idx]
        laps.append(
            Lap(
                index=lap_idx + 1,
                start_idx=start_idx,
                end_idx=end_idx,
                start_time=times[start_idx],
                end_time=times[end_idx],
                lap_time=lap_time,
                is_out_lap=False,
                is_in_lap=is_in,
                sector_times=(0.0, 0.0, 0.0),
            )
        )

    return laps


def _haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def compute_sectors(
    laps: List[Lap], lat: np.ndarray, lon: np.ndarray, times: np.ndarray
) -> None:
    for lap in laps:
        if lap.is_out_lap or lap.is_in_lap:
            lap.sector_times = (0.0, 0.0, 0.0)
            continue

        dist = [0.0]
        for i in range(lap.start_idx + 1, lap.end_idx + 1):
            d = _haversine(lat[i - 1], lon[i - 1], lat[i], lon[i])
            dist.append(dist[-1] + d)
        dist = np.array(dist)
        total = dist[-1] if len(dist) else 0.0
        if total <= 0:
            lap.sector_times = (0.0, 0.0, 0.0)
            continue

        sector_marks = [total / 3, 2 * total / 3, total]
        sector_times = []
        last_idx = 0
        for mark in sector_marks:
            idx = int(np.searchsorted(dist, mark))
            idx = min(max(idx, last_idx + 1), len(dist) - 1)
            t = times[lap.start_idx + idx] - times[lap.start_idx + last_idx]
            sector_times.append(t)
            last_idx = idx
        lap.sector_times = (sector_times[0], sector_times[1], sector_times[2])


def detect_corners(
    laps: List[Lap],
    times: np.ndarray,
    speed: np.ndarray,
    long_acc: np.ndarray,
    lean: np.ndarray,
) -> List[LapCorners]:
    positive_acc = long_acc[long_acc > 0]
    negative_acc = long_acc[long_acc < 0]
    accel_threshold = np.percentile(positive_acc, 60) if positive_acc.size else 0.4
    brake_threshold = np.percentile(-negative_acc, 60) if negative_acc.size else 0.4
    lean_abs = np.abs(lean)
    lean_threshold = max(5.0, float(np.percentile(lean_abs, 60)))

    results: List[LapCorners] = []

    for lap in laps:
        corners: List[CornerMetrics] = []
        if lap.is_out_lap or lap.is_in_lap:
            results.append(LapCorners(lap_index=lap.index, corners=corners))
            continue

        state = "straight"
        corner_idx = 0
        pre_corner_max_speed = 0.0
        entry_speed = 0.0
        min_speed = float("inf")
        max_lean = 0.0
        corner_start_idx = lap.start_idx

        for i in range(lap.start_idx + 1, lap.end_idx + 1):
            la = long_acc[i]
            ln = abs(lean[i])

            if state == "straight" and la < -brake_threshold:
                state = "braking"
                window_start = max(lap.start_idx, i - 25)
                pre_corner_max_speed = float(np.max(speed[window_start : i + 1]))
                corner_start_idx = i

            elif state in ("braking", "straight") and ln >= lean_threshold:
                if state == "straight":
                    # No braking detected before lean — compute pre-corner speed from window
                    window_start = max(lap.start_idx, i - 25)
                    pre_corner_max_speed = float(np.max(speed[window_start : i + 1]))
                    corner_start_idx = i
                state = "corner"
                entry_speed = float(speed[i])
                min_speed = float(speed[i])
                max_lean = float(abs(lean[i]))

            elif state == "corner":
                min_speed = min(min_speed, float(speed[i]))
                max_lean = max(max_lean, float(abs(lean[i])))
                # Exit via positive long_acc, or via lean returning to straight (handles zero long_acc sessions)
                if la > accel_threshold or (
                    ln < lean_threshold and float(speed[i]) > min_speed
                ):
                    corners.append(
                        CornerMetrics(
                            corner_index=corner_idx,
                            pre_corner_max_speed=pre_corner_max_speed,
                            entry_speed=entry_speed,
                            min_speed=min_speed,
                            max_lean=max_lean,
                            start_idx=corner_start_idx,
                            end_idx=i,
                        )
                    )
                    corner_idx += 1
                    state = "straight"

        results.append(LapCorners(lap_index=lap.index, corners=corners))

    return results
