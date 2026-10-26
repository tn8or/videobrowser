from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TrackMap:
    points: np.ndarray  # Nx2 normalized (0..1)
    bounds: Tuple[float, float, float, float]  # min_x, min_y, max_x, max_y


def build_track_map(lat: np.ndarray, lon: np.ndarray, mask: np.ndarray) -> TrackMap:
    lats = lat[mask]
    lons = lon[mask]
    if lats.size == 0:
        return TrackMap(points=np.zeros((0, 2), dtype=np.float32), bounds=(0, 0, 1, 1))

    min_lat, max_lat = float(np.min(lats)), float(np.max(lats))
    min_lon, max_lon = float(np.min(lons)), float(np.max(lons))

    span_lat = max(max_lat - min_lat, 1e-9)
    span_lon = max(max_lon - min_lon, 1e-9)

    x = (lons - min_lon) / span_lon
    y = 1.0 - (lats - min_lat) / span_lat
    points = np.stack([x, y], axis=1).astype(np.float32)
    return TrackMap(points=points, bounds=(min_lon, min_lat, max_lon, max_lat))


def map_point(lat: float, lon: float, bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = bounds
    span_lat = max(max_lat - min_lat, 1e-9)
    span_lon = max(max_lon - min_lon, 1e-9)
    x = (lon - min_lon) / span_lon
    y = 1.0 - (lat - min_lat) / span_lat
    return float(x), float(y)
