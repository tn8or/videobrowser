#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from overlay.overlay_render import (
    Color,
    CornerDisplay,
    FrameState,
    LapRow,
    RenderConfig,
    build_map_layer,
    build_static_layer,
    render_frame,
    WHITE,
    GREEN,
    PURPLE,
)
from overlay.track_map import TrackMap
from overlay.vbo_parser import parse_vbo

import numpy as np


def _build_frame_states(
    track_map: TrackMap, num_variants: int = 10
) -> List[FrameState]:
    states: List[FrameState] = []
    for i in range(num_variants):
        frac = i / max(num_variants - 1, 1)
        lean = -45.0 + 90.0 * frac
        speed = 50.0 + 250.0 * frac
        lap_number = i + 1
        lap_time = 60.0 + i * 0.3

        corner_display = CornerDisplay(
            pre_corner_max_speed=180.0 + i * 2,
            entry_speed=100.0 + i * 3,
            min_speed=55.0 + i,
            max_lean=42.0 + i * 0.5,
            best_pre_corner_max_speed=190.0 + i,
            best_entry_speed=110.0 + i * 2,
            best_min_speed=60.0 + i,
            best_max_lean=45.0 + i * 0.3,
            color=[WHITE, GREEN, PURPLE][i % 3],
        )

        last_laps = [
            LapRow(
                lap_number=max(1, lap_number - j),
                lap_time=lap_time - j * 0.2,
                sector_times=(18.0, 22.0, 20.0),
                color=[WHITE, GREEN, PURPLE][j % 3],
            )
            for j in range(4)
        ]

        px = 0.2 + 0.6 * frac
        py = 0.3 + 0.4 * math.sin(frac * math.pi * 2)

        state = FrameState(
            lap_number=lap_number,
            lap_time=lap_time,
            on_track_faster=frac > 0.5,
            last_laps=last_laps,
            speed_kmh=speed,
            best_speed_kmh=speed + 5.0,
            corner_display=corner_display if i % 2 == 0 else None,
            lean_angle=lean,
            map_pos=(px, py),
            is_in_lap=(i % 5 == 4),
        )
        states.append(state)

    return states


def _build_empty_states(num_variants: int = 10) -> List[FrameState]:
    states: List[FrameState] = []
    for i in range(num_variants):
        state = FrameState(
            lap_number=None,
            lap_time=None,
            on_track_faster=None,
            last_laps=[],
            speed_kmh=0.0,
            best_speed_kmh=None,
            corner_display=None,
            lean_angle=0.0,
            map_pos=None,
            is_in_lap=False,
        )
        states.append(state)
    return states


def run_benchmark(
    vbo_path: str,
    width: int = 1920,
    height: int = 1080,
    fps: int = 25,
    frames: int = 250,
    font_path: str = "/System/Library/Fonts/Supplemental/Arial.ttf",
) -> None:
    session = parse_vbo(vbo_path)
    lat = session.data["lat"]
    lon = session.data["lng"]

    active_mask = np.ones_like(lat, dtype=bool)
    from overlay.track_map import build_track_map
    track_map = build_track_map(lat, lon, active_mask)

    config = RenderConfig(
        width=width, height=height, fps=fps, font_path=font_path
    )

    map_layer = build_map_layer(config, track_map)
    states = _build_frame_states(track_map, num_variants=frames)

    print(f"Resolution:    {width}×{height}")
    print(f"FPS target:    {fps}")
    print(f"Frames:        {frames}")
    print(f"VBO:           {Path(vbo_path).name}")
    print(f"Font:          {Path(font_path).name}")
    print()

    budget = 1000.0 / fps

    for s in states[:5]:
        render_frame(config, track_map, s, map_layer=map_layer)

    t0 = time.perf_counter()
    for s in states:
        render_frame(config, track_map, s, map_layer=map_layer)
    elapsed = time.perf_counter() - t0

    ms_per_frame = elapsed / frames * 1000
    effective_fps = frames / elapsed

    print(f"{'Without static_layer:':<20} {elapsed:8.3f}s  total")
    print(f"{'':20} {ms_per_frame:8.3f} ms/frame")
    print(f"{'':20} {effective_fps:8.1f} fps")
    print(f"{'':20} {budget:8.1f} ms/frame budget @{fps}fps")
    print(f"{'':20} {'OK' if ms_per_frame < budget else 'OVER BUDGET':>10}")
    print()

    static_surf = build_static_layer(config, track_map, map_layer=map_layer)
    for s in states[:5]:
        render_frame(config, track_map, s, map_layer=map_layer, static_layer=static_surf)

    t0s = time.perf_counter()
    for s in states:
        render_frame(config, track_map, s, map_layer=map_layer, static_layer=static_surf)
    elapsed_static = time.perf_counter() - t0s

    ms_static = elapsed_static / frames * 1000
    speedup = elapsed / elapsed_static if elapsed_static > 0 else 0
    static_fps = frames / elapsed_static

    print(f"{'With static_layer:':<20} {elapsed_static:8.3f}s  total")
    print(f"{'':20} {ms_static:8.3f} ms/frame")
    print(f"{'':20} {static_fps:8.1f} fps")
    print(f"{'':20} {speedup:8.2f}x speedup")
    print()

    t1 = time.perf_counter()
    for s in states:
        render_frame(config, track_map, s)
    elapsed_no_map = time.perf_counter() - t1
    ms_no_map = elapsed_no_map / frames * 1000
    print(f"{'Without map_layer:':<20} {elapsed_no_map:8.3f}s  total")
    print(f"{'':20} {ms_no_map:8.3f} ms/frame")
    print()

    empty_states = _build_empty_states(frames)
    t2 = time.perf_counter()
    for s in empty_states:
        render_frame(config, track_map, s)
    elapsed_empty = time.perf_counter() - t2
    ms_empty = elapsed_empty / frames * 1000
    print(f"{'Minimal state:':<20} {elapsed_empty:8.3f}s  total")
    print(f"{'':20} {ms_empty:8.3f} ms/frame")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark overlay render_frame()")
    parser.add_argument("vbo", help="Path to a .vbo file")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument(
        "--font",
        default="/System/Library/Fonts/Supplemental/Arial.ttf",
        help="Path to TTF font",
    )
    args = parser.parse_args()

    run_benchmark(
        vbo_path=args.vbo,
        width=args.width,
        height=args.height,
        fps=args.fps,
        frames=args.frames,
        font_path=args.font,
    )


if __name__ == "__main__":
    main()
