#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

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
from overlay.track_map import TrackMap, build_track_map
from overlay.vbo_parser import parse_vbo


def _bytes_to_rgba(data: bytes, width: int, height: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))


def _build_frame_states(
    track_map: TrackMap, num_frames: int = 50
) -> List[FrameState]:
    states: List[FrameState] = []
    for i in range(num_frames):
        frac = i / max(num_frames - 1, 1)
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

        states.append(FrameState(
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
        ))
    return states


def _diff_frames(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float, int]:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    pixel_diff = np.sum(diff, axis=2)
    changed_pixels = int(np.count_nonzero(pixel_diff))
    max_diff = float(np.max(pixel_diff)) if pixel_diff.size else 0.0
    return pixel_diff, max_diff, changed_pixels


def _save_raw_as_pgm(path: Path, data: np.ndarray) -> None:
    height, width = data.shape[:2]
    gray = np.mean(data[:, :, :3].astype(np.float32), axis=2).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(f"P5\n{width} {height}\n255\n".encode())
        f.write(gray.tobytes())


def run_comparison(
    vbo_path: str,
    width: int = 1920,
    height: int = 1080,
    frames: int = 50,
    font_path: str = "/System/Library/Fonts/Supplemental/Arial.ttf",
    output_dir: Path | None = None,
) -> None:
    session = parse_vbo(vbo_path)
    lat = session.data["lat"]
    lon = session.data["lng"]
    active_mask = np.ones_like(lat, dtype=bool)
    track_map = build_track_map(lat, lon, active_mask)

    config = RenderConfig(
        width=width, height=height, fps=25, font_path=font_path
    )

    map_layer = build_map_layer(config, track_map)
    static_layer = build_static_layer(config, track_map, map_layer=map_layer)
    states = _build_frame_states(track_map, num_frames=frames)

    print(f"Resolution:  {width}x{height}")
    print(f"Frames:      {frames}")
    print(f"VBO:         {Path(vbo_path).name}")
    print()

    frames_no_cache: List[bytes] = []
    frames_with_static: List[bytes] = []
    frames_with_map_only: List[bytes] = []

    for s in states:
        frames_no_cache.append(render_frame(config, track_map, s))
        frames_with_static.append(
            render_frame(config, track_map, s, map_layer=map_layer, static_layer=static_layer)
        )
        frames_with_map_only.append(
            render_frame(config, track_map, s, map_layer=map_layer)
        )

    print("=" * 60)
    print("Comparison: no layers vs static_layer + map_layer")
    print("=" * 60)

    total_changed = 0
    total_pixels = width * height * frames
    max_pixel_diff_global = 0.0
    mismatched_frames: List[int] = []

    for i, (a_data, b_data) in enumerate(zip(frames_no_cache, frames_with_static)):
        a = _bytes_to_rgba(a_data, width, height)
        b = _bytes_to_rgba(b_data, width, height)
        pixel_diff, max_diff, changed = _diff_frames(a, b)
        total_changed += changed
        if max_diff > max_pixel_diff_global:
            max_pixel_diff_global = max_diff
        if changed > 0:
            mismatched_frames.append(i)

        if output_dir is not None and changed > 0:
            out_a = output_dir / f"frame_{i:04d}_no_cache.pgm"
            out_b = output_dir / f"frame_{i:04d}_static.pgm"
            out_d = output_dir / f"frame_{i:04d}_diff.png.pgm"
            _save_raw_as_pgm(out_a, a)
            _save_raw_as_pgm(out_b, b)
            if pixel_diff.max() > 0:
                diff_viz = np.clip(pixel_diff * 10, 0, 255).astype(np.uint8)
                diff_rgb = np.stack([diff_viz] * 3, axis=2)
                _save_raw_as_pgm(out_d, diff_rgb)

    pct_changed = 100.0 * total_changed / total_pixels if total_pixels > 0 else 0
    print(f"  Total pixels compared: {total_pixels:,}")
    print(f"  Changed pixels:        {total_changed:,} ({pct_changed:.4f}%)")
    print(f"  Max per-pixel diff:    {max_pixel_diff_global:.0f} (0-765 scale)")
    print(f"  Frames with any diff:  {len(mismatched_frames)}/{frames}")
    print()

    print("=" * 60)
    print("Comparison: static_layer vs map_layer only")
    print("=" * 60)

    total_changed_2 = 0
    mismatched_2: List[int] = []
    for i, (a_data, b_data) in enumerate(zip(frames_with_map_only, frames_with_static)):
        a = _bytes_to_rgba(a_data, width, height)
        b = _bytes_to_rgba(b_data, width, height)
        _, _, changed = _diff_frames(a, b)
        total_changed_2 += changed
        if changed > 0:
            mismatched_2.append(i)

    pct2 = 100.0 * total_changed_2 / total_pixels if total_pixels > 0 else 0
    print(f"  Changed pixels:        {total_changed_2:,} ({pct2:.4f}%)")
    print(f"  Frames with any diff:  {len(mismatched_2)}/{frames}")
    print()

    print("=" * 60)
    print("Timing: render methods")
    print("=" * 60)

    for s in states[:3]:
        render_frame(config, track_map, s)

    t0 = time.perf_counter()
    for s in states:
        render_frame(config, track_map, s)
    t_no = time.perf_counter() - t0

    t0 = time.perf_counter()
    for s in states:
        render_frame(config, track_map, s, map_layer=map_layer, static_layer=static_layer)
    t_static = time.perf_counter() - t0

    t0 = time.perf_counter()
    for s in states:
        render_frame(config, track_map, s, map_layer=map_layer)
    t_map = time.perf_counter() - t0

    print(f"  No layers:      {t_no:.3f}s ({t_no / frames * 1000:.2f} ms/frame)")
    print(f"  Static + map:   {t_static:.3f}s ({t_static / frames * 1000:.2f} ms/frame)")
    print(f"  Map only:       {t_map:.3f}s ({t_map / frames * 1000:.2f} ms/frame)")
    speedup = t_no / t_static if t_static > 0 else 0
    print(f"  Static speedup: {speedup:.2f}x vs no layers")

    if mismatched_frames:
        print(f"\n  NOTE: {len(mismatched_frames)} frames differ between no-cache and static-layer paths.")
        print("  This is EXPECTED — no-cache redraws all text backgrounds each frame,")
        print("  while static-layer uses pre-rendered backgrounds. Dynamic content")
        print("  (lap times, speed, corner data) is overlaid on both paths.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare render output between cached and non-cached overlay paths"
    )
    parser.add_argument("vbo", help="Path to a .vbo file")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument(
        "--font",
        default="/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Save frame diffs here for visual inspection",
    )
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    run_comparison(
        vbo_path=args.vbo,
        width=args.width,
        height=args.height,
        frames=args.frames,
        font_path=args.font,
        output_dir=out,
    )


if __name__ == "__main__":
    main()
