#!../.venv/bin/python
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _ensure_import_path() -> None:
    root = Path(__file__).resolve().parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch render overlays for all VBO files in a directory"
    )
    parser.add_argument(
        "--input-dir", default=".", help="Directory containing .vbo files"
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Output directory for standalone .mov overlay files",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width. For standalone overlays defaults to 1920 (height locked to 16:9); for video compositing defaults to native video resolution.",
    )
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--font",
        default="/System/Library/Fonts/Supplemental/Arial.ttf",
        help="Path to TTF font",
    )
    parser.add_argument(
        "--stats", default="./stats.sqlite", help="Stats SQLite file path"
    )
    parser.add_argument(
        "--preview-30s",
        action="store_true",
        help="Render 30 seconds from the middle of each file",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only collect stats into the SQLite file (no rendering)",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Delete the stats DB before processing, forcing a full re-import of all sessions",
    )
    parser.add_argument(
        "--filter",
        default=None,
        metavar="PATTERN",
        help="Only process files whose name contains PATTERN (e.g. '02-08-2025')",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory of video files (e.g. GoPro .mov/.mp4). "
            "When provided, overlay is composited directly onto each video "
            "based on UTC timestamp matching and written to --outvideo-dir."
        ),
    )
    parser.add_argument(
        "--outvideo-dir",
        default="./outvideo",
        metavar="DIR",
        help="Output directory for composited video files (default: ./outvideo)",
    )
    parser.add_argument(
        "--gpu-accel",
        choices=["auto", "on", "off"],
        default="auto",
        help=(
            "GPU acceleration mode for video compositing: "
            "'auto' uses best available hardware, "
            "'on' requires GPU (CUDA overlay on NVIDIA), "
            "'off' uses software-only encoding/decoding (default: auto)"
        ),
    )

    args = parser.parse_args()

    _ensure_import_path()
    from overlay.cli import render_vbo, capture_stats
    from overlay.overlay_render import RenderConfig
    from overlay.vbo_parser import parse_vbo

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    stats_path = Path(args.stats)

    if args.reset_db and stats_path.exists():
        stats_path.unlink()
        print(f"Removed {stats_path} for fresh re-import")

    all_vbo_files = sorted(input_dir.glob("*.vbo"))
    if not all_vbo_files:
        raise SystemExit(f"No .vbo files found in {input_dir}")

    # Always capture stats for every session in the directory so historical
    # "best" comparisons work even when rendering is filtered to a subset.
    for vbo_file in all_vbo_files:
        try:
            capture_stats(vbo_file, stats_path)
        except ValueError as exc:
            print(f"Skipping stats for {vbo_file.name}: {exc}")

    if args.stats_only:
        print(f"Stats captured in {stats_path}")
        return

    # ------------------------------------------------------------------
    # Video compositing mode
    # ------------------------------------------------------------------
    if args.video_dir is not None:
        from overlay.video_compositor import scan_and_composite

        video_dir = Path(args.video_dir)
        outvideo_dir = Path(args.outvideo_dir)

        # Parse VBO sessions (apply --filter if given so only relevant
        # sessions are loaded, but use all sessions for stats accuracy).
        vbo_files_for_video = all_vbo_files
        if args.filter:
            vbo_files_for_video = [f for f in all_vbo_files if args.filter in f.name]
        if not vbo_files_for_video:
            raise SystemExit(
                f"No .vbo files matching '{args.filter}' found in {input_dir}"
            )

        print(f"Loading {len(vbo_files_for_video)} VBO session(s)…")
        sessions = []
        for vbo_file in vbo_files_for_video:
            try:
                sessions.append(parse_vbo(str(vbo_file)))
            except (ValueError, KeyError) as exc:
                print(f"  Skipping {vbo_file.name}: {exc}")

        scan_and_composite(
            video_dir=video_dir,
            output_dir=outvideo_dir,
            sessions=sessions,
            font_path=args.font,
            stats_path=stats_path,
            overlay_width=args.width,
            gpu_accel=args.gpu_accel,
        )
        return

    # ------------------------------------------------------------------
    # Standalone overlay rendering mode (original behaviour)
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    standalone_width = args.width if args.width is not None else 1920
    height = int(round(standalone_width * 9 / 16))
    config = RenderConfig(
        width=standalone_width, height=height, fps=args.fps, font_path=args.font
    )

    render_files = all_vbo_files
    if args.filter:
        render_files = [f for f in all_vbo_files if args.filter in f.name]
    if not render_files:
        raise SystemExit(f"No .vbo files matching '{args.filter}' found in {input_dir}")

    preview_seconds = 30 if args.preview_30s else None
    for vbo_file in render_files:
        render_vbo(
            vbo_file, output_dir, config, stats_path,
            preview_seconds=preview_seconds,
        )


if __name__ == "__main__":
    main()
