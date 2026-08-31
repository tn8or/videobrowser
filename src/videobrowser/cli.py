from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from videobrowser.db import ScanDB
from videobrowser.events import events_to_json
from videobrowser.gopro import group_recordings, iter_video_paths
from videobrowser.motion import DEFAULT_THRESHOLD
from videobrowser.pipeline import ScanConfig, pick_device, scan_recording
from videobrowser.profiles import CameraMap
from videobrowser.riders import RiderTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="videobrowser",
        description="Find nearby riders, overtakes, and kneepuck moments in GoPro track-day video.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser("scan", help="Scan video files and export highlight clips")
    scan.add_argument("inputs", nargs="+", type=Path, help="Video files or folders")
    scan.add_argument("-o", "--out", type=Path, required=True, help="Output directory")
    scan.add_argument("--fps", type=float, default=5.0, help="Analysis frame rate (default 5)")
    scan.add_argument("--size", type=int, default=640, help="Max analysis side in pixels")
    scan.add_argument("--pad", type=float, default=10.0, help="Seconds of pad around each event")
    scan.add_argument(
        "--profile",
        default="auto",
        help="auto (classify from frames) | cockpit | side | side_left | rear | rear_right | ground | unknown",
    )
    scan.add_argument(
        "--map",
        type=Path,
        default=None,
        help="Optional cameras.json override. Not inferred from GoPro filenames.",
    )
    scan.add_argument("--model", default="yolo11n.pt", help="Ultralytics model path or name")
    scan.add_argument("--device", default="auto", help="auto | mps | cpu | coreml")
    scan.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    scan.add_argument("--force", action="store_true", help="Reprocess recordings already in the SQLite log")
    scan.add_argument("--no-clips", action="store_true", help="Detect only; do not cut clips")
    scan.add_argument("--previews", type=int, default=8, help="Annotated preview JPEGs per recording")
    scan.add_argument(
        "--lean-gate",
        type=float,
        default=20.0,
        help="Min IMU lean degrees to run kneepuck (ignored if no GPMF)",
    )
    scan.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Only analyze the first N seconds of each recording (debug)",
    )
    scan.add_argument(
        "--min-motion",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Drop events while the camera is parked (0 disables). Default 4.",
    )
    scan.add_argument(
        "--max-clip-seconds",
        type=float,
        default=24.0,
        help="Max length of each exported clip (default 24)",
    )
    scan.add_argument(
        "--max-clips",
        type=int,
        default=12,
        help="Max clips per recording (highest-scoring peaks, default 12)",
    )
    return parser


def scan_cmd(args: argparse.Namespace) -> int:
    inputs = [p.expanduser() for p in args.inputs]
    paths = iter_video_paths(inputs)
    if not paths:
        print("no video files found", file=sys.stderr)
        return 1
    recordings = group_recordings(paths)
    camera_map = CameraMap.load(args.map) if args.map else CameraMap({}, {})
    if args.map:
        print(f"camera map override: {args.map}")

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    db = ScanDB.open(out / "scan.sqlite")
    device = pick_device(args.device)
    tracker = RiderTracker(model_path=args.model, device=device, conf=args.conf, imgsz=args.size)
    print(f"detector: {args.model} on {device}")

    config = ScanConfig(
        out_dir=out,
        fps=args.fps,
        max_side=args.size,
        pad=args.pad,
        profile=args.profile,
        camera_map=camera_map,
        model=args.model,
        device=device,
        conf=args.conf,
        force=args.force,
        no_clips=args.no_clips,
        preview_limit=args.previews,
        lean_gate=args.lean_gate,
        max_seconds=args.max_seconds,
        min_motion=args.min_motion,
        max_clip_seconds=args.max_clip_seconds,
        max_clips=args.max_clips,
    )

    all_events = []
    try:
        for recording in recordings:
            all_events.extend(scan_recording(recording, config, db, tracker))
    finally:
        db.close()

    payload = events_to_json(
        all_events,
        clips=[
            {"path": str(p)}
            for p in sorted((out / "clips").glob("*.mp4"))
        ]
        if (out / "clips").exists()
        else [],
    )
    (out / "events.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {out / 'events.json'}  ({len(all_events)} events)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "scan":
        return scan_cmd(args)
    parser.error(f"unknown command {args.cmd}")
    return 2
