from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from videobrowser.db import ScanDB
from videobrowser.events import events_to_json
from videobrowser.gopro import (
    drop_duplicates,
    group_recordings,
    iter_video_paths,
    recording_day,
    sort_by_start,
)
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
    scan.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Video files or folders (folders are searched recursively)",
    )
    scan.add_argument("-o", "--out", type=Path, required=True, help="Output directory")
    scan.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="TEXT",
        help=(
            "Skip any file or folder whose relative path contains TEXT "
            "(repeatable, case-insensitive substring)"
        ),
    )
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
    scan.add_argument(
        "--no-overlay",
        "--no-racebox",
        dest="no_overlay",
        action="store_true",
        help="No overlays at all: skip the RaceBox stage and the date/time on verticals",
    )
    scan.add_argument(
        "--no-vertical",
        action="store_true",
        help="Do not write 1080x1920 follow-cam copies of clips to <out>/vertical",
    )
    scan.add_argument(
        "--no-landscape",
        action="store_true",
        help="Do not write landscape clips or RaceBox overlay renders (verticals only)",
    )
    scan.add_argument(
        "--no-fetch",
        action="store_true",
        help="Do not auto-download RaceBox sessions before scanning",
    )
    scan.add_argument(
        "--sessions-dir",
        type=Path,
        default=None,
        help="Directory of RaceBox .vbo sessions (default: <out>/sessions)",
    )
    scan.add_argument(
        "--font",
        default="/System/Library/Fonts/Supplemental/Arial.ttf",
        help="TTF font used for overlay text",
    )
    scan.add_argument(
        "--stats",
        type=Path,
        default=None,
        help="Stats SQLite for lap bests (default: <out>/stats.sqlite)",
    )
    scan.add_argument(
        "--no-shorts",
        action="store_true",
        help="Do not build the per-day shorts under <out>/shorts after the scan",
    )

    shorts = sub.add_parser(
        "shorts",
        help="Build per-day vertical shorts (mixed action + themes) from a finished scan",
    )
    shorts.add_argument("-o", "--out", type=Path, required=True, help="Scan output directory")
    shorts.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Source folders; only needed for outputs scanned before shorts existed",
    )
    shorts.add_argument("--date", action="append", default=[], metavar="YYYY-MM-DD", help="Only this day (repeatable)")
    shorts.add_argument("--short-seconds", type=float, default=50.0, help="Max length of each short (default 50)")
    shorts.add_argument("--max-action", type=int, default=6, help="Max mixed action shorts per day (default 6)")
    shorts.add_argument("--no-audio-sync", action="store_true", help="Sync cameras by GPS only")
    shorts.add_argument("--no-crash", action="store_true", help="Skip crash detection from telemetry")
    shorts.add_argument("--dry-run", action="store_true", help="Write moments.json and index.html only")
    shorts.add_argument(
        "--sessions-dir",
        type=Path,
        default=None,
        help="RaceBox .vbo sessions for absolute time (default: <out>/sessions)",
    )
    return parser


def shorts_cmd(args: argparse.Namespace) -> int:
    from videobrowser.shorts import build_shorts

    out = args.out.expanduser().resolve()
    if not (out / "scan.sqlite").exists():
        print(f"no scan.sqlite in {out}", file=sys.stderr)
        return 1
    built = build_shorts(
        out,
        inputs=[p.expanduser() for p in args.inputs],
        sessions_dir=(args.sessions_dir or (out / "sessions")).expanduser(),
        dates=args.date or None,
        max_seconds=args.short_seconds,
        max_action_shorts=args.max_action,
        audio_sync=not args.no_audio_sync,
        crashes=not args.no_crash,
        render=not args.dry_run,
    )
    if built and not args.dry_run:
        db = ScanDB.open(out / "scan.sqlite")
        try:
            db.clear_shorts_pending(list(built))
        finally:
            db.close()
    return 0


def scan_cmd(args: argparse.Namespace) -> int:
    inputs = [p.expanduser() for p in args.inputs]
    out = args.out.expanduser().resolve()
    paths = iter_video_paths(
        inputs, exclude_roots=[out], exclude_substrings=args.exclude
    )
    if args.exclude:
        print("exclude: " + ", ".join(args.exclude))
    if not paths:
        print("no video files found", file=sys.stderr)
        return 1
    # Oldest first, so each day's shorts can be built as soon as its last
    # recording is scanned instead of after the whole library.
    recordings = drop_duplicates(sort_by_start(group_recordings(paths, input_roots=inputs)))
    camera_map = CameraMap.load(args.map) if args.map else CameraMap({}, {})
    if args.map:
        print(f"camera map override: {args.map}")

    days = {recording_day(r) for r in recordings}
    print(
        f"found {len(paths)} video(s) in {len(recordings)} recording(s) "
        f"over {len(days)} day(s), oldest first"
    )
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
        vertical=not args.no_vertical,
        landscape=not args.no_landscape,
        overlay=not args.no_overlay,
        font_path=args.font,
    )

    if not args.no_overlay:
        try:
            from videobrowser.racebox_stage import prepare

            print("racebox: setting up overlays…", flush=True)
            sessions_dir = (args.sessions_dir or (out / "sessions")).expanduser()
            stats_path = (args.stats or (out / "stats.sqlite")).expanduser()
            config.racebox = prepare(
                out_dir=out,
                sessions_dir=sessions_dir,
                font_path=args.font,
                stats_path=stats_path,
                do_fetch=not args.no_fetch,
            )
        except Exception as exc:
            print(f"racebox setup failed, continuing without overlays: {exc}")
            config.racebox = None

    shorts_on = not (args.no_shorts or args.no_vertical or args.no_clips)
    sessions_dir = (args.sessions_dir or (out / "sessions")).expanduser()
    shorts_failed = False

    def build_days(dates: list[str]) -> None:
        """Build shorts for *dates*; a day stays pending until its build finishes."""
        nonlocal shorts_failed
        if not (shorts_on and dates):
            return
        try:
            from videobrowser.shorts import build_shorts

            build_shorts(out, sessions_dir=sessions_dir, dates=dates)
        except Exception as exc:
            # Detection is already saved in scan.sqlite; the day stays pending
            # so the next scan retries it.
            print(f"shorts failed: {exc}", file=sys.stderr)
            shorts_failed = True
            return
        db.clear_shorts_pending(dates)

    def shorts_for(scanned: list[str]) -> None:
        """Build shorts for the days the just-finished recordings landed on."""
        dates = db.date_labels(scanned)
        db.add_shorts_pending(dates)
        build_days(dates)

    all_events = []
    try:
        current_day: str | None = None
        scanned: list[str] = []
        for recording in recordings:
            day = recording_day(recording)
            if day != current_day:
                shorts_for(scanned)
                current_day, scanned = day, []
            skipped = db.is_processed(recording.recording_id) and not config.force
            all_events.extend(scan_recording(recording, config, db, tracker))
            if not skipped:
                scanned.append(recording.recording_id)
        shorts_for(scanned)
        # Days left over from an earlier scan whose shorts failed or were off.
        build_days(db.shorts_pending())
    finally:
        db.close()

    payload = events_to_json(
        all_events,
        clips=[
            {"path": str(p)}
            for p in sorted((out / "clips").rglob("*.mp4"))
        ]
        if (out / "clips").exists()
        else [],
    )
    (out / "events.json").write_text(json.dumps(payload, indent=2))
    print(f"wrote {out / 'events.json'}  ({len(all_events)} events)")
    return 1 if shorts_failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "scan":
        return scan_cmd(args)
    if args.cmd == "shorts":
        return shorts_cmd(args)
    parser.error(f"unknown command {args.cmd}")
    return 2
