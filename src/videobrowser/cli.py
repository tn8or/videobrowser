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
from videobrowser.telemetry import DEFAULT_MIN_TRACK_KMH, STOPPED_KMH
from videobrowser.vehicles import DEFAULT_MAX_SHARE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="videobrowser",
        description="Find nearby riders, overtakes, and kneepuck moments in GoPro track-day video.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    scan = sub.add_parser(
        "scan",
        help="Analyse video files (events, clip windows, rider boxes) into scan.sqlite; cuts nothing",
    )
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
    scan.add_argument(
        "--follow-links",
        action="store_true",
        help="Include linked files that point outside the input folders",
    )
    scan.add_argument(
        "--date",
        action="append",
        default=[],
        metavar="YYYY-MM-DD",
        help="Only recordings from this day (repeatable)",
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
    scan.add_argument("--device", default="auto", help="auto | cuda | mps | cpu | coreml")
    scan.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    scan.add_argument(
        "--tensorrt",
        action="store_true",
        help="On cuda: run the detector as a TensorRT FP16 engine (built once next to the model)",
    )
    scan.add_argument("--force", action="store_true", help="Reprocess recordings already in the SQLite log")
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
        "--min-speed",
        type=float,
        default=STOPPED_KMH,
        help="Drop events below this GoPro GPS speed in km/h, i.e. standing still (0 disables). Default 10.",
    )
    scan.add_argument(
        "--max-vehicle-share",
        type=float,
        default=DEFAULT_MAX_SHARE,
        help="Drop events with a car, van or truck in view this share of the time, i.e. the paddock (0 disables). Default 0.4.",
    )
    scan.add_argument(
        "--max-clip-seconds",
        type=float,
        default=24.0,
        help="Max length of each clip window (default 24)",
    )
    scan.add_argument(
        "--max-clips",
        type=int,
        default=12,
        help="Max clip windows per recording (highest-scoring peaks, default 12)",
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
    # Clip options from when scan also cut clips; accepted so old command
    # lines keep working, and ignored.
    for flag in ("--no-clips", "--no-overlay", "--no-racebox", "--no-vertical", "--no-landscape", "--no-shorts"):
        scan.add_argument(flag, action="store_true", dest=f"old_{flag[2:].replace('-', '_')}", help=argparse.SUPPRESS)

    for name, what in (
        ("horizontals", "landscape clips under <out>/clips plus the RaceBox overlay renders"),
        ("verticals", "1080x1920 follow-cam clips under <out>/vertical"),
    ):
        cmd = sub.add_parser(name, help=f"Cut {what} from a finished scan")
        cmd.add_argument("-o", "--out", type=Path, required=True, help="Scan output directory")
        cmd.add_argument("--date", action="append", default=[], metavar="YYYY-MM-DD", help="Only this day (repeatable)")
        cmd.add_argument(
            "--jobs",
            type=int,
            default=None,
            help="Clips cut at once (default: CPU threads / 4, at most 4)",
        )
        cmd.add_argument(
            "--no-overlay",
            "--no-racebox",
            dest="no_overlay",
            action="store_true",
            help="No overlays at all (RaceBox, telemetry, date/time)",
        )
        cmd.add_argument("--force", action="store_true", help="Cut clips again even if the file exists")
        cmd.add_argument(
            "--min-speed",
            type=float,
            default=DEFAULT_MIN_TRACK_KMH,
            help="Leave out clips below this speed in km/h (standing still, paddock, pit lane), "
            "as the shorts do (0 disables). Default 40.",
        )
        cmd.add_argument(
            "--max-vehicle-share",
            type=float,
            default=DEFAULT_MAX_SHARE,
            help="Leave out clips with a car, van or truck in view this share of the time (0 disables). Default 0.4.",
        )
        cmd.add_argument(
            "--sessions-dir",
            type=Path,
            default=None,
            help="Directory of RaceBox .vbo sessions (default: <out>/sessions)",
        )
        cmd.add_argument(
            "--font",
            default="/System/Library/Fonts/Supplemental/Arial.ttf",
            help="TTF font used for overlay text",
        )
        if name == "horizontals":
            cmd.add_argument(
                "--stats",
                type=Path,
                default=None,
                help="Stats SQLite for lap bests (default: <out>/stats.sqlite)",
            )

    facing_cmd_p = sub.add_parser(
        "facing",
        help="Re-read which way each scanned camera looks (front/rear) from the picture, without a full scan",
    )
    facing_cmd_p.add_argument("-o", "--out", type=Path, required=True, help="Scan output directory")
    facing_cmd_p.add_argument("--date", action="append", default=[], metavar="YYYY-MM-DD", help="Only this day (repeatable)")
    facing_cmd_p.add_argument("--size", type=int, default=640, help="Max analysis side in pixels")
    facing_cmd_p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="CAMERA=front|rear",
        help="Fix the facing by hand for every recording whose id contains CAMERA, e.g. GX=front "
        "(repeatable; kept for later scans)",
    )
    facing_cmd_p.add_argument(
        "--clear",
        action="store_true",
        help="Forget every hand-set facing (for all days, also with --date) and measure again; "
        "combine with --set to start over",
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
    shorts.add_argument(
        "--min-speed",
        type=float,
        default=DEFAULT_MIN_TRACK_KMH,
        help="Track speed in km/h: slower moments (standing still, paddock, pit lane) are left out (0 disables). Default 40.",
    )
    shorts.add_argument(
        "--max-vehicle-share",
        type=float,
        default=DEFAULT_MAX_SHARE,
        help="Drop moments with a car, van or truck in view this share of the time, i.e. the paddock (0 disables). Default 0.4.",
    )
    shorts.add_argument("--dry-run", action="store_true", help="Write moments.json and index.html only")
    shorts.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Clips cut from the GoPro files at once (default: CPU threads / 4, at most 4)",
    )
    shorts.add_argument(
        "--sessions-dir",
        type=Path,
        default=None,
        help="RaceBox .vbo sessions for absolute time (default: <out>/sessions)",
    )

    sub.add_parser(
        "racebox-login",
        help="Log in to racebox.pro in Chrome (solve the captcha) and save the session for fetches",
    )
    rb_fetch = sub.add_parser(
        "racebox-fetch",
        help="Download new RaceBox .vbo sessions without scanning (scan does this itself)",
    )
    rb_fetch.add_argument("sessions_dir", type=Path, help="Folder to save .vbo files in")
    rb_fetch.add_argument("--headful", action="store_true", help="Show the browser window")
    return parser


def shorts_cmd(args: argparse.Namespace) -> int:
    from videobrowser.exports import default_jobs
    from videobrowser.shorts import build_shorts

    out = args.out.expanduser().resolve()
    if not (out / "scan.sqlite").exists():
        print(f"no scan.sqlite in {out}", file=sys.stderr)
        return 1
    build_shorts(
        out,
        inputs=[p.expanduser() for p in args.inputs],
        sessions_dir=(args.sessions_dir or (out / "sessions")).expanduser(),
        dates=args.date or None,
        max_seconds=args.short_seconds,
        max_action_shorts=args.max_action,
        audio_sync=not args.no_audio_sync,
        crashes=not args.no_crash,
        render=not args.dry_run,
        min_speed=args.min_speed,
        max_vehicle_share=args.max_vehicle_share,
        jobs=max(1, args.jobs) if args.jobs else default_jobs(),
    )
    return 0


def facing_cmd(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from videobrowser.exports import recording_from_meta
    from videobrowser.pipeline import facing_for_recording
    from videobrowser.probe import ProbeError
    from videobrowser.profiles import PROFILES, UNKNOWN

    out = args.out.expanduser().resolve()
    if not (out / "scan.sqlite").exists():
        print(f"no scan.sqlite in {out}", file=sys.stderr)
        return 1
    overrides = []
    for item in args.set:
        camera, _, way = item.partition("=")
        if not camera or way not in ("front", "rear"):
            print(f"--set wants CAMERA=front or CAMERA=rear, got {item!r}", file=sys.stderr)
            return 2
        overrides.append((camera, way))
    db = ScanDB.open(out / "scan.sqlite")
    try:
        if args.clear:
            db.clear_facing_overrides()
        for camera, way in overrides:
            db.set_facing_override(camera, way)
        for meta in db.recording_metas():
            if args.date and meta["date_label"] not in args.date:
                continue
            rid = meta["recording_id"]
            fixed = db.facing_override(rid)
            if fixed:
                print(f"{rid}: {fixed} (set by hand)")
                db.set_facing(rid, fixed)
                continue
            try:
                recording = recording_from_meta(meta)
            except (ProbeError, OSError) as exc:
                print(f"{rid}: source unavailable ({exc}); skipped")
                continue
            print(rid)
            profile = replace(PROFILES.get(meta["profile"], UNKNOWN), rotation=int(meta["rotation"]))
            db.set_facing(rid, facing_for_recording(recording, args.size, profile))
    finally:
        db.close()
    print("next: videobrowser verticals --force / shorts -o " + str(out))
    return 0


def exports_cmd(args: argparse.Namespace) -> int:
    from videobrowser.exports import default_jobs, export_horizontals, export_verticals

    out = args.out.expanduser().resolve()
    if not (out / "scan.sqlite").exists():
        print(f"no scan.sqlite in {out}", file=sys.stderr)
        return 1
    common = dict(
        dates=args.date or None,
        overlay=not args.no_overlay,
        sessions_dir=(args.sessions_dir or (out / "sessions")).expanduser(),
        font_path=args.font,
        force=args.force,
        jobs=max(1, args.jobs) if args.jobs else default_jobs(),
        min_speed=args.min_speed,
        max_vehicle_share=args.max_vehicle_share,
    )
    if args.cmd == "horizontals":
        export_horizontals(out, stats_path=args.stats, **common)
    else:
        export_verticals(out, **common)
    return 0


def scan_cmd(args: argparse.Namespace) -> int:
    inputs = [p.expanduser() for p in args.inputs]
    out = args.out.expanduser().resolve()
    paths = iter_video_paths(
        inputs,
        exclude_roots=[out],
        exclude_substrings=args.exclude,
        follow_links=args.follow_links,
    )
    if args.exclude:
        print("exclude: " + ", ".join(args.exclude))
    if not paths:
        print("no video files found", file=sys.stderr)
        return 1
    recordings = drop_duplicates(sort_by_start(group_recordings(paths, input_roots=inputs)))
    if args.date:
        recordings = [r for r in recordings if recording_day(r) in args.date]
        print("date: " + ", ".join(args.date))
        if not recordings:
            print("no recordings on " + ", ".join(args.date), file=sys.stderr)
            return 1
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
    tracker = RiderTracker(
        model_path=args.model, device=device, conf=args.conf, imgsz=args.size, tensorrt=args.tensorrt
    )
    if args.tensorrt and device == "cuda":
        tracker.load()
    print(f"detector: {args.model} on {device}{' (TensorRT)' if getattr(tracker, 'engine', False) else ''}")

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
        preview_limit=args.previews,
        lean_gate=args.lean_gate,
        max_seconds=args.max_seconds,
        min_motion=args.min_motion,
        min_speed=args.min_speed,
        max_vehicle_share=args.max_vehicle_share,
        max_clip_seconds=args.max_clip_seconds,
        max_clips=args.max_clips,
    )
    if any(v for k, v in vars(args).items() if k.startswith("old_")):
        print("note: scan no longer cuts clips; use horizontals, verticals and shorts")

    # RaceBox speed and turns stand in for footage without GoPro GPS.
    sessions_dir = (args.sessions_dir or (out / "sessions")).expanduser()
    if not args.no_fetch:
        try:
            from videobrowser.racebox_stage import fetch_sessions

            fetch_sessions(sessions_dir)
        except Exception as exc:
            print(f"racebox: fetch failed ({exc})")
    from videobrowser.racebox_align import load_track

    config.racebox_track = load_track(sessions_dir)

    all_events = []
    try:
        for recording in recordings:
            all_events.extend(scan_recording(recording, config, db, tracker))
    finally:
        db.close()

    (out / "events.json").write_text(json.dumps(events_to_json(all_events, clips=[]), indent=2))
    print(f"wrote {out / 'events.json'}  ({len(all_events)} events)")
    print("next: videobrowser horizontals / verticals / shorts -o " + str(out))
    return 0


def racebox_login_cmd(args: argparse.Namespace) -> int:
    from videobrowser.racebox import fetch

    fetch.interactive_login()
    return 0


def racebox_fetch_cmd(args: argparse.Namespace) -> int:
    from videobrowser.racebox_stage import fetch_sessions

    fetch_sessions(args.sessions_dir.expanduser().resolve(), headful=args.headful)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "scan":
        return scan_cmd(args)
    if args.cmd == "shorts":
        return shorts_cmd(args)
    if args.cmd == "facing":
        return facing_cmd(args)
    if args.cmd in ("horizontals", "verticals"):
        return exports_cmd(args)
    if args.cmd == "racebox-login":
        return racebox_login_cmd(args)
    if args.cmd == "racebox-fetch":
        return racebox_fetch_cmd(args)
    parser.error(f"unknown command {args.cmd}")
    return 2
