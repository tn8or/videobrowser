"""Near-finished vertical shorts per day, built from a finished scan.

``videobrowser shorts -o <out>`` reads ``scan.sqlite``, puts every camera of a
day on one clock (``sync.py``), groups events into themed moments
(``themes.py``) and writes:

- ``shorts/YYYY-MM-DD/action-01.mp4`` ...: the day's best moments, several
  shorts of up to ``--short-seconds`` each, best moments first
- ``shorts/YYYY-MM-DD/overhalinger-01.mp4`` etc.: one series per theme
- ``shorts/YYYY-MM-DD/index.html``: the shorts plus every clip, by theme
- ``shorts/YYYY-MM-DD/moments.json``: what was found, for checking by hand

Each moment is cut from the vertical follow-cam clip that covers it (crop and
HUD included). A moment no vertical clip covers (a rear-camera hand-over that
did not make the clip list, an own crash) is cut from the source with a
centred 9:16 crop instead. A short label names the theme on each piece.
"""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from videobrowser.clips import _REENCODE_PIPELINES
from videobrowser.db import ScanDB
from videobrowser.events import ClipWindow
from videobrowser.probe import ProbeError, ffprobe_json, local_date_label, probe
from videobrowser.sync import CameraClock, ChapterRef, align
from videobrowser.themes import (
    LABELS,
    THEME_ORDER,
    CamEvent,
    Moment,
    Segment,
    crash_moment,
    detect_crashes,
    facing,
    find_moments,
)

OUT_W, OUT_H, OUT_FPS = 1080, 1920, 30
# Label shown at the start of each piece, seconds.
LABEL_S = 1.8
FILE_NAMES = {
    "action": "action",
    "overtake": "overhalinger",
    "overtaken": "overhalet",
    "catching_up": "indhentninger",
    "kneepuck": "knae",
    "crash": "styrt",
    "crash_other": "mulige-styrt",
    "off_track": "grus",
}
# Themes need this many moments for a short of their own (a crash or an
# off-track excursion needs one).
MIN_THEME_MOMENTS = {"crash": 1, "crash_other": 1, "off_track": 1}
_TZ = ZoneInfo("Europe/Copenhagen")


@dataclass
class ShortPlan:
    name: str
    theme: str
    moments: list[Moment]

    @property
    def duration(self) -> float:
        return sum(m.duration for m in self.moments)


@dataclass(frozen=True)
class ClipRow:
    recording_id: str
    t_start: float
    t_end: float
    types: str
    vertical_path: Path | None
    clip_path: Path | None


# --- planning --------------------------------------------------------------


def _pack(moments: list[Moment], max_seconds: float) -> list[list[Moment]]:
    """Greedy fill, in the given order, into lists of at most *max_seconds*."""
    shorts: list[list[Moment]] = []
    current: list[Moment] = []
    length = 0.0
    for m in moments:
        if current and length + m.duration > max_seconds:
            shorts.append(current)
            current, length = [], 0.0
        current.append(m)
        length += m.duration
    if current:
        shorts.append(current)
    return shorts


def plan_shorts(
    moments: list[Moment], max_seconds: float = 50.0, max_action_shorts: int = 6
) -> list[ShortPlan]:
    plans: list[ShortPlan] = []
    best_first = sorted(moments, key=lambda m: -m.score)
    for i, group in enumerate(_pack(best_first, max_seconds)[:max_action_shorts], start=1):
        group.sort(key=lambda m: m.utc_start)
        plans.append(ShortPlan(f"{FILE_NAMES['action']}-{i:02d}", "action", group))
    for theme in THEME_ORDER:
        themed = sorted((m for m in moments if m.theme == theme), key=lambda m: m.utc_start)
        if len(themed) < MIN_THEME_MOMENTS.get(theme, 2):
            continue
        for i, group in enumerate(_pack(themed, max_seconds), start=1):
            plans.append(ShortPlan(f"{FILE_NAMES[theme]}-{i:02d}", theme, group))
    return plans


# --- loading the scan ------------------------------------------------------

_NAME_RE = re.compile(r"^(?P<rid>.+)_(?P<start>\d{5})_(?P<types>[a-z_+]+)\.mp4$")


def _clips_from_folder(out_dir: Path, rid: str, date_label: str) -> list[ClipRow]:
    """Clip rows rebuilt from file names (scans from before the clips table)."""
    rows: list[ClipRow] = []
    vdir = out_dir / "vertical" / date_label
    cdir = out_dir / "clips" / date_label
    for path in sorted(vdir.glob(f"{rid}_*.mp4")) if vdir.exists() else []:
        match = _NAME_RE.match(path.name)
        if not match or match.group("rid") != rid:
            continue
        try:
            duration = probe(path).duration
        except ProbeError:
            continue
        start = float(match.group("start"))
        landscape = cdir / path.name
        rows.append(
            ClipRow(rid, start, start + duration, match.group("types"), path,
                    landscape if landscape.exists() else None)
        )
    return rows


def backfill_meta(db: ScanDB, inputs: list[Path], out_dir: Path) -> None:
    """Fill ``recording_meta`` for recordings scanned before it existed."""
    from videobrowser.gopro import group_recordings, iter_video_paths
    from videobrowser.telemetry import creation_utc_prefer_gps

    known = {row["recording_id"] for row in db.recording_metas()}
    profiles = {
        row["recording_id"]: row["profile"]
        for row in db.conn.execute("SELECT recording_id, profile FROM recordings")
    }
    paths = iter_video_paths(inputs, exclude_roots=[out_dir])
    for recording in group_recordings(paths, input_roots=inputs):
        rid = recording.recording_id
        if rid in known or rid not in profiles:
            continue
        info0 = recording.chapters[0].info
        start = creation_utc_prefer_gps(info0)
        db.save_recording_meta(
            rid,
            profiles[rid],
            info0.rotation,
            start,
            local_date_label(start),
            recording.duration,
            [
                {"path": str(ch.path), "offset": ch.offset, "duration": ch.duration}
                for ch in recording.chapters
            ],
        )
        print(f"  meta {rid}: backfilled from {recording.chapters[0].path.name}")


@dataclass
class DayData:
    date_label: str
    clocks: dict[str, CameraClock]
    profiles: dict[str, str]
    rotations: dict[str, int]
    events: list
    clips: list[ClipRow]


def load_days(db: ScanDB, out_dir: Path) -> dict[str, DayData]:
    days: dict[str, DayData] = {}
    for row in db.recording_metas():
        chapters = [
            ChapterRef(Path(c["path"]), float(c["offset"]), float(c["duration"]))
            for c in json.loads(row["chapters"])
        ]
        day = days.setdefault(row["date_label"], DayData(row["date_label"], {}, {}, {}, [], []))
        rid = row["recording_id"]
        day.clocks[rid] = CameraClock(rid, row["start_utc"], float(row["duration"]), chapters)
        day.profiles[rid] = row["profile"]
        day.rotations[rid] = int(row["rotation"])
    owner = {rid: day for day in days.values() for rid in day.clocks}
    for row in db.all_events():
        if row["recording_id"] in owner:
            owner[row["recording_id"]].events.append(row)
    with_rows: set[str] = set()
    for row in db.all_clips():
        day = owner.get(row["recording_id"])
        if day is None:
            continue
        with_rows.add(row["recording_id"])
        day.clips.append(
            ClipRow(
                row["recording_id"], row["t_start"], row["t_end"], row["types"],
                Path(row["vertical_path"]) if row["vertical_path"] else None,
                Path(row["clip_path"]) if row["clip_path"] else None,
            )
        )
    for rid, day in owner.items():
        if rid not in with_rows:
            day.clips.extend(_clips_from_folder(out_dir, rid, day.date_label))
    return days


def _crash_moments(day: DayData) -> list[Moment]:
    from videobrowser.telemetry import _walk_grav_accl, extract_gpmd_bytes, speed_series

    found: list[Moment] = []
    for rid, clock in day.clocks.items():
        if clock.start_utc is None:
            continue
        for chapter in clock.chapters:
            try:
                info = probe(chapter.path)
            except (ProbeError, OSError):
                continue
            if not info.has_gpmd or info.gpmd_index is None:
                continue
            blob = extract_gpmd_bytes(chapter.path, info.gpmd_index)
            if not blob:
                continue
            speed = speed_series(info, blob)
            _grav, accl, _stamps = _walk_grav_accl(blob)
            if not speed or not accl:
                continue
            accl_t = np.linspace(0.0, info.duration, len(accl))
            for t in detect_crashes(
                np.array([s.t for s in speed]),
                np.array([s.speed_kmh for s in speed]),
                accl_t,
                np.asarray(accl),
            ):
                print(f"  crash? {rid} t={chapter.offset + t:.1f}s")
                found.append(crash_moment(rid, chapter.offset + t, clock.start_utc))
    # Several cameras feel the same crash: keep one moment per incident.
    found.sort(key=lambda m: m.utc_start)
    merged: list[Moment] = []
    for m in found:
        if merged and m.utc_start - merged[-1].utc_start < 15.0:
            front = facing(day.profiles[m.segments[0].recording_id]) == "front"
            if front and facing(day.profiles[merged[-1].segments[0].recording_id]) != "front":
                merged[-1] = m
            continue
        merged.append(m)
    return merged


def day_moments(day: DayData, crashes: bool = True) -> list[Moment]:
    events = [
        CamEvent(
            recording_id=row["recording_id"],
            facing=facing(day.profiles[row["recording_id"]]),
            type=row["type"],
            t_start=float(row["t_start"]),
            t_end=float(row["t_end"]),
            score=float(row["score"]),
            utc_offset=float(day.clocks[row["recording_id"]].start_utc or 0.0),
        )
        for row in day.events
    ]
    moments = find_moments(events)
    if crashes:
        moments.extend(_crash_moments(day))
    return moments


# --- rendering -------------------------------------------------------------


def _clock_label(utc: float) -> str:
    return datetime.fromtimestamp(utc, tz=timezone.utc).astimezone(_TZ).strftime("%H:%M")


def render_label(text: str, sub: str, dest: Path) -> Path:
    """Transparent 1080x1920 PNG with a pill-shaped label below the HUD top row."""
    import cairo

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, OUT_W, OUT_H)
    ctx = cairo.Context(surface)
    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
    ctx.set_font_size(78)
    ext = ctx.text_extents(text)
    ctx.set_font_size(44)
    sub_ext = ctx.text_extents(sub)
    width = max(ext.width, sub_ext.width) + 120
    height = 210
    x = (OUT_W - width) / 2
    y = OUT_H * 0.25
    r = 40
    ctx.new_sub_path()
    ctx.arc(x + width - r, y + r, r, -np.pi / 2, 0)
    ctx.arc(x + width - r, y + height - r, r, 0, np.pi / 2)
    ctx.arc(x + r, y + height - r, r, np.pi / 2, np.pi)
    ctx.arc(x + r, y + r, r, np.pi, 3 * np.pi / 2)
    ctx.close_path()
    ctx.set_source_rgba(0, 0, 0, 0.6)
    ctx.fill()
    ctx.set_source_rgb(1, 1, 1)
    ctx.set_font_size(78)
    ctx.move_to((OUT_W - ext.width) / 2 - ext.x_bearing, y + 105)
    ctx.show_text(text)
    ctx.set_font_size(44)
    ctx.set_source_rgba(1, 1, 1, 0.8)
    ctx.move_to((OUT_W - sub_ext.width) / 2 - sub_ext.x_bearing, y + 175)
    ctx.show_text(sub)
    dest.parent.mkdir(parents=True, exist_ok=True)
    surface.write_to_png(str(dest))
    return dest


def _has_audio(path: Path) -> bool:
    try:
        data = ffprobe_json(path)
    except Exception:
        return False
    return any(s.get("codec_type") == "audio" for s in data.get("streams", []))


def _source_for(segment: Segment, clips: list[ClipRow]) -> tuple[Path, float, float] | None:
    """(vertical file, start inside it, duration) covering most of *segment*."""
    best: tuple[float, ClipRow] | None = None
    for clip in clips:
        if clip.recording_id != segment.recording_id or clip.vertical_path is None:
            continue
        if not clip.vertical_path.exists():
            continue
        inter = min(segment.t_end, clip.t_end) - max(segment.t_start, clip.t_start)
        if inter > 0 and (best is None or inter > best[0]):
            best = (inter, clip)
    if best is None or best[0] < 0.6 * segment.duration:
        return None
    clip = best[1]
    lo = max(segment.t_start, clip.t_start)
    hi = min(segment.t_end, clip.t_end)
    return clip.vertical_path, lo - clip.t_start, hi - lo


def _from_source(day: DayData, segment: Segment, dest: Path) -> Path:
    """Centred 9:16 cut straight from the GoPro files (no HUD)."""
    from videobrowser.gopro import Chapter, Recording
    from videobrowser.vertical import export_vertical

    clock = day.clocks[segment.recording_id]
    recording = Recording(segment.recording_id, "", segment.recording_id)
    for i, ref in enumerate(clock.chapters, start=1):
        info = probe(ref.path)
        recording.chapters.append(
            Chapter(ref.path, "", i, segment.recording_id, ref.duration, ref.offset, info)
        )
    end = min(segment.t_end, clock.duration)
    window = ClipWindow(segment.t_start, end, segment.recording_id, [], 0.0, [])
    return export_vertical(recording, window, dest, day.rotations[segment.recording_id], [])


def _encode_piece(src: Path, start: float, duration: float, label_png: Path, dest: Path) -> None:
    audio = _has_audio(src)
    inputs = ["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(src), "-loop", "1", "-i", str(label_png)]
    if not audio:
        inputs += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
    graph = (
        f"[0:v]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
        f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={OUT_FPS}[base];"
        f"[1:v]format=rgba[lab];"
        f"[base][lab]overlay=0:0:enable='lt(t,{LABEL_S})':shortest=1,format=yuv420p[vout]"
    )
    audio_map = ["-map", "0:a:0"] if audio else ["-map", "2:a:0"]
    last: RuntimeError | None = None
    for decode_args, encode_args in _REENCODE_PIPELINES:
        enc = ["20M" if a == "60M" else a for a in encode_args]
        cmd = (
            ["ffmpeg", "-hide_banner", "-y", "-loglevel", "error", *decode_args]
            + inputs
            + ["-filter_complex", graph, "-map", "[vout]", *audio_map]
            + ["-t", f"{duration:.3f}", *enc]
            + ["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]
            + ["-movflags", "+faststart", str(dest)]
        )
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return
        last = RuntimeError(proc.stderr.strip() or "ffmpeg failed")
        dest.unlink(missing_ok=True)
    raise last if last is not None else RuntimeError("no encode pipeline available")


def render_short(plan: ShortPlan, day: DayData, dest: Path) -> Path | None:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        pieces: list[Path] = []
        blank = render_label_blank(work / "blank.png")
        total = len(plan.moments)
        for i, moment in enumerate(plan.moments, start=1):
            sub = _clock_label(moment.utc_start)
            if plan.theme != "action":
                sub = f"{i}/{total}  ·  {sub}"
            label = render_label(moment.label, sub, work / f"label{i:02d}.png")
            for j, segment in enumerate(moment.segments):
                piece = work / f"p{i:02d}_{j}.mp4"
                found = _source_for(segment, day.clips)
                try:
                    if found is None:
                        src = _from_source(day, segment, work / f"src{i:02d}_{j}.mp4")
                        found = (src, 0.0, probe(src).duration)
                    path, start, duration = found
                    if duration < 0.5:
                        continue
                    # Label only on the first camera of a moment.
                    _encode_piece(path, start, duration, label if j == 0 else blank, piece)
                    pieces.append(piece)
                except Exception as exc:
                    print(f"  short {plan.name}: skip {segment.recording_id}@{segment.t_start:.0f}s ({exc})")
        if not pieces:
            return None
        listing = work / "list.txt"
        listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in pieces))
        dest.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(dest)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "concat failed")
    return dest


def render_label_blank(dest: Path) -> Path:
    import cairo

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, OUT_W, OUT_H)
    surface.write_to_png(str(dest))
    return dest


# --- index page ------------------------------------------------------------


def _moment_json(m: Moment, day: DayData) -> dict:
    return {
        "theme": m.theme,
        "label": m.label,
        "time": _clock_label(m.utc_start),
        "utc_start": round(m.utc_start, 2),
        "score": round(m.score, 3),
        "segments": [
            {
                "recording_id": s.recording_id,
                "camera": facing(day.profiles[s.recording_id]),
                "t_start": round(s.t_start, 2),
                "t_end": round(s.t_end, 2),
            }
            for s in m.segments
        ],
    }


def write_index(
    day_dir: Path, day: DayData, plans: list[ShortPlan], written: dict[str, Path]
) -> Path:
    def rel(path: Path) -> str:
        return html.escape(os.path.relpath(path, day_dir))

    def video(path: Path, caption: str) -> str:
        return (
            f'<figure><video src="{rel(path)}" controls preload="metadata"></video>'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )

    parts = [
        "<!doctype html><html lang=da><meta charset=utf-8>",
        "<meta name=viewport content='width=device-width,initial-scale=1'>",
        f"<title>Shorts {html.escape(day.date_label)}</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee}"
        "h1,h2{font-weight:600}.grid{display:flex;flex-wrap:wrap;gap:12px}"
        "figure{margin:0;width:200px}video{width:200px;aspect-ratio:9/16;background:#000;border-radius:8px}"
        ".wide video{aspect-ratio:16/9;width:320px}.wide{width:320px}"
        "figcaption{font-size:13px;opacity:.8;margin-top:4px}</style>",
        f"<h1>{html.escape(day.date_label)}</h1>",
    ]
    if written:
        parts.append("<h2>Shorts</h2><div class=grid>")
        for plan in plans:
            if plan.name in written:
                caption = f"{plan.name} · {len(plan.moments)} øjeblikke · {plan.duration:.0f}s"
                parts.append(video(written[plan.name], caption))
        parts.append("</div>")
    by_type: dict[str, list[ClipRow]] = {}
    for clip in sorted(day.clips, key=lambda c: (c.recording_id, c.t_start)):
        for kind in clip.types.split("+"):
            by_type.setdefault(kind, []).append(clip)
    names = {
        "approaching_rider": "Rytter nærmer sig",
        "receding_rider": "Rytter fjerner sig",
        "nearby_rider": "Rytter tæt på",
        "kneepuck_visible": "Knæ i asfalten",
        "rider_down": "Muligt styrt",
        "off_track": "Ude i gruset",
    }
    for kind, clips in by_type.items():
        parts.append(f"<h2>Alle klip: {html.escape(names.get(kind, kind))}</h2><div class=grid>")
        for clip in clips:
            clock = day.clocks.get(clip.recording_id)
            when = _clock_label(clock.utc(clip.t_start)) if clock and clock.start_utc else "?"
            cam = facing(day.profiles.get(clip.recording_id, ""))
            caption = f"{when} · {cam} · {clip.recording_id}"
            if clip.vertical_path and clip.vertical_path.exists():
                parts.append(video(clip.vertical_path, caption))
            elif clip.clip_path and clip.clip_path.exists():
                parts.append(f"<div class=wide>{video(clip.clip_path, caption)}</div>")
        parts.append("</div>")
    dest = day_dir / "index.html"
    dest.write_text("\n".join(parts))
    return dest


# --- entry point -----------------------------------------------------------


def _racebox_offsets(sessions_dir: Path | None):
    if sessions_dir is None or not sessions_dir.exists() or not any(sessions_dir.glob("*.vbo")):
        return None
    try:
        from videobrowser.racebox_stage import _gopro_session_offset, load_sessions
        from videobrowser.sync import _chapter_blob

        sessions = load_sessions(sessions_dir)
    except Exception as exc:
        print(f"shorts: RaceBox sessions unavailable ({exc})")
        return None
    if not sessions:
        return None

    def offset(clock: CameraClock) -> float | None:
        blob = _chapter_blob(clock.chapters[0]) if clock.chapters else None
        return _gopro_session_offset(blob, sessions) if blob else None

    return offset


def build_shorts(
    out_dir: Path,
    inputs: list[Path] | None = None,
    sessions_dir: Path | None = None,
    dates: list[str] | None = None,
    max_seconds: float = 50.0,
    max_action_shorts: int = 6,
    audio_sync: bool = True,
    crashes: bool = True,
    render: bool = True,
) -> dict[str, list[Path]]:
    db = ScanDB.open(out_dir / "scan.sqlite")
    try:
        if inputs:
            backfill_meta(db, inputs, out_dir)
        days = load_days(db, out_dir)
    finally:
        db.close()
    if not days:
        print("shorts: nothing scanned yet (or scanned before this version; pass the input folders)")
        return {}
    rb_offset = _racebox_offsets(sessions_dir)
    results: dict[str, list[Path]] = {}
    for label, day in sorted(days.items()):
        if dates and label not in dates:
            continue
        print(f"shorts {label}: {len(day.clocks)} camera recording(s), {len(day.events)} events")
        align(list(day.clocks.values()), racebox_offset=rb_offset, use_audio=audio_sync)
        moments = day_moments(day, crashes=crashes)
        plans = plan_shorts(moments, max_seconds, max_action_shorts)
        day_dir = out_dir / "shorts" / label
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "moments.json").write_text(
            json.dumps(
                {
                    "cameras": {
                        rid: {
                            "camera": facing(day.profiles[rid]),
                            "profile": day.profiles[rid],
                            "start_utc": c.start_utc,
                            "sync": c.source,
                        }
                        for rid, c in day.clocks.items()
                    },
                    "moments": [_moment_json(m, day) for m in sorted(moments, key=lambda m: m.utc_start)],
                    "shorts": [
                        {"name": p.name, "theme": p.theme, "seconds": round(p.duration, 1),
                         "moments": [_clock_label(m.utc_start) for m in p.moments]}
                        for p in plans
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        counts = {t: sum(1 for m in moments if m.theme == t) for t in LABELS}
        print("  moments: " + ", ".join(f"{t}={n}" for t, n in counts.items() if n))
        written: dict[str, Path] = {}
        if render:
            for plan in plans:
                # Render beside the old file and swap it in only once complete,
                # so a failed or interrupted run keeps the previous short.
                final = day_dir / f"{plan.name}.mp4"
                partial = day_dir / f".{plan.name}.partial.mp4"
                try:
                    path = render_short(plan, day, partial)
                except Exception as exc:
                    print(f"  short {plan.name} failed: {exc}")
                    partial.unlink(missing_ok=True)
                    continue
                if path is not None:
                    os.replace(partial, final)
                    written[plan.name] = final
                    print(f"  wrote {final.relative_to(out_dir)} ({plan.duration:.0f}s)")
            # Shorts this plan no longer produces (fewer moments than last run).
            names = {plan.name for plan in plans}
            for old in day_dir.glob("*.mp4"):
                if old.stem not in names and not old.name.startswith("."):
                    old.unlink()
        write_index(day_dir, day, plans, written)
        results[label] = list(written.values())
    return results
