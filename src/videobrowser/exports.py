"""Clips cut from a finished scan.

``videobrowser scan`` only analyses: it stores events, clip windows, rider
boxes and scene motion in ``scan.sqlite``. These commands cut from that:

- ``videobrowser horizontals -o <out>``: landscape clips under ``clips/``
  plus the RaceBox overlay renders (highlights, full sessions, fastest lap)
- ``videobrowser verticals -o <out>``: 1080x1920 follow-cam clips under
  ``vertical/``, the crop following the riders found by the scan

Windows where the bike stands still, rides below ``--min-speed`` or has a
car in view (the paddock) are left out, by the same rule the shorts use.

Overlays go only on recordings whose start time is sure (GoPro GPS time or
a clear RaceBox match); ``--no-overlay`` turns them off altogether. A clip
whose file already exists is kept unless ``--force``; files of windows a
re-scan dropped are removed.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from videobrowser.db import ScanDB
from videobrowser.events import ClipWindow, Event
from videobrowser.gopro import Chapter, Recording
from videobrowser.pipeline import _clip_name, turns_for_recording
from videobrowser.probe import ProbeError, probe
from videobrowser.riders import TrackPoint
from videobrowser.telemetry import DEFAULT_MIN_TRACK_KMH
from videobrowser.vehicles import DEFAULT_MAX_SHARE

DEFAULT_FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def recording_from_meta(row) -> Recording:
    """The recording a ``recording_meta`` row describes, its files probed."""
    rid = row["recording_id"]
    recording = Recording(rid, "", rid)
    for i, ref in enumerate(json.loads(row["chapters"]), start=1):
        path = Path(ref["path"])
        recording.chapters.append(
            Chapter(path, "", i, rid, float(ref["duration"]), float(ref["offset"]), probe(path))
        )
    return recording


# Start sources found on the RaceBox log: the scan's start is RaceBox time.
RACEBOX_PLACED = ("racebox-motion", "racebox-line")


def start_source(row, recording: Recording) -> str:
    """How the recording's start is known; scans from before this was stored
    count as sure only with GoPro telemetry."""
    keys = row.keys() if hasattr(row, "keys") else []
    if "start_source" in keys and row["start_source"]:
        return row["start_source"]
    return "gps" if any(ch.info.has_gpmd for ch in recording.chapters) else "clock"


def windows_for(db: ScanDB, rid: str) -> list[tuple[Any, ClipWindow]]:
    """(clips row, window) pairs, each window carrying the events inside it."""
    events = [
        Event(e["type"], e["t_start"], e["t_end"], e["score"], rid, e["track_id"])
        for e in db.events_for(rid)
    ]
    out = []
    for row in db.clips_for(rid):
        inside = [e for e in events if e.t_end >= row["t_start"] and e.t_start <= row["t_end"]]
        window = ClipWindow(
            row["t_start"], row["t_end"], rid, row["types"].split("+"), row["score"], inside
        )
        out.append((row, window))
    return out


def tracks_for(db: ScanDB, rid: str) -> dict[int, list[TrackPoint]]:
    tracks: dict[int, list[TrackPoint]] = {}
    for track_id, t, x1, y1, x2, y2, w, h in db.tracks(rid):
        point = TrackPoint(t, x1, y1, x2, y2, 1.0, 0, int(track_id), int(w), int(h))
        tracks.setdefault(point.track_id, []).append(point)
    return tracks


def _clear_others(directory: Path, rid: str, keep: set[str]) -> None:
    """Remove *rid*'s files in *directory* that no current window names."""
    if not directory.exists():
        return
    for path in directory.glob(f"{rid}_*.mp4"):
        if path.name not in keep:
            path.unlink(missing_ok=True)


def default_jobs() -> int:
    """Clips cut at once: ffmpeg does the work, so a few run side by side.

    About one per four CPU threads, at most 4 (consumer NVIDIA cards allow a
    handful of NVENC sessions at a time).
    """
    return max(1, min(4, (os.cpu_count() or 4) // 4))


def _run_jobs(
    items: Iterable[Any], work: Callable[[Any], Any], jobs: int
) -> Iterable[tuple[Any, Any, Exception | None]]:
    """(item, result, error) for each item, in order, *jobs* at a time.

    Callers handle the results on their own thread, so the SQLite
    connection is only used from there.
    """

    def _safe(item):
        try:
            return work(item), None
        except Exception as exc:
            return None, exc

    items = list(items)
    if jobs <= 1 or len(items) <= 1:
        for item in items:
            yield (item, *_safe(item))
        return
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        for item, (result, error) in zip(items, pool.map(_safe, items)):
            yield item, result, error


def _days(db: ScanDB, dates: list[str] | None):
    for row in db.recording_metas():
        if dates and row["date_label"] not in dates:
            continue
        yield row


def _track_days(
    db: ScanDB, out_dir: Path, dates: list[str] | None, sessions_dir: Path, min_speed: float, max_share: float
):
    """The scan's days with speed laid on (GoPro GPS, else RaceBox), for
    ``_on_track``; None when both checks are off."""
    if min_speed <= 0 and max_share <= 0:
        return None
    from videobrowser.racebox_align import load_track
    from videobrowser.shorts import load_days, place_without_gps

    days = load_days(db, out_dir)
    track = load_track(sessions_dir)
    for label, day in days.items():
        if not dates or label in dates:
            place_without_gps(day, track)
    return days


def _on_track(
    db: ScanDB, days, meta, pairs: list, column: str, min_speed: float, max_share: float
) -> list:
    """*pairs* without the windows the shorts would leave out (standing
    still, paddock); their file paths are cleared."""
    if days is None:
        return pairs
    from videobrowser.shorts import clip_drop_reason

    day = days[meta["date_label"]]
    kept, dropped = [], {}
    for row, window in pairs:
        why = clip_drop_reason(day, meta["recording_id"], window.t_start, window.t_end, min_speed, max_share)
        if why is None:
            kept.append((row, window))
            continue
        dropped[why] = dropped.get(why, 0) + 1
        db.set_clip_path(row["id"], column, None)
    if dropped:
        print(f"{meta['recording_id']}: left out " + ", ".join(f"{n} {why}" for why, n in dropped.items()))
    return kept


def _rating_records(days, meta, done: list, kind: str, min_speed: float) -> list:
    """Ratings-database rows (``ratings.ClipRecord``) for the clip windows
    cut for one recording; *done* is (window, file) pairs."""
    from videobrowser.ratings import ClipRecord
    from videobrowser.themes import Moment, Segment, facing

    rid = meta["recording_id"]
    day = days.get(meta["date_label"]) if days else None
    records = []
    for window, path in done:
        if day is not None and rid in day.clocks:
            from videobrowser.shorts import moment_record

            start = day.offset(rid)
            m = Moment(
                "clip", start + window.t_start, start + window.t_end, window.score,
                [Segment(rid, window.t_start, window.t_end)],
            )
            record = moment_record(m, day, [(kind, path)], min_speed)
            record.features["event_types"] = sorted(set(record.features["event_types"]) | set(window.types))
        else:
            looks = meta["facing"] if "facing" in meta.keys() and meta["facing"] else facing(meta["profile"])
            record = ClipRecord(
                date_label=meta["date_label"],
                recording_id=rid,
                recording_duration=float(meta["duration"]),
                theme="clip",
                t_start=window.t_start,
                t_end=window.t_end,
                utc_start=(meta["start_utc"] or 0.0) + window.t_start,
                camera=looks,
                features={
                    "theme": "clip",
                    "event_types": sorted(set(window.types)),
                    "duration": round(window.t_end - window.t_start, 2),
                    "cameras": [looks],
                    "kneepuck": "kneepuck" in window.types,
                },
                heuristic_score=window.score,
                files=[(kind, path)],
            )
        records.append(record)
    return records


def _register(
    days, meta, done: list, kind: str, folder: Path, min_speed: float, ratings: bool, verdicts=None
) -> None:
    if not ratings:
        return
    from videobrowser.ratings import register_all

    try:
        records = _rating_records(days, meta, done, kind, min_speed)
    except Exception as exc:  # a feature that fails must not stop the cutting
        print(f"  ratings: not saved ({exc})")
        return
    for record in records:
        why = (verdicts or {}).get(record.key)
        if why:  # kept to explore although the learned score rates it low
            record.why.append(why)
    register_all(records, forget=[(kind, folder, f"{meta['recording_id']}_")])


def _score_verdicts(db, days, metas, kind, min_speed, max_share, min_score, top, explore) -> dict | None:
    """(recording id, clips row id), and clip key, -> ``learn.gate`` verdict for every
    window the speed and car checks keep, per day; None when the score gate
    is off or there is no learned score yet."""
    if min_score <= 0 and top <= 0:
        return None
    import sqlite3

    from videobrowser.learn import current, gate
    from videobrowser.ratings import RatingsDB
    from videobrowser.shorts import clip_drop_reason

    try:
        rdb = RatingsDB.open()
        try:
            found = current(rdb)
        finally:
            rdb.close()
    except (sqlite3.Error, OSError):
        found = None
    if found is None:
        print("--min-score/--top: no learned score yet (videobrowser train); nothing skipped")
        return None
    model = found[1]
    per_day: dict[str, list] = {}
    ids: dict[str, tuple] = {}
    for meta in metas:
        rid = meta["recording_id"]
        day = days.get(meta["date_label"]) if days else None
        for row, window in windows_for(db, rid):
            if day is not None and clip_drop_reason(day, rid, window.t_start, window.t_end, min_speed, max_share):
                continue
            [record] = _rating_records(days, meta, [(window, None)], kind, min_speed)
            per_day.setdefault(meta["date_label"], []).append((record.key, model.predict(record.features), "clip"))
            ids[record.key] = (rid, row["id"])
    verdicts: dict[tuple, str | None] = {}
    for items in per_day.values():
        for key, why in gate(items, min_score, top, explore).items():
            verdicts[ids[key]] = why
            verdicts[key] = why  # by clip key too, for the ratings database
    return verdicts


def _skip_low(db, verdicts, meta, pairs: list, column: str) -> list:
    """*pairs* without the windows the learned score leaves out."""
    if verdicts is None:
        return pairs
    kept = []
    for row, window in pairs:
        why = verdicts.get((meta["recording_id"], row["id"]))
        if why is not None and why.startswith("left out"):
            db.set_clip_path(row["id"], column, None)
            continue
        kept.append((row, window))
    if len(kept) < len(pairs):
        print(f"{meta['recording_id']}: left out {len(pairs) - len(kept)} by the learned score")
    return kept


def _racebox_context(out_dir: Path, sessions_dir: Path, font_path: str, stats_path: Path | None):
    """Overlay context with the RaceBox sessions, or a session-less one."""
    from videobrowser import racebox_stage as rbs

    try:
        return rbs.prepare(
            out_dir=out_dir,
            sessions_dir=sessions_dir,
            font_path=font_path,
            stats_path=stats_path or (out_dir / "stats.sqlite"),
            do_fetch=False,
        )
    except Exception as exc:
        print(f"racebox: sessions unavailable ({exc}); overlays from GoPro telemetry only")
        return rbs.overlay_context(out_dir, font_path)


def export_horizontals(
    out_dir: Path,
    dates: list[str] | None = None,
    overlay: bool = True,
    sessions_dir: Path | None = None,
    font_path: str = DEFAULT_FONT,
    stats_path: Path | None = None,
    force: bool = False,
    jobs: int = 1,
    min_speed: float = DEFAULT_MIN_TRACK_KMH,
    max_vehicle_share: float = DEFAULT_MAX_SHARE,
    ratings: bool = True,
    min_score: float = 0.0,
    top: float = 0.0,
    explore: float = 0.1,
) -> int:
    """Cut landscape clips (and RaceBox overlay renders); returns how many
    were cut. *ratings*: register them in the local ratings database."""
    from videobrowser.clips import export_clip

    sessions_dir = sessions_dir or (out_dir / "sessions")
    ctx = None
    cut = 0
    db = ScanDB.open(out_dir / "scan.sqlite")
    try:
        days = _track_days(db, out_dir, dates, sessions_dir, min_speed, max_vehicle_share)
        metas = list(_days(db, dates))
        verdicts = _score_verdicts(
            db, days, metas, "horizontal", min_speed, max_vehicle_share, min_score, top, explore
        )
        for i, meta in enumerate(metas, 1):
            rid = meta["recording_id"]
            at = f"[file {i} of {len(metas)}]"
            pairs = windows_for(db, rid)
            if not pairs:
                continue
            try:
                recording = recording_from_meta(meta)
            except (ProbeError, OSError) as exc:
                print(f"{at} {rid}: source unavailable ({exc}); skipped")
                continue
            pairs = _on_track(db, days, meta, pairs, "clip_path", min_speed, max_vehicle_share)
            pairs = _skip_low(db, verdicts, meta, pairs, "clip_path")
            clip_dir = out_dir / "clips" / meta["date_label"]
            _clear_others(clip_dir, rid, {_clip_name(w) for _, w in pairs})
            todo = []
            done = []
            for row, window in pairs:
                dest = clip_dir / _clip_name(window)
                if force or not dest.exists():
                    todo.append((row, window))
                else:
                    db.set_clip_path(row["id"], "clip_path", str(dest))
                    done.append((window, dest))

            def cut_one(item, recording=recording, clip_dir=clip_dir, rotation=meta["rotation"]):
                _, window = item
                return export_clip(recording, window, clip_dir / _clip_name(window), rotation=rotation)

            new = 0
            for (row, window), dest, exc in _run_jobs(todo, cut_one, jobs):
                if exc is not None:
                    print(f"  clip failed {_clip_name(window)}: {exc}")
                    db.set_clip_path(row["id"], "clip_path", None)
                    continue
                db.set_clip_path(row["id"], "clip_path", str(dest))
                done.append((window, dest))
                new += 1
            cut += new
            print(f"{at} {rid}: {new} landscape clip(s) cut, {len(pairs) - new} kept")
            _register(days, meta, done, "horizontal", clip_dir, min_speed, ratings, verdicts)
            if not overlay:
                continue
            source = start_source(meta, recording)
            if source == "clock":
                print("  overlays skipped: no sure RaceBox match for this recording")
                continue
            if ctx is None:
                ctx = _racebox_context(out_dir, sessions_dir, font_path, stats_path)
                ctx.keep_existing = not force
                ctx.jobs = jobs
            try:
                from videobrowser.racebox_stage import run_for_recording

                start = meta["start_utc"] if source in RACEBOX_PLACED else None
                run_for_recording(recording, [w for _, w in pairs], meta["rotation"], ctx, start)
            except Exception as exc:
                print(f"  racebox overlays failed: {exc}")
    finally:
        db.close()
    return cut


def export_verticals(
    out_dir: Path,
    dates: list[str] | None = None,
    overlay: bool = True,
    sessions_dir: Path | None = None,
    font_path: str = DEFAULT_FONT,
    force: bool = False,
    jobs: int = 1,
    min_speed: float = DEFAULT_MIN_TRACK_KMH,
    max_vehicle_share: float = DEFAULT_MAX_SHARE,
    ratings: bool = True,
    min_score: float = 0.0,
    top: float = 0.0,
    explore: float = 0.1,
) -> int:
    """Cut 1080x1920 follow-cam clips; returns how many were cut.
    *ratings*: register them in the local ratings database."""
    from videobrowser.racebox_align import load_track
    from videobrowser.themes import facing
    from videobrowser.vertical import export_vertical, focus_samples

    sessions_dir = sessions_dir or (out_dir / "sessions")
    ctx = None
    track = None
    track_loaded = False
    cut = 0
    db = ScanDB.open(out_dir / "scan.sqlite")
    try:
        days = _track_days(db, out_dir, dates, sessions_dir, min_speed, max_vehicle_share)
        riding = db.riding()
        metas = list(_days(db, dates))
        verdicts = _score_verdicts(
            db, days, metas, "vertical", min_speed, max_vehicle_share, min_score, top, explore
        )
        for i, meta in enumerate(metas, 1):
            rid = meta["recording_id"]
            at = f"[file {i} of {len(metas)}]"
            pairs = windows_for(db, rid)
            if not pairs:
                continue
            try:
                recording = recording_from_meta(meta)
            except (ProbeError, OSError) as exc:
                print(f"{at} {rid}: source unavailable ({exc}); skipped")
                continue
            pairs = _on_track(db, days, meta, pairs, "vertical_path", min_speed, max_vehicle_share)
            pairs = _skip_low(db, verdicts, meta, pairs, "vertical_path")
            vdir = out_dir / "vertical" / meta["date_label"]
            _clear_others(vdir, rid, {_clip_name(w) for _, w in pairs})
            todo = []
            done = []
            for row, window in pairs:
                if force or not (vdir / _clip_name(window)).exists():
                    todo.append((row, window))
                else:
                    db.set_clip_path(row["id"], "vertical_path", str(vdir / _clip_name(window)))
                    done.append((window, vdir / _clip_name(window)))
            if not todo:
                print(f"{at} {rid}: {len(pairs)} vertical clip(s) kept")
                _register(days, meta, done, "vertical", vdir, min_speed, ratings, verdicts)
                continue

            source = start_source(meta, recording)
            if source == "gps":
                turns = turns_for_recording(recording)
            elif source in RACEBOX_PLACED:
                if not track_loaded:
                    track, track_loaded = load_track(sessions_dir), True
                turns = track.turns(meta["start_utc"], recording.duration) if track else []
            else:
                turns = []
            if not riding.get(rid):
                # No part of the bike in view: a helmet camera (or one in the
                # hand). The head already looks into the corner, so the crop
                # stays on the riders instead of turning further.
                turns = []
            tracks = tracks_for(db, rid)
            looks = meta["facing"] if "facing" in meta.keys() and meta["facing"] else facing(meta["profile"])

            factory_for = None
            if overlay and source != "clock":
                from videobrowser import racebox_stage as rbs

                if ctx is None:
                    ctx = _racebox_context(out_dir, sessions_dir, font_path, None)
                start = meta["start_utc"] if source in RACEBOX_PLACED else None
                if start is None:
                    start = rbs.recording_start_utc(recording, ctx)

                def factory_for(window, start=start):
                    return lambda w, h, fps: rbs.overlay_source(
                        ctx, recording, window, w, h, fps, start, "vertical"
                    )
            elif overlay:
                print("  overlays skipped: no sure RaceBox match for this recording")

            def cut_one(
                item,
                recording=recording,
                vdir=vdir,
                rotation=meta["rotation"],
                tracks=tracks,
                turns=turns,
                looks=looks,
                factory_for=factory_for,
            ):
                _, window = item
                return export_vertical(
                    recording,
                    window,
                    vdir / _clip_name(window),
                    rotation,
                    focus_samples(window, tracks, turns, looks),
                    overlay_factory=factory_for(window) if factory_for else None,
                )

            new = 0
            for (row, window), dest, exc in _run_jobs(todo, cut_one, jobs):
                if exc is not None:
                    print(f"  vertical failed {_clip_name(window)}: {exc}")
                    db.set_clip_path(row["id"], "vertical_path", None)
                    continue
                db.set_clip_path(row["id"], "vertical_path", str(dest))
                done.append((window, dest))
                new += 1
            cut += new
            print(f"{at} {rid}: {new} vertical clip(s) cut, {len(pairs) - len(todo)} kept")
            _register(days, meta, done, "vertical", vdir, min_speed, ratings, verdicts)
    finally:
        db.close()
    return cut
