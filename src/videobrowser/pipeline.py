from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from tqdm import tqdm

from videobrowser.clips import export_clip
from videobrowser.db import ScanDB
from videobrowser.decode import FrameDecoder, decode_one_frame
from videobrowser.events import (
    SPEED_EXEMPT,
    ClipWindow,
    Event,
    clip_windows,
    dumps_extra,
    merge_same_type,
)
from videobrowser.gopro import Chapter, Recording, chapter_at
from videobrowser.incidents import offtrack_fraction, score_incidents
from videobrowser.motion import DEFAULT_THRESHOLD, event_is_moving, scene_motion
from videobrowser.preview import write_previews
from videobrowser.profiles import CameraMap, Profile, Rect, apply_ego_mask, guess_profile_from_samples, resolve_profile
from videobrowser.probe import local_date_label
from videobrowser.puck import detect_puck
from videobrowser.riders import RiderTracker, TrackPoint, score_tracks
from videobrowser.telemetry import (
    STOPPED_KMH,
    LeanSample,
    SpeedSample,
    TurnSample,
    creation_utc_prefer_gps,
    lean_series,
    leaned,
    on_track,
    speed_series,
    turn_series,
)


@dataclass
class ScanConfig:
    out_dir: Path
    fps: float = 5.0
    max_side: int = 640
    pad: float = 10.0
    profile: str = "auto"
    camera_map: CameraMap = field(default_factory=lambda: CameraMap({}, {}))
    model: str = "yolo11n.pt"
    device: str = "auto"
    conf: float = 0.25
    force: bool = False
    no_clips: bool = False
    preview_limit: int = 8
    lean_gate: float = 20.0
    max_seconds: float | None = None
    min_motion: float = DEFAULT_THRESHOLD
    # Drop events below this GPS speed (standing still); 0 disables. The
    # shorts apply the stricter track speed, so paddock clips stay available.
    min_speed: float = STOPPED_KMH
    max_clip_seconds: float = 24.0
    max_clips: int = 12
    # Optional RaceboxContext; when set, overlay clips are produced after cuts.
    racebox: Any = None
    # 1080x1920 follow-cam copies of each clip under ``vertical/``.
    vertical: bool = True
    # Landscape clips under ``clips/`` plus the RaceBox overlay renders.
    landscape: bool = True
    # Date/time/telemetry overlay on vertical clips (off with ``--no-overlay``).
    overlay: bool = True
    font_path: str = "/System/Library/Fonts/Supplemental/Arial.ttf"


def pick_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _sample_times(recording: Recording) -> list[tuple[Chapter, float]]:
    """(chapter, local time) spread over the whole recording, all chapters.

    Times used to be taken from the recording's total duration but decoded from
    the first chapter, so on chaptered recordings the later samples fell past
    its end and only the first minutes (often the pits) decided the rotation.
    """
    duration = recording.duration or recording.chapters[0].duration or 0.0
    # Odd count so rotation and mount votes cannot tie; several so frames from
    # straights outvote ones leaned deep into a corner.
    fractions = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    picks = []
    for frac in fractions:
        t = duration * frac
        chapter = chapter_at(recording, t)
        if chapter is None:
            continue
        local = t - chapter.offset
        if chapter.duration > 8:
            local = min(max(local, 4.0), chapter.duration - 2.0)
        picks.append((chapter, local))
    return picks


def _sample_frames(recording: Recording, max_side: int, rotation: int = 0) -> list:
    frames = []
    for chapter, t in _sample_times(recording):
        image = decode_one_frame(
            chapter.path,
            t,
            max_side,
            chapter.info.width,
            chapter.info.height,
            rotation=rotation,
        )
        if image is not None:
            frames.append(image)
    return frames


def _lean_for_recording(recording: Recording) -> list[LeanSample]:
    samples: list[LeanSample] = []
    for chapter in recording.chapters:
        for sample in lean_series(chapter.info):
            samples.append(LeanSample(t=sample.t + chapter.offset, lean_deg=sample.lean_deg))
    samples.sort(key=lambda s: s.t)
    return samples


def speed_for_recording(recording: Recording) -> list[SpeedSample]:
    samples: list[SpeedSample] = []
    for chapter in recording.chapters:
        for sample in speed_series(chapter.info):
            samples.append(SpeedSample(t=sample.t + chapter.offset, speed_kmh=sample.speed_kmh))
    samples.sort(key=lambda s: s.t)
    return samples


def turns_for_recording(recording: Recording) -> list[TurnSample]:
    samples: list[TurnSample] = []
    for chapter in recording.chapters:
        for sample in turn_series(chapter.info):
            samples.append(TurnSample(t=sample.t + chapter.offset, yaw_rate=sample.yaw_rate))
    samples.sort(key=lambda s: s.t)
    return samples


def _puck_events(hits: list[tuple[float, float]], recording_id: str) -> list[Event]:
    if not hits:
        return []
    events: list[Event] = []
    start_t, start_score = hits[0]
    last_t, peak = start_t, start_score
    count = 1
    for t, score in hits[1:]:
        if t - last_t <= 0.5:
            last_t = t
            peak = max(peak, score)
            count += 1
            continue
        if count >= 3:
            events.append(
                Event(
                    type="kneepuck_visible",
                    t_start=start_t,
                    t_end=last_t,
                    score=peak,
                    recording_id=recording_id,
                )
            )
        start_t, last_t, peak, count = t, t, score, 1
    if count >= 3:
        events.append(
            Event(
                type="kneepuck_visible",
                t_start=start_t,
                t_end=last_t,
                score=peak,
                recording_id=recording_id,
            )
        )
    return events


def _in_roi(point: TrackPoint, roi: Rect | None) -> bool:
    """True when the centre of *point*'s box lies inside *roi* (frame fractions)."""
    if roi is None:
        return False
    cx = 0.5 * (point.x1 + point.x2) / max(1, point.frame_w)
    cy = 0.5 * (point.y1 + point.y2) / max(1, point.frame_h)
    return roi.x <= cx <= roi.x + roi.w and roi.y <= cy <= roi.y + roi.h


def _merge_tracks(
    *parts: dict[int, list[TrackPoint]],
) -> dict[int, list[TrackPoint]]:
    merged: dict[int, list[TrackPoint]] = {}
    for part in parts:
        for track_id, points in part.items():
            merged.setdefault(track_id, []).extend(points)
    for points in merged.values():
        points.sort(key=lambda p: p.t)
    return merged


def _clip_name(window: ClipWindow) -> str:
    types = "+".join(window.types)
    return f"{window.recording_id}_{int(window.t_start):05d}_{types}.mp4"


def _clear_stale_outputs(directory: Path, recording_id: str, suffix: str) -> None:
    """Remove a recording's previous outputs so re-runs don't leave near-duplicate files.

    Clip filenames encode the (sub-second-varying) window start, so a re-scan with
    slightly shifted windows would otherwise pile up copies of the same moment.
    """
    if not directory.exists():
        return
    for path in directory.glob(f"{recording_id}_*{suffix}"):
        try:
            path.unlink()
        except OSError as exc:
            print(f"  could not remove stale {path.name}: {exc}")


def _export_verticals(
    recording: Recording,
    windows: list[ClipWindow],
    profile: Profile,
    config: ScanConfig,
    tracks: dict[int, list[TrackPoint]],
    turns: list[TurnSample],
    rec_start_utc: float | None,
    out_dir: Path,
) -> None:
    from videobrowser.themes import facing
    from videobrowser.vertical import export_vertical, focus_samples

    overlay_factory = None
    ctx = None
    if config.overlay:
        try:
            from videobrowser import racebox_stage as rbs

            ctx = config.racebox or rbs.overlay_context(config.out_dir, config.font_path)
            if rec_start_utc is None:
                rec_start_utc = rbs.recording_start_utc(recording, ctx)
        except Exception as exc:
            print(f"  vertical: overlay unavailable ({exc}); exporting without it")
            ctx = None

    _clear_stale_outputs(out_dir, recording.recording_id, ".mp4")
    for window in windows:
        dest = out_dir / _clip_name(window)
        if ctx is not None:

            def overlay_factory(w, h, fps, window=window):
                return rbs.overlay_source(
                    ctx, recording, window, w, h, fps, rec_start_utc, "vertical"
                )

        try:
            export_vertical(
                recording,
                window,
                dest,
                profile.rotation,
                focus_samples(window, tracks, turns, facing(profile.name)),
                overlay_factory=overlay_factory,
            )
        except Exception as exc:
            print(f"  vertical failed {dest.name}: {exc}")


def scan_recording(
    recording: Recording,
    config: ScanConfig,
    db: ScanDB,
    tracker: RiderTracker | None,
) -> list[Event]:
    if db.is_processed(recording.recording_id) and not config.force:
        print(f"skip {recording.recording_id} (already processed)")
        return []

    meta = recording.chapters[0].info.rotation if recording.chapters else 0
    samples = _sample_frames(recording, config.max_side, rotation=meta)
    if config.profile == "auto":
        profile = guess_profile_from_samples(samples)
        visual = profile.rotation
        profile = replace(profile, rotation=(meta + visual) % 360)
        print(
            f"{recording.recording_id}  profile={profile.name}  "
            f"rot={profile.rotation} (meta={meta} visual={visual})  "
            f"ego={profile.ego_side}  "
            f"{recording.duration:.1f}s  chapters={len(recording.chapters)}  "
            f"pipelines={','.join(profile.pipelines)}"
        )
    else:
        profile = resolve_profile(
            config.profile,
            config.camera_map,
            recording.recording_id,
            recording.chapters[0].path.stem,
            meta,
            samples[0] if samples else None,
            samples,
        )
        if profile.rotation == 0 and meta:
            profile = replace(profile, rotation=meta)
        print(
            f"{recording.recording_id}  profile={profile.name}  "
            f"rot={profile.rotation}  ego={profile.ego_side}  "
            f"{recording.duration:.1f}s  chapters={len(recording.chapters)}  "
            f"pipelines={','.join(profile.pipelines)}"
        )
    db.mark_recording(recording.recording_id, profile.name, recording.duration, processed=False)

    if tracker is not None:
        tracker.reset()
    tracks: dict[int, list[TrackPoint]] = {}
    # Boxes inside the puck ROI: followed by the vertical crop, not scored.
    focus_tracks: dict[int, list[TrackPoint]] = {}
    puck_hits: list[tuple[float, float]] = []
    motion_samples: list[tuple[float, float]] = []
    prev_image = None
    lean_samples = _lean_for_recording(recording) if "puck" in profile.pipelines else []

    remaining = config.max_seconds
    for chapter in recording.chapters:
        chapter_limit = remaining
        if remaining is not None:
            if remaining <= 0:
                break
        expected = int(
            min(chapter.duration, chapter_limit if chapter_limit is not None else chapter.duration)
            * config.fps
        )
        with FrameDecoder(
            chapter.path,
            chapter.info,
            fps=config.fps,
            max_side=config.max_side,
            rotation=profile.rotation,
            max_seconds=chapter_limit,
        ) as decoder:
            frame_iter = decoder.frames()
            first = next(frame_iter, None)
            backend = decoder.backend
            print(f"  decode {backend} {decoder.width}x{decoder.height} @ {config.fps:g}fps")

            def _frames():
                if first is not None:
                    yield first
                    yield from frame_iter

            for frame in tqdm(
                _frames(),
                total=max(expected, 1),
                desc=chapter.path.name,
                unit="fr",
            ):
                global_t = chapter.offset + frame.t
                if prev_image is not None:
                    motion_samples.append(
                        (
                            global_t,
                            scene_motion(prev_image, frame.image, profile.ego_masks),
                        )
                    )
                prev_image = frame.image
                if "riders" in profile.pipelines and tracker is not None:
                    masked = apply_ego_mask(frame.image, profile)
                    points = tracker.update(masked)
                    for point in points:
                        point.t = global_t
                        point.offtrack = offtrack_fraction(frame.image, point)
                        # The puck ROI is where the camera rider's own knee
                        # shows, so boxes centred there do not count as other
                        # riders. They are not blacked out before tracking
                        # either: riders ahead pass through that area too, and
                        # the vertical crop needs to see them.
                        bucket = (
                            focus_tracks
                            if _in_roi(point, profile.puck_roi)
                            else tracks
                        )
                        bucket.setdefault(point.track_id, []).append(point)
                if "puck" in profile.pipelines and profile.puck_roi is not None:
                    if leaned(lean_samples, global_t, config.lean_gate) or not lean_samples:
                        hit = detect_puck(frame.image, profile.puck_roi)
                        if hit:
                            puck_hits.append((global_t, hit.score))
        if remaining is not None:
            remaining -= chapter.duration

    events = []
    if "riders" in profile.pipelines:
        events.extend(score_tracks(tracks, recording.recording_id))
        events.extend(score_incidents(tracks, recording.recording_id))
    if "puck" in profile.pipelines:
        events.extend(_puck_events(puck_hits, recording.recording_id))
    events = merge_same_type(events)
    before_motion = len(events)
    events = [e for e in events if event_is_moving(e, motion_samples, config.min_motion)]
    dropped = before_motion - len(events)
    if config.min_speed > 0 and events:
        speeds = speed_for_recording(recording)
        before_speed = len(events)
        events = [
            e
            for e in events
            if e.type in SPEED_EXEMPT or on_track(speeds, e.t_start, e.t_end, config.min_speed)
        ]
        slow = before_speed - len(events)
        if slow:
            print(f"  dropped {slow} below {config.min_speed:g} km/h (standing still)")

    duration = recording.duration if config.max_seconds is None else min(
        recording.duration, config.max_seconds
    )
    windows = clip_windows(
        events,
        pad=config.pad,
        duration=duration,
        max_len=config.max_clip_seconds,
        max_clips=config.max_clips,
    )
    start_utc = (
        creation_utc_prefer_gps(recording.chapters[0].info) if recording.chapters else None
    )
    date_label = local_date_label(start_utc)
    clip_dir = config.out_dir / "clips" / date_label
    write_landscape = config.landscape and not config.no_clips
    if write_landscape:
        _clear_stale_outputs(clip_dir, recording.recording_id, ".mp4")
    clip_records: list[tuple[ClipWindow, Path | None]] = []
    for window in windows:
        dest = clip_dir / _clip_name(window)
        path: Path | None = None
        if write_landscape:
            try:
                path = export_clip(recording, window, dest, rotation=profile.rotation)
            except RuntimeError as exc:
                print(f"  clip failed {dest.name}: {exc}")
        clip_records.append((window, path))

    rec_start_utc: float | None = None
    if config.racebox is not None and write_landscape:
        try:
            from videobrowser.racebox_stage import run_for_recording

            rec_start_utc = run_for_recording(
                recording, windows, profile.rotation, config.racebox
            )
        except Exception as exc:
            print(f"  racebox overlays failed: {exc}")

    if config.vertical and not config.no_clips and windows:
        _export_verticals(
            recording,
            windows,
            profile,
            config,
            _merge_tracks(tracks, focus_tracks),
            turns_for_recording(recording),
            rec_start_utc,
            config.out_dir / "vertical" / date_label,
        )

    preview_dir = config.out_dir / "previews" / date_label
    if config.preview_limit and events:
        _clear_stale_outputs(preview_dir, recording.recording_id, ".jpg")
        try:
            write_previews(
                events,
                recording,
                profile,
                preview_dir,
                tracker if "riders" in profile.pipelines else None,
                config.preview_limit,
            )
        except Exception as exc:
            print(f"  previews failed: {exc}")

    db.replace_events(
        recording.recording_id,
        [
            (
                e.recording_id,
                e.type,
                e.t_start,
                e.t_end,
                e.score,
                e.track_id,
                next(
                    (
                        str(path)
                        for window, path in clip_records
                        if path and e in window.events
                    ),
                    None,
                ),
                dumps_extra(e.extra),
            )
            for e in events
        ],
    )
    vertical_dir = config.out_dir / "vertical" / date_label
    db.replace_clips(
        recording.recording_id,
        [
            (
                window.t_start,
                window.t_end,
                "+".join(window.types),
                window.score,
                str(path) if path else None,
                str(vertical_dir / _clip_name(window))
                if (vertical_dir / _clip_name(window)).exists()
                else None,
            )
            for window, path in clip_records
        ],
    )
    db.save_recording_meta(
        recording.recording_id,
        profile.name,
        profile.rotation,
        start_utc,
        date_label,
        recording.duration,
        [
            {"path": str(ch.path), "offset": ch.offset, "duration": ch.duration}
            for ch in recording.chapters
        ],
    )
    db.mark_recording(recording.recording_id, profile.name, recording.duration, processed=True)
    extra = f"  dropped {dropped} stationary" if dropped else ""
    print(
        f"  events={len(events)} clips={sum(1 for _, p in clip_records if p)} "
        f"types={sorted({e.type for e in events}) or '-'}{extra}"
    )
    return events
