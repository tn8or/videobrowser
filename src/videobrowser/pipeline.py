from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from tqdm import tqdm

from videobrowser.db import ScanDB
from videobrowser.decode import FrameDecoder, decode_frames, decode_one_frame
from videobrowser.events import (
    SPEED_EXEMPT,
    ClipWindow,
    Event,
    clip_windows,
    dumps_extra,
    merge_same_type,
)
from videobrowser.gopro import Chapter, Recording, chapter_at
from videobrowser.incidents import (
    GroundAhead,
    offtrack_fraction,
    own_offtrack_events,
    road_colour,
    score_incidents,
)
from videobrowser.motion import (
    DEFAULT_THRESHOLD,
    FACING_MIN_DIVERGENCE,
    MountWatch,
    event_is_moving,
    facing_from_flow,
    flow_divergence,
    per_second,
    riding_share,
    scene_motion,
)
from videobrowser.preview import write_previews
from videobrowser.profiles import (
    CameraMap,
    Profile,
    Rect,
    apply_ego_mask,
    guess_profile_from_samples,
    resolve_profile,
)
from videobrowser.probe import local_date_label
from videobrowser.puck import detect_puck
from videobrowser.riders import RiderTracker, TrackPoint, score_passes, score_tracks
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
from videobrowser.themes import facing
from videobrowser.vehicles import (
    DEFAULT_MAX_SHARE,
    EVENT_PAD_S,
    in_paddock,
    is_vehicle,
    vehicle_in_view,
)
from videobrowser.vehicles import per_second as vehicles_per_second


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
    preview_limit: int = 8
    lean_gate: float = 20.0
    max_seconds: float | None = None
    min_motion: float = DEFAULT_THRESHOLD
    # Drop events below this GPS speed (standing still); 0 disables. The
    # shorts apply the stricter track speed, so paddock clips stay available.
    min_speed: float = STOPPED_KMH
    # Drop events with a car, van or truck in view this share of the time
    # (the paddock); 0 disables.
    max_vehicle_share: float = DEFAULT_MAX_SHARE
    max_clip_seconds: float = 24.0
    max_clips: int = 12
    # RaceBox speed/turn track (racebox_align.RaceboxTrack) for footage
    # without GoPro GPS.
    racebox_track: Any = None


def pick_device(requested: str) -> str:
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
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


# Short seeks run side by side, one per CPU thread: each is its own ffmpeg,
# mostly start-up and waiting on the decoder.
_SEEK_WORKERS = os.cpu_count() or 4

# Frames sent to YOLO per call. One frame per call spent most of the time on
# per-call overhead rather than on the GPU (~22 ms a frame on an RTX 5070).
_DETECT_BATCH = 16


def _parallel(work, items: list) -> list:
    """work(item) for each item, in order, a few at a time."""
    if len(items) <= 1:
        return [work(item) for item in items]
    with ThreadPoolExecutor(max_workers=_SEEK_WORKERS) as pool:
        return list(pool.map(work, items))


def _sample_frames(recording: Recording, max_side: int, rotation: int = 0) -> list:
    def _one(pick):
        chapter, t = pick
        return decode_one_frame(
            chapter.path,
            t,
            max_side,
            chapter.info.width,
            chapter.info.height,
            rotation=rotation,
        )

    return [image for image in _parallel(_one, _sample_times(recording)) if image is not None]


# Moments sampled for which way the camera looks (the ones in the pits or
# standing still do not vote).
_FACING_SAMPLES = 101


def facing_for_recording(recording: Recording, max_side: int, profile: Profile) -> str | None:
    """"front" or "rear" from how the scene moves between two consecutive
    frames at moments spread over the recording, else None (the mount guess
    from ``profile`` stands). Logs the votes either way."""
    duration = recording.duration or 0.0
    picks = []
    for k in range(_FACING_SAMPLES):
        t = duration * (k + 0.5) / _FACING_SAMPLES
        chapter = chapter_at(recording, t)
        if chapter is None:
            continue
        local = min(max(t - chapter.offset, 0.0), max(0.0, chapter.duration - 0.5))
        picks.append((chapter, local))

    def _one(pick):
        chapter, local = pick
        pair = decode_frames(
            chapter.path, local, 2, max_side, chapter.info.width, chapter.info.height,
            rotation=profile.rotation,
        )
        if len(pair) == 2:
            return flow_divergence(pair[0], pair[1], profile.ego_masks)
        return None

    divergences = [d for d in _parallel(_one, picks) if d is not None]
    found = facing_from_flow(divergences)
    ahead = sum(1 for d in divergences if d > FACING_MIN_DIVERGENCE)
    back = sum(1 for d in divergences if d < -FACING_MIN_DIVERGENCE)
    print(
        f"  facing by scene flow: {found or 'unclear'} ({ahead} ahead / {back} back, "
        f"{len(divergences)} moments; mount guess {profile.name} = {facing(profile.name)})"
    )
    return found


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


def gps_for_recording(
    recording: Recording,
    config: ScanConfig,
    motion_samples: list[tuple[float, float]],
    start_utc: float | None,
) -> tuple[list[SpeedSample], list[TurnSample], float | None, str]:
    """Speed and turn samples from the GoPro GPS, else from RaceBox, plus the
    recording's start (UTC), corrected when it was found on the RaceBox log,
    and how that start is known: "gps", "racebox-motion" or "clock".

    Without GoPro telemetry the recording is laid on the RaceBox log by its
    scene motion (``racebox_align``); without a clear match there is none.
    """
    speeds = speed_for_recording(recording)
    if speeds or config.racebox_track is None:
        if speeds:
            return speeds, turns_for_recording(recording), start_utc, "gps"
        return [], [], start_utc, "clock"
    from videobrowser.racebox_align import clock_guesses, motion_match

    match = motion_match(
        per_second(motion_samples), config.racebox_track, clock_guesses(start_utc)
    )
    found = match.start_utc
    if found is None:
        print(f"  no GPS; no clear RaceBox match from scene motion: {match.reason}")
        return [], [], start_utc, "clock"
    shift = found - start_utc if start_utc is not None else 0.0
    print(f"  no GPS; laid on RaceBox by scene motion (camera clock {shift:+.1f} s, {match.reason})")
    track = config.racebox_track
    # A zero shift is still a RaceBox match: the camera clock was on the second.
    return (
        track.speeds(found, recording.duration),
        track.turns(found, recording.duration),
        found,
        "racebox-motion",
    )


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


# Share of frames in an event riding on a moving bike, without GPS speed.
RIDING_MIN_SHARE = 0.5
# Running off the track slows the bike by itself: its speed is judged on the
# approach, this many seconds before it.
APPROACH_S = 6.0


def _paced_span(event: Event) -> tuple[float, float]:
    """The part of *event* whose speed tells track from paddock."""
    if event.type == "own_off_track":
        return max(0.0, event.t_start - APPROACH_S), event.t_start
    return event.t_start, event.t_end


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


def scan_recording(
    recording: Recording,
    config: ScanConfig,
    db: ScanDB,
    tracker: RiderTracker | None,
    position: str = "",
) -> list[Event]:
    """*position* (e.g. "[file 3 of 12] ") leads the recording's first line."""
    if db.is_processed(recording.recording_id) and not config.force:
        print(f"{position}skip {recording.recording_id} (already processed)")
        return []

    meta = recording.chapters[0].info.rotation if recording.chapters else 0
    samples = _sample_frames(recording, config.max_side, rotation=meta)
    if config.profile == "auto":
        profile = guess_profile_from_samples(samples)
        visual = profile.rotation
        profile = replace(profile, rotation=(meta + visual) % 360)
        print(
            f"{position}{recording.recording_id}  profile={profile.name}  "
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
            f"{position}{recording.recording_id}  profile={profile.name}  "
            f"rot={profile.rotation}  ego={profile.ego_side}  "
            f"{recording.duration:.1f}s  chapters={len(recording.chapters)}  "
            f"pipelines={','.join(profile.pipelines)}"
        )
    db.mark_recording(recording.recording_id, profile.name, recording.duration, processed=False)
    camera_facing = db.facing_override(recording.recording_id)
    if camera_facing:
        print(f"  facing: {camera_facing} (set by hand)")
    else:
        camera_facing = facing_for_recording(recording, config.max_side, profile)

    if tracker is not None:
        tracker.reset()
    tracks: dict[int, list[TrackPoint]] = {}
    # Boxes inside the puck ROI: followed by the vertical crop, not scored.
    focus_tracks: dict[int, list[TrackPoint]] = {}
    puck_hits: list[tuple[float, float]] = []
    motion_samples: list[tuple[float, float]] = []
    # (t, vehicle in view) per frame the detector saw.
    vehicle_samples: list[tuple[float, bool]] = []
    # The ground the bike rides onto, for running off the track ourselves.
    looks_ahead = (camera_facing or facing(profile.name)) == "front"
    ground = GroundAhead() if looks_ahead else None
    mount = MountWatch()
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

            # Where the chapter's time goes, logged after it: waiting on the
            # decoder, in the detector, and the per-frame work around it.
            waited = detected = 0.0
            began = time.perf_counter()

            def _frames():
                nonlocal waited
                if first is not None:
                    yield first
                while True:
                    t0 = time.perf_counter()
                    frame = next(frame_iter, None)
                    waited += time.perf_counter() - t0
                    if frame is None:
                        return
                    yield frame

            use_riders = "riders" in profile.pipelines and tracker is not None

            def _detected():
                """(frame, rider boxes) in order, YOLO run _DETECT_BATCH frames at a time."""
                batch = []
                for frame in _frames():
                    batch.append(frame)
                    if len(batch) < _DETECT_BATCH:
                        continue
                    yield from _flush(batch)
                    batch = []
                if batch:
                    yield from _flush(batch)

            def _flush(batch):
                nonlocal detected
                if not use_riders:
                    for frame in batch:
                        yield frame, []
                    return
                t0 = time.perf_counter()
                found = tracker.update_batch([apply_ego_mask(f.image, profile) for f in batch])
                detected += time.perf_counter() - t0
                yield from zip(batch, found)

            for frame, points in tqdm(
                _detected(),
                total=max(expected, 1),
                desc=chapter.path.name,
                unit="fr",
            ):
                global_t = chapter.offset + frame.t
                if prev_image is not None:
                    motion_samples.append(
                        (global_t, scene_motion(prev_image, frame.image, profile.ego_masks))
                    )
                prev_image = frame.image
                mount.update(global_t, frame.image)
                if ground is not None:
                    ground.update(global_t, frame.image)
                if use_riders:
                    vehicle_samples.append((global_t, vehicle_in_view(points)))
                    points = [p for p in points if not is_vehicle(p)]
                    road = road_colour(frame.image) if points else None
                    for point in points:
                        point.t = global_t
                        point.offtrack = offtrack_fraction(frame.image, point, road)
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
            total = time.perf_counter() - began
            print(
                f"  time {total:.0f}s: decode wait {waited:.0f}s, detector {detected:.0f}s, "
                f"other {max(0.0, total - waited - detected):.0f}s"
            )
        if remaining is not None:
            remaining -= chapter.duration

    events = []
    if "riders" in profile.pipelines:
        events.extend(score_tracks(tracks, recording.recording_id))
        events.extend(score_passes(tracks, recording.recording_id))
        events.extend(score_incidents(tracks, recording.recording_id))
    if ground is not None:
        events.extend(own_offtrack_events(ground.samples, recording.recording_id))
    if "puck" in profile.pipelines:
        events.extend(_puck_events(puck_hits, recording.recording_id))
    events = merge_same_type(events)
    before_motion = len(events)
    events = [e for e in events if event_is_moving(e, motion_samples, config.min_motion)]
    dropped = before_motion - len(events)
    if config.max_vehicle_share > 0 and events:
        before_vehicles = len(events)
        events = [
            e
            for e in events
            if not in_paddock(
                vehicle_samples,
                [(e.t_start - EVENT_PAD_S, e.t_end + EVENT_PAD_S)],
                config.max_vehicle_share,
            )
        ]
        parked = before_vehicles - len(events)
        if parked:
            print(f"  dropped {parked} with a car in view (paddock)")
    start_utc = (
        creation_utc_prefer_gps(recording.chapters[0].info) if recording.chapters else None
    )
    # How the start is known decides whether exports may put an overlay on
    # it: GoPro GPS time and a clear RaceBox match are sure, a bare camera
    # clock may be seconds off.
    speeds, turns, start_utc, start_source = gps_for_recording(
        recording, config, motion_samples, start_utc
    )
    riding = mount.per_second()
    if riding is None:
        print("  riding: no part of a bike holds still in the picture (helmet camera or in hand?)")
    # Without GoPro GPS the picture decides whether the camera rode on a
    # moving bike: a camera in the hand moves the picture as much as the
    # track streaming past, and RaceBox time laid on by scene motion may be
    # off. RaceBox speed stands in only when the picture cannot tell.
    by_picture = start_source != "gps" and riding is not None
    if config.min_speed > 0 and events:
        before_speed = len(events)
        if by_picture:
            events = [
                e
                for e in events
                if e.type in SPEED_EXEMPT
                or (riding_share(riding, *_paced_span(e)) or 0.0) >= RIDING_MIN_SHARE
            ]
        else:
            events = [
                e
                for e in events
                if e.type in SPEED_EXEMPT
                or on_track(speeds, *_paced_span(e), config.min_speed)
            ]
        slow = before_speed - len(events)
        if slow and by_picture:
            print(f"  dropped {slow} with the camera not riding on a moving bike")
        elif slow:
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
    date_label = local_date_label(start_utc)
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
                None,
                dumps_extra(e.extra),
            )
            for e in events
        ],
    )
    # Clip windows only: ``horizontals`` and ``verticals`` cut them later.
    db.replace_clips(
        recording.recording_id,
        [(w.t_start, w.t_end, "+".join(w.types), w.score, None, None) for w in windows],
    )
    # The rider boxes the vertical crop follows, inside the windows only.
    db.save_tracks(
        recording.recording_id,
        [
            point
            for points in _merge_tracks(tracks, focus_tracks).values()
            for point in points
            if any(w.t_start <= point.t <= w.t_end for w in windows)
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
        start_source,
        facing=camera_facing,
    )
    db.save_motion(recording.recording_id, per_second(motion_samples))
    db.save_vehicles(recording.recording_id, vehicles_per_second(vehicle_samples))
    db.save_riding(recording.recording_id, riding)
    db.mark_recording(recording.recording_id, profile.name, recording.duration, processed=True)
    extra = f"  dropped {dropped} stationary" if dropped else ""
    print(
        f"  events={len(events)} clip windows={len(windows)} "
        f"types={sorted({e.type for e in events}) or '-'}{extra}"
    )
    return events
