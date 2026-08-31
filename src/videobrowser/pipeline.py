from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from tqdm import tqdm

from videobrowser.clips import export_clip
from videobrowser.db import ScanDB
from videobrowser.decode import FrameDecoder, decode_one_frame
from videobrowser.events import ClipWindow, Event, clip_windows, dumps_extra, merge_same_type
from videobrowser.gopro import Recording
from videobrowser.motion import DEFAULT_THRESHOLD, event_is_moving, scene_motion
from videobrowser.preview import write_previews
from videobrowser.profiles import CameraMap, Profile, apply_ego_mask, guess_profile_from_samples, resolve_profile
from videobrowser.puck import detect_puck
from videobrowser.riders import RiderTracker, TrackPoint, score_tracks
from videobrowser.telemetry import LeanSample, lean_series, leaned


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
    max_clip_seconds: float = 24.0
    max_clips: int = 12


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


def _sample_frames(recording: Recording, max_side: int, rotation: int = 0) -> list:
    chapter = recording.chapters[0]
    duration = recording.duration or chapter.duration or 0.0
    fractions = (0.22, 0.5, 0.78)
    frames = []
    for frac in fractions:
        t = duration * frac
        if duration > 8:
            t = min(max(t, 4.0), duration - 2.0)
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
            backend = "videotoolbox" if decoder.hwaccel else "software"
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
                    if profile.puck_roi is not None:
                        x1, y1, x2, y2 = profile.puck_roi.to_pixels(
                            masked.shape[1], masked.shape[0]
                        )
                        masked[y1:y2, x1:x2] = 0
                    points = tracker.update(masked)
                    for point in points:
                        point.t = global_t
                        tracks.setdefault(point.track_id, []).append(point)
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
    if "puck" in profile.pipelines:
        events.extend(_puck_events(puck_hits, recording.recording_id))
    events = merge_same_type(events)
    before_motion = len(events)
    events = [e for e in events if event_is_moving(e, motion_samples, config.min_motion)]
    dropped = before_motion - len(events)

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
    clip_dir = config.out_dir / "clips"
    if not config.no_clips:
        _clear_stale_outputs(clip_dir, recording.recording_id, ".mp4")
    clip_records: list[tuple[ClipWindow, Path | None]] = []
    for window in windows:
        dest = clip_dir / _clip_name(window)
        path: Path | None = None
        if not config.no_clips:
            try:
                path = export_clip(recording, window, dest, rotation=profile.rotation)
            except RuntimeError as exc:
                print(f"  clip failed {dest.name}: {exc}")
        clip_records.append((window, path))

    preview_dir = config.out_dir / "previews"
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
    db.mark_recording(recording.recording_id, profile.name, recording.duration, processed=True)
    extra = f"  dropped {dropped} stationary" if dropped else ""
    print(
        f"  events={len(events)} clips={sum(1 for _, p in clip_records if p)} "
        f"types={sorted({e.type for e in events}) or '-'}{extra}"
    )
    return events
