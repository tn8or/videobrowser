from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from videobrowser.clips import grab_frame
from videobrowser.events import Event
from videobrowser.gopro import Recording, chapter_at
from videobrowser.profiles import Profile, Rect, apply_ego_mask
from videobrowser.puck import detect_puck


def _annotate_riders(image: np.ndarray, boxes: list[tuple[int, int, int, int, str]]) -> np.ndarray:
    canvas = image.copy()
    for x1, y1, x2, y2, label in boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 80), 2)
        cv2.putText(
            canvas,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 80),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _annotate_puck(image: np.ndarray, roi: Rect | None) -> np.ndarray:
    canvas = image.copy()
    if roi is None:
        return canvas
    h, w = canvas.shape[:2]
    x1, y1, x2, y2 = roi.to_pixels(w, h)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 80, 255), 1)
    hit = detect_puck(canvas, roi)
    if hit:
        bx1, by1, bx2, by2 = hit.bbox
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 255, 255), 2)
        cv2.putText(
            canvas,
            f"puck {hit.score:.2f}",
            (bx1, max(16, by1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def write_previews(
    events: list[Event],
    recording: Recording,
    profile: Profile,
    out_dir: Path,
    tracker,
    limit: int,
) -> list[Path]:
    if limit <= 0 or not events:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(events, key=lambda e: e.score, reverse=True)[:limit]
    written: list[Path] = []
    for i, event in enumerate(ranked):
        chapter = chapter_at(recording, event.t_peak)
        if chapter is None:
            continue
        local_t = max(0.0, event.t_peak - chapter.offset)
        dest = out_dir / f"{recording.recording_id}_{i:02d}_{event.type}.jpg"
        try:
            grab_frame(chapter.path, local_t, dest, rotation=profile.rotation)
        except RuntimeError:
            continue
        image = cv2.imread(str(dest))
        if image is None:
            continue
        if event.type.endswith("rider") and tracker is not None:
            masked = apply_ego_mask(image, profile)
            points = tracker.detect(masked)
            boxes = [
                (int(p.x1), int(p.y1), int(p.x2), int(p.y2), f"id{p.track_id}")
                for p in points
            ]
            image = _annotate_riders(image, boxes)
        if event.type.startswith("kneepuck"):
            image = _annotate_puck(image, profile.puck_roi)
        cv2.imwrite(str(dest), image)
        written.append(dest)
    return written
