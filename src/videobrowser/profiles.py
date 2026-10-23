from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from videobrowser.decode import rotate_bgr


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    def clamp(self) -> Rect:
        x = min(max(self.x, 0.0), 1.0)
        y = min(max(self.y, 0.0), 1.0)
        return Rect(x, y, min(self.w, 1.0 - x), min(self.h, 1.0 - y))

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        r = self.clamp()
        x1 = int(r.x * width)
        y1 = int(r.y * height)
        x2 = int((r.x + r.w) * width)
        y2 = int((r.y + r.h) * height)
        return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


@dataclass(frozen=True)
class Profile:
    name: str
    rotation: int
    ego_masks: tuple[Rect, ...]
    puck_roi: Rect | None
    pipelines: tuple[str, ...]
    ego_side: str = "none"


COCKPIT = Profile(
    name="cockpit",
    rotation=0,
    ego_masks=(
        Rect(0.0, 0.72, 1.0, 0.28),
        Rect(0.0, 0.0, 0.06, 1.0),
        Rect(0.94, 0.0, 0.06, 1.0),
    ),
    puck_roi=None,
    pipelines=("riders",),
    ego_side="bottom",
)

SIDE = Profile(
    name="side",
    rotation=0,
    ego_masks=(Rect(0.48, 0.0, 0.52, 1.0),),
    # Knee meets the tarmac beside the fairing (upper-mid), not empty lower-left asphalt.
    puck_roi=Rect(0.26, 0.04, 0.28, 0.50),
    pipelines=("puck", "riders"),
    ego_side="right",
)

SIDE_LEFT = Profile(
    name="side_left",
    rotation=0,
    ego_masks=(Rect(0.0, 0.0, 0.52, 1.0),),
    puck_roi=Rect(0.46, 0.04, 0.28, 0.50),
    pipelines=("puck", "riders"),
    ego_side="left",
)

REAR = Profile(
    name="rear",
    rotation=0,
    ego_masks=(Rect(0.0, 0.0, 0.28, 1.0),),
    puck_roi=None,
    pipelines=("riders",),
    ego_side="left",
)

REAR_RIGHT = Profile(
    name="rear_right",
    rotation=0,
    ego_masks=(Rect(0.72, 0.0, 0.28, 1.0),),
    puck_roi=None,
    pipelines=("riders",),
    ego_side="right",
)

GROUND = Profile(
    name="ground",
    rotation=0,
    ego_masks=(Rect(0.0, 0.82, 0.22, 0.18),),
    puck_roi=None,
    pipelines=("riders",),
    ego_side="none",
)

UNKNOWN = Profile(
    name="unknown",
    rotation=0,
    ego_masks=(),
    puck_roi=Rect(0.25, 0.04, 0.50, 0.50),
    pipelines=("riders", "puck"),
    ego_side="none",
)

# rear_rotated kept as a manual override; auto infers rotation from the sky.
REAR_ROTATED = replace(REAR, name="rear_rotated", rotation=90)

PROFILES: dict[str, Profile] = {
    p.name: p
    for p in (COCKPIT, SIDE, SIDE_LEFT, REAR, REAR_RIGHT, REAR_ROTATED, GROUND, UNKNOWN)
}


@dataclass
class CameraMap:
    recordings: dict[str, str]
    stems: dict[str, str]

    @classmethod
    def load(cls, path: Path | None) -> CameraMap:
        if path is None or not path.exists():
            return cls(recordings={}, stems={})
        data = json.loads(path.read_text())
        recs = {str(k).upper(): str(v) for k, v in (data.get("recordings") or {}).items()}
        stems = {str(k).upper(): str(v) for k, v in (data.get("stems") or {}).items()}
        return cls(recordings=recs, stems=stems)

    def lookup(self, recording_id: str, stem: str) -> str | None:
        for key in (recording_id.upper(), recording_id.split("-")[-1].upper(), stem.upper()):
            if key in self.stems:
                return self.stems[key]
            if key in self.recordings:
                return self.recordings[key]
        return None


def apply_ego_mask(image: np.ndarray, profile: Profile) -> np.ndarray:
    if not profile.ego_masks:
        return image
    masked = image.copy()
    h, w = masked.shape[:2]
    for rect in profile.ego_masks:
        x1, y1, x2, y2 = rect.to_pixels(w, h)
        masked[y1:y2, x1:x2] = 0
    return masked


def _sky_weights(bgr: np.ndarray) -> np.ndarray:
    """Per-pixel sky likelihood: 1 for blue sky, 0.65 for overcast white."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    blue = (hue > 85) & (hue < 140) & (sat > 25) & (val > 90)
    overcast = (val > 175) & (sat < 55)
    return np.maximum(blue.astype(np.float32), overcast.astype(np.float32) * 0.65)


def _sky_score(bgr: np.ndarray) -> float:
    if bgr.size == 0:
        return 0.0
    return float(_sky_weights(bgr).mean())


def _sky_direction(image: np.ndarray) -> float | None:
    """Angle in degrees (90 = up, 0 = right) from ground towards a visible sky.

    Taken from the summed gradient of the sky mask, which points across the
    horizon whatever the frame's shape, so a horizon tilted by lean still gives
    the right side. None when there is no coherent sky.
    """
    h, w = image.shape[:2]
    if h < 8 or w < 8:
        return None
    sky = _sky_weights(image)
    frac = float(sky.mean())
    if frac < 0.04 or frac > 0.9:
        return None
    sky = cv2.GaussianBlur(sky, (0, 0), max(1.0, min(h, w) / 60))
    gx = cv2.Sobel(sky, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sky, cv2.CV_32F, 0, 1, ksize=3)
    sx, sy = float(gx.sum()), float(gy.sum())
    total = float(np.hypot(gx, gy).sum())
    # Scattered patches (glare, white fairing, sunlit concrete) cancel out;
    # only a horizon running across the frame gives a coherent direction.
    if total <= 0 or np.hypot(sx, sy) / total < 0.15:
        return None
    return float(np.degrees(np.arctan2(-sy, sx)) % 360)


def _rotation_for(angle: float) -> int | None:
    """Rotation putting a sky at *angle* on top; None near a diagonal."""
    side = int(((angle + 45) % 360) // 90)  # 0 right, 1 top, 2 left, 3 bottom
    if abs((angle - side * 90 + 180) % 360 - 180) > 30:
        return None
    return (270, 0, 90, 180)[side]


def _sky_rotation(image: np.ndarray) -> int | None:
    """Rotation that puts a clearly visible sky on top, or None without one."""
    angle = _sky_direction(image)
    return None if angle is None else _rotation_for(angle)


def infer_rotation(image: np.ndarray) -> int:
    """Rotate so the sky (if any) ends up at the top. Downward mounts stay at 0."""
    rotation = _sky_rotation(image)
    return 0 if rotation is None else rotation


def infer_rotation_from_samples(images: list[np.ndarray]) -> int:
    """One rotation for the whole recording, voted across *images*.

    Frames without a clear sky abstain. Turning the footage sideways needs a
    majority of all frames: one frame with a bright fairing, white bridge or
    sunlit tarmac along an edge must not rotate a landscape mount by 90.
    Upright vs upside down counts frames with the sky below against frames
    with it above, including diagonal ones, so frames leaned into left and
    right corners still count. Sideways frames that lost the majority stay out
    of that vote and cannot flip the recording.
    """
    angles = [a for a in (_sky_direction(image) for image in images) if a is not None]
    if not angles:
        return 0
    rotations = [_rotation_for(a) for a in angles]
    counts = Counter(rotations)
    sideways = max((90, 270), key=lambda r: counts[r])
    if counts[sideways] * 2 > len(images):
        return sideways
    vertical = [np.sin(np.radians(a)) for a, r in zip(angles, rotations) if r not in (90, 270)]
    below = sum(1 for v in vertical if v < 0)
    above = sum(1 for v in vertical if v > 0)
    return 180 if below > above else 0


def _has_gauge(gray: np.ndarray) -> bool:
    h, w = gray.shape[:2]
    if h < 40 or w < 40:
        return False
    crop = gray[int(h * 0.62) :, int(w * 0.2) : int(w * 0.8)]
    blur = cv2.GaussianBlur(crop, (9, 9), 2)
    min_r = max(8, crop.shape[0] // 10)
    max_r = max(min_r + 1, crop.shape[0] // 2)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=crop.shape[1] // 4,
        param1=80,
        param2=22,
        minRadius=min_r,
        maxRadius=max_r,
    )
    return circles is not None


def _ego_side(gray: np.ndarray) -> str:
    h, w = gray.shape
    left = gray[:, : int(w * 0.28)]
    right = gray[:, int(w * 0.72) :]
    bottom = gray[int(h * 0.75) :, :]
    mid = gray[int(h * 0.28) : int(h * 0.68), int(w * 0.28) : int(w * 0.72)]
    left_m, right_m = float(left.mean()), float(right.mean())
    bottom_m, mid_m = float(bottom.mean()), float(mid.mean())
    # Side/rear mounts: a dark fairing or tire on one edge. Check before the
    # dash band — low side cameras fill the bottom with dark asphalt.
    if left_m < 80 and left_m < right_m * 0.78 and left_m < mid_m * 0.9:
        return "left"
    if right_m < 80 and right_m < left_m * 0.78 and right_m < mid_m * 0.9:
        return "right"
    if bottom_m < 80 and mid_m > 40 and bottom_m < mid_m * 0.58:
        return "bottom"
    return "none"


def classify_upright(image: np.ndarray) -> tuple[str, str]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    sky_top = _sky_score(image[: max(1, h // 3), :])
    ego = _ego_side(gray)
    dark_frac = float((gray < 40).mean())
    left_m = float(gray[:, : int(w * 0.28)].mean())
    right_m = float(gray[:, int(w * 0.72) :].mean())
    bot_l = float(gray[int(h * 0.75) :, : int(w * 0.4)].mean())
    bot_r = float(gray[int(h * 0.75) :, int(w * 0.6) :].mean())
    mid_m = float(gray[int(h * 0.28) : int(h * 0.68), int(w * 0.28) : int(w * 0.72)].mean())
    # Dash sits across the full bottom. Side/rear mounts are dark on one side only.
    if bot_l < 75 and bot_r < 75 and mid_m > ((bot_l + bot_r) / 2) * 1.2:
        return "cockpit", "bottom"
    # Close-up fairing/tire in the lens is much darker than a distant rear view.
    if ego == "right" and right_m < 40:
        return "side", "right"
    if ego == "left" and left_m < 40:
        return "side", "left"
    if ego in {"left", "right"}:
        return "rear", ego
    if sky_top >= 0.35 and dark_frac < 0.12:
        return "ground", "none"
    if ego == "bottom" or (dark_frac >= 0.22 and sky_top < 0.2):
        return "cockpit", "bottom"
    if sky_top >= 0.08:
        return "ground", "none"
    return "unknown", ego


def guess_profile(image: np.ndarray, metadata_rotation: int = 0) -> str:
    """Name-only guess for tests and simple callers."""
    if metadata_rotation:
        image = rotate_bgr(image, metadata_rotation)
    rotation = infer_rotation(image)
    upright = rotate_bgr(image, rotation)
    name, _ = classify_upright(upright)
    return name


def profile_from_guess(name: str, rotation: int, ego_side: str) -> Profile:
    if name == "cockpit":
        return replace(COCKPIT, rotation=rotation)
    if name == "side":
        base = SIDE_LEFT if ego_side == "left" else SIDE
        return replace(base, rotation=rotation, name="side")
    if name == "rear":
        if ego_side == "right":
            return replace(REAR_RIGHT, rotation=rotation, name="rear")
        if ego_side == "bottom":
            return replace(
                REAR,
                rotation=rotation,
                name="rear",
                ego_masks=(Rect(0.0, 0.72, 1.0, 0.28),),
                ego_side="bottom",
            )
        return replace(REAR, rotation=rotation)
    if name == "ground":
        return replace(GROUND, rotation=rotation)
    return replace(UNKNOWN, rotation=rotation)


def guess_profile_from_samples(images: list[np.ndarray]) -> Profile:
    if not images:
        return UNKNOWN
    # Rotation is decided once for the recording, not per mount guess: a lone
    # frame must not win a tie between differently classified frames.
    rotation = infer_rotation_from_samples(images)
    votes = [classify_upright(rotate_bgr(image, rotation)) for image in images]
    name = Counter(v[0] for v in votes).most_common(1)[0][0]
    ego = Counter(v[1] for v in votes if v[0] == name).most_common(1)[0][0]
    return profile_from_guess(name, rotation, ego)


def resolve_profile(
    name: str | None,
    camera_map: CameraMap,
    recording_id: str,
    stem: str,
    metadata_rotation: int,
    sample: np.ndarray | None,
    samples: list[np.ndarray] | None = None,
) -> Profile:
    chosen = (name or "").strip().lower()
    if chosen and chosen != "auto":
        if chosen not in PROFILES:
            raise ValueError(f"unknown profile {chosen!r}; choose from {sorted(PROFILES)}")
        return PROFILES[chosen]
    # Optional explicit --map only. Filenames are not a mount signal.
    mapped = camera_map.lookup(recording_id, stem)
    if mapped:
        if mapped not in PROFILES:
            raise ValueError(f"cameras map points at unknown profile {mapped!r}")
        return PROFILES[mapped]
    frames = list(samples or [])
    if sample is not None:
        frames.append(sample)
    if frames:
        guessed = guess_profile_from_samples(frames)
        if metadata_rotation % 180 == 90 and guessed.rotation == 0:
            return replace(guessed, rotation=metadata_rotation)
        return guessed
    if metadata_rotation % 180 == 90:
        return replace(REAR, rotation=metadata_rotation)
    return UNKNOWN
