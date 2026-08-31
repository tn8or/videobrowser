from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from videobrowser.profiles import Rect


@dataclass(frozen=True)
class PuckDetection:
    cx: float
    cy: float
    area: float
    score: float
    bbox: tuple[int, int, int, int]


def _roi_crop(image: np.ndarray, roi: Rect) -> tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = roi.to_pixels(w, h)
    return image[y1:y2, x1:x2], x1, y1


def detect_puck(image: np.ndarray, roi: Rect) -> PuckDetection | None:
    """White/grey kneepuck on tarmac next to the fairing.

    At speed the puck is motion-blurred, so circularity is modest. Far-left
    buildings, sky, and empty lower tarmac are kept out by the ROI.
    """
    crop, ox, oy = _roi_crop(image, roi)
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Scuffed slider is off-white; motion blur pulls V down.
    lower = np.array([0, 0, 155], dtype=np.uint8)
    upper = np.array([180, 80, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ch, cw = crop.shape[:2]
    crop_area = float(ch * cw)
    best: PuckDetection | None = None
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 120 or area > min(5000.0, 0.22 * crop_area):
            continue
        peri = cv2.arcLength(contour, True)
        if peri < 20:
            continue
        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < 0.16:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = bw / max(bh, 1)
        if aspect < 0.4 or aspect > 2.8:
            continue
        fill = area / max(float(bw * bh), 1.0)
        if fill < 0.28:
            continue
        y0, y1 = max(0, y - 8), min(ch, y + bh + 8)
        x0, x1 = max(0, x - 8), min(cw, x + bw + 8)
        neigh_h = hsv[y0:y1, x0:x1, 0]
        neigh_s = hsv[y0:y1, x0:x1, 1]
        neigh_v = hsv[y0:y1, x0:x1, 2]
        if neigh_v.size == 0:
            continue
        sky = float(((neigh_h > 85) & (neigh_h < 140) & (neigh_v > 90)).mean())
        if sky > 0.22:
            continue
        score = float(circularity * min(1.0, area / 700.0))
        if best is None or score > best.score:
            best = PuckDetection(
                cx=float(ox + x + bw / 2),
                cy=float(oy + y + bh / 2),
                area=area,
                score=score,
                bbox=(ox + x, oy + y, ox + x + bw, oy + y + bh),
            )
    return best
