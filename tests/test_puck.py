import numpy as np

from videobrowser.profiles import SIDE, Rect
from videobrowser.puck import detect_puck

ROI = SIDE.puck_roi


def test_detects_white_ellipse_on_asphalt():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:, :360] = (110, 110, 110)
    image[:, 360:] = (20, 20, 20)
    yy, xx = np.ogrid[:360, :640]
    mask = ((xx - 300) / 24) ** 2 + ((yy - 120) / 16) ** 2 <= 1
    image[mask] = (250, 250, 250)
    hit = detect_puck(image, ROI)
    assert hit is not None
    assert 270 < hit.cx < 330
    assert 90 < hit.cy < 150


def test_ignores_empty_asphalt():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:, :360] = (110, 110, 110)
    image[:, 360:] = (15, 15, 15)
    assert detect_puck(image, ROI) is None


def test_ignores_white_building_far_left():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:, :360] = (110, 110, 110)
    image[:, 360:] = (20, 20, 20)
    image[20:90, 40:140] = 250
    assert detect_puck(image, ROI) is None


def test_ignores_cloud_when_leaned():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:, :] = (110, 110, 110)
    image[:, 360:] = 20
    image[170:300, 80:220] = (210, 150, 70)
    yy, xx = np.ogrid[:360, :640]
    cloud = ((xx - 160) / 16) ** 2 + ((yy - 230) / 12) ** 2 <= 1
    image[cloud] = 250
    assert detect_puck(image, ROI) is None


def test_ignores_long_track_line():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    image[:, :360] = (110, 110, 110)
    image[240:252, 80:300] = 250
    assert detect_puck(image, ROI) is None
