import numpy as np

from videobrowser.finish_line import LineHit, LineWatch, find_line, pass_from_frames

W, H = 640, 360


def _road(seed=0):
    """Grey asphalt with grain."""
    rng = np.random.default_rng(seed)
    grey = np.clip(rng.normal(90, 8, (H, W)), 0, 255).astype(np.uint8)
    return np.dstack([grey] * 3)


def _band(image, row, angle_deg, x0=0.15, x1=0.85, thick=5, value=230):
    """A painted line across the road: centre *row* (share of height)."""
    slope = np.tan(np.radians(angle_deg))
    for x in range(int(x0 * W), int(x1 * W)):
        y = int(row * H + slope * (x - W / 2))
        image[max(0, y - thick // 2) : y + thick // 2 + 1, x] = value
    return image


def test_line_across_the_road_is_found_with_its_roll():
    hit = find_line(_band(_road(), 0.7, -6))
    assert hit is not None
    assert hit.run >= 0.6
    assert abs(hit.row - 0.7) < 0.02
    assert abs(hit.angle - -6) <= 2


def test_lines_along_the_track_and_grain_are_not_lines():
    image = _road()
    # Lane edges: steep in the picture, running into the distance.
    for x0, x1 in ((100, 280), (540, 360)):
        for y in range(int(0.4 * H), H):
            x = int(x0 + (x1 - x0) * (H - y) / (0.6 * H))
            image[y, x - 2 : x + 3] = 230
    assert find_line(image) is None


def test_a_wall_is_not_a_line():
    image = _road()
    image[int(0.5 * H) : int(0.8 * H), 50:600] = 220
    assert find_line(image) is None


def test_a_short_stripe_is_not_a_line():
    assert find_line(_band(_road(), 0.7, 0, x0=0.45, x1=0.6)) is None


def test_pass_is_the_last_frame_the_line_comes_down_in():
    rows = [0.5, 0.6, 0.72, 0.86, None, None]
    times = [10.0 + i * 0.04 for i in range(len(rows))]
    images = [_band(_road(i), r, -4) if r is not None else _road(i) for i, r in enumerate(rows)]
    found = pass_from_frames(times, images)
    assert found is not None
    assert abs(found.t - 10.12) < 1e-9
    assert abs(found.row - 0.86) < 0.02


def test_watch_keeps_the_last_sighting_of_each_pass():
    watch = LineWatch()
    for t in (5.0, 5.2, 5.4, 90.0, 90.2):
        watch.hits.append((t, LineHit(0.6, 0.7, 0.0)))
    assert watch.candidates() == [5.4, 90.2]
