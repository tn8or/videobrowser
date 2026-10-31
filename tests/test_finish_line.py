import numpy as np

from videobrowser.finish_line import LineHit, find_line, passes_from_hits

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


def test_passes_follow_each_line_down_and_skip_the_bike():
    # Two lines coming down one after the other, and a fairing edge that
    # stays put at 0.5.
    rows = [[0.45], [0.55], [0.7, 0.42], [0.52], [0.66], [], [], []]
    times = [10.0 + i * 0.04 for i in range(len(rows))]
    fairing = LineHit(0.5, 0.5, 0.0)
    hits = [[LineHit(0.6, r, -4.0) for r in rs] + [fairing] for rs in rows]
    passes = passes_from_hits(times, hits, 25.0)
    assert [round(p.t, 2) for p in passes] == [10.1, 10.18]
    assert [round(p.row, 2) for p in passes] == [0.7, 0.66]


def test_one_sighting_is_no_pass():
    hits = [[LineHit(0.6, 0.6, 0.0)], [], [], [], [], [], []]
    assert passes_from_hits([i * 0.04 for i in range(7)], hits, 25.0) == []
