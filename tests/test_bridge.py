import numpy as np
import pytest

from videobrowser.bridge import EdgeHit, find_edges, passes_from_edges


def _sky(seed=0):
    rng = np.random.default_rng(seed)
    image = np.empty((360, 640, 3), np.uint8)
    image[:] = (230, 200, 170)  # BGR: light blue
    image[220:] = rng.integers(90, 130, (140, 640, 1), dtype=np.uint8)  # road
    return image


def test_bridge_underside_is_found_and_the_horizon_is_not():
    image = _sky()
    image[:60, 100:560] = (150, 150, 150)
    image[60:66, 100:560] = (60, 60, 60)  # dark underside over the sky
    hits = find_edges(image)
    assert any(abs(h.row - 66 / 360) < 0.02 and h.run > 0.6 for h in hits)
    # Sky over road is bright above: no edge.
    assert not any(h.row > 0.5 for h in hits)


def test_short_dark_patch_is_no_bridge():
    image = _sky()
    image[40:60, 300:360] = (60, 60, 60)
    assert find_edges(image) == []


def _rise(t0, rows, fps=12.5):
    return [(t0 + i / fps, [EdgeHit(0.5, r, 0.0)]) for i, r in enumerate(rows)]


def test_an_edge_rising_out_of_the_top_is_a_pass():
    frames = _rise(10.0, [0.35, 0.3, 0.22, 0.12, 0.04])
    t = [f[0] for f in frames]
    assert passes_from_edges(t, [f[1] for f in frames], 12.5) == [t[-1] + 0.04]


def test_still_or_falling_edges_are_no_pass():
    still = _rise(10.0, [0.1, 0.1, 0.1, 0.1])
    falling = _rise(20.0, [0.05, 0.1, 0.2, 0.3])
    frames = still + falling
    assert passes_from_edges([f[0] for f in frames], [f[1] for f in frames], 12.5) == []


def test_two_edges_of_one_bridge_are_one_pass():
    frames = _rise(10.0, [0.35, 0.25, 0.12, 0.04])
    for (t, hits), r in zip(frames, [0.45, 0.4, 0.3, 0.2]):
        hits.append(EdgeHit(0.4, r, 0.0))
    frames += [(10.32 + 0.08 * i, [EdgeHit(0.4, r, 0.0)]) for i, r in enumerate([0.1, 0.03], 1)]
    out = passes_from_edges([f[0] for f in frames], [f[1] for f in frames], 12.5)
    assert len(out) == 1


def test_a_fence_leaving_after_the_bridge_does_not_time_it():
    bridge = _rise(10.0, [0.35, 0.25, 0.12, 0.04])
    fence = _rise(10.08, [0.4, 0.3, 0.2, 0.12, 0.05, 0.02, 0.01])
    frames = {}
    for t, hits in bridge + fence:
        frames.setdefault(round(t, 2), []).extend(hits)
    t = sorted(frames)
    assert passes_from_edges(t, [frames[k] for k in t], 12.5) == [bridge[-1][0] + 0.04]


def test_a_steep_edge_off_to_one_side_is_no_bridge():
    image = _sky()
    for x in range(100, 560):
        y = int(40 + (x - 100) * 0.45)  # about 24 degrees
        image[:y, x] = (60, 60, 60)
    assert find_edges(image) == []


def test_a_windscreen_edge_that_holds_its_row_is_no_pass():
    # Fixed near the top while the bike leans a little: never a rising track.
    frames = [(10.0 + i * 0.08, [EdgeHit(0.6, 0.12 + 0.005 * (i % 2), 0.0)]) for i in range(20)]
    frames += [(11.6, [EdgeHit(0.6, 0.3, 0.0)]), (11.68, [EdgeHit(0.6, 0.1, 0.0)])]
    assert passes_from_edges([f[0] for f in frames], [f[1] for f in frames], 12.5) == pytest.approx([11.72])


def test_a_roof_rising_slowly_in_the_paddock_is_no_pass():
    rows = list(np.linspace(0.3, 0.05, 30))  # 2.4 s to clear the top
    frames = _rise(10.0, rows)
    assert passes_from_edges([f[0] for f in frames], [f[1] for f in frames], 12.5) == []
