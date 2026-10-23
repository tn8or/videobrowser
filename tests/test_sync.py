import numpy as np

from videobrowser.sync import ENV_RATE, CameraClock, best_lag, clusters, track_offset


def _clock(rid, start, duration):
    return CameraClock(rid, start, duration, [])


def test_clusters_group_overlapping_recordings():
    groups = clusters([_clock("A", 0, 100), _clock("B", 50, 100), _clock("C", 400, 10)])
    assert [[c.recording_id for c in g] for g in groups] == [["A", "B"], ["C"]]


def test_best_lag_finds_shift_and_its_sign():
    rng = np.random.default_rng(1)
    signal = rng.standard_normal(ENV_RATE * 120)
    k = 17  # other(t) == reference(t + k)
    reference = signal[200 : 200 + ENV_RATE * 100]
    other = signal[200 + k : 200 + k + ENV_RATE * 100] + 0.5 * rng.standard_normal(ENV_RATE * 100)
    lag, peak, runner = best_lag(reference, other)
    assert abs(lag - k / ENV_RATE) < 1e-9
    assert peak > runner


def _lap(t):
    # A 60 s loop around a 500 m circle.
    angle = 2 * np.pi * t / 60.0
    return 55.0 + 0.0045 * np.sin(angle), 12.0 + 0.008 * np.cos(angle)


def test_track_offset_lines_up_two_cameras():
    t = np.arange(0, 300, 0.1)
    lat, lon = _lap(t)
    reference = np.column_stack([1000 + t, lat, lon])
    # Same ride, but this camera's clock reads 2.4 s late.
    other = np.column_stack([1000 + t + 2.4, lat, lon])
    assert abs(track_offset(other, reference) + 2.4) < 0.06
