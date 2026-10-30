import numpy as np
import pytest

from videobrowser.racebox_align import (
    RaceboxTrack,
    clock_guesses,
    day_match,
    motion_match,
    motion_start_utc,
    track_from_sessions,
)

T0 = 1_785_000_000.0


def _track(starts, length=1200.0):
    """Sessions at *starts* (s after T0), 10 Hz, pit exit/entry at each end."""
    utc, speed = [], []
    for s0 in starts:
        t = np.arange(0, length, 0.1)
        v = 110 + 50 * np.sin(t / 15.0)
        v[t < 60] = np.linspace(0, 60, (t < 60).sum())
        v[t > length - 60] = np.linspace(60, 0, (t > length - 60).sum())
        utc.append(T0 + s0 + t)
        speed.append(v)
    utc, speed = np.concatenate(utc), np.concatenate(speed)
    return RaceboxTrack(utc, speed, np.zeros(len(utc)))


def _motion(track, start, seconds, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(seconds):
        u = start + i + 0.5
        on = np.min(np.abs(track.utc - u)) <= 5
        v = float(np.interp(u, track.utc, track.speed_kmh)) if on else 0.0
        m = (1.0 if v < 5 else 4 + min(v, 120) / 8) + rng.normal(0, 1.5)
        out.append((i + 0.5, max(0.0, m)))
    return out


def test_motion_lays_recording_on_racebox_within_a_second():
    track = _track([0, 2700, 6000])
    start = T0 + 2700 - 300.4  # camera rolling 5 min before the session
    motion = _motion(track, start, 1800)
    # Camera clock off by 40 s, read either way round.
    for guesses in ([start + 40], [start - 7200 + 40, start + 40]):
        assert motion_start_utc(motion, track, guesses) == pytest.approx(start, abs=1.0)


def test_fine_pass_without_a_score_keeps_the_coarse_start(monkeypatch):
    import videobrowser.racebox_align as ra

    track = _track([0, 2700, 6000])
    start = T0 + 2700 - 300.4
    motion = _motion(track, start, 1800)
    monkeypatch.setattr(ra, "FINE_SEARCH_S", 0)
    coarse = motion_start_utc(motion, track, [start + 40])
    monkeypatch.setattr(ra, "FINE_SEARCH_S", 20)
    calls = []
    real = ra._pearson

    def no_fine_score(speed, x, known, ks):
        calls.append(len(ks))
        out = real(speed, x, known, ks)
        return out if len(calls) == 1 else out * 0 - 1.0

    monkeypatch.setattr(ra, "_pearson", no_fine_score)
    found = motion_start_utc(motion, track, [start + 40])
    assert found == coarse


def test_no_match_when_sessions_are_indistinguishable_and_no_clock():
    track = _track([0, 3000, 6000])
    motion = _motion(track, T0 + 3000 - 300.4, 1800)
    assert motion_start_utc(motion, track, []) is None


def test_match_says_why_a_recording_was_not_placed():
    track = _track([0, 2700])
    motion = _motion(track, T0 + 2700 - 300.4, 1800)
    far = motion_match(motion, track, [T0 + 86400.0])
    assert far.start_utc is None and far.reason.startswith("no RaceBox log near the camera clock")
    short = motion_match(motion[:60], track, [T0])
    assert short.start_utc is None and "only 60 s" in short.reason
    ok = motion_match(motion, track, [T0 + 2700 - 300.4 + 40])
    assert ok.start_utc is not None and ok.reason.startswith("r=")


def test_clock_guesses_try_local_and_utc_readings():
    summer = 1_785_485_689.0  # 2026-07-31, CEST
    assert clock_guesses(summer) == [summer, summer + 7200]
    assert clock_guesses(None) == []


def test_track_from_overlapping_sessions_is_strictly_increasing():
    class Session:
        def __init__(self, t0, n):
            t = T0 + t0 + np.arange(n) * 0.1
            self.data = {
                "time": t,
                "lat": 55.0 + np.arange(n) * 1e-5,
                "lng": np.full(n, 10.0),
                "velocity": np.full(n, 100.0),
            }

    # Second session starts before the first ends, and is listed first.
    track = track_from_sessions([Session(50, 1000), Session(0, 1000)])
    assert track is not None
    assert np.all(np.diff(track.utc) > 0)
    assert track.utc[0] == pytest.approx(T0) and track.utc[-1] == pytest.approx(T0 + 50 + 99.9)


def _handled(motion, rng, share=0.3):
    """Camera picked up in the paddock now and then: bursts of 10-60 s of
    big motion over about *share* of the time standing still."""
    out = [list(p) for p in motion]
    i = 0
    while i < len(out):
        if out[i][1] < 3 and rng.random() < share / 35:
            for j in range(i, min(len(out), i + int(rng.uniform(10, 60)))):
                out[j][1] += rng.uniform(20, 40)
            i += 60
        i += 1
    return [tuple(p) for p in out]


def test_whole_day_places_paddock_only_recordings_too():
    rng = np.random.default_rng(7)
    track = _track([0, 3300, 7400, 10500, 14800])  # 20 min out, irregular gaps
    error = 25.0  # camera clock error
    recs = []
    for i, (s0, length) in enumerate([(-600, 1500), (1800, 1200), (6900, 1500), (10200, 2400)]):
        start = T0 + s0 + 0.2
        recs.append((start + error, _handled(_motion(track, start, length, seed=i), rng)))
    # 1800 s + 1200 s lies between two sessions: paddock only, no match alone.
    assert motion_start_utc(recs[1][1], track, clock_guesses(recs[1][0])) is None
    match = day_match(recs, track)
    # Session edges only: right to within a minute or two, not a second.
    assert match.start_utc == pytest.approx(recs[0][0] - error, abs=120.0)


# --- finish line --------------------------------------------------------------

from videobrowser.racebox_align import LINE_LEAD_S, finish_passes, line_crossings, line_place  # noqa: E402

LAPS = np.array([76.1, 74.9, 77.1, 75.6])
# Knutstorp's straight at 176 km/h: grid and start lines 13-50 m before the
# finish line, each passing this long before it.
BEFORE = (1.02, 0.74, 0.58, 0.50, 0.26)


def _line_track(first=100.0, laps=LAPS):
    utc = T0 + np.arange(0.0, 1000.0, 0.1)
    crossings = T0 + first + np.r_[0.0, np.cumsum(laps)]
    return RaceboxTrack(utc, np.full(len(utc), 176.0), np.zeros(len(utc)), crossings)


def _picture(track, start, seed=5, misses=()):
    """Line passes the picture would show for a recording starting at *start*:
    the grid, the start and the finish line on every lap, and lines in the
    corners now and then."""
    rng = np.random.default_rng(seed)
    out = []
    for i, c in enumerate(track.crossings):
        finish = c - start - LINE_LEAD_S
        out.extend(finish - b for b in BEFORE)
        if i not in misses:
            out.append(finish + rng.normal(0, 0.02))
    out.extend(rng.uniform(0, 500, 40))
    calls = []

    def find(t0, t1):
        calls.append((t0, t1))
        return sorted(t for t in out if t0 <= t <= t1)

    return find, calls


def test_crossings_are_interpolated_between_samples():
    # Northward along a meridian at 10 m/s, sampled every 0.04 s; the line
    # runs east-west at 100 m.
    t = np.arange(0, 20, 0.04)
    lat = 56.0 + t * 10 / 111_320.0
    lon = np.full_like(t, 13.0)
    line = (56.0 + 100 / 111_320.0, 12.9999, 56.0 + 100 / 111_320.0, 13.0001)
    (crossing,) = line_crossings(T0 + t, lat, lon, line)
    assert abs(crossing - (T0 + 10.0)) < 1e-3


def test_crossings_off_the_line_ends_do_not_count():
    t = np.arange(0, 20, 0.04)
    lat = 56.0 + t * 10 / 111_320.0
    lon = np.full_like(t, 13.001)  # beside the line, as on a parallel road
    line = (56.0 + 100 / 111_320.0, 12.9999, 56.0 + 100 / 111_320.0, 13.0001)
    assert line_crossings(T0 + t, lat, lon, line) == []


def test_finish_is_the_last_line_before_clear_track():
    passes = [10.0 - b for b in BEFORE] + [10.0, 30.0]
    assert finish_passes(passes, 176 / 3.6) == [10.0, 30.0]


def test_finish_line_fixes_a_start_half_a_lap_off():
    track = _line_track()
    start = T0 + 10.0
    find, calls = _picture(track, start)
    match = line_place(start + 28.9, 450.0, track, find)
    assert match.start_utc is not None
    assert abs(match.start_utc - start) < 0.05
    assert len(match.passes) == len(track.crossings)
    # One wide window, the rest narrow.
    assert sum(t1 - t0 for t0, t1 in calls) < 80 + 2 * 25 * len(track.crossings)


def test_a_missed_line_on_one_lap_still_places():
    track = _line_track()
    start = T0 + 10.0
    find, _ = _picture(track, start, misses=(2,))
    match = line_place(start - 12.0, 450.0, track, find)
    assert abs(match.start_utc - start) < 0.05
    assert len(match.passes) == len(track.crossings) - 1


def test_lines_in_the_corners_alone_place_nothing():
    track = _line_track()
    rng = np.random.default_rng(1)
    noise = sorted(rng.uniform(0, 450, 60))
    match = line_place(T0 + 10.0, 450.0, track, lambda t0, t1: [t for t in noise if t0 <= t <= t1])
    assert match.start_utc is None


def test_one_crossing_places_only_near_the_motion_start():
    track = _line_track(first=100.0, laps=np.array([]))
    start = T0 + 10.0
    finish = track.crossings[0] - start - LINE_LEAD_S

    def find(t0, t1):
        return [t for t in (finish,) if t0 <= t <= t1]

    assert abs(line_place(start + 4.0, 150.0, track, find).start_utc - start) < 1e-6
    assert line_place(start + 30.0, 150.0, track, find).start_utc is None


def test_track_carries_the_sessions_finish_line_crossings():
    class Session:
        def __init__(self):
            t = np.arange(0, 20, 0.04)
            self.data = {
                "time": T0 + t,
                "lat": 56.0 + t * 10 / 111_320.0,
                "lng": np.full_like(t, 13.0),
                "velocity": np.full_like(t, 36.0),
            }
            self.start_line = (56.0 + 100 / 111_320.0, 12.9999, 56.0 + 100 / 111_320.0, 13.0001)

    track = track_from_sessions([Session()])
    assert len(track.crossings) == 1
    assert abs(track.crossings[0] - (T0 + 10.0)) < 1e-3
