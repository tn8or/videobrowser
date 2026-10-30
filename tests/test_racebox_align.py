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


def test_a_pass_past_the_end_of_the_recording_places_nothing():
    track = _line_track(first=100.0, laps=np.array([]))
    start = T0 + 10.0
    duration = track.crossings[0] - start - LINE_LEAD_S - 0.01

    def find(t0, t1):
        return [duration + 0.01] if t0 <= duration <= t1 else []

    assert line_place(start, duration, track, find).start_utc is None


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


# --- venue landmark (the bridge) ------------------------------------------------

from videobrowser.racebox_align import along_track, landmark_match, venue_landmark_m  # noqa: E402


def test_venues_are_told_by_their_start_line_in_racebox_minutes():
    assert venue_landmark_m((3359.21818, -786.87418, 3359.20913, -786.86331)) == (56.0, 666.0)  # Knutstorp
    assert venue_landmark_m((3435.68832, -815.965, 3435.69406, -815.94955)) == (23.0,)  # Anderstorp
    assert venue_landmark_m((3300.0, -600.0, 3300.01, -600.0)) == ()


def test_landmark_is_metres_along_the_track_past_each_crossing():
    # North at 10 m/s, then 20 m/s from t=12, in RaceBox minutes.
    t = np.arange(0, 30, 0.04)
    v = np.where(t < 12, 10.0, 20.0)
    north = np.r_[0.0, np.cumsum(v[:-1] * 0.04)]
    lat = (56.0 + north / 111_320.0) * 60.0
    lon = np.full_like(t, 13.0 * 60.0)
    (u,) = along_track(T0 + t, lat, lon, [T0 + 10.0], 51.0)
    # 20 m at 10 m/s, then 31 m at 20 m/s.
    assert abs(u - (T0 + 12.0 + 31.0 / 20.0)) < 0.01


def test_bridge_passes_place_a_recording_half_a_lap_off():
    landmarks = T0 + 100.0 + np.r_[0.0, np.cumsum(LAPS)]
    start = T0 + 10.0
    rng = np.random.default_rng(2)
    passes = list(landmarks - start + rng.normal(0, 0.05, len(landmarks)))[1:]  # first one missed
    match = landmark_match(passes, landmarks, [start + 28.9])
    assert abs(match.start_utc - start) < 0.1
    assert len(match.passes) == len(landmarks) - 1


def test_one_bridge_pass_is_not_enough():
    landmarks = T0 + 100.0 + np.r_[0.0, np.cumsum(LAPS)]
    assert landmark_match([90.0], landmarks, [T0 + 10.0]).start_utc is None


from videobrowser.racebox_align import BRIDGE_REFINE_S, bridge_place  # noqa: E402


def _bridge_track(first=100.0, laps=LAPS):
    track = _line_track(first, laps)
    # The bridge 56 m on at 176 km/h.
    track.landmarks = track.crossings + 56.0 / (176.0 / 3.6)
    return track


def _bridges(track, start, lag=0.03, misses=()):
    """Bridge passes the picture would show (a hair after RaceBox's), plus
    the windows asked for."""
    out = [u - start + lag for i, u in enumerate(track.landmarks) if i not in misses]
    calls = []

    def find(t0, t1):
        calls.append((t0, t1))
        return [t for t in out if t0 <= t <= t1]

    return find, calls


def test_bridge_places_a_recording_motion_put_29_s_off_and_the_line_refines_it():
    track = _bridge_track()
    start = T0 + 10.0
    bridges, calls = _bridges(track, start, lag=0.08)
    lines, _ = _picture(track, start)
    match = bridge_place(start + 28.9, 500.0, track, bridges, lines)
    assert abs(match.start_utc - start) < 0.03
    assert "finish line" in match.reason
    # Windows around a few landmark passes, not the whole recording.
    assert sum(b - a for a, b in calls) <= 3 * 80.0 + 1e-6


def test_bridge_start_stands_when_the_line_is_in_shadow():
    track = _bridge_track()
    start = T0 + 10.0
    bridges, _ = _bridges(track, start, lag=0.08)
    match = bridge_place(start - 20.0, 500.0, track, bridges, lambda a, b: [])
    assert abs(match.start_utc - (start - 0.08)) < 0.01
    assert "no finish line" in match.reason


def test_line_refine_ignores_the_checkered_line_before_the_finish():
    track = _bridge_track()
    start = T0 + 10.0
    bridges, _ = _bridges(track, start)
    # Only the checkered line, 0.58 s before the finish line, shows.
    checkered = [c - start - LINE_LEAD_S - 0.58 for c in track.crossings]
    match = bridge_place(start, 500.0, track, bridges, lambda a, b: [t for t in checkered if a <= t <= b])
    assert abs(match.start_utc - (start - 0.03)) < 0.01
    assert 0.58 > BRIDGE_REFINE_S > 0.26


def test_one_bridge_places_nothing():
    track = _bridge_track(laps=np.array([]))
    start = T0 + 10.0
    bridges, _ = _bridges(track, start, lag=0.0)
    assert bridge_place(start + 4.0, 200.0, track, bridges).start_utc is None


def test_line_refine_absorbs_a_camera_that_sees_the_bridge_leave_early():
    track = _bridge_track()
    start = T0 + 10.0
    # This camera loses the bridge 0.25 s before RaceBox's time.
    bridges, _ = _bridges(track, start, lag=-0.25)
    lines, _ = _picture(track, start)
    match = bridge_place(start, 500.0, track, bridges, lines)
    assert abs(match.start_utc - start) < 0.03
    assert "finish line" in match.reason and "not enough" not in match.reason


def test_no_bridge_in_the_picture_places_nothing():
    track = _bridge_track()
    assert bridge_place(T0, 500.0, track, lambda a, b: []).start_utc is None


def test_line_refine_takes_the_finish_line_over_the_grid_line_before_it():
    track = _bridge_track()
    start = T0 + 10.0
    bridges, _ = _bridges(track, start, lag=0.05)
    lines, _ = _picture(track, start)
    match = bridge_place(start, 500.0, track, bridges, lines)
    assert abs(match.start_utc - start) < 0.03


def test_one_line_pass_does_not_move_the_bridge_start():
    track = _bridge_track()
    start = T0 + 10.0
    bridges, _ = _bridges(track, start, lag=0.08)
    first = track.crossings[0] - start - LINE_LEAD_S
    match = bridge_place(start, 500.0, track, bridges, lambda a, b: [first] if a <= first <= b else [])
    assert abs(match.start_utc - (start - 0.08)) < 0.01
    assert "not enough" in match.reason


from videobrowser.racebox_align import bridge_place_anywhere  # noqa: E402


def _day(seed=3):
    """A day at Knutstorp: 7 sessions of 10 laps, two bridges a lap."""
    rng = np.random.default_rng(seed)
    utc = T0 + np.arange(0.0, 8 * 3600.0, 0.5)
    crossings, landmarks = [], []
    t = T0
    for _ in range(7):
        t += 3600.0
        for _ in range(10):
            lap = 77.0 + rng.normal(0, 0.5)
            crossings.append(t)
            landmarks += [t + 1.15, t + 0.333 * lap]
            t += lap
    track = RaceboxTrack(utc, np.full(len(utc), 176.0), np.zeros(len(utc)), np.array(crossings), np.array(landmarks))
    return track, rng


def test_whole_recording_finds_the_bridge_hours_off_the_camera_clock():
    track, rng = _day()
    start = track.landmarks[30] - 150.0
    passes = [u - start + rng.normal(0, 0.03) for u in track.landmarks if 0 < u - start < 900]
    passes += list(rng.uniform(0, 900, 28))  # paddock and other bridges
    clock = start - 7230.5  # local time read as UTC
    match = bridge_place_anywhere(clock, 900.0, track, lambda a, b: sorted(passes), lambda a, b: [])
    assert abs(match.start_utc - start) < 0.05


def test_whole_recording_of_the_paddock_places_nothing():
    track, rng = _day()
    for seed in range(5):
        fake = sorted(np.random.default_rng(seed).uniform(0, 1500, 40))
        assert bridge_place_anywhere(T0 + 3 * 3600, 1500.0, track, lambda a, b: fake).start_utc is None


def test_whole_recording_needs_more_than_a_lap():
    track, rng = _day()
    start = track.landmarks[30] - 20.0
    passes = [u - start for u in track.landmarks if 0 < u - start < 100]
    calls = []

    def find(a, b):
        calls.append((a, b))
        return passes

    assert bridge_place_anywhere(start, 100.0, track, find).start_utc is None
    # Too short to be worth decoding.
    assert calls == []


def test_line_refine_ignores_lines_past_the_window():
    track = _bridge_track()
    start = T0 + 10.0
    bridges, _ = _bridges(track, start, lag=0.05)
    # Something line-like 0.4 s after every finish line (Anderstorp).
    lines = [c - start - LINE_LEAD_S for c in track.crossings]
    lines += [t + 0.4 for t in lines]
    match = bridge_place(start, 500.0, track, bridges, lambda a, b: [t for t in lines if a <= t <= b])
    assert abs(match.start_utc - start) < 0.01


def test_whole_recording_is_not_decoded_without_enough_racebox_laps():
    track, rng = _day()
    track.landmarks = track.landmarks[:4]
    calls = []
    match = bridge_place_anywhere(T0, 900.0, track, lambda a, b: calls.append((a, b)) or [])
    assert match.start_utc is None and "at most 4 landmark passes" in match.reason
    assert calls == []


def test_whole_recording_is_not_decoded_when_the_camera_sees_no_sky():
    track, rng = _day()
    calls = []
    match = bridge_place_anywhere(
        track.landmarks[30] - 7230.0, 900.0, track, lambda a, b: calls.append((a, b)) or [], sky=lambda: 0.005
    )
    assert "pitched too low" in match.reason
    assert calls == []
