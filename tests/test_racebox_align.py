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
