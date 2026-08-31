from videobrowser.events import Event, clip_windows, merge_same_type


def _event(kind: str, start: float, end: float, score: float = 0.5) -> Event:
    return Event(type=kind, t_start=start, t_end=end, score=score, recording_id="GX-0009")


def test_merge_same_type_gap():
    events = merge_same_type(
        [
            _event("nearby_rider", 1, 2),
            _event("nearby_rider", 2.4, 3),
            _event("nearby_rider", 10, 11),
        ]
    )
    nearby = [e for e in events if e.type == "nearby_rider"]
    assert len(nearby) == 2
    assert nearby[0].t_start == 1
    assert nearby[0].t_end == 3


def test_clip_windows_pad_and_merge():
    events = [
        _event("nearby_rider", 20, 22, 0.4),
        _event("approaching_rider", 21, 24, 0.9),
    ]
    windows = clip_windows(events, pad=10, duration=100)
    assert len(windows) == 1
    assert windows[0].t_start == 12.5
    assert windows[0].t_end == 32.5
    assert "approaching_rider" in windows[0].types
    assert windows[0].score == 0.9


def test_clip_windows_do_not_span_whole_session():
    events = [
        _event("nearby_rider", 10, 400, 0.2),
        _event("approaching_rider", 50, 52, 0.9),
        _event("approaching_rider", 200, 202, 0.8),
    ]
    windows = clip_windows(events, pad=10, duration=500, max_len=24, max_clips=12)
    assert len(windows) >= 2
    assert all(w.t_end - w.t_start <= 24.5 for w in windows)
    peaks = [0.5 * (e.t_start + e.t_end) for e in events[1:]]
    covered = [any(w.t_start <= p <= w.t_end for w in windows) for p in peaks]
    assert all(covered)
