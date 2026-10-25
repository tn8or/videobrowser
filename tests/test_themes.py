import pytest
import numpy as np

from videobrowser.themes import CamEvent, detect_crashes, facing, find_moments


def _ev(rid, face, kind, start, end, score=0.5, offset=1000.0):
    return CamEvent(rid, face, kind, start, end, score, offset)


def test_facing_from_profile():
    assert facing("rear") == facing("rear_right") == "rear"
    assert facing("cockpit") == facing("side") == "front"


def test_front_then_rear_is_an_overtake():
    # Rear camera started 5 s later, so its recording time runs 5 s behind.
    events = [
        _ev("F", "front", "approaching_rider", 100, 104, 0.9, offset=1000.0),
        _ev("R", "rear", "receding_rider", 100.5, 103, 0.7, offset=1005.0),
    ]
    moments = find_moments(events)
    themes = [m.theme for m in moments]
    assert themes == ["overtake"]
    (m,) = moments
    assert [s.recording_id for s in m.segments] == ["F", "R"]
    assert m.segments[1].t_start < 100.5


def test_rear_then_front_is_being_overtaken():
    events = [
        _ev("R", "rear", "approaching_rider", 50, 53, 0.8),
        _ev("F", "front", "receding_rider", 54, 57, 0.6),
    ]
    assert [m.theme for m in find_moments(events)] == ["overtaken"]


def test_handover_too_late_is_only_catching_up():
    events = [
        _ev("F", "front", "approaching_rider", 100, 104, 0.9),
        _ev("R", "rear", "receding_rider", 120, 123, 0.7),
    ]
    themes = sorted(m.theme for m in find_moments(events))
    assert themes == ["action", "catching_up"]


def test_short_approach_is_action_and_kneepuck_has_its_own_theme():
    events = [
        _ev("F", "front", "approaching_rider", 10, 10.8, 0.4),
        _ev("S", "front", "kneepuck_visible", 40, 41, 0.6),
    ]
    themes = sorted(m.theme for m in find_moments(events))
    assert themes == ["action", "kneepuck"]


def test_action_overlapping_a_theme_is_dropped():
    events = [
        _ev("F", "front", "approaching_rider", 100, 104, 0.9),
        _ev("F", "front", "nearby_rider", 101, 104, 0.3),
    ]
    assert [m.theme for m in find_moments(events)] == ["catching_up"]


def _telemetry(stop: bool, jolt: float):
    speed_t = np.arange(0, 60, 0.1)
    speed = np.full_like(speed_t, 90.0)
    if stop:
        speed[speed_t > 30] = 0.0
    accl_t = np.arange(0, 60, 0.005)
    accl = np.tile([0.0, 9.81, 0.0], (len(accl_t), 1))
    accl[np.argmin(np.abs(accl_t - 29.9))] = [9.81 * jolt, 9.81, 0.0]
    return speed_t, speed, accl_t, accl


def test_crash_needs_jolt_and_stop():
    assert detect_crashes(*_telemetry(stop=True, jolt=5.0)) == [pytest.approx(29.9)]
    assert detect_crashes(*_telemetry(stop=False, jolt=5.0)) == []  # kerb strike
    assert detect_crashes(*_telemetry(stop=True, jolt=1.2)) == []  # braking to a stop


def test_incident_events_become_their_own_themes():
    events = [
        _ev("F", "front", "rider_down", 100, 102, 2.0),
        _ev("F", "front", "off_track", 300, 302, 1.0),
    ]
    themes = sorted(m.theme for m in find_moments(events))
    assert themes == ["crash_other", "off_track"]


def test_rider_pass_is_a_duel():
    moments = find_moments([_ev("F", "front", "rider_pass", 30, 31.4, 1.5)])
    assert [m.theme for m in moments] == ["duel"]
    assert moments[0].label == "DUEL"
