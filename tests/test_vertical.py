import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from videobrowser.events import ClipWindow, Event
from videobrowser.gopro import Chapter, Recording
from videobrowser.overlay_bridge import ensure_on_path
from videobrowser.probe import probe
from videobrowser.riders import TrackPoint
from videobrowser.vertical import (
    MAX_PAN_PER_S,
    FocusSample,
    crop_centers,
    crop_offsets,
    crop_size,
    export_vertical,
    focus_samples,
    sendcmd_script,
)

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg missing")


def _pt(t, track_id, cx, h=0.3, w=0.1, frame_w=640, frame_h=360):
    half_w = w * frame_w / 2
    half_h = h * frame_h / 2
    x = cx * frame_w
    return TrackPoint(
        t=t,
        x1=x - half_w,
        y1=frame_h / 2 - half_h,
        x2=x + half_w,
        y2=frame_h / 2 + half_h,
        conf=0.9,
        cls=3,
        track_id=track_id,
        frame_w=frame_w,
        frame_h=frame_h,
    )


def _window(events=(), t0=0.0, t1=10.0):
    return ClipWindow(t0, t1, "GX-0001", ["nearby_rider"], 1.0, list(events))


def _event(track_id):
    return Event("nearby_rider", 1.0, 2.0, 0.5, "GX-0001", track_id=str(track_id))


def test_crop_size_is_9_by_16_and_even():
    assert crop_size(1920, 1080) == (608, 1080)
    assert crop_size(3840, 2160) == (1214, 2160)
    assert crop_size(1080, 1920) == (1080, 1920)
    w, h = crop_size(1440, 1920)
    assert w <= 1440 and h <= 1920 and w % 2 == 0 and h % 2 == 0


def test_focus_prefers_event_track_over_bigger_bystander():
    tracks = {
        1: [_pt(1.0, 1, 0.2, h=0.2)],
        2: [_pt(1.0, 2, 0.8, h=0.6, w=0.3)],
    }
    samples = focus_samples(_window([_event(1)]), tracks)
    assert [round(s.cx, 2) for s in samples] == [0.2]


def test_focus_prefers_riders_in_view_over_puck_and_ignores_tiny_boxes():
    tracks = {
        5: [_pt(1.0, 5, 0.5)],  # rider ahead, e.g. through a corner exit
        6: [_pt(2.0, 6, 0.1, h=0.01)],  # too far away to follow
    }
    samples = focus_samples(_window(), tracks, puck_positions=[(1.0, 0.85), (2.0, 0.8)])
    # Rider wins at t=1; at t=2 nobody followable is in view, so the puck steers.
    assert [(s.t, round(s.cx, 2)) for s in samples] == [(1.0, 0.5), (2.0, 0.8)]


def test_focus_keeps_current_target_until_clearly_smaller():
    tracks = {
        1: [_pt(0.0, 1, 0.3, w=0.10), _pt(0.2, 1, 0.3, w=0.10), _pt(0.4, 1, 0.3, w=0.05)],
        2: [_pt(0.0, 2, 0.7, w=0.09), _pt(0.2, 2, 0.7, w=0.12), _pt(0.4, 2, 0.7, w=0.12)],
    }
    samples = focus_samples(_window(), tracks)
    # Starts on the larger rider, stays while the other is only a bit larger,
    # switches once it is less than half the size of the biggest.
    assert [round(s.cx, 1) for s in samples] == [0.3, 0.3, 0.7]


def test_focus_ignores_points_outside_window():
    tracks = {1: [_pt(0.5, 1, 0.2), _pt(5.0, 1, 0.6), _pt(20.0, 1, 0.9)]}
    samples = focus_samples(_window(t0=1.0, t1=10.0), tracks)
    assert [s.t for s in samples] == [5.0]


def test_crop_centers_fallback_is_clamped():
    centers = crop_centers([], 0.0, 10, 25.0, half_width=0.2, fallback=0.05)
    assert np.allclose(centers, 0.2)


def test_crop_centers_clamp_to_frame_and_limit_pan_speed():
    fps = 25.0
    samples = [FocusSample(0.0, 0.0), FocusSample(1.0, 0.0), FocusSample(1.04, 1.0)]
    centers = crop_centers(samples, 0.0, 100, fps, half_width=0.16)
    assert centers.min() >= 0.16 - 1e-9 and centers.max() <= 0.84 + 1e-9
    assert np.max(np.abs(np.diff(centers))) <= MAX_PAN_PER_S / fps + 1e-9
    # Gets there in the end and holds after the last sample.
    assert centers[-1] == pytest.approx(0.84, abs=1e-3)


def test_crop_centers_start_centred_when_subject_appears_late():
    fps = 25.0
    # Subject only shows up 4 s into the clip, at the far right.
    samples = [FocusSample(4.0, 0.9), FocusSample(5.0, 0.9)]
    centers = crop_centers(samples, 0.0, 150, fps, half_width=0.16)
    # Opens centred instead of on the right edge, then pans over to it.
    assert centers[0] == pytest.approx(0.5, abs=1e-3)
    assert np.allclose(centers[: int(2.0 * fps)], 0.5, atol=0.01)
    assert centers[-1] == pytest.approx(0.84, abs=1e-3)
    assert np.max(np.abs(np.diff(centers))) <= MAX_PAN_PER_S / fps + 1e-9


def test_crop_centers_start_on_subject_already_in_view():
    samples = [FocusSample(0.2, 0.3), FocusSample(1.0, 0.3)]
    centers = crop_centers(samples, 0.0, 50, 25.0, half_width=0.16)
    assert np.allclose(centers, 0.3)


def test_crop_centers_hold_ends_and_follow_steady_subject():
    # First sample within LEAD_IN_S of the start: held back to frame 0.
    samples = [FocusSample(0.8, 0.4), FocusSample(3.0, 0.4)]
    centers = crop_centers(samples, 0.0, 125, 25.0, half_width=0.16)
    assert np.allclose(centers, 0.4)


def test_crop_offsets_are_even_and_in_frame():
    offsets = crop_offsets(np.array([0.0, 0.5, 1.0, 0.3337]), 1920, 608)
    assert offsets.tolist()[0] == 0
    assert offsets.tolist()[2] == 1920 - 608
    assert all(x % 2 == 0 for x in offsets.tolist())
    assert 0 <= offsets.max() <= 1920 - 608


def test_sendcmd_only_emits_changes():
    script = sendcmd_script(np.array([10, 10, 12, 12, 14]), 25.0)
    assert script.splitlines() == [
        "0.0000 crop@vc x 10;",
        "0.0800 crop@vc x 12;",
        "0.1600 crop@vc x 14;",
    ]


def test_vertical_layout_stays_inside_safe_zone():
    ensure_on_path()
    from overlay import overlay_render as orr

    config = orr.RenderConfig(
        width=1080, height=1920, fps=30, font_path="DejaVuSans.ttf", variant="vertical"
    )
    state = orr.FrameState(
        lap_number=3,
        lap_time=91.2,
        on_track_faster=True,
        last_laps=[],
        speed_kmh=187.4,
        best_speed_kmh=None,
        corner_display=None,
        lean_angle=-48.0,
        map_pos=(0.5, 0.5),
        wall_time_utc=1_780_000_000.0,
    )
    track = orr.TrackMap(
        points=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32), bounds=(0, 0, 1, 1)
    )
    frame = orr.render_frame(config, track, state)
    alpha = np.frombuffer(frame, dtype=np.uint8).reshape(1920, 1080, 4)[:, :, 3]
    ys, xs = np.nonzero(alpha)
    assert len(ys) > 0
    top = int(1920 * 0.12)
    # Clock starts `top` below the top edge; speed/lean end `top` above the bottom.
    assert ys.min() >= top - 2
    assert abs((1920 - ys.max()) - top) <= 4
    # Speed sits bottom-left and lean bottom-right on the same line; the lean
    # gauge (box + needle) is not drawn.
    bottom = ys > 1920 // 2
    assert xs[bottom].min() < 1080 * 0.15
    assert xs[bottom].max() > 1080 * 0.85
    band = alpha[1920 // 2 :]
    assert not band[:, int(1080 * 0.55) : int(1080 * 0.75)].any()
    assert ys[bottom].min() > 1920 - top - 120
    # Compact variant: no lap timer text in the vertical HUD.
    assert orr.is_compact(config)


def _make_moving_box(path: Path) -> None:
    # White 60x120 box on black, sliding left -> right over 2 s at 25 fps.
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=640x360:d=2:r=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
            "-f", "lavfi", "-i", "color=c=white:s=60x120:d=2:r=25",
            "-filter_complex", "[0:v][2:v]overlay=x='40+t*250':y=120:shortest=1[v]",
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(path),
        ],
        check=True,
    )


def _recording(src: Path) -> Recording:
    info = probe(src)
    return Recording(
        recording_id="GX-0001",
        prefix="GX",
        recording="0001",
        chapters=[
            Chapter(
                path=src, prefix="GX", chapter=1, recording="0001",
                duration=info.duration, offset=0.0, info=info,
            )
        ],
    )


def _frame_gray(path: Path, t: float, w: int, h: int) -> np.ndarray:
    raw = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(path),
            "-frames:v", "1", "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w)


@needs_ffmpeg
def test_export_vertical_follows_subject(tmp_path: Path):
    src = tmp_path / "src.mp4"
    _make_moving_box(src)
    recording = _recording(src)
    window = ClipWindow(0.0, 2.0, "GX-0001", ["nearby_rider"], 1.0, [])
    samples = [
        FocusSample(t, (70 + t * 250) / 640.0) for t in np.arange(0.0, 2.0, 0.2)
    ]
    dest = tmp_path / "vertical" / "out.mp4"
    export_vertical(recording, window, dest, 0, samples)
    out = probe(dest)
    assert (out.width, out.height) == (1080, 1920)
    assert out.duration == pytest.approx(2.0, abs=0.15)
    # Mid-clip the box is inside the crop, near the middle horizontally.
    frame = _frame_gray(dest, 1.2, 1080, 1920)
    cols = np.nonzero(frame[960] > 128)[0]
    assert cols.size > 0
    assert 1080 * 0.2 < cols.mean() < 1080 * 0.8


class _SolidOverlay:
    def frame_for(self, idx):
        img = np.zeros((1920, 1080, 4), dtype=np.uint8)
        img[100:200, 100:200] = (0, 0, 255, 255)  # opaque red square (BGRA)
        return img.tobytes()


@needs_ffmpeg
def test_export_vertical_composites_overlay(tmp_path: Path):
    src = tmp_path / "src.mp4"
    _make_moving_box(src)
    dest = tmp_path / "out.mp4"
    export_vertical(
        _recording(src),
        ClipWindow(0.0, 1.0, "GX-0001", [], 1.0, []),
        dest,
        0,
        [],
        overlay_factory=lambda w, h, fps: _SolidOverlay(),
    )
    frame = _frame_gray(dest, 0.5, 1080, 1920)
    assert frame[150, 150] > 40  # red shows up as mid grey on black
    assert frame[1800, 540] < 30


@pytest.mark.parametrize("overlay", [True, False])
def test_export_verticals_overlay_follows_config(tmp_path: Path, monkeypatch, overlay):
    from videobrowser import pipeline, racebox_stage, vertical
    from videobrowser.profiles import UNKNOWN

    calls = []
    monkeypatch.setattr(racebox_stage, "overlay_context", lambda out, font: object())
    monkeypatch.setattr(racebox_stage, "recording_start_utc", lambda rec, ctx: None)
    monkeypatch.setattr(
        vertical, "export_vertical", lambda *a, **kw: calls.append(kw["overlay_factory"])
    )
    config = pipeline.ScanConfig(out_dir=tmp_path, overlay=overlay)
    pipeline._export_verticals(
        Recording(recording_id="GX-0001", prefix="GX", recording="0001", chapters=[]),
        [_window()],
        UNKNOWN,
        config,
        {},
        [],
        None,
        tmp_path,
    )
    assert len(calls) == 1
    assert (calls[0] is not None) is overlay


def test_tracks_in_puck_roi_are_followed_but_not_scored():
    from videobrowser.pipeline import _in_roi, _merge_tracks
    from videobrowser.profiles import UNKNOWN

    ahead = _pt(1.0, 1, 0.5, h=0.1)  # centre-top of frame: inside the puck ROI
    beside = _pt(1.0, 2, 0.9, h=0.3)
    assert _in_roi(ahead, UNKNOWN.puck_roi)
    assert not _in_roi(beside, UNKNOWN.puck_roi)
    assert not _in_roi(ahead, None)

    later = _pt(2.0, 1, 0.8, h=0.1)
    merged = _merge_tracks({1: [later], 2: [beside]}, {1: [ahead]})
    assert [p.t for p in merged[1]] == [1.0, 2.0]
    assert merged[2] == [beside]
