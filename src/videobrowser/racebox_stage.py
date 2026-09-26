"""RaceBox overlay stage for the videobrowser scan pipeline.

Given the recordings videobrowser already scans, this module:

* auto-fetches ``.vbo`` sessions from racebox.pro (optional),
* imports their laps into a shared stats SQLite for best-lap colouring,
* recuts each detected highlight from source with a compact telemetry overlay,
* renders a full per-session clip (out-lap to in-lap) and a fastest-lap clip
  with the full overlay (corner speeds disabled).

Overlay data source per clip, in priority order:
    1. a RaceBox session whose UTC range overlaps the clip,
    2. the clip's own GoPro GPMD telemetry (speed + lean),
    3. date/time only (from the clip's creation timestamp).

The overlay is composited on top of an upright clip produced by
:func:`videobrowser.clips.export_clip`, so rotation handling stays with the
existing exporter and the overlay is always drawn for an upright frame.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from videobrowser.clips import export_clip
from videobrowser.events import ClipWindow
from videobrowser.gopro import Recording, chapter_at
from videobrowser.overlay_bridge import ensure_on_path, load as _load_bridge
from videobrowser.probe import ProbeError, probe, local_date_label
from videobrowser.telemetry import (
    creation_utc_prefer_gps,
    extract_gpmd_bytes,
    gopro_clock_offset,
    gps_fixes,
    lean_at,
    lean_series,
    speed_at,
    speed_series,
)

_DEFAULT_FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
_FASTEST_LAP_PAD = 10.0  # seconds of lead-in / lead-out around the fastest lap


@dataclass
class RaceboxContext:
    sessions: list
    stats_path: Path
    font_path: str
    out_dir: str
    rb: dict
    gpu_caps: Any
    enabled: bool = True
    highlights_dir: str = "highlights"
    full_dir: str = "full_sessions"
    fastest_dir: str = "fastest_laps"


# ---------------------------------------------------------------------------
# Setup: fetch, load, stats import
# ---------------------------------------------------------------------------


def _rb() -> dict:
    return _load_bridge()


def fetch_sessions(sessions_dir: Path, headful: bool = False, max_clicks: int = 200) -> None:
    """Download new RaceBox VBO sessions into *sessions_dir* (best effort)."""
    root = ensure_on_path()
    fetcher_dir = root / "fetcher"
    user_f = fetcher_dir / "username"
    pass_f = fetcher_dir / "password"
    if not user_f.exists() or not pass_f.exists():
        print("racebox: fetch skipped (missing fetcher/username or fetcher/password)")
        return
    try:
        if str(fetcher_dir) not in sys.path:
            sys.path.insert(0, str(fetcher_dir))
        import download_sessions as dl  # type: ignore
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on env
        print(f"racebox: fetch skipped ({exc})")
        return

    sessions_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Scope the download tracker to the destination folder so pointing at a
        # fresh sessions dir actually re-downloads, rather than being skipped
        # because the file was previously saved somewhere else.
        tracker = dl.DownloadTracker(sessions_dir / "downloads.db")
        username = dl.read_secret(user_f)
        password = dl.read_secret(pass_f)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not headful)
            context = browser.new_context(accept_downloads=True)
            page = context.new_page()
            try:
                dl.login(page, username, password)
                dl.load_all_sessions(page)
                downloaded, skipped = dl.download_via_clicks(
                    page, sessions_dir, False, max_clicks, tracker
                )
                print(f"racebox: fetched {downloaded} new session(s), {skipped} skipped")
            finally:
                browser.close()
    except Exception as exc:  # pragma: no cover - network/UI dependent
        print(f"racebox: fetch failed ({exc})")


def load_sessions(sessions_dir: Path) -> list:
    rb = _rb()
    sessions: list = []
    if not sessions_dir.exists():
        return sessions
    for vbo in sorted(sessions_dir.glob("*.vbo")):
        try:
            sessions.append(rb["parse_vbo"](str(vbo)))
        except Exception as exc:
            print(f"racebox: skip {vbo.name} ({exc})")
    return sessions


def import_stats(sessions_dir: Path, stats_path: Path) -> None:
    rb = _rb()
    if not sessions_dir.exists():
        return
    for vbo in sorted(sessions_dir.glob("*.vbo")):
        try:
            rb["capture_stats"](vbo, stats_path)
        except Exception as exc:
            print(f"racebox: stats import skipped for {vbo.name} ({exc})")


def prepare(
    out_dir: Path,
    sessions_dir: Path,
    font_path: str,
    stats_path: Path,
    do_fetch: bool,
    headful: bool = False,
) -> RaceboxContext:
    """Fetch/load sessions, import stats, and build a ready-to-use context."""
    rb = _rb()
    print(f"racebox: sessions dir {sessions_dir}", flush=True)
    sessions_dir.mkdir(parents=True, exist_ok=True)
    if do_fetch:
        fetch_sessions(sessions_dir, headful=headful)
    vbos = sorted(sessions_dir.glob("*.vbo"))
    print(f"racebox: loading {len(vbos)} session file(s)", flush=True)
    sessions: list = []
    store = None
    if vbos:
        print(f"racebox: opening stats {stats_path}", flush=True)
        store = rb["StatsStore"](stats_path)
    try:
        for i, vbo in enumerate(vbos, 1):
            print(f"racebox: [{i}/{len(vbos)}] {vbo.name}", flush=True)
            try:
                session = rb["parse_vbo"](str(vbo))
            except Exception as exc:
                print(f"racebox: skip {vbo.name} ({exc})")
                continue
            sessions.append(session)
            if store is not None:
                try:
                    rb["ingest_session_stats"](session, store)
                except Exception as exc:
                    print(f"racebox: stats import skipped for {vbo.name} ({exc})")
    finally:
        if store is not None:
            store.close()
    print(f"racebox: {len(sessions)} session(s) loaded from {sessions_dir}", flush=True)
    try:
        gpu_caps = rb["detect_gpu_capabilities"]()
    except Exception:
        gpu_caps = None
    return RaceboxContext(
        sessions=sessions,
        stats_path=stats_path,
        font_path=font_path or _DEFAULT_FONT,
        out_dir=str(out_dir),
        rb=rb,
        gpu_caps=gpu_caps,
    )


# ---------------------------------------------------------------------------
# Frame sources
# ---------------------------------------------------------------------------


class _SessionFrameSource:
    """Wraps a RaceBox ``_SessionRenderer`` keyed to absolute UTC time."""

    def __init__(self, renderer, start_utc: float, fps: float) -> None:
        self.renderer = renderer
        self.start_utc = start_utc
        self.fps = fps

    def frame_for(self, idx: int) -> Optional[bytes]:
        frame = self.renderer.get_frame(self.start_utc + idx / self.fps)
        if frame is None:
            return None
        return frame if isinstance(frame, bytes) else frame.tobytes()


class _GpmdFrameSource:
    """Compact overlay driven by GoPro GPMD speed/lean, or date/time only."""

    def __init__(
        self,
        ctx: RaceboxContext,
        config,
        lean_samples: list,
        speed_samples: list,
        local_start: float,
        fps: float,
        clip_start_utc: Optional[float],
    ) -> None:
        self.rb = ctx.rb
        self.config = config
        self.lean_samples = lean_samples
        self.speed_samples = speed_samples
        self.local_start = local_start
        self.fps = fps
        self.clip_start_utc = clip_start_utc
        self._empty_map = self.rb["TrackMap"](
            points=np.zeros((0, 2), dtype=np.float32), bounds=(0.0, 0.0, 1.0, 1.0)
        )

    def frame_for(self, idx: int) -> Optional[bytes]:
        local_t = self.local_start + idx / self.fps
        lean = lean_at(self.lean_samples, local_t) if self.lean_samples else None
        speed = speed_at(self.speed_samples, local_t) if self.speed_samples else None
        wall = self.clip_start_utc + idx / self.fps if self.clip_start_utc is not None else None
        if lean is None and speed is None and wall is None:
            return None
        state = self.rb["FrameState"](
            lap_number=None,
            lap_time=None,
            on_track_faster=None,
            last_laps=[],
            speed_kmh=speed,
            best_speed_kmh=None,
            corner_display=None,
            lean_angle=lean,
            map_pos=None,
            is_in_lap=False,
            wall_time_utc=wall,
        )
        frame = self.rb["render_frame"](self.config, self._empty_map, state)
        return frame if isinstance(frame, bytes) else frame.tobytes()


# ---------------------------------------------------------------------------
# ffmpeg overlay pass
# ---------------------------------------------------------------------------


def _render_config(ctx: RaceboxContext, width: int, height: int, fps: float, variant: str):
    return ctx.rb["RenderConfig"](
        width=width,
        height=height,
        fps=max(1, int(round(fps))),
        font_path=ctx.font_path,
        variant=variant,
        show_corners=False,
    )


def _overlay_pass(
    ctx: RaceboxContext,
    src: Path,
    out_path: Path,
    width: int,
    height: int,
    fps: float,
    codec: str,
    n_frames: int,
    source,
) -> bool:
    """Composite overlay frames from *source* onto *src*, writing *out_path*."""
    rb = ctx.rb
    encoder_args, encoder_desc = rb["select_encoder_args"](codec, ctx.gpu_caps)
    fps_num, fps_den = rb["fps_fraction"](fps)
    filter_complex = rb["build_filter_complex"](width, height, width, height, False)
    # Decode the source video on the hardware media engine when possible; the
    # frames are copied to system memory so the software overlay filter still
    # works. This keeps CPU usage down for the decode half of the pass.
    hw_decode = rb["resolve_gpu_decode_args"](codec, ctx.gpu_caps, False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "info", "-stats",
        *hw_decode,
        # *src* is already rotation-baked. Default autorotate would apply a
        # leftover GoPro Display Matrix and invert the footage under an
        # upright overlay.
        "-noautorotate",
        "-display_rotation", "0",
        "-i", str(src),
        "-f", "rawvideo",
        "-pix_fmt", "bgra",
        "-s", f"{width}x{height}",
        "-r", f"{fps_num}/{fps_den}",
        "-i", "pipe:0",
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "0:a?",
        *encoder_args,
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ]

    decode_desc = "software"
    if hw_decode:
        try:
            decode_desc = hw_decode[hw_decode.index("-hwaccel") + 1]
        except (ValueError, IndexError):
            decode_desc = "hardware"
    print(f"  overlay {out_path.name}: encode={encoder_desc}, decode={decode_desc}")

    log_path = Path(ctx.out_dir) / "ffmpeg.log"
    empty = b"\x00" * (width * height * 4)
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n=== {out_path} ===\n{' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=log)
        try:
            for idx in range(max(1, n_frames)):
                frame = source.frame_for(idx)
                if proc.stdin is None:
                    break
                try:
                    proc.stdin.write(frame if frame is not None else empty)
                except BrokenPipeError:
                    break
        finally:
            if proc.stdin:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
        ret = proc.wait()
    if ret != 0:
        out_path.unlink(missing_ok=True)
        print(f"  racebox overlay ffmpeg exited {ret} for {out_path.name} (see {log_path})")
        return False
    return True


# ---------------------------------------------------------------------------
# Session matching helpers
# ---------------------------------------------------------------------------


def _session_bounds(session) -> tuple[float, float]:
    times = session.data["time"]
    return float(times[0]), float(times[-1])


def _gopro_session_offset(blob: bytes, sessions: list) -> float:
    """Seconds to add to GoPro GPS time so it sits on the RaceBox track.

    GoPro ``GPSU`` stamps run ahead of the fix they label, and the lead is
    different on every file (about 1–3 s here). The overlay clock is that
    stamp, so speed and lean show a moment the picture has not reached yet.
    """
    fixes = gps_fixes(blob)
    if len(fixes) < 40:
        return 0.0
    t0 = float(fixes[0, 0]) - 30.0
    t1 = float(fixes[-1, 0]) + 30.0
    times: list[np.ndarray] = []
    lats: list[np.ndarray] = []
    lons: list[np.ndarray] = []
    for session in sessions:
        data = getattr(session, "data", None)
        if not data or "time" not in data or "lat" not in data or "lng" not in data:
            continue
        utc = np.asarray(data["time"], dtype=np.float64)
        if utc.size < 2 or float(utc[-1]) < t0 or float(utc[0]) > t1:
            continue
        times.append(utc)
        lats.append(np.asarray(data["lat"], dtype=np.float64))
        lons.append(np.asarray(data["lng"], dtype=np.float64))
    if not times:
        return 0.0
    return gopro_clock_offset(
        fixes[:, 0],
        fixes[:, 1],
        fixes[:, 2],
        np.concatenate(times),
        np.concatenate(lats),
        np.concatenate(lons),
    )


def _sessions_overlapping(sessions: list, t0: float, t1: float) -> list:
    hits = []
    for s in sessions:
        st, en = _session_bounds(s)
        if st < t1 and en > t0:
            hits.append(s)
    return hits


def _new_session_renderer(ctx: RaceboxContext, session, config, start_utc: float):
    renderer = ctx.rb["SessionRenderer"](session, config, ctx.stats_path)
    if start_utc > renderer.t_start:
        renderer.skip_to(start_utc)
    return renderer


# ---------------------------------------------------------------------------
# Per-clip renderers
# ---------------------------------------------------------------------------


def _pick_highlight_source(
    ctx: RaceboxContext,
    recording: Recording,
    window: ClipWindow,
    width: int,
    height: int,
    fps: float,
    clip_start_utc: Optional[float],
    clip_end_utc: Optional[float],
):
    # 1. Overlapping RaceBox session -> compact overlay with track map.
    if clip_start_utc is not None and clip_end_utc is not None:
        matches = _sessions_overlapping(ctx.sessions, clip_start_utc, clip_end_utc)
        if matches:
            config = _render_config(ctx, width, height, fps, "compact")
            try:
                renderer = _new_session_renderer(ctx, matches[0], config, clip_start_utc)
                return _SessionFrameSource(renderer, clip_start_utc, fps)
            except Exception as exc:
                print(f"  racebox: session renderer failed ({exc}); falling back")

    # 2. GoPro GPMD telemetry from the source chapter -> speed + lean.
    chapter = chapter_at(recording, window.t_start)
    if chapter is not None and chapter.info.has_gpmd:
        try:
            blob = extract_gpmd_bytes(chapter.info.path, chapter.info.gpmd_index)
            lean = lean_series(chapter.info, blob)
            speed = speed_series(chapter.info, blob)
        except Exception:
            lean, speed = [], []
        if lean or speed:
            config = _render_config(ctx, width, height, fps, "compact")
            local_start = window.t_start - chapter.offset
            return _GpmdFrameSource(ctx, config, lean, speed, local_start, fps, clip_start_utc)

    # 3. Date/time only (requires a wall-clock timestamp).
    if clip_start_utc is not None:
        config = _render_config(ctx, width, height, fps, "compact")
        return _GpmdFrameSource(ctx, config, [], [], 0.0, fps, clip_start_utc)

    return None


def _highlight_name(window: ClipWindow) -> str:
    types = "+".join(window.types) if window.types else "clip"
    return f"{window.recording_id}_{int(window.t_start):05d}_{types}.mp4"


def _render_highlight(
    ctx: RaceboxContext,
    recording: Recording,
    window: ClipWindow,
    rotation: int,
    rec_start_utc: Optional[float],
    out_dir: Path,
) -> None:
    out_path = out_dir / _highlight_name(window)
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp) / "clip.mp4"
        try:
            export_clip(recording, window, temp, rotation=rotation)
        except Exception as exc:
            print(f"  racebox: highlight cut failed {out_path.name}: {exc}")
            return
        try:
            meta = probe(temp)
        except ProbeError as exc:
            print(f"  racebox: probe failed {out_path.name}: {exc}")
            return

        fps = meta.fps or 30.0
        clip_start_utc = rec_start_utc + window.t_start if rec_start_utc is not None else None
        clip_end_utc = (
            clip_start_utc + (window.t_end - window.t_start)
            if clip_start_utc is not None
            else None
        )
        source = _pick_highlight_source(
            ctx, recording, window, meta.width, meta.height, fps, clip_start_utc, clip_end_utc
        )
        if source is None:
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(temp, out_path)
            return
        n_frames = int(meta.duration * fps) + 1
        _overlay_pass(
            ctx, temp, out_path, meta.width, meta.height, fps, meta.codec, n_frames, source
        )


def _render_session_span(
    ctx: RaceboxContext,
    recording: Recording,
    session,
    rotation: int,
    rec_start_utc: float,
    rec_end_utc: float,
    u0: float,
    u1: float,
    out_path: Path,
) -> None:
    if u1 - u0 < 1.0:
        return
    g0 = u0 - rec_start_utc
    g1 = u1 - rec_start_utc
    window = ClipWindow(
        t_start=g0,
        t_end=g1,
        recording_id=recording.recording_id,
        types=["full"],
        score=0.0,
        events=[],
    )
    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp) / "clip.mp4"
        try:
            export_clip(recording, window, temp, rotation=rotation)
        except Exception as exc:
            print(f"  racebox: cut failed {out_path.name}: {exc}")
            return
        try:
            meta = probe(temp)
        except ProbeError as exc:
            print(f"  racebox: probe failed {out_path.name}: {exc}")
            return
        fps = meta.fps or 30.0
        config = _render_config(ctx, meta.width, meta.height, fps, "full")
        try:
            renderer = _new_session_renderer(ctx, session, config, u0)
        except Exception as exc:
            print(f"  racebox: renderer failed {out_path.name}: {exc}")
            return
        source = _SessionFrameSource(renderer, u0, fps)
        n_frames = int(meta.duration * fps) + 1
        _overlay_pass(
            ctx, temp, out_path, meta.width, meta.height, fps, meta.codec, n_frames, source
        )


def _render_full_session(
    ctx: RaceboxContext,
    recording: Recording,
    session,
    rotation: int,
    rec_start_utc: float,
    rec_end_utc: float,
    out_dir: Path,
) -> None:
    st, en = _session_bounds(session)
    u0 = max(st, rec_start_utc)
    u1 = min(en, rec_end_utc)
    out_path = out_dir / f"{Path(session.path).stem}__{recording.recording_id}.mp4"
    _render_session_span(
        ctx, recording, session, rotation, rec_start_utc, rec_end_utc, u0, u1, out_path
    )


def _render_fastest_lap(
    ctx: RaceboxContext,
    recording: Recording,
    session,
    rotation: int,
    rec_start_utc: float,
    rec_end_utc: float,
    out_dir: Path,
) -> None:
    st, en = _session_bounds(session)
    # Build a throwaway renderer just to reuse its lap detection.
    probe_config = _render_config(ctx, 16, 16, 30.0, "full")
    try:
        laps = ctx.rb["SessionRenderer"](session, probe_config, ctx.stats_path).laps
    except Exception as exc:
        print(f"  racebox: lap detection failed for {Path(session.path).name}: {exc}")
        return
    timed = [lap for lap in laps if not lap.is_out_lap and not lap.is_in_lap]
    if not timed:
        return
    fastest = min(timed, key=lambda lap: lap.lap_time)
    timed_sorted = sorted(timed, key=lambda lap: lap.start_time)
    lap_number = timed_sorted.index(fastest) + 1

    u0 = max(fastest.start_time - _FASTEST_LAP_PAD, rec_start_utc, st)
    u1 = min(fastest.end_time + _FASTEST_LAP_PAD, rec_end_utc, en)
    out_path = (
        out_dir
        / f"{Path(session.path).stem}__lap{lap_number}__{recording.recording_id}.mp4"
    )
    _render_session_span(
        ctx, recording, session, rotation, rec_start_utc, rec_end_utc, u0, u1, out_path
    )


# ---------------------------------------------------------------------------
# Entry point called from the scan pipeline
# ---------------------------------------------------------------------------


def run_for_recording(
    recording: Recording,
    windows: list[ClipWindow],
    rotation: int,
    ctx: RaceboxContext,
) -> None:
    if ctx is None or not ctx.enabled:
        return
    if not recording.chapters:
        return

    out_dir = Path(ctx.out_dir)

    info0 = recording.chapters[0].info
    blob = None
    if info0.has_gpmd and info0.gpmd_index is not None:
        blob = extract_gpmd_bytes(info0.path, info0.gpmd_index)
    rec_start_utc = creation_utc_prefer_gps(info0, blob)
    if rec_start_utc is not None and blob and ctx.sessions:
        offset = _gopro_session_offset(blob, ctx.sessions)
        if abs(offset) >= 0.05:
            print(f"  racebox clock aligned by {offset:+.2f}s")
            rec_start_utc += offset
    rec_end_utc = rec_start_utc + recording.duration if rec_start_utc is not None else None
    date_label = local_date_label(rec_start_utc)

    print(
        f"  racebox overlays: {len(windows)} highlight(s)"
        + (", start UTC known" if rec_start_utc is not None else ", no UTC timestamp")
    )

    highlights_dir = out_dir / ctx.highlights_dir / date_label
    for window in windows:
        _render_highlight(ctx, recording, window, rotation, rec_start_utc, highlights_dir)

    if rec_start_utc is None or rec_end_utc is None or not ctx.sessions:
        return

    matched = _sessions_overlapping(ctx.sessions, rec_start_utc, rec_end_utc)
    for session in matched:
        session_date = local_date_label(session.session_start_utc.timestamp())
        _render_full_session(
            ctx,
            recording,
            session,
            rotation,
            rec_start_utc,
            rec_end_utc,
            out_dir / ctx.full_dir / session_date,
        )
        _render_fastest_lap(
            ctx,
            recording,
            session,
            rotation,
            rec_start_utc,
            rec_end_utc,
            out_dir / ctx.fastest_dir / session_date,
        )
