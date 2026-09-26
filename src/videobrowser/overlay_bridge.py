"""Bridge that exposes the standalone ``racebox/overlay`` package to videobrowser.

The RaceBox overlay code lives under ``<repo>/racebox/overlay`` as its own
package (``overlay``) with relative imports. Rather than move or repackage it,
we add ``racebox/`` to ``sys.path`` on demand and re-export the handful of
symbols the merged pipeline needs.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path


def _racebox_root() -> Path:
    # src/videobrowser/overlay_bridge.py -> repo root is three parents up.
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "racebox"


@lru_cache(maxsize=1)
def ensure_on_path() -> Path:
    """Add ``racebox/`` to ``sys.path`` so ``import overlay.*`` resolves."""
    root = _racebox_root()
    if not root.exists():
        raise ModuleNotFoundError(f"racebox project not found at {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def load():
    """Import and return the RaceBox overlay symbols used by the pipeline."""
    ensure_on_path()

    from overlay.cli import capture_stats, ingest_session_stats  # noqa: WPS433 (runtime import by design)
    from overlay.lap_timing import Lap, compute_sectors, detect_laps
    from overlay.overlay_render import (
        FrameState,
        LapRow,
        RenderConfig,
        render_frame,
    )
    from overlay.stats import StatsStore
    from overlay.track_map import TrackMap, build_track_map, map_point
    from overlay.vbo_parser import VboSession, parse_vbo
    from overlay.video_compositor import (
        GpuCapabilities,
        VideoMetadata,
        _SessionRenderer,
        _build_filter_complex,
        _detect_gpu_capabilities,
        _fps_fraction,
        _resolve_gpu_decode_args,
        _select_encoder_args,
        get_video_metadata,
    )

    return {
        "capture_stats": capture_stats,
        "ingest_session_stats": ingest_session_stats,
        "Lap": Lap,
        "compute_sectors": compute_sectors,
        "detect_laps": detect_laps,
        "FrameState": FrameState,
        "LapRow": LapRow,
        "RenderConfig": RenderConfig,
        "render_frame": render_frame,
        "StatsStore": StatsStore,
        "TrackMap": TrackMap,
        "build_track_map": build_track_map,
        "map_point": map_point,
        "VboSession": VboSession,
        "parse_vbo": parse_vbo,
        "GpuCapabilities": GpuCapabilities,
        "VideoMetadata": VideoMetadata,
        "SessionRenderer": _SessionRenderer,
        "build_filter_complex": _build_filter_complex,
        "detect_gpu_capabilities": _detect_gpu_capabilities,
        "fps_fraction": _fps_fraction,
        "resolve_gpu_decode_args": _resolve_gpu_decode_args,
        "select_encoder_args": _select_encoder_args,
        "get_video_metadata": get_video_metadata,
    }
