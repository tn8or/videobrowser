from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProbeError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    duration: float
    width: int
    height: int
    fps: float
    rotation: int
    codec: str
    has_gpmd: bool
    gpmd_index: int | None


def ffprobe_json(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, check=True, text=True)
    except FileNotFoundError as exc:
        raise ProbeError("ffprobe not found on PATH") from exc
    except subprocess.CalledProcessError as exc:
        raise ProbeError(exc.stderr.strip() or "ffprobe failed") from exc
    return json.loads(proc.stdout)


def normalize_rotation(degrees: float) -> int:
    """Snap display-matrix angles like -180 / 90.4 to 0, 90, 180, or 270."""
    snapped = int(round(float(degrees) / 90.0)) * 90
    return snapped % 360


def _stream_rotation(stream: dict[str, Any]) -> int:
    tags = stream.get("tags") or {}
    for key in ("rotate", "ROTATE"):
        if key in tags:
            try:
                return int(round(float(tags[key]))) % 360
            except ValueError:
                pass
    for item in stream.get("side_data_list") or []:
        if "rotation" in item:
            try:
                # displaymatrix rotation is often -90 / 90
                raw = float(item["rotation"])
                return int(round(raw)) % 360
            except (TypeError, ValueError):
                pass
    return 0


def _fps(stream: dict[str, Any]) -> float:
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(key) or "0/0"
        if isinstance(value, str) and "/" in value:
            num, den = value.split("/", 1)
            try:
                n, d = float(num), float(den)
                if d:
                    return n / d
            except ValueError:
                continue
        try:
            fps = float(value)
            if fps > 0:
                return fps
        except (TypeError, ValueError):
            continue
    return 25.0


def probe(path: Path) -> VideoInfo:
    data = ffprobe_json(path)
    video = next((s for s in data.get("streams") or [] if s.get("codec_type") == "video"), None)
    if video is None:
        raise ProbeError(f"no video stream in {path}")
    fmt = data.get("format") or {}
    duration = float(video.get("duration") or fmt.get("duration") or 0.0)
    gpmd_index = None
    for stream in data.get("streams") or []:
        tag = f"{stream.get('codec_tag_string') or ''} {stream.get('codec_name') or ''}".lower()
        handler = str((stream.get("tags") or {}).get("handler_name") or "").lower()
        if "gpmd" in tag or "goprometa" in handler or "gopro met" in handler:
            gpmd_index = int(stream["index"])
            break
    return VideoInfo(
        path=path,
        duration=duration,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_fps(video),
        rotation=normalize_rotation(_stream_rotation(video)),
        codec=str(video.get("codec_name") or ""),
        has_gpmd=gpmd_index is not None,
        gpmd_index=gpmd_index,
    )
