from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import os
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import cairo

from .track_map import TrackMap

Color = Tuple[int, int, int, int]


@dataclass
class RenderConfig:
    width: int
    height: int
    fps: int
    font_path: str
    # "full" draws the complete telemetry HUD; "compact" draws only the
    # date/time, track map, speedometer and lean indicator.
    variant: str = "full"
    # Corner Pre/Entry/Min metrics are disabled by default in the merged
    # pipeline; keep the flag so the drawing code can be re-enabled later.
    show_corners: bool = False
    # IANA timezone used to render the wall-clock date/time (no zone abbr).
    timezone: str = "Europe/Copenhagen"


@dataclass
class LapRow:
    lap_number: int
    lap_time: float
    sector_times: Tuple[float, float, float]
    color: Color


@dataclass
class CornerDisplay:
    pre_corner_max_speed: float | None
    entry_speed: float | None
    min_speed: float | None
    max_lean: float | None
    best_pre_corner_max_speed: float | None
    best_entry_speed: float | None
    best_min_speed: float | None
    best_max_lean: float | None
    color: Color


@dataclass
class FrameState:
    lap_number: int | None
    lap_time: float | None
    on_track_faster: bool | None
    last_laps: List[LapRow]
    speed_kmh: float | None
    best_speed_kmh: float | None
    corner_display: CornerDisplay | None
    lean_angle: float | None
    map_pos: Tuple[float, float] | None
    is_in_lap: bool = False
    # UTC epoch seconds for the date/time element; None hides it.
    wall_time_utc: float | None = None


WHITE: Color = (255, 255, 255, 255)
GREEN: Color = (0, 220, 0, 255)
PURPLE: Color = (170, 70, 255, 255)
RED: Color = (255, 80, 80, 255)
GRAY: Color = (180, 180, 180, 200)
DIM_BG: Color = (30, 30, 30, 51)
# Stable strings used to size the date/time square so it does not jump.
_CLOCK_DATE_SAMPLE = "0000-00-00"
_CLOCK_TIME_SAMPLE = "00:00:00"


def format_time(seconds: float | None) -> str:
    if seconds is None:
        return "--:--.---"
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    return f"{minutes:02d}:{secs:06.3f}"


def _rgba(color: Color) -> Tuple[float, float, float, float]:
    return (color[0] / 255, color[1] / 255, color[2] / 255, color[3] / 255)


def _set_source_rgba(ctx: cairo.Context, color: Color) -> None:
    r, g, b, a = _rgba(color)
    ctx.set_source_rgba(r, g, b, a)


_FONT_FAMILY_CACHE: Dict[str, str] = {}


def _get_font_family(font_path: str) -> str:
    cached = _FONT_FAMILY_CACHE.get(font_path)
    if cached is None:
        base = os.path.splitext(os.path.basename(font_path))[0]
        cached = base
        _FONT_FAMILY_CACHE[font_path] = cached
    return cached


def _select_font(
    ctx: cairo.Context, font_path: str, size: float,
    slant: int = cairo.FONT_SLANT_NORMAL,
    weight: int = cairo.FONT_WEIGHT_NORMAL,
) -> None:
    family = _get_font_family(font_path)
    ctx.select_font_face(family, slant, weight)
    ctx.set_font_size(size)


def _text_bg_pad(font_size: float) -> tuple[float, float]:
    """Horizontal / vertical padding used by speed, lean, and the clock plaque."""
    return max(4.0, font_size * 0.25), max(2.0, font_size * 0.18)


def _draw_text_with_bg(
    ctx: cairo.Context, x: float, y: float, text: str,
    font_path: str, font_size: float, fill: Color,
    bg: bool = True,
) -> None:
    if not text:
        return
    _select_font(ctx, font_path, font_size)
    ext = ctx.text_extents(text)
    if bg:
        pad_x, pad_y = _text_bg_pad(font_size)
        bx = x + ext.x_bearing - pad_x
        by = y + ext.y_bearing - pad_y
        bw = ext.width + 2 * pad_x
        bh = ext.height + 2 * pad_y
        _set_source_rgba(ctx, DIM_BG)
        ctx.rectangle(bx, by, bw, bh)
        ctx.fill()
    r, g, b, a = _rgba(fill)
    ctx.set_source_rgba(r, g, b, a)
    ctx.move_to(x, y)
    ctx.show_text(text)


_TZ_CACHE: Dict[str, ZoneInfo] = {}


def _tz(name: str) -> ZoneInfo:
    tz = _TZ_CACHE.get(name)
    if tz is None:
        try:
            tz = ZoneInfo(name)
        except ZoneInfoNotFoundError:
            # Windows (and some minimal images) ship without an IANA database.
            # The tzdata package on PyPI provides the same keys.
            try:
                import tzdata  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    f"timezone {name!r} is unavailable; install the tzdata package"
                ) from exc
            tz = ZoneInfo(name)
        _TZ_CACHE[name] = tz
    return tz


def format_wall_clock_parts(
    wall_time_utc: float, tz_name: str = "Europe/Copenhagen"
) -> tuple[str, str]:
    """Return ``(YYYY-MM-DD, HH:MM:SS)`` in *tz_name*, with no zone abbreviation."""
    local = datetime.fromtimestamp(float(wall_time_utc), tz=timezone.utc).astimezone(
        _tz(tz_name)
    )
    return local.strftime("%Y-%m-%d"), local.strftime("%H:%M:%S")


def format_wall_clock(wall_time_utc: float, tz_name: str = "Europe/Copenhagen") -> str:
    """Format a UTC epoch timestamp in *tz_name* as ``YYYY-MM-DD HH:MM:SS``."""
    date_s, time_s = format_wall_clock_parts(wall_time_utc, tz_name)
    return f"{date_s} {time_s}"


def _text_x_centered(
    ctx: cairo.Context, font_path: str, font_size: float, text: str, center_x: float
) -> float:
    _select_font(ctx, font_path, font_size)
    ext = ctx.text_extents(text)
    return center_x - ext.width / 2 - ext.x_bearing


def _clock_block_metrics(ctx: cairo.Context, font_path: str, font_size: float) -> dict:
    """Size of the date/time plaque (stable, based on sample strings).

    Vertical padding matches the speed / lean text boxes so the plaque
    hugs the two lines instead of being forced into a tall square.
    """
    pad_x, pad_y = _text_bg_pad(font_size)
    line_gap = max(2.0, font_size * 0.12)
    _select_font(ctx, font_path, font_size)
    date_ext = ctx.text_extents(_CLOCK_DATE_SAMPLE)
    time_ext = ctx.text_extents(_CLOCK_TIME_SAMPLE)
    text_w = max(date_ext.width, time_ext.width)
    text_h = date_ext.height + line_gap + time_ext.height
    return {
        "pad_x": pad_x,
        "pad_y": pad_y,
        "line_gap": line_gap,
        "width": text_w + 2 * pad_x,
        "height": text_h + 2 * pad_y,
        "text_h": text_h,
    }


def _draw_datetime(
    ctx: cairo.Context,
    config: RenderConfig,
    layout: dict,
    wall_time_utc: float,
    bg: bool,
) -> None:
    """Draw date and time inside one plaque at the top-left."""
    date_s, time_s = format_wall_clock_parts(wall_time_utc, config.timezone)
    fs = layout["font_size"]
    left = layout["clock_left"]
    top = layout["clock_top"]
    width = layout["clock_width"]
    height = layout["clock_height"]
    if bg:
        _set_source_rgba(ctx, DIM_BG)
        ctx.rectangle(left, top, width, height)
        ctx.fill()

    metrics = _clock_block_metrics(ctx, config.font_path, fs)
    inner_top = top + metrics["pad_y"]
    pad_x = metrics["pad_x"]
    _select_font(ctx, config.font_path, fs)
    date_ext = ctx.text_extents(date_s)
    time_ext = ctx.text_extents(time_s)
    date_x = left + pad_x - date_ext.x_bearing
    time_x = left + pad_x - time_ext.x_bearing
    date_y = inner_top - date_ext.y_bearing
    time_y = inner_top + date_ext.height + metrics["line_gap"] - time_ext.y_bearing
    _draw_text_with_bg(
        ctx, date_x, date_y, date_s, config.font_path, fs, WHITE, bg=False
    )
    _draw_text_with_bg(
        ctx, time_x, time_y, time_s, config.font_path, fs, WHITE, bg=False
    )


def _compute_layout(config: RenderConfig):
    margin = int(min(config.width, config.height) * 0.10)
    base_size = max(18, int(min(config.width, config.height) * 0.03))
    font_size = float(base_size)
    font_large_size = base_size * 1.6
    font_small_size = base_size * 0.75

    bl_x = float(margin)
    speed_y = config.height - margin - int(base_size * 4.0)
    # Right edge of the lean box matches the speed's left-edge inset.
    box_size = int(base_size * 1.6)
    box_x = config.width - margin - box_size
    lean_text_y = float(speed_y)
    box_y = lean_text_y + int(base_size * 1.2)
    bottom_line_y = int(speed_y + base_size * 3.8)
    max_box = max(6, bottom_line_y - int(box_y) - 6)
    if box_size > max_box:
        box_size = max_box
        box_x = config.width - margin - box_size
    scratch = cairo.ImageSurface(cairo.FORMAT_A8, 8, 8)
    scratch_ctx = cairo.Context(scratch)
    clock = _clock_block_metrics(scratch_ctx, config.font_path, font_size)
    clock_left = bl_x
    clock_top = float(margin)
    clock_width = clock["width"]
    clock_height = clock["height"]
    x0 = float(margin)
    y0 = clock_top + clock_height + int(base_size * 0.5)
    list_y = y0 + int(base_size * 4.4)

    return {
        "margin": margin, "base_size": base_size,
        "font_size": font_size, "font_large_size": font_large_size,
        "font_small_size": font_small_size,
        "x0": x0, "y0": y0, "list_y": list_y,
        "bl_x": bl_x, "speed_y": speed_y,
        "clock_left": clock_left, "clock_top": clock_top,
        "clock_width": clock_width, "clock_height": clock_height,
        "box_x": float(box_x), "box_y": float(box_y), "box_size": box_size,
        "lean_text_y": lean_text_y, "lean_cx": box_x + box_size / 2,
        "br_x": float(box_x), "br_y": lean_text_y,
        "bottom_line_y": bottom_line_y,
    }


def _map_geometry(config: RenderConfig) -> tuple[int, int, int, int]:
    """Square track-map box in the top-right, sized from the short side."""
    margin = int(min(config.width, config.height) * 0.10)
    side = int(min(config.width, config.height) * 0.25)
    return config.width - side - margin, margin, side, side


def build_map_layer(
    config: RenderConfig, track_map: TrackMap, map_padding: int = 20
) -> cairo.ImageSurface:
    map_x, map_y, map_w, map_h = _map_geometry(config)

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, config.width, config.height)
    ctx = cairo.Context(surface)
    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.set_source_rgba(0, 0, 0, 0)
    ctx.paint()
    ctx.set_operator(cairo.OPERATOR_OVER)

    if track_map.points.size > 0:
        _set_source_rgba(ctx, WHITE)
        ctx.set_line_width(4)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        pts = track_map.points
        ctx.move_to(map_x + pts[0][0] * map_w, map_y + pts[0][1] * map_h)
        for i in range(1, len(pts)):
            ctx.line_to(map_x + pts[i][0] * map_w, map_y + pts[i][1] * map_h)
        ctx.stroke()

    return surface


def build_static_layer(
    config: RenderConfig,
    track_map: TrackMap,
    map_layer: Optional[cairo.ImageSurface] = None,
) -> cairo.ImageSurface:
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, config.width, config.height)
    ctx = cairo.Context(surface)
    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.set_source_rgba(0, 0, 0, 0)
    ctx.paint()
    ctx.set_operator(cairo.OPERATOR_OVER)

    font_path = config.font_path
    L = _compute_layout(config)
    base_size = L["base_size"]
    font_size = L["font_size"]
    font_large_size = L["font_large_size"]
    font_small_size = L["font_small_size"]
    x0 = L["x0"]
    y0 = L["y0"]
    list_y = L["list_y"]
    bl_x = L["bl_x"]
    speed_y = L["speed_y"]
    br_x = L["br_x"]
    br_y = L["br_y"]
    bottom_line_y = L["bottom_line_y"]
    box_x = L["box_x"]
    box_y = L["box_y"]
    box_size = L["box_size"]
    lean_cx = L["lean_cx"]

    def _bg_rect(px: float, py: float, text: str, fs: float) -> None:
        _select_font(ctx, font_path, fs)
        ext = ctx.text_extents(text)
        pad_x, pad_y = _text_bg_pad(fs)
        _set_source_rgba(ctx, DIM_BG)
        ctx.rectangle(
            px + ext.x_bearing - pad_x,
            py + ext.y_bearing - pad_y,
            ext.width + 2 * pad_x,
            ext.height + 2 * pad_y,
        )
        ctx.fill()

    if config.variant != "compact":
        _bg_rect(x0, y0, "Lap 999", font_size)
        _bg_rect(x0, y0 + base_size + 6, "09:59.999", font_large_size)
        _bg_rect(x0, int(y0 + base_size * 2.8), "ON TRACK", font_small_size)

        max_lap_row = "L999 09:59.999  |  09:59.999 09:59.999 09:59.999"
        for i in range(4):
            _bg_rect(x0, list_y + i * int(base_size * 0.9), max_lap_row, font_small_size)

    _set_source_rgba(ctx, DIM_BG)
    ctx.rectangle(L["clock_left"], L["clock_top"], L["clock_width"], L["clock_height"])
    ctx.fill()
    _bg_rect(bl_x, speed_y, "999.9 km/h", font_large_size)
    if config.show_corners:
        _bg_rect(bl_x, int(speed_y + base_size * 2.2), "Pre: 999.9 | Best 999.9", font_small_size)
        _bg_rect(bl_x, int(speed_y + base_size * 3.0), "Entry: 999.9 | Best 999.9", font_small_size)
        _bg_rect(bl_x, int(speed_y + base_size * 3.8), "Min: 999.9 | Best 999.9", font_small_size)

    lean_tx = _text_x_centered(ctx, font_path, font_large_size, "90°", lean_cx)
    _bg_rect(lean_tx, br_y, "90°", font_large_size)
    if config.show_corners:
        _bg_rect(br_x, bottom_line_y, "Prev Max 90.0° | Best 90.0°", font_small_size)

    _set_source_rgba(ctx, GRAY)
    ctx.set_line_width(2)
    ctx.rectangle(box_x, box_y, box_size, box_size)
    ctx.stroke()

    if map_layer is not None:
        ctx.set_source_surface(map_layer, 0, 0)
        ctx.paint()

    return surface


def render_frame(
    config: RenderConfig,
    track_map: TrackMap,
    state: FrameState,
    map_padding: int = 20,
    map_layer: Optional[cairo.ImageSurface] = None,
    static_layer: Optional[cairo.ImageSurface] = None,
) -> bytes:
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, config.width, config.height)
    ctx = cairo.Context(surface)

    ctx.set_operator(cairo.OPERATOR_SOURCE)
    ctx.set_source_rgba(0, 0, 0, 0)
    ctx.paint()
    ctx.set_operator(cairo.OPERATOR_OVER)

    font_path = config.font_path
    L = _compute_layout(config)
    base_size = L["base_size"]
    font_size = L["font_size"]
    font_large_size = L["font_large_size"]
    font_small_size = L["font_small_size"]
    x0 = L["x0"]
    y0 = L["y0"]
    list_y = L["list_y"]
    bl_x = L["bl_x"]
    speed_y = L["speed_y"]
    br_x = L["br_x"]
    br_y = L["br_y"]
    bottom_line_y = L["bottom_line_y"]
    box_x = L["box_x"]
    box_y = L["box_y"]
    box_size = L["box_size"]
    lean_cx = L["lean_cx"]
    use_cached_bg = static_layer is not None

    def draw_text_bg(
        px: float, py: float, text: str, fs: float, fill: Color
    ) -> None:
        _draw_text_with_bg(ctx, px, py, text, font_path, fs, fill, bg=not use_cached_bg)

    if static_layer is not None:
        ctx.set_source_surface(static_layer, 0, 0)
        ctx.paint()
    elif map_layer is not None:
        ctx.set_source_surface(map_layer, 0, 0)
        ctx.paint()

    if state.wall_time_utc is not None:
        _draw_datetime(ctx, config, L, state.wall_time_utc, bg=not use_cached_bg)

    compact = config.variant == "compact"

    if not compact:
        if state.is_in_lap:
            draw_text_bg(x0, y0, "In Lap", font_size, GRAY)
            if state.lap_number is not None and state.lap_time is not None:
                draw_text_bg(
                    x0, y0 + base_size + 6, format_time(state.lap_time), font_large_size, GRAY
                )
        elif state.lap_number is not None and state.lap_time is not None:
            draw_text_bg(x0, y0, f"Lap {state.lap_number}", font_size, WHITE)
            draw_text_bg(
                x0, y0 + base_size + 6, format_time(state.lap_time), font_large_size, WHITE
            )
            if state.on_track_faster is not None:
                label = "ON TRACK" if state.on_track_faster else "BEHIND"
                color = GREEN if state.on_track_faster else RED
                draw_text_bg(x0, int(y0 + base_size * 2.8), label, font_small_size, color)
        else:
            draw_text_bg(x0, y0, "Out Lap", font_size, GRAY)

        cur_list_y = list_y
        for row in state.last_laps:
            text = f"L{row.lap_number} {format_time(row.lap_time)}"
            sector_text = " ".join(format_time(s) for s in row.sector_times)
            draw_text_bg(x0, cur_list_y, f"{text}  |  {sector_text}", font_small_size, row.color)
            cur_list_y += int(base_size * 0.9)

    map_x, map_y, map_w, map_h = _map_geometry(config)
    if (
        map_layer is None
        and not use_cached_bg
        and static_layer is None
        and track_map.points.size > 0
    ):
        _set_source_rgba(ctx, WHITE)
        ctx.set_line_width(4)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        pts = track_map.points
        ctx.move_to(map_x + pts[0][0] * map_w, map_y + pts[0][1] * map_h)
        for i in range(1, len(pts)):
            ctx.line_to(map_x + pts[i][0] * map_w, map_y + pts[i][1] * map_h)
        ctx.stroke()

    if state.map_pos is not None:
        px, py = state.map_pos
        cx = map_x + px * map_w
        cy = map_y + py * map_h
        _set_source_rgba(ctx, RED)
        ctx.arc(cx, cy, 8, 0, 2 * math.pi)
        ctx.fill()

    if state.speed_kmh is not None:
        draw_text_bg(bl_x, speed_y, f"{state.speed_kmh:05.1f} km/h", font_large_size, WHITE)

    if state.corner_display is not None:
        metrics = state.corner_display
        c = metrics.color
        draw_text_bg(
            bl_x,
            int(speed_y + base_size * 2.2),
            f"Pre: {metrics.pre_corner_max_speed:.1f} | Best {metrics.best_pre_corner_max_speed:.1f}",
            font_small_size,
            c,
        )
        draw_text_bg(
            bl_x,
            int(speed_y + base_size * 3.0),
            f"Entry: {metrics.entry_speed:.1f} | Best {metrics.best_entry_speed:.1f}",
            font_small_size,
            c,
        )
        draw_text_bg(
            bl_x,
            int(speed_y + base_size * 3.8),
            f"Min: {metrics.min_speed:.1f} | Best {metrics.best_min_speed:.1f}",
            font_small_size,
            c,
        )

    if state.lean_angle is not None:
        lean_value = int(round(abs(state.lean_angle)))
        lean_text = f"{lean_value}°"
        lean_tx = _text_x_centered(
            ctx, font_path, font_large_size, lean_text, lean_cx
        )
        draw_text_bg(lean_tx, br_y, lean_text, font_large_size, WHITE)

        if not use_cached_bg:
            _set_source_rgba(ctx, GRAY)
            ctx.set_line_width(2)
            ctx.rectangle(box_x, box_y, box_size, box_size)
            ctx.stroke()
        center_x = lean_cx
        center_y = box_y + box_size / 2
        # Positive lean = bike rolling right (viewed from behind) → `/`.
        angle = max(-60.0, min(60.0, state.lean_angle))
        radians = math.radians(angle)
        line_len = box_size * 0.45
        dx = math.sin(radians) * line_len
        dy = -math.cos(radians) * line_len
        _set_source_rgba(ctx, WHITE)
        ctx.set_line_width(3)
        ctx.move_to(center_x - dx, center_y - dy)
        ctx.line_to(center_x + dx, center_y + dy)
        ctx.stroke()

    if state.corner_display is not None and state.corner_display.max_lean is not None:
        best_lean = state.corner_display.best_max_lean
        if best_lean is not None:
            draw_text_bg(
                br_x,
                bottom_line_y,
                f"Prev Max {state.corner_display.max_lean:.1f}° | Best {best_lean:.1f}°",
                font_small_size,
                state.corner_display.color,
            )
        else:
            draw_text_bg(
                br_x,
                bottom_line_y,
                f"Prev Max {state.corner_display.max_lean:.1f}°",
                font_small_size,
                state.corner_display.color,
            )

    surface.flush()
    return bytes(surface.get_data())
