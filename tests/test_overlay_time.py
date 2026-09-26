from datetime import datetime, timezone

from videobrowser.overlay_bridge import ensure_on_path


def _overlay_render():
    ensure_on_path()
    from overlay import overlay_render

    return overlay_render


def test_wall_clock_is_cest_offset_in_summer():
    format_wall_clock = _overlay_render().format_wall_clock
    utc = datetime(2026, 8, 2, 8, 42, 1, tzinfo=timezone.utc)
    assert format_wall_clock(utc.timestamp()) == "2026-08-02 10:42:01"


def test_wall_clock_is_cet_offset_in_winter():
    format_wall_clock = _overlay_render().format_wall_clock
    utc = datetime(2026, 1, 15, 8, 42, 0, tzinfo=timezone.utc)
    assert format_wall_clock(utc.timestamp()) == "2026-01-15 09:42:00"


def test_wall_clock_parts_split_date_and_time():
    parts = _overlay_render().format_wall_clock_parts
    utc = datetime(2026, 7, 31, 10, 51, 19, tzinfo=timezone.utc)
    assert parts(utc.timestamp()) == ("2026-07-31", "12:51:19")


def test_lean_box_uses_same_edge_inset_as_speed():
    overlay = _overlay_render()
    config = overlay.RenderConfig(
        width=1920,
        height=1080,
        fps=25,
        font_path="/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    layout = overlay._compute_layout(config)
    assert layout["bl_x"] == layout["margin"]
    assert layout["box_x"] + layout["box_size"] == config.width - layout["margin"]


def test_clock_block_aligns_with_speed_and_map():
    overlay = _overlay_render()
    config = overlay.RenderConfig(
        width=1920,
        height=1080,
        fps=25,
        font_path="/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    layout = overlay._compute_layout(config)
    _map_x, map_y, _map_w, _map_h = overlay._map_geometry(config)
    assert layout["clock_left"] == layout["bl_x"]
    assert layout["clock_top"] == map_y
    assert layout["clock_width"] > 0
    assert layout["clock_height"] > 0
    assert layout["clock_height"] < layout["clock_width"]
