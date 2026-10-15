# Vertical (1080x1920) follow-cam clips: plan and status

Goal: every `videobrowser scan` also writes a 9:16 1080x1920 copy of each
highlight clip to `<out>/vertical/YYYY-MM-DD/` (next to `clips/` and
`highlights/`), cropped/panned to follow the action the scan detected, with a
compact telemetry overlay laid out for Reels / TikTok / Shorts.

## Decisions (agreed with Tommy)

- Clips with no rider to follow (e.g. kneepuck only): crop around the puck
  position, otherwise a fixed crop (puck ROI centre for side mounts, else centre).
- Vertical clips are also made with `--no-racebox`: no RaceBox session data,
  but GoPro GPMD speed/lean and date/time when available.
- `--no-vertical` disables the stage. `--no-clips` also skips it.
- 1080p sources only give a 608x1080 crop (upscaled, a bit soft); 4K is fine.

## Design

1. **Focus samples** (`src/videobrowser/vertical.py: focus_samples`): per
   analysed frame (5 fps) inside the clip window, pick the subject:
   tracks referenced by the window's events -> kneepuck position -> any other
   rider with box height >= 5% of frame. Largest box wins, but the current
   target is kept while its area is >= 50% of the largest (hysteresis).
   Output: horizontal centre as fraction of frame width.
2. **Crop path** (`crop_centers`): interpolate to clip fps (hold first/last
   sample), clamp to reachable range, zero-phase EMA (tau 0.35 s), then
   rate-limit pan to 0.6 frame widths/s.
3. **Render** (`export_vertical`): single-chapter windows decode straight from
   the source with accurate `-ss`; multi-chapter windows are joined with
   `export_clip` first. ffmpeg graph:
   `rotation -> sendcmd (per-frame crop x) -> crop@vc -> scale 1080:1920 -> overlay`.
   Overlay BGRA frames are piped on stdin like `racebox_stage._overlay_pass`.
   Encoders reuse `clips._reencode_pipelines` (VideoToolbox / NVENC / x264) at 20M.
4. **Overlay** (`racebox/overlay/overlay_render.py`): new `variant="vertical"`
   = compact HUD (clock, map, speed, lean) with a phone-safe layout: clock
   top-left and map top-right below the top 12%; speed and lean side by side
   bottom-left above the bottom 25% and clear of the right button column.
5. **Plumbing**: `pipeline.scan_recording` records puck x positions, calls
   `_export_verticals` after the RaceBox stage; `racebox_stage` gained
   `overlay_context` (session-less ctx), `recording_start_utc`,
   `overlay_source(..., variant)`, and `run_for_recording` returns the aligned
   UTC start so it is not computed twice.

## Status / remaining (stopped 2026-10-15)

Done (committed, untested on real footage):
- [x] `src/videobrowser/vertical.py`, overlay `vertical` variant, pipeline + CLI wiring
- [x] `tests/test_vertical.py`: path logic, layout and overlay tests pass

Next steps:
- [ ] Fix `test_export_vertical_follows_subject`. The first version used
      `drawbox=x='40+t*250'`, but in drawbox `t` means thickness, so the source
      had no box and the test failed for that reason (crop offsets looked right:
      30 -> 360 px). It was switched to an `overlay` of a white `color` source,
      but that ffmpeg command then hung (likely the overlay never ending; try
      adding `:shortest=1` / `eof_action=endall`, or `-t 2`). Debug helper idea:
      spy on `vertical._render` to print `offsets`.
- [ ] Run the full suite: `pip install -e ".[dev]" && pytest`
- [ ] README + INSTRUCTIONS: document `vertical/` output and `--no-vertical`
- [ ] Try on real footage; tune `SMOOTH_TAU`, `MAX_PAN_PER_S`, `SWITCH_RATIO`
      in `vertical.py`, and check the overlay safe zones on a phone
- [ ] Verify `sendcmd` + `crop` commands on the CI image's ffmpeg (bookworm 5.1)
- [ ] Open a draft PR from `claude/project-thread-i75obm`
