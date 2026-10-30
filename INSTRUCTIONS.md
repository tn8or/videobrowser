# Project instructions

Product criteria and answers for the unified **videobrowser** repo (GoPro
highlight detection + RaceBox telemetry overlays). Usage lives in
[README.md](README.md).

## Criteria

1. Scan GoPro track-day footage, detect interesting moments (nearby riders,
   approaching/receding riders, kneepuck), and cut highlight clips.
2. Parse RaceBox `.vbo` files and render a telemetry overlay composited onto
   matching video (highlights, full sessions, fastest laps).
3. Overlay must scale cleanly from 720p to UHD (drawn at the source frame size).
4. Track map in top-right, max ~25% width/height, pointer for current position.
5. Full overlay, top-left: lap number, active lap timer, ON TRACK / BEHIND vs
   previous best.
6. Below current lap time: last 4 laps with sector times and colours
   (white = not faster, green = fastest of the day, purple = fastest overall).
7. Bottom-left: speedometer. Bottom-right: lean angle.
8. Persist track stats with timestamps in SQLite so re-running an old `.vbo`
   yields a dashboard correct for that historical time.
9. Prefer maximal performance; hardware decode/encode when available.
10. Match clips to sessions by overlapping UTC time ranges (GoPro creation time
    interpreted as Europe/Copenhagen wall time).

## Answers / decisions

1. **Primary workflow:** `videobrowser scan` — detect, cut, and composite
   overlays onto video in one pass. The standalone ProRes 4444 alpha `.mov`
   output for Final Cut Pro has been removed.
2. **Overlay frame rate:** match the source clip FPS in the scan pipeline.
3. **Fonts:** configurable TTF (`--font`).
4. **Corner metrics:** Pre/Entry/Min corner speeds and corner lean comparisons
   are **disabled** in the merged pipeline (`show_corners=False`). Drawing code
   remains for optional re-enable later.
5. **Track identification:** track name from VBO metadata.
6. **Stats file:** SQLite; default `<out>/stats.sqlite`, overridable with
   `--stats` so day/overall bests accumulate across runs.
7. **Time zone:** `Europe/Copenhagen` for wall-clock display and GoPro creation
   date interpretation.
8. **Multi-session comparisons:** yes — sector times on each lap row; day-best
   (green) and overall-best (purple) from the stats store.
9. **Privacy:** none beyond keeping the racebox.pro login session local in
   `~/.videobrowser/racebox_auth.json` (outside the repo).
10. **Outputs from a scan:**
    - `clips/` — raw rotation-corrected highlights (no overlay)
    - `highlights/` — same windows with **compact** overlay
    - `full_sessions/` — out-lap → in-lap with **full** overlay
    - `fastest_laps/` — fastest timed lap ±10s with **full** overlay
11. **Compact overlay:** date/time, track map, speed, lean (no lap list).
12. **Out-lap:** do not show timers or lap stats (not on first timed lap yet).
13. **In-lap:** keep last lap timers on screen but do not count a new lap.
14. **In/out laps:** excluded from the mini-map path.
15. **Telemetry fallback** when no overlapping `.vbo`: GoPro GPMD speed/lean if
    present, otherwise date/time only. Full-session / fastest-lap clips require
    a matched `.vbo`.
16. **Auto-fetch:** optional Playwright download of new sessions before a scan;
    skipped cleanly if credentials or Playwright are missing (`--no-fetch` to
    force local-only).
17. **License:** not chosen. Do not add a `LICENSE` file or a `license` field
    until it is. The detector dependencies are AGPL-3.0. See
    [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
