# videobrowser

Local highlight extraction **and** RaceBox telemetry overlays for motorcycle
track-day GoPro footage.

`videobrowser scan` scans GoPro recordings, detects interesting moments (nearby
riders, approaching/receding riders, kneepuck), and cuts short highlight clips.
In the same run it can also overlay RaceBox telemetry (lap timer, track map,
speed, lean) onto three separate output sets.

Design criteria and product decisions live in [INSTRUCTIONS.md](INSTRUCTIONS.md).

---

## Requirements

- Python 3.11–3.14
- `ffmpeg` and `ffprobe` on your `PATH`
  - macOS: VideoToolbox decode/encode when available
  - Windows/Linux with NVIDIA: CUDA decode + NVENC encode (`h264_nvenc` / `hevc_nvenc`)
  - otherwise software (`libx264` / `libx265`)
- Python packages (installed with the project):
  - `ultralytics`, `torch`, `opencv-python-headless`, `numpy`, `tqdm`, `lap` — detection
  - `pycairo` — overlay rendering
  - `playwright` (optional, `fetch` extra) — auto-downloading RaceBox sessions

### Install

```bash
python -m venv .venv
source .venv/bin/activate

# core + overlay rendering
pip install -e .

# optional: RaceBox session auto-download
pip install -e ".[fetch]"
playwright install chromium
```

Credentials for auto-fetch (optional): put the racebox.pro login in
`racebox/fetcher/username` and `racebox/fetcher/password`.

---

## Quick start

```bash
# Scan footage, cut highlights, and produce RaceBox overlays
videobrowser scan inputclips -o output

# A whole library — folders are searched recursively
videobrowser scan ~/Movies/GoPro -o output

# Detection only, no RaceBox overlays
videobrowser scan inputclips -o output --no-racebox
```

`videobrowser` is the installed console script. You can also run:

```bash
python -m videobrowser scan inputclips -o output
```

---

## What a scan produces

Everything is written under the `-o/--out` directory:

```
output/
  clips/YYYY-MM-DD/          # raw rotation-corrected highlight cuts (no overlay)
  previews/YYYY-MM-DD/       # annotated JPEG previews of detected events
  highlights/YYYY-MM-DD/     # highlight clips re-cut with a COMPACT telemetry overlay
  full_sessions/YYYY-MM-DD/  # one clip per RaceBox session, out-lap -> in-lap (FULL overlay)
  fastest_laps/YYYY-MM-DD/   # fastest timed lap per session, +/-10s (FULL overlay)
  sessions/                  # downloaded/.vbo files (default location)
  stats.sqlite               # lap bests used for green/purple lap colouring
  scan.sqlite                # detection log
  events.json                # events + clip list from the last run
```

Date folders use the clip's GoPro creation timestamp in `Europe/Copenhagen`
(e.g. `highlights/2026-08-01/`). Full-session and fastest-lap clips use the
RaceBox session date. Recordings with no timestamp go under `unknown/`.

### Overlay variants

- **Compact** (`highlights/`): date/time, track map, speed, lean.
- **Full** (`full_sessions/`, `fastest_laps/`): date/time, lap number + running
  lap timer with ON TRACK / BEHIND, last four laps with sector times, track map,
  speed, lean.

Corner entry/exit speeds are disabled in all variants.

### Overlay data source per clip

For each clip the overlay data is chosen in this order:

1. A RaceBox `.vbo` session whose UTC time range overlaps the clip.
2. The clip's own GoPro GPMD telemetry (speed + lean, no track map/laps).
3. Date/time only (from the clip's creation timestamp).

`full_sessions/` and `fastest_laps/` require a matched `.vbo` (laps come from the
session's start/finish line), so they are only produced for sessions that overlap
your footage. Clips with no telemetry are still exported to `highlights/` — just
without an overlay.

---

## Modes / common workflows

### 1. Full run with auto-fetch (default)

Downloads any new RaceBox sessions, imports lap stats, scans footage, and writes
all three overlay sets.

```bash
videobrowser scan inputclips -o output
```

Auto-fetch needs credentials in `racebox/fetcher/username` and
`racebox/fetcher/password` plus `playwright install chromium`. If they are
missing, fetching is skipped with a message and the scan continues using whatever
`.vbo` files are already present.

### 2. Use local sessions, no download

```bash
videobrowser scan inputclips -o output --no-fetch --sessions-dir /path/to/vbos
```

`--sessions-dir` defaults to `<out>/sessions`. Drop your `.vbo` files there (or
point at another folder) and skip the network with `--no-fetch`.

### 3. Detection only (no overlays)

```bash
videobrowser scan inputclips -o output --no-racebox
```

Produces `clips/` and `previews/` only.

### 4. Custom font / stats location

```bash
videobrowser scan inputclips -o output \
  --font /System/Library/Fonts/Supplemental/Arial.ttf \
  --stats /path/to/stats.sqlite
```

`--stats` defaults to `<out>/stats.sqlite`. Point multiple runs at the same file
so day-best (green) and overall-best (purple) lap colouring accumulates across
track days.

---

## Useful flags

| Flag | Default | Purpose |
|------|---------|---------|
| `-o, --out` | required | Output directory |
| `--fps` | 5 | Analysis frame rate |
| `--size` | 640 | Max analysis side (px) |
| `--pad` | 10 | Seconds padded around each detected peak |
| `--profile` | `auto` | Camera mount profile (`auto`, `cockpit`, `side`, `rear`, `ground`, ...) |
| `--map` | none | `cameras.json` mount override |
| `--model` | `yolo11n.pt` | Ultralytics model |
| `--device` | `auto` | `auto` / `mps` / `cpu` / `coreml` |
| `--conf` | 0.25 | YOLO confidence threshold |
| `--exclude` | none | Repeatable; skip files/folders whose path contains the text |
| `--no-clips` | off | Detect only; do not cut clips (also skips overlays) |
| `--previews` | 8 | Annotated preview JPEGs per recording |
| `--max-clip-seconds` | 24 | Max highlight length |
| `--max-clips` | 12 | Max highlights per recording |
| `--no-racebox` | off | Disable the RaceBox overlay stage entirely |
| `--no-fetch` | off | Do not auto-download RaceBox sessions |
| `--sessions-dir` | `<out>/sessions` | Folder of `.vbo` sessions |
| `--font` | Arial | TTF font for overlay text |
| `--stats` | `<out>/stats.sqlite` | Lap-bests SQLite |

Run `videobrowser scan --help` for the complete list.

---

## How clips are matched to sessions

Video files carry a UTC creation timestamp (`com.apple.quicktime.creationdate`,
treated as `Europe/Copenhagen`). Each RaceBox `.vbo` stores UTC sample times.
A clip is matched to a session when their UTC time ranges overlap, so one session
can map to several clips (multiple GoPro chapters / camera angles) or a single
clip, automatically.

---

## Project layout

```
src/videobrowser/     # CLI, scan pipeline, clip cutting, RaceBox stage bridge
racebox/overlay/      # VBO parser, lap timing, Cairo overlay renderer
racebox/fetcher/      # Playwright downloader for racebox.pro sessions
tests/                # pytest suite
```

The scan pipeline drives everything through `videobrowser`. Overlay rendering
code still lives under `racebox/` and is loaded via
`videobrowser.overlay_bridge`.

---

## Standalone RaceBox tooling

The RaceBox package under `racebox/` can still be used on its own — for example
to render ProRes 4444 alpha overlays for Final Cut Pro, or to download sessions
manually.

```bash
# Download sessions only
python racebox/fetcher/download_sessions.py --output-dir output/sessions

# Render one alpha overlay .mov per .vbo (FCP workflow)
python -m overlay.cli \
  --input "path/to/session.vbo" \
  --output-dir ./output \
  --width 1920 --height 1080 --fps 25 \
  --font /path/to/font.ttf \
  --stats ./stats.sqlite
# run from racebox/ so `overlay` is importable, or set PYTHONPATH=racebox

# Batch all VBOs in a folder (optional: composite onto matching video)
python racebox/run_all.py \
  --input-dir output/sessions \
  --output-dir ./overlay-out \
  --stats ./stats.sqlite
```

For the headful fetcher (if Bike Mode is required on the RaceBox site):

```bash
python racebox/fetcher/download_sessions.py --output-dir output/sessions --headful
```

---

## Development

```bash
pip install -e ".[dev]"
pytest
```
