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

# Detection only, no overlays
videobrowser scan inputclips -o output --no-overlay
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
  vertical/YYYY-MM-DD/       # 1080x1920 follow-cam copy of each highlight (VERTICAL overlay)
  shorts/YYYY-MM-DD/         # ready-to-post shorts per day + index.html overview (see "Shorts")
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

- **Vertical** (`vertical/`): the compact elements laid out for 9:16 phone
  video, kept clear of the Reels / TikTok / Shorts UI. The 9:16 crop pans to
  follow the rider that triggered the clip (else any rider in view, else the
  kneepuck). It opens centred unless the subject is already in view. Without RaceBox sessions the overlay uses GoPro telemetry or
  date/time; `--no-overlay` leaves it off entirely.

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
videobrowser scan inputclips -o output --no-overlay
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
| `--no-overlay` | off | No overlays: skip the RaceBox stage and date/time on verticals (alias `--no-racebox`) |
| `--no-vertical` | off | Do not write 1080x1920 follow-cam copies to `vertical/` |
| `--no-landscape` | off | Do not write landscape clips or RaceBox overlay renders (only `vertical/`) |
| `--no-fetch` | off | Do not auto-download RaceBox sessions |
| `--sessions-dir` | `<out>/sessions` | Folder of `.vbo` sessions |
| `--font` | Arial | TTF font for overlay text |
| `--stats` | `<out>/stats.sqlite` | Lap-bests SQLite |
| `--no-shorts` | off | Do not build `shorts/` after the scan |

Run `videobrowser scan --help` for the complete list.

---

## Shorts

`scan` works through the recordings oldest first (by file creation time) and
builds a day's shorts as soon as that day's last recording is scanned, so the
first days are ready while the rest of a library is still running. Days with
no newly scanned recordings are left as they are. `videobrowser shorts -o
output` rebuilds them on its own. The day's vertical clips are cut into
near-finished 9:16 shorts:

```
shorts/2026-08-01/
  action-01.mp4 ...        # the day's best moments, several shorts of <= 50 s
  overhalinger-01.mp4      # one series per theme (needs at least 2 moments)
  overhalet-01.mp4
  indhentninger-01.mp4
  knae-01.mp4
  styrt-01.mp4             # own crash, from GoPro telemetry
  mulige-styrt-01.mp4      # other riders: possible crashes to check
  grus-01.mp4              # riders out on grass or gravel
  index.html               # the shorts + every clip of the day, by type
  moments.json             # what was found and how each camera was synced
```

**Cameras on one clock.** Every recording of a day starts from its GoPro GPS
time. Overlapping recordings (front, rear, helmet) are then laid on top of
each other by matching their GPS tracks, or on the RaceBox log when a session
overlaps. Engine sound refines the result when both cameras' loudness
envelopes match clearly; otherwise GPS stands. Turn the audio step off with
`--no-audio-sync`.

**Themes.** Event meaning depends on which way the camera faces (rear
profiles: `rear`, `rear_right`, `rear_rotated`; everything else is front):

| Theme | Rule | Reliability |
|-------|------|-------------|
| Overhaling | front `approaching` ends, rear `receding` starts within -1.5..+5 s | good with front + rear |
| Bliver overhalet | rear `approaching` ends, front `receding` starts within -1.5..+5 s | good with front + rear |
| Indhenter | front `approaching` for 1.5 s+ that is not an overtake | good |
| Knæ i asfalten | `kneepuck_visible` | as the puck detector |
| Styrt (own) | accelerometer jolt >= 3x gravity, moving >= 30 km/h before, stopped after | needs real crash data to tune |
| Muligt styrt (others) | `rider_down`: a bike box loses its rider while a person box appears beside it, or the bike box turns low and wide while out on grass/gravel | candidates to check by hand |
| Ude i gruset | `off_track`: the ground under a rider's box turns from grey asphalt to green grass or pale gravel for 0.6 s+ | candidates to check by hand |
| Action | any other event, for the mixed shorts | |

`rider_down` and `off_track` are found during the scan (`incidents.py`), so
footage scanned before they existed needs `scan --force` to get them. Each piece is taken from the vertical clip that covers it (crop + HUD);
a piece no clip covers (e.g. the rear half of an overtake) is cut from the
source with a centred crop. A label with the theme and time opens each moment.

Outputs scanned before this feature have no per-recording metadata yet; pass
the source folders once: `videobrowser shorts -o output inputclips`.

| Flag | Default | Purpose |
|------|---------|---------|
| `--date` | all | Only this day (repeatable) |
| `--short-seconds` | 50 | Max length of each short |
| `--max-action` | 6 | Max mixed action shorts per day |
| `--no-audio-sync` | off | Sync cameras by GPS only |
| `--no-crash` | off | Skip crash detection |
| `--dry-run` | off | Only write `moments.json` and `index.html` |

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
