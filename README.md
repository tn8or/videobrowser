# videobrowser

Local highlight extraction **and** RaceBox telemetry overlays for motorcycle
track-day GoPro footage.

Work happens in steps, each its own command:

1. `videobrowser scan` analyses GoPro recordings: it detects interesting
   moments (nearby riders, approaching/receding riders, kneepuck), places each
   recording on the RaceBox log where it can, and stores everything in
   `scan.sqlite`. It cuts no clips.
2. `videobrowser horizontals` cuts landscape highlight clips and the RaceBox
   telemetry overlays (lap timer, track map, speed, lean).
3. `videobrowser verticals` cuts 1080x1920 follow-cam clips.
4. `videobrowser shorts` lays out per-day shorts as folders of clips.

Steps 2-4 read only the scan, so each can be re-run on its own.

Design criteria and product decisions live in [INSTRUCTIONS.md](INSTRUCTIONS.md).

---

## Requirements

- Python 3.11–3.14
- `ffmpeg` and `ffprobe` on your `PATH`
  - macOS: VideoToolbox decode/encode when available
  - Windows/Linux with NVIDIA: CUDA decode + NVENC encode (`h264_nvenc` / `hevc_nvenc`).
    Scan scales frames on the GPU (`scale_cuda`, in the gyan.dev and BtbN
    Windows builds) and logs `decode cuda`; `decode cuda-copy` means the build
    lacks it and frames are scaled on the CPU instead.
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

On Windows/Linux with an NVIDIA card, pip installs a CPU-only `torch`. Swap it
for a CUDA build so the detector runs on the GPU (`cu128` covers RTX 50-series):

```bash
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available())"   # should print True
```

`scan --tensorrt` then runs the detector as a TensorRT FP16 engine, usually
2-3x faster than PyTorch on RTX cards. The first run builds
`yolo11n-640-fp16.engine` next to the model, which takes a few minutes, and
ultralytics installs `tensorrt` and `onnx` itself if they are missing. Delete
the `.engine` file after a driver or TensorRT update. If the build fails, scan
falls back to PyTorch.

`scan` then logs `detector: yolo11n.pt on cuda`.

Log in to racebox.pro once for auto-fetch: `python racebox/fetcher/download_sessions.py --login`
opens Chrome, you log in (and solve the captcha), and the session is saved to
`racebox/fetcher/auth_state.json`. No username/password files are needed; set
`CHROME_BIN` if Chrome isn't found.

---

## Quick start

```bash
# Analyse footage (a whole library: folders are searched recursively)
videobrowser scan ~/Movies/GoPro -o output

# Then cut what you want from it (-o is always the same scan folder)
videobrowser horizontals -o output            # landscape clips + RaceBox overlays
videobrowser verticals -o output --no-overlay # 9:16 follow-cam clips, no overlay
videobrowser shorts -o output                 # per-day shorts (folders of clips)
videobrowser facing -o output                 # re-read front/rear per camera, if a short has them swapped
```

`videobrowser` is the installed console script. You can also run:

```bash
python -m videobrowser scan inputclips -o output
```

---

## What each step produces

Everything is written under the `-o/--out` directory:

```
output/
  scan.sqlite                # scan: events, clip windows, rider boxes, motion, RaceBox placement
  events.json                # scan: events from the last run
  previews/YYYY-MM-DD/       # scan: annotated JPEG previews of detected events
  sessions/                  # scan: downloaded .vbo files (default location)
  clips/YYYY-MM-DD/          # horizontals: rotation-corrected highlight cuts (no overlay)
  highlights/YYYY-MM-DD/     # horizontals: highlight clips with a COMPACT telemetry overlay
  full_sessions/YYYY-MM-DD/  # horizontals: one clip per RaceBox session, out-lap -> in-lap (FULL overlay)
  fastest_laps/YYYY-MM-DD/   # horizontals: fastest timed lap per session, +/-10s (FULL overlay)
  stats.sqlite               # horizontals: lap bests used for green/purple lap colouring
  vertical/YYYY-MM-DD/       # verticals: 1080x1920 follow-cam copy of each highlight (VERTICAL overlay)
  shorts/YYYY-MM-DD/         # shorts: folders of clips per day + index.html overview (see "Shorts")
```

`horizontals` and `verticals` keep clips and overlay renders whose file
already exists, and only make the missing ones (re-cut all with `--force`) and remove files of clip windows a re-scan dropped. Both
take `--date` (repeatable) and `--no-overlay`. Overlays only go on recordings
whose start time is sure: GoPro GPS time, or a clear RaceBox match from the
scan. Rider following on verticals needs a scan from this version (the rider
boxes are stored by `scan`); older scans get the corner-looking crop only. Scans from before this version
also have no stored RaceBox placement, so footage with GoPro telemetry counts
as sure and the rest gets no overlay; run `scan --force` once after upgrading.

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
  follow the rider that triggered the clip (else any rider in view), and
  keeps that rider as they get smaller. Boxes that sit still on the frame
  edge (the camera's own exhaust or fairing) are never followed, nor scored
  as riders. Only after nobody has been in view
  for 1.5 s does it look into the corner like a rider would, from the GoPro
  GPS turn rate (left in a left-hander on a front camera, mirrored on a rear
  one). It opens centred unless the subject is already in view. Without RaceBox sessions the overlay uses GoPro telemetry or
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

`scan` downloads any new RaceBox sessions and analyses the footage;
`horizontals` imports lap stats and writes all three overlay sets.

```bash
videobrowser scan inputclips -o output
videobrowser horizontals -o output
```

Auto-fetch needs a saved login (`download_sessions.py --login`) plus
`playwright install chromium`. If there is no login, fetching is skipped with a
message and the scan continues using whatever `.vbo` files are already present.

### 2. Use local sessions, no download

```bash
videobrowser scan inputclips -o output --no-fetch --sessions-dir /path/to/vbos
```

`--sessions-dir` defaults to `<out>/sessions`. Drop your `.vbo` files there (or
point at another folder) and skip the network with `--no-fetch`.

### 3. Clips without overlays

```bash
videobrowser horizontals -o output --no-overlay
videobrowser verticals -o output --no-overlay
```

### 4. Custom font / stats location

```bash
videobrowser horizontals -o output \
  --font /System/Library/Fonts/Supplemental/Arial.ttf \
  --stats /path/to/stats.sqlite
```

`--stats` defaults to `<out>/stats.sqlite`. Point multiple runs at the same file
so day-best (green) and overall-best (purple) lap colouring accumulates across
track days.

---

## Useful flags (scan)

| Flag | Default | Purpose |
|------|---------|---------|
| `-o, --out` | required | Output directory |
| `--fps` | 5 | Analysis frame rate |
| `--size` | 640 | Max analysis side (px) |
| `--pad` | 10 | Seconds padded around each detected peak |
| `--profile` | `auto` | Camera mount profile (`auto`, `cockpit`, `side`, `rear`, `ground`, ...) |
| `--map` | none | `cameras.json` mount override |
| `--model` | `yolo11n.pt` | Ultralytics model |
| `--device` | `auto` | `auto` / `cuda` / `mps` / `cpu` / `coreml` |
| `--conf` | 0.25 | YOLO confidence threshold |
| `--exclude` | none | Repeatable; skip files/folders whose path contains the text |
| `--date YYYY-MM-DD` | all days | Repeatable; only scan recordings from that day |
| `--follow-links` | off | Include linked files that point outside the input folders (skipped links are listed in the output) |
| `--previews` | 8 | Annotated preview JPEGs per recording |
| `--max-clip-seconds` | 24 | Max clip window length |
| `--max-clips` | 12 | Max clip windows per recording |
| `--no-fetch` | off | Do not auto-download RaceBox sessions |
| `--sessions-dir` | `<out>/sessions` | Folder of `.vbo` sessions |
| `--min-speed` | 10 | Drop events below this GoPro GPS speed in km/h (standing still); 0 disables |

The clip options scan used to take (`--no-overlay`, `--no-vertical`,
`--no-landscape`, `--no-clips`, `--no-shorts`) are still accepted and ignored.
`horizontals` and `verticals` take `--date`, `--no-overlay` (alias
`--no-racebox`), `--force`, `--sessions-dir`, `--font` and `--jobs` (clips cut
at once, default CPU threads / 4, at most 4); `horizontals` also
`--stats` (default `<out>/stats.sqlite`). Run `videobrowser <command> --help`
for the complete list.

---

## Shorts

`scan` works through the recordings oldest first (by file creation time).
Copies of the same recording in several folders (same start time, length and
file sizes) are scanned once. `videobrowser shorts -o output` then builds the
shorts of every scanned day (`--date` for one). Run `verticals` first so the
shorts can copy whole vertical clips; without them every piece is cut from
the source. Each short is a folder of the day's 9:16
source clips, numbered in order, to put together in an editor (joining them
with ffmpeg gives clicks in the sound):

```
shorts/2026-08-01/
  action-01/ ...           # the day's best moments, several shorts of <= 50 s
    01_103112_action_front_<clip>.mp4
    02a_104501_overtake_front_<clip>.mp4   # one moment, two cameras
    02b_104501_overtake_rear.mp4
    klip.txt               # each file, its moment and the part to use
  overhalinger-01/         # one series per theme (needs at least 2 moments)
  overhalet-01/
  indhentninger-01/
  knae-01/
  styrt-01/                # own crash, from GoPro telemetry
  mulige-styrt-01/         # other riders: possible crashes to check
  grus-01/                 # riders out on grass or gravel
  index.html               # the shorts + every clip of the day, by type
  moments.json             # what was found and how each camera was synced
```

**Cameras on one clock.** Every recording of a day starts from its GoPro GPS
time. Overlapping recordings (front, rear, helmet) are then laid on top of
each other by matching their GPS tracks, or on the RaceBox log when a session
overlaps. Engine sound refines the result when both cameras' loudness
envelopes match clearly; otherwise GPS stands. Turn the audio step off with
`--no-audio-sync`.

**Themes.** Event meaning depends on which way the camera faces. The scan
reads it from the picture: at 101 moments spread over each recording it
compares two consecutive frames, and the scene grows towards a camera looking
ahead and shrinks into one looking back (`facing by scene flow: ...` in the
log, with the votes). Without a clear reading the mount guess decides (`rear`,
`rear_right`, `rear_rotated` look back, everything else ahead).
`videobrowser facing -o output` (optionally `--date`) re-reads it for an
existing scan without a full `scan --force`. When a camera stays unclear,
fix it by hand: `videobrowser facing -o output --set GX=front --set GH=rear`
applies to every recording whose id contains that text, now and in later
scans. `videobrowser facing -o output --clear` forgets those and measures
again:

| Theme | Rule | Reliability |
|-------|------|-------------|
| Overhaling | front `approaching` ends, rear `receding` starts within -1.5..+5 s | good with front + rear |
| Bliver overhalet | rear `approaching` ends, front `receding` starts within -1.5..+5 s | good with front + rear |
| Indhenter | front `approaching` for 1.5 s+ that is not an overtake | good |
| Duel | `rider_pass`: two other riders' bike boxes in view swap which one is nearer (box height, 15%+ apart, each order held 0.6 s, swap within 3 s); scores higher when they also swap left/right. Boxes that jump between frames (tracker swapped ids) do not count | needs tuning on real footage |
| Knæ i asfalten | `kneepuck_visible` | as the puck detector |
| Styrt (own) | accelerometer jolt >= 3x gravity, moving >= 30 km/h before, stopped after (GPS samples without a fix do not count, and the picture must stop moving too) | needs real crash data to tune |
| Muligt styrt (others) | `rider_down`: a bike box loses its rider while a person box appears beside it, or the bike box turns low and wide while out on grass/gravel | candidates to check by hand |
| Ude i gruset | `off_track`: the ground under a rider's box turns from asphalt to green grass or pale gravel for 0.6 s+; ground the colour of the road under the camera's own bike never counts (sunlit asphalt can look like gravel) | candidates to check by hand |
| Action | any other event, for the mixed shorts | |

`rider_down` and `off_track` are found during the scan (`incidents.py`), so
footage scanned before they existed needs `scan --force` to get them; the
same goes for `rider_pass`. Each piece is the whole vertical clip that covers it (crop + HUD),
copied; `klip.txt` says which part the moment is. A piece no clip covers
(e.g. the rear half of an overtake) is cut from the source with a crop that
looks into the corner. For verticals without the telemetry HUD run
`verticals --no-overlay`.
The scan drops events where the bike stands still (below 10 km/h GoPro GPS
speed, `scan --min-speed`). In the shorts, moments where the bike stands
still are left out, and moments below 40 km/h (paddock, pit lane;
`shorts --min-speed`) get at most one per action short and none in the
theme shorts. Only own crashes skip this check. Footage without GPS (e.g.
`.mov` files re-wrapped by Final Cut, which drop the GoPro telemetry track)
falls back to the scene motion measured during the scan (stored from this
version on, so rescan with `--force`). `--min-speed 0` turns the checks off.

With RaceBox sessions in `--sessions-dir` (default `<out>/sessions`), footage without GoPro GPS is laid on the RaceBox log by
matching its scene motion against the RaceBox speed (pit stops and out-laps
make the pattern), searched within a few minutes of the camera clock. A clear
match gives it RaceBox speed and turn rate, so the speed checks and the
look-into-the-corner crop work as with GoPro GPS, and its start time is
corrected for the shorts. The scan log says why a recording was not matched
(e.g. `weak match r=0.31`, `ambiguous r=0.62 vs 0.58 elsewhere`, `no RaceBox
log near the camera clock ...`). Scene motion is median-filtered over a
minute and capped first, so a camera picked up in the paddock does not look
like a session.

The shorts then match each camera's remaining recordings of the day at once
(one clock error per camera): hours of footage with pit stops between
sessions match where a single recording does not, paddock-only recordings
included. That placement is only good to a minute or so at session edges, so
it supplies speed (paddock or track) but not start time or turn rate. A
recording still unplaced is checked against the RaceBox log at both readings
of its camera clock: a moment with the bike in under both is dropped as
standing still, and one where it may be riding counts as slow (at most one
per action short), since it is not sure.

Overlays (RaceBox and telemetry HUD) only go on footage whose link to the
RaceBox log is sure: GoPro GPS time, or a clear scene-motion match. Other
footage is exported without overlay rather than with one that may be off.

Outputs scanned before this feature have no per-recording metadata yet; pass
the source folders once: `videobrowser shorts -o output inputclips`.

| Flag | Default | Purpose |
|------|---------|---------|
| `--date` | all | Only this day (repeatable) |
| `--short-seconds` | 50 | Max length of each short |
| `--max-action` | 6 | Max mixed action shorts per day |
| `--no-audio-sync` | off | Sync cameras by GPS only |
| `--no-crash` | off | Skip crash detection |
| `--min-speed` | 40 | Track speed (km/h): slower moments get at most one per action short, none in theme shorts |
| `--dry-run` | off | Only write `moments.json` and `index.html` |
| `--jobs` | CPU threads / 4, max 4 | Clips cut from the GoPro files at once |

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
