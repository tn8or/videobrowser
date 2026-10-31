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

Log in to racebox.pro once for auto-fetch: `videobrowser racebox-login`
opens Chrome, you log in (and solve the captcha), and the session is saved to
`~/.videobrowser/racebox_auth.json`. Set `CHROME_BIN` if Chrome isn't found.
A login saved by older versions in `racebox/fetcher/auth_state.json` is copied
there automatically on the first fetch; after that the old `racebox/` folder
can be deleted.

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
  follow the rider that triggered the clip (else any bike in view; a person
  only when no bike is, so a marshal does not pull it away), and
  keeps that rider as they get smaller. Boxes that sit still on the frame
  edge (the camera's own exhaust or fairing) are never followed, nor scored
  as riders. Only after nobody has been in view
  for 3 s does it look into the corner like a rider would, from the GoPro
  GPS turn rate (left in a left-hander on a front camera, mirrored on a rear
  one). A helmet camera (no part of the bike in view) never does: the head
  already looks into the corner. It opens centred unless the subject is already in view. Without RaceBox sessions the overlay uses GoPro telemetry or
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

Auto-fetch needs a saved login (`videobrowser racebox-login`) plus
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
at once, default CPU threads / 4, at most 4), plus `--min-speed` (40) and
`--max-vehicle-share` (0.4), which leave out clips standing still or in the
paddock, as the shorts do; `horizontals` also
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
  afkoersler-01/           # own bike over the edge of the track
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
| Afkørsel (own) | `own_off_track`, front camera only: the middle of the picture (the ground the bike rides onto) turns from asphalt to grass or gravel for 0.8 s+ after 3 s+ on asphalt; asphalt the colour of the last road frames never counts. Judged on the pace of the 6 s before it, since it slows the bike | needs tuning on real footage |
| Action | any other event, for the mixed shorts; also `kerb_close` (a red-and-white kerb covers 4%+ of the lower frame for 4+ frames) | |

`rider_down`, `off_track` and `own_off_track` are found during the scan (`incidents.py`), so
footage scanned before they existed needs `scan --force` to get them; the
same goes for `rider_pass`. Each piece is the whole vertical clip that covers it (crop + HUD),
copied; `klip.txt` says which part the moment is. A piece no clip covers
(e.g. the rear half of an overtake) is cut from the source with a crop that
looks into the corner. For verticals without the telemetry HUD run
`verticals --no-overlay`.
The scan drops events where the bike stands still (below 10 km/h GoPro GPS
speed, `scan --min-speed`). `horizontals`, `verticals` and the shorts then
all leave out clips and moments where the bike stands still or rides below
40 km/h (paddock, pit lane; `--min-speed` on each), by the same rule; the
shorts only cut tighter. Only own crashes skip this check. `--min-speed 0`
turns the checks off.

Footage without GoPro GPS (e.g. `.mov` files re-wrapped by Final Cut, trimmed
or exported clips, other riders' clips) is judged by the picture: does the
camera ride on a moving bike? Scene motion alone cannot tell, since a camera
waved about in the paddock moves the picture as much as the track streaming
past. What a camera on a bike has, and one in the hand lacks, is a part of
the picture that stays put while the rest moves (fairing, dash, tail). The
scan finds those blocks per recording (still and textured in most moving
frames) and stores, per second, the share of frames where the scene streams
past while they hold still (`riding` in `scan.sqlite`). A moment is on track
when half its frames ride. This beats RaceBox speed laid on by scene motion,
whose timing can be off. Change is measured at 320 px against each block's
own texture, so soft, low-contrast scenes stream too. The scan log says how
close each recording came (`riding: ... streaming median ..., N bike blocks
holding still`). When a recording shows no part of a bike (helmet camera,
all in the hand), RaceBox speed decides where the recording is placed on the
log. Scene motion alone never makes a moment track, since a camera in the
hand moves as much; with another rider (a person on a bike) in view while
the scene streams past at track pace it does, which is what keeps helmet
camera moments without RaceBox speed. Without any of these the moment is
left out. Rescan
with `--force` to get it.

**Something to see.** Every moment (and every clip in `horizontals` and
`verticals`) needs another rider in view (a bike box 3%+ of the frame
height with a person on it, so not a parked bike or the camera's own bike
filmed by hand) or a red-and-white kerb close by; own
crashes and own off-tracks are exempt. Moments are ranked by how much there
is to see per second: each rider in view (up to four) counts fully when its
box is 15%+ of the frame height and less the farther off it is, and a kerb
close by counts half. Overtakes and duels weigh 1.5x, a knee down adds one;
own crashes and off-tracks keep their own score. The scan stores this per
second (`action` in `scan.sqlite`, with the nearest rider's height); days
scanned before it need `scan --force` to get the check.

**Session over.** When a session ends the marshal posts blink a red lamp
and everyone rides into the pit lane. The scan looks for a small, bright red
spot above the track (not on a rider) that comes, goes and comes back within
a few seconds, and stores it per second (`action` in `scan.sqlite`); moments
and clips with it are left out. Every piece of a short is at least 8 s long,
widened around its middle; a moment on a recording shorter than that is
left out. Days scanned before need `scan --force`.

**Why a moment is in.** `klip.txt` ends each line with why the moment was
kept (riders in view, kerb, and where the pace came from: `GPS 87 km/h`,
`RaceBox 74 km/h` or `riding 92% of the time`). `moments.json` has the same
under `why`, and lists the moments left out under `left_out`, with the
reason last.

**Recordings without a date** (no creation time, or a placeholder such as
1904 or 1970 written by an editor) go to `shorts/unknown/`. Each gets a
timeline of its own there, so they never pair up as overtakes or push each
other's moments out; file names and `klip.txt` show the time into the
recording instead of the clock.

Cars, vans and trucks mark the paddock: there are none on track. The scan
asks the detector for them beside the riders (they never become rider
tracks) and drops events with a vehicle in view at least 40% of the time
(`scan --max-vehicle-share`), so they never become clip windows for
`horizontals`, `verticals` or the shorts. The shorts check their moments
again from the per-second share stored in the scan (`shorts
--max-vehicle-share`). Only boxes at least 8% of the frame high count, so a
car far off behind the fence does not, and riding past a marshal's car for a
second or two stays under the share. `0` turns the check off; days scanned
before this need `scan --force`.

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

Scene motion lands within seconds at best, and sometimes half a lap off.
At a venue with bridges over the track, the scan then looks for them:
Knutstorp's "Porsche Center Malmö" and "Förenade Bil" bridges, and
Anderstorp's "PORSCHE" bridge. A bridge's underside is a long, nearly level
edge (12°, 20° from a side mount), dark above with sky below, rising up the picture and leaving over the
top edge. Every mount sees it, shadow and fairing don't hide it, and
nothing else on the lap looks like it. RaceBox gives the time the bike is
under each bridge from the distance along the track past the start line:
56 m and 666 m at Knutstorp (where the underside leaves the picture), and
0 m at Anderstorp (RaceBox's line taken as the painted line under the
bridge, the second of two 18 m apart). The picture is searched at 12.5 fps ±40 s around three
of those times, and the start most bridge passes agree with (two or more;
one alone lines up with some bridge near almost any start) wins. Where scene motion found nothing,
or nothing with a bridge near, a recording of 250 s or more is searched
whole and laid on
every bridge pass within a day and a half of the camera clock (camera clocks
set to local time are hours off); that needs six bridge passes on RaceBox's
and 1.5 times the next best start. It is skipped (it decodes the whole
recording) when RaceBox has fewer than six bridge passes in any stretch as
long as the recording, or when the camera is pitched so far down that under
2 % of the picture is sky.

The painted finish line then sets the start to the frame: every frame is
decoded ±0.35 s around each crossing (where the bridge leaves the picture
depends on the camera's height and pitch), and the finish line is the last line
with 30 m of clear track after it (grid and start lines lie 13-50 m before
it). Two crossings or more must agree; otherwise, as where the line is in
shadow, the bridge start stands. The scan log says `laid on RaceBox by the
bridge (bridge: 8 of 8 landmark passes ...; finish line 3 of 4 crossings,
-0.04 s, ...)` or why not.

The bridges also tell the shorts when a recording without GPS is on track:
the pit lane does not pass under them, so from the first bridge pass until
100 s without one the bike is on track, and moments outside count as
paddock. A recording placed on RaceBox goes by RaceBox's bridge times; one
the scan searched whole by the passes it saw (one missed pass tolerated).

Venues without a bridge (Padborg Park) go by the finish line alone: every
frame ±40 s around the fastest crossing, then ±0.5 s around each other
crossing for every start that gives. The start most crossings agree with
(two or more) wins; a recording with one crossing only is placed by it
within 10 s of scene motion. The scan log says `laid on RaceBox by the
finish line (4 of 5 crossings on the line, spread 0.02 s, ...)` or why
not. Either way the start source is then `racebox-line`, and overlays and
shorts use it. Days scanned before this need `scan --force`.

To check one recording by hand, `videobrowser finish-line VIDEO [VIDEO ...]
--sessions-dir output/sessions --near START_UTC --frames passes/` prints the
bridge passes and lines found in each window searched, the placement, and
saves the three frames before each pass with the edges or lines it saw
drawn. `START_UTC` is the scan's `start_utc` (epoch s) for the recording;
`--no-bridge` goes by the finish line alone.

The shorts then match each camera's remaining recordings of the day at once
(one clock error per camera): hours of footage with pit stops between
sessions match where a single recording does not, paddock-only recordings
included. That placement is only good to a minute or so at session edges, so
it supplies speed (paddock or track) but not start time or turn rate. A
recording still unplaced is checked against the RaceBox log at both readings
of its camera clock: a moment with the bike in under both is dropped as
standing still, and one where it may be riding counts as slow and is
left out too, since it is not sure.

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
| `--min-speed` | 40 | Track speed (km/h): slower moments (standing still, paddock) are left out |
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
src/videobrowser/          # CLI, scan pipeline, clip cutting, RaceBox stage
src/videobrowser/racebox/  # VBO parser, lap timing, stats, Cairo overlay renderer, racebox.pro fetcher
tests/                     # pytest suite
```

---

## Manual RaceBox download

`videobrowser scan` fetches new sessions itself. To download them on their own:

```bash
videobrowser racebox-fetch output/sessions
# add --headful to watch the browser (e.g. if Bike Mode needs a manual click)
```

---

## Development

```bash
pip install -e ".[dev]"
pytest
```
