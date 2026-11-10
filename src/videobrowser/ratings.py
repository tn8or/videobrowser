"""Local ratings database: every moment the pipeline cut, and its votes.

``shorts``, ``horizontals`` and ``verticals`` register each moment they
produce here, with a snapshot of the detector features behind it, so votes
(``vote``) can later train a score (VIDE-1). The file lives at
``~/.videobrowser/ratings.sqlite`` (``VIDEOBROWSER_RATINGS`` overrides it),
apart from ``scan.sqlite``: a scan can be thrown away and redone, votes are
hand work and must outlive it. One file covers every output folder.

A moment is keyed by its day, recording, the recording's length, its first
piece's start in that recording (to the second) and its theme
(``clip_key``). A rescan
moves windows a little, so a new moment with no vote takes the latest vote
of an old one on the same recording that overlaps it by half or more
(``INHERIT_IOU``), and votes outlive detector changes.

Votes are append-only rows; the latest per moment counts. A vote also goes
to the same footage in the other outputs (overlapping moments on the same
recording), so voting on a short's moment rates its vertical clip too. File paths are
kept per machine (``clip_files``), since the Mac and the PC see the same
footage under different paths.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ENV_PATH = "VIDEOBROWSER_RATINGS"
SCHEMA_VERSION = 1
# A new moment takes an old moment's vote when they overlap this much
# (intersection over union of their recording time).
INHERIT_IOU = 0.5
VOTES = (-1, 0, 1)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- One row per moment (all its cameras together). t_start/t_end are the
-- first piece's recording time, which a re-sync leaves alone.
CREATE TABLE IF NOT EXISTS clips (
    clip_key TEXT PRIMARY KEY,
    date_label TEXT NOT NULL,
    recording_id TEXT NOT NULL,
    camera TEXT,
    theme TEXT NOT NULL,
    t_start REAL NOT NULL,
    t_end REAL NOT NULL,
    utc_start REAL NOT NULL,
    features TEXT NOT NULL,
    heuristic_score REAL,
    why TEXT NOT NULL DEFAULT '[]',
    model_score REAL,
    model_version INTEGER,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS clips_by_recording ON clips(date_label, recording_id);

-- Where this machine keeps a moment's files: "short", "vertical" or
-- "horizontal".
CREATE TABLE IF NOT EXISTS clip_files (
    clip_key TEXT NOT NULL,
    machine_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    -- When the file's first frame was filmed, on the day's shared clock (s);
    -- the rating page plays a moment's cameras in step with it.
    start REAL,
    PRIMARY KEY (clip_key, machine_id, kind, path)
);

-- Append-only: the latest row per moment counts. vote 0 clears a vote.
-- inherited_from names the old moment a rescanned one took its vote from.
CREATE TABLE IF NOT EXISTS votes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clip_key TEXT NOT NULL,
    vote INTEGER NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    machine_id TEXT NOT NULL,
    voted_at TEXT NOT NULL,
    inherited_from TEXT
);
CREATE INDEX IF NOT EXISTS votes_by_clip ON votes(clip_key, voted_at);

CREATE TABLE IF NOT EXISTS models (
    version INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at TEXT NOT NULL,
    n_votes INTEGER NOT NULL,
    weights TEXT NOT NULL,
    metrics TEXT NOT NULL
);
"""


def default_path() -> Path:
    env = os.environ.get(ENV_PATH)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".videobrowser" / "ratings.sqlite"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def clip_key(date_label: str, recording_id: str, recording_duration: float, t_start: float, theme: str) -> str:
    """Stable id of a moment. GoPro file names repeat across days, hence
    the day and the recording's length beside its name. The start is
    recording time, which re-syncing the cameras leaves alone."""
    raw = "|".join(
        [date_label, recording_id, str(round(recording_duration)), str(round(t_start)), theme]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Intersection over union of two time spans."""
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


@dataclass
class ClipRecord:
    """A moment to register: where it is, what the detectors saw, and the
    files this machine wrote for it."""

    date_label: str
    recording_id: str
    recording_duration: float
    theme: str
    t_start: float
    t_end: float
    utc_start: float
    camera: str | None = None
    features: dict = field(default_factory=dict)
    heuristic_score: float | None = None
    # (kind, path) or (kind, path, start on the day clock)
    files: list[tuple] = field(default_factory=list)
    # Why the pipeline kept it (``Moment.why``), shown on the rating page.
    why: list[str] = field(default_factory=list)
    # The learned score (``learn.py``) and its model version, when known.
    model_score: float | None = None
    model_version: int | None = None

    @property
    def key(self) -> str:
        return clip_key(self.date_label, self.recording_id, self.recording_duration, self.t_start, self.theme)


@dataclass
class RatingsDB:
    path: Path
    conn: sqlite3.Connection
    machine_id: str

    @classmethod
    def open(cls, path: Path | None = None) -> RatingsDB:
        path = path or default_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        columns = {r["name"] for r in conn.execute("PRAGMA table_info(clips)")}
        if "why" not in columns:  # files from before the rating page
            conn.execute("ALTER TABLE clips ADD COLUMN why TEXT NOT NULL DEFAULT '[]'")
        if "start" not in {r["name"] for r in conn.execute("PRAGMA table_info(clip_files)")}:
            conn.execute("ALTER TABLE clip_files ADD COLUMN start REAL")
        row = conn.execute("SELECT value FROM meta WHERE key = 'machine_id'").fetchone()
        if row is None:
            machine_id = uuid.uuid4().hex
            conn.execute("INSERT INTO meta(key, value) VALUES ('machine_id', ?)", (machine_id,))
        else:
            machine_id = row["value"]
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
        return cls(path, conn, machine_id)

    def close(self) -> None:
        self.conn.close()

    # --- moments ---------------------------------------------------------

    def register(self, record: ClipRecord, commit: bool = True) -> str:
        """Add or refresh a moment and this machine's files for it; a new
        moment with no vote takes one from an old moment it overlaps."""
        key = record.key
        now = _now()
        self.conn.execute(
            """
            INSERT INTO clips(clip_key, date_label, recording_id, camera, theme, t_start, t_end,
                              utc_start, features, heuristic_score, why, first_seen, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(clip_key) DO UPDATE SET
                camera = excluded.camera, t_start = excluded.t_start, t_end = excluded.t_end,
                features = excluded.features, heuristic_score = excluded.heuristic_score,
                why = excluded.why, last_seen = excluded.last_seen
            """,
            (
                key, record.date_label, record.recording_id, record.camera, record.theme,
                record.t_start, record.t_end, record.utc_start,
                json.dumps(record.features, sort_keys=True), record.heuristic_score,
                json.dumps(record.why, ensure_ascii=False), now, now,
            ),
        )
        if record.model_score is not None:
            self.conn.execute(
                "UPDATE clips SET model_score = ?, model_version = ? WHERE clip_key = ?",
                (record.model_score, record.model_version, key),
            )
        for kind, path, *start in record.files:
            self.conn.execute(
                "INSERT INTO clip_files(clip_key, machine_id, kind, path, start) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(clip_key, machine_id, kind, path) DO UPDATE SET start = excluded.start",
                (key, self.machine_id, kind, str(path), start[0] if start else None),
            )
        if self.latest_vote(key) is None:
            self._inherit(key, record)
        if commit:
            self.conn.commit()
        return key

    def _inherit(self, key: str, record: ClipRecord) -> None:
        """Give a new moment the vote of the best-overlapping old one."""
        for other in self._overlapping(record.date_label, record.recording_id, (record.t_start, record.t_end), key):
            old = self.latest_vote(other)
            if old is None:
                continue
            self.conn.execute(
                "INSERT INTO votes(clip_key, vote, tags, machine_id, voted_at, inherited_from) VALUES (?, ?, ?, ?, ?, ?)",
                (key, old["vote"], old["tags"], self.machine_id, old["voted_at"], other),
            )
            return

    def forget_files(self, kind: str, under: Path, name_prefix: str = "") -> None:
        """Drop this machine's *kind* file rows in folder *under* (and below)
        whose file name starts with *name_prefix*, before a command writes
        those files again."""
        folder = Path(under)
        rows = self.conn.execute(
            "SELECT clip_key, path FROM clip_files WHERE machine_id = ? AND kind = ?",
            (self.machine_id, kind),
        ).fetchall()
        gone = []
        for r in rows:
            path = Path(r["path"])
            if folder in path.parents and path.name.startswith(name_prefix):
                gone.append((r["clip_key"], r["path"]))
        self.conn.executemany(
            "DELETE FROM clip_files WHERE clip_key = ? AND machine_id = ? AND kind = ? AND path = ?",
            [(k, self.machine_id, kind, p) for k, p in gone],
        )
        self.conn.commit()

    def clip(self, key: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM clips WHERE clip_key = ?", (key,)).fetchone()

    def files(self, key: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT kind, path FROM clip_files WHERE clip_key = ? AND machine_id = ? ORDER BY kind, path",
            (key, self.machine_id),
        ).fetchall()

    def key_for_path(self, path: Path) -> str | None:
        row = self.conn.execute(
            "SELECT clip_key FROM clip_files WHERE machine_id = ? AND path = ?",
            (self.machine_id, str(path)),
        ).fetchone()
        return row["clip_key"] if row else None

    # --- votes -----------------------------------------------------------

    def vote(self, key: str, vote: int, tags: list[str] | None = None, spread: bool = True) -> list[str]:
        """Vote on *key*. With *spread*, the same footage in the other
        outputs (moments on the same recording overlapping it by
        ``INHERIT_IOU`` or more, e.g. its vertical and landscape clips) gets
        the vote too. Returns every key that got the vote, *key* first."""
        if vote not in VOTES:
            raise ValueError(f"vote must be one of {VOTES}, not {vote!r}")
        clip = self.clip(key)
        if clip is None:
            raise KeyError(key)
        now = _now()
        tag_json = json.dumps(sorted(set(tags or [])))
        rows = [(key, vote, tag_json, self.machine_id, now, None)]
        if spread:
            rows += [
                (other, vote, tag_json, self.machine_id, now, key)
                for other in self._overlapping(clip["date_label"], clip["recording_id"],
                                               (clip["t_start"], clip["t_end"]), key)
            ]
        self.conn.executemany(
            "INSERT INTO votes(clip_key, vote, tags, machine_id, voted_at, inherited_from) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        self.conn.commit()
        return [r[0] for r in rows]

    def _overlapping(self, date_label: str, recording_id: str, span: tuple[float, float], key: str) -> list[str]:
        """Moments on the same recording overlapping *span* by
        ``INHERIT_IOU`` or more, best overlap first."""
        found = []
        for row in self.conn.execute(
            "SELECT clip_key, t_start, t_end FROM clips WHERE date_label = ? AND recording_id = ? AND clip_key != ?",
            (date_label, recording_id, key),
        ):
            iou = overlap(span, (row["t_start"], row["t_end"]))
            if iou >= INHERIT_IOU:
                found.append((iou, row["clip_key"]))
        return [k for _, k in sorted(found, reverse=True)]

    def latest_vote(self, key: str) -> sqlite3.Row | None:
        """The vote that counts for *key* (None when never voted; a cleared
        vote is a row with vote 0)."""
        return self.conn.execute(
            "SELECT * FROM votes WHERE clip_key = ? ORDER BY voted_at DESC, id DESC LIMIT 1",
            (key,),
        ).fetchone()

    def stats(self) -> dict[str, int]:
        latest = self.conn.execute(
            """
            SELECT vote FROM (
                SELECT vote, ROW_NUMBER() OVER (
                    PARTITION BY clip_key ORDER BY voted_at DESC, id DESC
                ) AS n FROM votes
            ) WHERE n = 1
            """
        ).fetchall()
        return {
            "moments": self.conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0],
            "up": sum(1 for r in latest if r["vote"] > 0),
            "down": sum(1 for r in latest if r["vote"] < 0),
        }


def register_all(
    records: list[ClipRecord], forget: list[tuple] = (), path: Path | None = None
) -> int:
    """Register *records* in one go, after dropping this machine's file rows
    under each (kind, folder[, file name prefix]) in *forget*. A broken ratings file must never
    stop clips from being cut, so errors are printed, not raised."""
    try:
        db = RatingsDB.open(path)
    except (sqlite3.Error, OSError) as exc:
        print(f"  ratings: not saved ({exc})")
        return 0
    try:
        for item in forget:
            db.forget_files(*item)
        from videobrowser.learn import current

        model = current(db)
        for record in records:
            if model is not None and record.model_score is None:
                record.model_score = model[1].predict(record.features)
                record.model_version = model[0]
            db.register(record, commit=False)
        db.conn.commit()
        return len(records)
    except (sqlite3.Error, OSError) as exc:
        print(f"  ratings: not saved ({exc})")
        return 0
    finally:
        db.close()
