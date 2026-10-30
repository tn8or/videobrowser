from __future__ import annotations

import sqlite3
from dataclasses import dataclass
import json
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
    recording_id TEXT PRIMARY KEY,
    profile TEXT NOT NULL,
    duration REAL NOT NULL,
    processed INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    recording_id TEXT NOT NULL,
    chapter INTEGER NOT NULL,
    duration REAL NOT NULL,
    processed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id TEXT NOT NULL,
    type TEXT NOT NULL,
    t_start REAL NOT NULL,
    t_end REAL NOT NULL,
    score REAL NOT NULL,
    track_id TEXT,
    clip_path TEXT,
    extra TEXT
);

-- Per-recording facts the ``shorts`` stage needs without re-scanning.
CREATE TABLE IF NOT EXISTS recording_meta (
    recording_id TEXT PRIMARY KEY,
    profile TEXT NOT NULL,
    rotation INTEGER NOT NULL DEFAULT 0,
    start_utc REAL,
    date_label TEXT NOT NULL,
    duration REAL NOT NULL,
    chapters TEXT NOT NULL
);

-- Scene motion per second (JSON [[t, motion], ...]): tells the shorts when
-- the bike stands still on footage without GPS.
CREATE TABLE IF NOT EXISTS motion (
    recording_id TEXT PRIMARY KEY,
    samples TEXT NOT NULL
);

-- Share of frames per second with a car, van or truck in view (JSON
-- [[t, share], ...]): tells the shorts when the bike is in the paddock.
CREATE TABLE IF NOT EXISTS vehicles (
    recording_id TEXT PRIMARY KEY,
    samples TEXT NOT NULL
);

-- Share of frames per second where the camera rides on a moving bike
-- (JSON [[t, share], ...], ``motion.MountWatch``): tells the shorts track
-- from paddock on footage without GPS. No row: the recording showed no part
-- of a bike that stays put (helmet camera, camera in the hand).
CREATE TABLE IF NOT EXISTS riding (
    recording_id TEXT PRIMARY KEY,
    samples TEXT NOT NULL
);

-- What there is to see per second (JSON [[t, riders, kerb], ...],
-- ``action.py``): the most other riders in one frame and the share of frames
-- with a kerb close by. Shorts need one of them in view.
CREATE TABLE IF NOT EXISTS action (
    recording_id TEXT PRIMARY KEY,
    samples TEXT NOT NULL
);

-- Rider boxes inside the clip windows (JSON [[track_id, t, x1, y1, x2, y2,
-- frame_w, frame_h], ...]): what the vertical crop follows, without
-- re-running the detector.
-- "front"/"rear" set by hand for every recording whose id contains camera
-- (e.g. "GX"); wins over the scene-flow reading.
CREATE TABLE IF NOT EXISTS facing_overrides (
    camera TEXT PRIMARY KEY,
    facing TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tracks (
    recording_id TEXT PRIMARY KEY,
    points TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id TEXT NOT NULL,
    t_start REAL NOT NULL,
    t_end REAL NOT NULL,
    types TEXT NOT NULL,
    score REAL NOT NULL,
    clip_path TEXT,
    vertical_path TEXT
);
"""


@dataclass
class ScanDB:
    path: Path
    conn: sqlite3.Connection

    @classmethod
    def open(cls, path: Path) -> ScanDB:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=30.0)
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(recording_meta)")}
        if "start_source" not in columns:
            # How start_utc was found: "gps", "racebox-motion" or "clock".
            conn.execute("ALTER TABLE recording_meta ADD COLUMN start_source TEXT")
        if "facing" not in columns:
            # "front"/"rear" from scene flow; NULL falls back to the profile.
            conn.execute("ALTER TABLE recording_meta ADD COLUMN facing TEXT")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS scans (id INTEGER PRIMARY KEY, started_at TEXT, note TEXT)"
        )
        conn.execute(
            "INSERT INTO scans(started_at, note) VALUES (?, ?)",
            (datetime.now(timezone.utc).isoformat(), path.name),
        )
        conn.commit()
        return cls(path=path, conn=conn)

    def close(self) -> None:
        self.conn.close()

    def is_processed(self, recording_id: str) -> bool:
        row = self.conn.execute(
            "SELECT processed FROM recordings WHERE recording_id = ?",
            (recording_id,),
        ).fetchone()
        return bool(row and row["processed"])

    def mark_recording(self, recording_id: str, profile: str, duration: float, processed: bool) -> None:
        self.conn.execute(
            """
            INSERT INTO recordings(recording_id, profile, duration, processed, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(recording_id) DO UPDATE SET
                profile=excluded.profile,
                duration=excluded.duration,
                processed=excluded.processed,
                updated_at=excluded.updated_at
            """,
            (
                recording_id,
                profile,
                duration,
                int(processed),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    def replace_events(self, recording_id: str, events: list[tuple]) -> None:
        self.conn.execute("DELETE FROM events WHERE recording_id = ?", (recording_id,))
        self.conn.executemany(
            """
            INSERT INTO events(recording_id, type, t_start, t_end, score, track_id, clip_path, extra)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            events,
        )
        self.conn.commit()

    def save_recording_meta(
        self,
        recording_id: str,
        profile: str,
        rotation: int,
        start_utc: float | None,
        date_label: str,
        duration: float,
        chapters: list[dict],
        start_source: str | None = None,
        facing: str | None = None,
    ) -> None:
        """*chapters*: ``[{"path", "offset", "duration"}, ...]`` in recording order.

        *start_source*: how *start_utc* was found ("gps", "racebox-motion"
        or "clock"); overlays go only on the first two. *facing*: "front"
        or "rear" when scene flow told, else None (the profile decides).
        """
        self.conn.execute(
            """
            INSERT INTO recording_meta(
                recording_id, profile, rotation, start_utc, date_label, duration, chapters,
                start_source, facing
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(recording_id) DO UPDATE SET
                profile=excluded.profile,
                rotation=excluded.rotation,
                start_utc=excluded.start_utc,
                date_label=excluded.date_label,
                duration=excluded.duration,
                chapters=excluded.chapters,
                start_source=excluded.start_source,
                facing=excluded.facing
            """,
            (
                recording_id, profile, rotation, start_utc, date_label, duration,
                json.dumps(chapters), start_source, facing,
            ),
        )
        self.conn.commit()

    def set_facing_override(self, camera: str, facing: str) -> None:
        self.conn.execute(
            "INSERT INTO facing_overrides(camera, facing) VALUES (?, ?) "
            "ON CONFLICT(camera) DO UPDATE SET facing=excluded.facing",
            (camera, facing),
        )
        self.conn.commit()

    def clear_facing_overrides(self) -> None:
        self.conn.execute("DELETE FROM facing_overrides")
        self.conn.commit()

    def facing_override(self, recording_id: str) -> str | None:
        """The hand-set facing whose camera text occurs in *recording_id*."""
        for row in self.conn.execute("SELECT camera, facing FROM facing_overrides ORDER BY length(camera) DESC"):
            if row["camera"] in recording_id:
                return row["facing"]
        return None

    def set_facing(self, recording_id: str, facing: str | None) -> None:
        self.conn.execute(
            "UPDATE recording_meta SET facing = ? WHERE recording_id = ?", (facing, recording_id)
        )
        self.conn.commit()

    def save_tracks(self, recording_id: str, points: list) -> None:
        """*points*: ``TrackPoint``-like objects (track_id, t, box, frame size)."""
        self.conn.execute(
            "INSERT OR REPLACE INTO tracks(recording_id, points) VALUES (?, ?)",
            (
                recording_id,
                json.dumps(
                    [
                        [p.track_id, round(p.t, 2), round(p.x1, 1), round(p.y1, 1),
                         round(p.x2, 1), round(p.y2, 1), p.frame_w, p.frame_h]
                        for p in points
                    ]
                ),
            ),
        )
        self.conn.commit()

    def tracks(self, recording_id: str) -> list[list]:
        row = self.conn.execute(
            "SELECT points FROM tracks WHERE recording_id = ?", (recording_id,)
        ).fetchone()
        return json.loads(row["points"]) if row else []

    def clips_for(self, recording_id: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM clips WHERE recording_id = ? ORDER BY t_start", (recording_id,)
            )
        )

    def events_for(self, recording_id: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM events WHERE recording_id = ? ORDER BY t_start", (recording_id,)
            )
        )

    def set_clip_path(self, clip_id: int, column: str, path: str | None) -> None:
        """Record an exported file: *column* is ``clip_path`` or ``vertical_path``."""
        if column not in ("clip_path", "vertical_path"):
            raise ValueError(column)
        self.conn.execute(f"UPDATE clips SET {column} = ? WHERE id = ?", (path, clip_id))
        self.conn.commit()

    def replace_clips(self, recording_id: str, clips: list[tuple]) -> None:
        """*clips*: ``(t_start, t_end, types, score, clip_path, vertical_path)`` rows."""
        self.conn.execute("DELETE FROM clips WHERE recording_id = ?", (recording_id,))
        self.conn.executemany(
            """
            INSERT INTO clips(recording_id, t_start, t_end, types, score, clip_path, vertical_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [(recording_id, *row) for row in clips],
        )
        self.conn.commit()

    def save_motion(self, recording_id: str, samples: list[tuple[float, float]]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO motion(recording_id, samples) VALUES (?, ?)",
            (recording_id, json.dumps([[round(t, 1), round(m, 2)] for t, m in samples])),
        )
        self.conn.commit()

    def motion(self) -> dict[str, list[tuple[float, float]]]:
        return {
            row["recording_id"]: [(float(t), float(m)) for t, m in json.loads(row["samples"])]
            for row in self.conn.execute("SELECT * FROM motion")
        }

    def save_vehicles(self, recording_id: str, samples: list[tuple[float, float]]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO vehicles(recording_id, samples) VALUES (?, ?)",
            (recording_id, json.dumps([[round(t, 1), round(v, 2)] for t, v in samples])),
        )
        self.conn.commit()

    def vehicles(self) -> dict[str, list[tuple[float, float]]]:
        return {
            row["recording_id"]: [(float(t), float(v)) for t, v in json.loads(row["samples"])]
            for row in self.conn.execute("SELECT * FROM vehicles")
        }

    def save_riding(self, recording_id: str, samples: list[tuple[float, float]] | None) -> None:
        if samples is None:
            self.conn.execute("DELETE FROM riding WHERE recording_id = ?", (recording_id,))
        else:
            self.conn.execute(
                "INSERT OR REPLACE INTO riding(recording_id, samples) VALUES (?, ?)",
                (recording_id, json.dumps([[round(t, 1), round(v, 2)] for t, v in samples])),
            )
        self.conn.commit()

    def riding(self) -> dict[str, list[tuple[float, float]]]:
        return {
            row["recording_id"]: [(float(t), float(v)) for t, v in json.loads(row["samples"])]
            for row in self.conn.execute("SELECT * FROM riding")
        }

    def save_action(self, recording_id: str, samples: list[tuple]) -> None:
        """(t, riders, kerb share, nearest rider height, red lamp) per
        second; scans from before have fewer values per row."""
        rows = [
            [round(row[0], 1), int(row[1]), round(row[2], 2)]
            + [round(v, 3) for v in row[3:4]]
            + [int(v) for v in row[4:5]]
            for row in samples
        ]
        self.conn.execute(
            "INSERT OR REPLACE INTO action(recording_id, samples) VALUES (?, ?)",
            (recording_id, json.dumps(rows)),
        )
        self.conn.commit()

    def action(self) -> dict[str, list[tuple]]:
        return {
            row["recording_id"]: [
                (
                    float(r[0]), int(r[1]), float(r[2]),
                    float(r[3]) if len(r) > 3 else 0.0, int(r[4]) if len(r) > 4 else 0,
                )
                for r in json.loads(row["samples"])
            ]
            for row in self.conn.execute("SELECT * FROM action")
        }

    def recording_metas(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM recording_meta ORDER BY start_utc"))

    def all_events(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM events ORDER BY recording_id, t_start"))

    def all_clips(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM clips ORDER BY recording_id, t_start"))
