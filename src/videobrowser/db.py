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

-- Rider boxes inside the clip windows (JSON [[track_id, t, x1, y1, x2, y2,
-- frame_w, frame_h], ...]): what the vertical crop follows, without
-- re-running the detector.
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
    ) -> None:
        """*chapters*: ``[{"path", "offset", "duration"}, ...]`` in recording order.

        *start_source*: how *start_utc* was found ("gps", "racebox-motion"
        or "clock"); overlays go only on the first two.
        """
        self.conn.execute(
            """
            INSERT INTO recording_meta(
                recording_id, profile, rotation, start_utc, date_label, duration, chapters, start_source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(recording_id) DO UPDATE SET
                profile=excluded.profile,
                rotation=excluded.rotation,
                start_utc=excluded.start_utc,
                date_label=excluded.date_label,
                duration=excluded.duration,
                chapters=excluded.chapters,
                start_source=excluded.start_source
            """,
            (
                recording_id, profile, rotation, start_utc, date_label, duration,
                json.dumps(chapters), start_source,
            ),
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

    def recording_metas(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM recording_meta ORDER BY start_utc"))

    def all_events(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM events ORDER BY recording_id, t_start"))

    def all_clips(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM clips ORDER BY recording_id, t_start"))
