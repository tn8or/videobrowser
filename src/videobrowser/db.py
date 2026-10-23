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
    ) -> None:
        """*chapters*: ``[{"path", "offset", "duration"}, ...]`` in recording order."""
        self.conn.execute(
            """
            INSERT INTO recording_meta(recording_id, profile, rotation, start_utc, date_label, duration, chapters)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(recording_id) DO UPDATE SET
                profile=excluded.profile,
                rotation=excluded.rotation,
                start_utc=excluded.start_utc,
                date_label=excluded.date_label,
                duration=excluded.duration,
                chapters=excluded.chapters
            """,
            (recording_id, profile, rotation, start_utc, date_label, duration, json.dumps(chapters)),
        )
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

    def recording_metas(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM recording_meta ORDER BY start_utc"))

    def all_events(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM events ORDER BY recording_id, t_start"))

    def all_clips(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM clips ORDER BY recording_id, t_start"))

    def date_labels(self, recording_ids: list[str]) -> list[str]:
        """Output days (``recording_meta.date_label``) of *recording_ids*."""
        if not recording_ids:
            return []
        marks = ",".join("?" * len(recording_ids))
        rows = self.conn.execute(
            f"SELECT DISTINCT date_label FROM recording_meta WHERE recording_id IN ({marks})",
            recording_ids,
        )
        return sorted(row["date_label"] for row in rows)
