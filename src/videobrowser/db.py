from __future__ import annotations

import sqlite3
from dataclasses import dataclass
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
"""


@dataclass
class ScanDB:
    path: Path
    conn: sqlite3.Connection

    @classmethod
    def open(cls, path: Path) -> ScanDB:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
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
