from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile

# Filesystems where SQLite WAL (mmap of the -shm sidecar) can hang forever.
_REMOTE_FS_TYPES = frozenset(
    {
        "smbfs",
        "smb",
        "cifs",
        "nfs",
        "nfs4",
        "afpfs",
        "afp",
        "webdav",
        "fuse",
        "osxfuse",
        "macfuse",
        "fuseblk",
        "ncpfs",
    }
)
_LOCAL_FS_TYPES = frozenset(
    {
        "apfs",
        "hfs",
        "hfs+",
        "ufs",
        "msdos",
        "exfat",
        "ntfs",
        "ext2",
        "ext3",
        "ext4",
        "xfs",
        "btrfs",
        "zfs",
        "tmpfs",
    }
)
_remote_dev_cache: dict[int, bool] = {}
_warned_remote: set[str] = set()
_SQLITE_TIMEOUT_S = 30.0


def _existing_ancestor(path: Path) -> Path:
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return probe


def _fstype(path: Path) -> str:
    if sys.platform == "darwin":
        try:
            result = subprocess.run(
                ["stat", "-f", "%T", str(path)],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if result.returncode == 0:
            return result.stdout.strip().lower()
    return ""


def _fstype_from_mount(path: Path) -> str:
    """Filesystem type of the longest mount prefix covering *path* (macOS/Linux `mount`)."""
    try:
        result = subprocess.run(
            ["mount"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    try:
        path_s = str(path.resolve())
    except OSError:
        path_s = str(path)
    best_type = ""
    best_len = -1
    for line in result.stdout.splitlines():
        if " on " not in line or " (" not in line:
            continue
        _src, rest = line.split(" on ", 1)
        mnt, opts = rest.rsplit(" (", 1)
        fstype = opts.split(",", 1)[0].strip().lower()
        if not fstype:
            continue
        if path_s == mnt or path_s.startswith(mnt.rstrip("/") + "/") or mnt == "/":
            if len(mnt) > best_len:
                best_len = len(mnt)
                best_type = fstype
    return best_type


def is_remote_filesystem(path: Path) -> bool:
    """True when *path* is on SMB/NFS/AFP or another non-local volume.

    SQLite WAL uses mmap on a shared-memory sidecar. That hangs on network
    filesystems rather than erroring, which looks like a stall right after
    videobrowser prints the detector line.

    macOS SMB mounts often report a bogus type from ``stat -f %T`` (``/`` or
    ``???``); the mount table is the reliable signal.
    """
    probe = _existing_ancestor(path)
    try:
        probe = probe.resolve()
    except OSError:
        pass
    try:
        dev = probe.stat().st_dev
    except OSError:
        return False
    cached = _remote_dev_cache.get(dev)
    if cached is not None:
        return cached
    fstype = _fstype_from_mount(probe) or _fstype(probe)
    if fstype in _REMOTE_FS_TYPES:
        remote = True
    elif fstype in _LOCAL_FS_TYPES:
        remote = False
    else:
        try:
            remote = dev != Path.home().stat().st_dev
        except OSError:
            remote = False
    _remote_dev_cache[dev] = remote
    return remote


def _sqlite_sidecars(db_path: Path) -> tuple[Path, Path]:
    return Path(f"{db_path}-wal"), Path(f"{db_path}-shm")


def _copy_sqlite_main_and_wal(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dest)
    else:
        dest.unlink(missing_ok=True)
    src_wal, _src_shm = _sqlite_sidecars(src)
    dest_wal, dest_shm = _sqlite_sidecars(dest)
    if src_wal.exists():
        shutil.copy2(src_wal, dest_wal)
    else:
        dest_wal.unlink(missing_ok=True)
    # The -shm file is reconstructed from WAL; copying it off SMB is the hang.
    dest_shm.unlink(missing_ok=True)


def _local_mirror_path(remote: Path) -> Path:
    key = hashlib.sha1(os.fsencode(str(remote.resolve()))).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "videobrowser-sqlite" / key / remote.name


def _warn_remote_once(path: Path) -> None:
    label = str(path)
    if label in _warned_remote:
        return
    _warned_remote.add(label)
    print(
        f"racebox: {path} is on a network volume; using a local SQLite working copy "
        "(WAL journaling hangs on SMB/NFS)",
        flush=True,
    )


@dataclass
class BestLapSummary:
    best_lap_time: float | None
    best_lap_timestamp: float | None
    best_sector_times: Tuple[float, float, float] | None


@dataclass
class CornerBest:
    pre_corner_max_speed: float | None
    entry_speed: float | None
    min_speed: float | None
    max_lean: float | None


class StatsStore:
    def __init__(self, db_path: Path):
        requested = Path(db_path).expanduser()
        try:
            requested = requested.resolve()
        except OSError:
            pass
        requested.parent.mkdir(parents=True, exist_ok=True)
        self.requested_path = requested
        self._sync_back_to: Path | None = None
        if is_remote_filesystem(requested):
            _warn_remote_once(requested)
            working = _local_mirror_path(requested)
            _copy_sqlite_main_and_wal(requested, working)
            self.db_path = working
            self._sync_back_to = requested
        else:
            self.db_path = requested
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), timeout=_SQLITE_TIMEOUT_S)
        self.conn.execute("PRAGMA busy_timeout=30000;")
        if self._sync_back_to is not None:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            self.conn.execute("PRAGMA journal_mode=DELETE;")
            self.conn.execute("PRAGMA synchronous=FULL;")
            wal, shm = _sqlite_sidecars(self.db_path)
            wal.unlink(missing_ok=True)
            shm.unlink(missing_ok=True)
        else:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._ensure_schema()

    def close(self) -> None:
        conn = getattr(self, "conn", None)
        if conn is None:
            return
        try:
            if self._sync_back_to is None:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.close()
        finally:
            self.conn = None
        if self._sync_back_to is not None:
            _copy_sqlite_main_and_wal(self.db_path, self._sync_back_to)
            wal, shm = _sqlite_sidecars(self._sync_back_to)
            wal.unlink(missing_ok=True)
            shm.unlink(missing_ok=True)

    def __enter__(self) -> "StatsStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_schema(self) -> None:
        with self.conn:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER NOT NULL,
                    session_start_utc REAL NOT NULL,
                    source_file TEXT NOT NULL,
                    created_at_utc REAL NOT NULL,
                    FOREIGN KEY(track_id) REFERENCES tracks(id)
                );

                CREATE TABLE IF NOT EXISTS laps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    lap_number INTEGER NOT NULL,
                    lap_time REAL NOT NULL,
                    timestamp_utc REAL NOT NULL,
                    is_out_lap INTEGER NOT NULL,
                    is_in_lap INTEGER NOT NULL,
                    sector1 REAL NOT NULL,
                    sector2 REAL NOT NULL,
                    sector3 REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );

                CREATE TABLE IF NOT EXISTS corners (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lap_id INTEGER NOT NULL,
                    corner_index INTEGER NOT NULL,
                    pre_corner_max_speed REAL NOT NULL,
                    entry_speed REAL NOT NULL,
                    min_speed REAL NOT NULL,
                    max_lean REAL NOT NULL,
                    FOREIGN KEY(lap_id) REFERENCES laps(id)
                );
                """)

    def _get_track_id(self, track_name: str) -> int:
        cur = self.conn.execute("SELECT id FROM tracks WHERE name = ?", (track_name,))
        row = cur.fetchone()
        if row:
            return row[0]
        cur = self.conn.execute("INSERT INTO tracks(name) VALUES (?)", (track_name,))
        return cur.lastrowid

    def session_exists(self, track_name: str, session_start_utc: datetime) -> bool:
        cur = self.conn.execute(
            """
            SELECT 1 FROM sessions s
            JOIN tracks t ON t.id = s.track_id
            WHERE t.name = ? AND s.session_start_utc = ?
            """,
            (track_name, session_start_utc.timestamp()),
        )
        return cur.fetchone() is not None

    def create_session(
        self, track_name: str, session_start_utc: datetime, source_file: str
    ) -> int:
        with self.conn:
            track_id = self._get_track_id(track_name)
            cur = self.conn.execute(
                "INSERT INTO sessions(track_id, session_start_utc, source_file, created_at_utc) VALUES (?,?,?,?)",
                (
                    track_id,
                    session_start_utc.timestamp(),
                    source_file,
                    datetime.now(timezone.utc).timestamp(),
                ),
            )
            return cur.lastrowid

    def insert_lap(
        self,
        session_id: int,
        lap_number: int,
        lap_time: float,
        timestamp_utc: float,
        is_out_lap: bool,
        is_in_lap: bool,
        sector_times: Tuple[float, float, float],
    ) -> int:
        with self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO laps(session_id, lap_number, lap_time, timestamp_utc, is_out_lap, is_in_lap, sector1, sector2, sector3)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    session_id,
                    lap_number,
                    lap_time,
                    timestamp_utc,
                    int(is_out_lap),
                    int(is_in_lap),
                    sector_times[0],
                    sector_times[1],
                    sector_times[2],
                ),
            )
            return cur.lastrowid

    def insert_corner(
        self,
        lap_id: int,
        corner_index: int,
        pre_corner_max_speed: float,
        entry_speed: float,
        min_speed: float,
        max_lean: float,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO corners(lap_id, corner_index, pre_corner_max_speed, entry_speed, min_speed, max_lean)
                VALUES (?,?,?,?,?,?)
                """,
                (
                    lap_id,
                    corner_index,
                    pre_corner_max_speed,
                    entry_speed,
                    min_speed,
                    max_lean,
                ),
            )

    def fetch_best_lap(self, track_name: str, up_to_utc: float) -> BestLapSummary:
        cur = self.conn.execute(
            """
            SELECT l.lap_time, l.timestamp_utc, l.sector1, l.sector2, l.sector3
            FROM laps l
            JOIN sessions s ON s.id = l.session_id
            JOIN tracks t ON t.id = s.track_id
            WHERE t.name = ? AND l.is_out_lap = 0 AND l.is_in_lap = 0 AND l.timestamp_utc <= ?
            ORDER BY l.lap_time ASC
            LIMIT 1
            """,
            (track_name, up_to_utc),
        )
        row = cur.fetchone()
        if not row:
            return BestLapSummary(None, None, None)
        return BestLapSummary(row[0], row[1], (row[2], row[3], row[4]))

    def fetch_best_day_lap(
        self, track_name: str, day_start_utc: float, day_end_utc: float
    ) -> float | None:
        cur = self.conn.execute(
            """
            SELECT MIN(l.lap_time)
            FROM laps l
            JOIN sessions s ON s.id = l.session_id
            JOIN tracks t ON t.id = s.track_id
            WHERE t.name = ? AND l.is_out_lap = 0 AND l.is_in_lap = 0
              AND l.timestamp_utc BETWEEN ? AND ?
            """,
            (track_name, day_start_utc, day_end_utc),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None

    def fetch_corner_best(
        self, track_name: str, corner_index: int, up_to_utc: float
    ) -> CornerBest:
        cur = self.conn.execute(
            """
            SELECT MAX(c.pre_corner_max_speed), MAX(c.entry_speed), MAX(c.min_speed), MAX(c.max_lean)
            FROM corners c
            JOIN laps l ON l.id = c.lap_id
            JOIN sessions s ON s.id = l.session_id
            JOIN tracks t ON t.id = s.track_id
            WHERE t.name = ? AND l.is_out_lap = 0 AND l.is_in_lap = 0
              AND l.timestamp_utc <= ? AND c.corner_index = ?
            """,
            (track_name, up_to_utc, corner_index),
        )
        row = cur.fetchone()
        return CornerBest(row[0], row[1], row[2], row[3])

    def fetch_corner_best_day(
        self,
        track_name: str,
        corner_index: int,
        day_start_utc: float,
        day_end_utc: float,
    ) -> CornerBest:
        cur = self.conn.execute(
            """
            SELECT MAX(c.pre_corner_max_speed), MAX(c.entry_speed), MAX(c.min_speed), MAX(c.max_lean)
            FROM corners c
            JOIN laps l ON l.id = c.lap_id
            JOIN sessions s ON s.id = l.session_id
            JOIN tracks t ON t.id = s.track_id
            WHERE t.name = ? AND l.is_out_lap = 0 AND l.is_in_lap = 0
              AND l.timestamp_utc BETWEEN ? AND ? AND c.corner_index = ?
            """,
            (track_name, day_start_utc, day_end_utc, corner_index),
        )
        row = cur.fetchone()
        return CornerBest(row[0], row[1], row[2], row[3])

    def fetch_recent_laps(
        self, track_name: str, up_to_utc: float, limit: int = 4
    ) -> List[Tuple[int, float, Tuple[float, float, float], float]]:
        cur = self.conn.execute(
            """
            SELECT l.lap_number, l.lap_time, l.sector1, l.sector2, l.sector3, l.timestamp_utc
            FROM laps l
            JOIN sessions s ON s.id = l.session_id
            JOIN tracks t ON t.id = s.track_id
            WHERE t.name = ? AND l.is_out_lap = 0 AND l.is_in_lap = 0 AND l.timestamp_utc <= ?
            ORDER BY l.timestamp_utc DESC
            LIMIT ?
            """,
            (track_name, up_to_utc, limit),
        )
        return [
            (row[0], row[1], (row[2], row[3], row[4]), row[5])
            for row in cur.fetchall()
        ]
