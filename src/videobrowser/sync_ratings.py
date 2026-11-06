"""Share the ratings database between machines through a folder (Dropbox).

The SQLite file itself never goes into Dropbox: two machines writing one
live database through a file-sync service corrupts it sooner or later. Each
machine instead appends what it registered and voted to its own change
file, ``<sync dir>/<machine id>.jsonl``, and reads the other machines'
files into its own database:

- a ``clip`` line per moment this machine registered (no file paths: those
  differ per machine),
- a ``vote`` line per vote cast (or spread) here.

Files are only ever appended to, so Dropbox never has two writers of one
file. Each machine remembers how far it has read every other file (and
written its own), so a sync only moves what is new, and a half-written last
line (Dropbox mid-upload) is left for the next sync. Votes keep their
machine and time; the latest per moment counts, wherever it was cast.
Models are not shared: each machine re-trains from the merged votes.

``videobrowser ratings sync [DIR]`` syncs once; with ``sync_dir`` in
``~/.videobrowser/config.toml`` (``VIDEOBROWSER_CONFIG`` overrides the
path), ``rate`` syncs on start and stop and ``train`` before it trains.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from videobrowser.ratings import RatingsDB

CONFIG_ENV = "VIDEOBROWSER_CONFIG"
CLIP_FIELDS = (
    "clip_key", "date_label", "recording_id", "camera", "theme", "t_start", "t_end", "utc_start",
    "features", "heuristic_score", "why", "first_seen", "last_seen",
)


def config_path() -> Path:
    env = os.environ.get(CONFIG_ENV)
    return Path(env).expanduser() if env else Path.home() / ".videobrowser" / "config.toml"


def configured_dir() -> Path | None:
    """``sync_dir`` from the config file, None when not set."""
    path = config_path()
    if not path.is_file():
        return None
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as exc:
        print(f"ratings sync: {path} unreadable ({exc})")
        return None
    value = data.get("sync_dir")
    return Path(value).expanduser() if value else None


@dataclass
class SyncResult:
    sent_clips: int = 0
    sent_votes: int = 0
    got_clips: int = 0
    got_votes: int = 0
    files: int = 0

    def __str__(self) -> str:
        return (
            f"sent {self.sent_clips} moment(s) and {self.sent_votes} vote(s); "
            f"got {self.got_clips} moment(s) and {self.got_votes} vote(s) from {self.files} other machine(s)"
        )


def _meta(db: RatingsDB, key: str, default: str = "") -> str:
    row = db.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def _set_meta(db: RatingsDB, key: str, value: str) -> None:
    db.conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _send(db: RatingsDB, folder: Path) -> tuple[int, int]:
    """Append what is new here to this machine's change file."""
    since = _meta(db, "sync_sent_seen", "")
    last_vote = int(_meta(db, "sync_sent_vote", "0") or 0)
    clips = db.conn.execute(
        f"""
        SELECT {", ".join("c." + f for f in CLIP_FIELDS)} FROM clips c
        WHERE c.last_seen > ? AND EXISTS (
            SELECT 1 FROM clip_files f WHERE f.clip_key = c.clip_key AND f.machine_id = ?
        ) ORDER BY c.last_seen
        """,
        (since, db.machine_id),
    ).fetchall()
    votes = db.conn.execute(
        "SELECT * FROM votes WHERE id > ? AND machine_id = ? ORDER BY id", (last_vote, db.machine_id)
    ).fetchall()
    if not clips and not votes:
        return 0, 0
    lines = [json.dumps({"type": "clip", **{f: c[f] for f in CLIP_FIELDS}}, ensure_ascii=False) for c in clips]
    lines += [
        json.dumps(
            {"type": "vote", "clip_key": v["clip_key"], "vote": v["vote"], "tags": v["tags"],
             "machine_id": v["machine_id"], "voted_at": v["voted_at"], "inherited_from": v["inherited_from"]},
            ensure_ascii=False,
        )
        for v in votes
    ]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{db.machine_id}.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
        f.flush()
        os.fsync(f.fileno())
    if clips:
        _set_meta(db, "sync_sent_seen", max(c["last_seen"] for c in clips))
    if votes:
        _set_meta(db, "sync_sent_vote", str(max(v["id"] for v in votes)))
    db.conn.commit()
    return len(clips), len(votes)


def _receive(db: RatingsDB, path: Path) -> tuple[int, int]:
    """Read another machine's change file from where the last sync stopped."""
    key = f"sync_read:{path.name}"
    offset = int(_meta(db, key, "0") or 0)
    with path.open("rb") as f:
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n") + 1  # a half-written last line waits for the next sync
    clips = votes = 0
    for raw in data[:end].splitlines():
        try:
            item = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue  # a damaged line: skip it, keep the rest
        if item.get("type") == "clip":
            values = [item.get(f) for f in CLIP_FIELDS]
            if any(values[i] is None for i in (0, 1, 2, 4, 5, 6, 7, 8, 11, 12)):
                continue
            # A newer snapshot (the other machine re-ran shorts) replaces an
            # older one; this machine's own first_seen stays.
            cur = db.conn.execute(
                f"""
                INSERT INTO clips({', '.join(CLIP_FIELDS)}) VALUES ({', '.join('?' * len(CLIP_FIELDS))})
                ON CONFLICT(clip_key) DO UPDATE SET
                    camera = excluded.camera, t_start = excluded.t_start, t_end = excluded.t_end,
                    features = excluded.features, heuristic_score = excluded.heuristic_score,
                    why = excluded.why, last_seen = excluded.last_seen
                WHERE excluded.last_seen > clips.last_seen
                """,
                values,
            )
            clips += cur.rowcount
        elif item.get("type") == "vote":
            if item.get("vote") not in (-1, 0, 1) or not item.get("clip_key") or not item.get("voted_at"):
                continue
            seen = db.conn.execute(
                "SELECT 1 FROM votes WHERE clip_key = ? AND machine_id = ? AND voted_at = ? AND vote = ?",
                (item["clip_key"], item.get("machine_id"), item["voted_at"], item["vote"]),
            ).fetchone()
            if seen:
                continue
            db.conn.execute(
                "INSERT INTO votes(clip_key, vote, tags, machine_id, voted_at, inherited_from) VALUES (?, ?, ?, ?, ?, ?)",
                (item["clip_key"], item["vote"], item.get("tags") or "[]", item.get("machine_id") or "?",
                 item["voted_at"], item.get("inherited_from")),
            )
            votes += 1
    _set_meta(db, key, str(offset + end))
    db.conn.commit()
    return clips, votes


def sync(folder: Path, db_path: Path | None = None) -> SyncResult:
    """Send this machine's news to *folder* and read the other machines'."""
    db = RatingsDB.open(db_path)
    try:
        result = SyncResult()
        result.sent_clips, result.sent_votes = _send(db, folder)
        for path in sorted(folder.glob("*.jsonl")) if folder.is_dir() else []:
            if path.stem == db.machine_id:
                continue
            got_clips, got_votes = _receive(db, path)
            result.got_clips += got_clips
            result.got_votes += got_votes
            result.files += 1
        return result
    finally:
        db.close()


def sync_if_configured(db_path: Path | None = None) -> SyncResult | None:
    """Sync with the configured folder, None when none is set. Never
    raises: a folder that is not there (Dropbox not running, a drive not
    mounted) is reported and skipped."""
    folder = configured_dir()
    if folder is None:
        return None
    try:
        result = sync(folder, db_path)
    except OSError as exc:
        print(f"ratings sync: {folder} not reachable ({exc})")
        return None
    print(f"ratings sync: {result}")
    return result
