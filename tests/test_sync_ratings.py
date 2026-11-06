import json
from pathlib import Path

from videobrowser.ratings import ClipRecord, RatingsDB
from videobrowser.sync_ratings import configured_dir, sync, sync_if_configured


def _rec(tmp_path, n, machine):
    clip = tmp_path / machine / f"clip{n}.mp4"
    clip.parent.mkdir(parents=True, exist_ok=True)
    clip.write_bytes(b"x")
    return ClipRecord("2026-08-01", f"GX{n:06d}", 600.0, "overtake", 10.0, 18.0, 1785571200.0 + n,
                      features={"riders": n}, heuristic_score=1.0, files=[("short", clip)], why=["w"])


def test_two_machines_share_moments_and_votes_latest_wins(tmp_path):
    shared = tmp_path / "Dropbox" / "videobrowser"
    mac, pc = tmp_path / "mac.sqlite", tmp_path / "pc.sqlite"
    a = RatingsDB.open(mac)
    key = a.register(_rec(tmp_path, 1, "mac"))
    a.vote(key, 1, ["great"])
    a.close()
    first = sync(shared, mac)
    assert (first.sent_clips, first.sent_votes) == (1, 1)
    # The PC gets the moment (without the Mac's paths) and the vote.
    got = sync(shared, pc)
    assert (got.got_clips, got.got_votes, got.files) == (1, 1, 1)
    b = RatingsDB.open(pc)
    assert json.loads(b.clip(key)["features"]) == {"riders": 1} and b.files(key) == []
    assert b.latest_vote(key)["vote"] == 1
    # The PC changes its mind later; the Mac follows.
    b.vote(key, -1, ["boring"])
    b.close()
    sync(shared, pc)
    assert sync(shared, mac).got_votes == 1
    a = RatingsDB.open(mac)
    latest = a.latest_vote(key)
    assert latest["vote"] == -1 and json.loads(latest["tags"]) == ["boring"]
    assert a.stats() == {"moments": 1, "up": 0, "down": 1}
    a.close()
    # Nothing new: nothing moves, and votes are never doubled.
    again = sync(shared, mac)
    assert (again.sent_clips, again.sent_votes, again.got_clips, again.got_votes) == (0, 0, 0, 0)
    assert sorted(p.name for p in shared.iterdir()) == sorted(
        [f"{RatingsDB.open(mac).machine_id}.jsonl", f"{RatingsDB.open(pc).machine_id}.jsonl"]
    )


def test_half_written_and_damaged_lines(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    other = shared / "othermachine.jsonl"
    clip = {"type": "clip", "clip_key": "k1", "date_label": "2026-08-01", "recording_id": "GX1", "camera": None,
            "theme": "duel", "t_start": 1.0, "t_end": 9.0, "utc_start": 1.0, "features": "{}",
            "heuristic_score": 1.0, "why": "[]", "first_seen": "2026-11-06T00:00:00", "last_seen": "2026-11-06T00:00:00"}
    vote = {"type": "vote", "clip_key": "k1", "vote": 1, "tags": "[]", "machine_id": "othermachine",
            "voted_at": "2026-11-06T01:00:00.000+00:00", "inherited_from": None}
    half = json.dumps({**vote, "vote": -1, "voted_at": "2026-11-06T02:00:00.000+00:00"})
    other.write_text(json.dumps(clip) + "\n" + "{not json\n" + json.dumps(vote) + "\n" + half[:30])
    db_path = tmp_path / "r.sqlite"
    assert sync(shared, db_path).got_votes == 1
    db = RatingsDB.open(db_path)
    assert db.latest_vote("k1")["vote"] == 1
    db.close()
    # Dropbox finishes the upload: the rest of the line arrives.
    other.write_text(other.read_text() + half[30:] + "\n")
    assert sync(shared, db_path).got_votes == 1
    db = RatingsDB.open(db_path)
    assert db.latest_vote("k1")["vote"] == -1
    db.close()


def test_config_and_unreachable_folder(tmp_path, capsys):
    import os

    assert configured_dir() is None and sync_if_configured() is None
    Path(os.environ["VIDEOBROWSER_CONFIG"]).write_text(f'sync_dir = "{(tmp_path / "box").as_posix()}"\n')
    assert configured_dir() == tmp_path / "box"
    assert sync_if_configured() is not None and "ratings sync: sent" in capsys.readouterr().out
    blocker = tmp_path / "file"
    blocker.write_text("x")
    Path(os.environ["VIDEOBROWSER_CONFIG"]).write_text(f'sync_dir = "{(blocker / "box").as_posix()}"\n')
    db = RatingsDB.open()
    db.vote(db.register(_rec(tmp_path, 3, "here")), 1)
    db.close()
    assert sync_if_configured() is None and "not reachable" in capsys.readouterr().out


def test_sync_command(tmp_path, capsys):
    from videobrowser.cli import main

    assert main(["ratings", "sync"]) == 1
    assert main(["ratings", "sync", str(tmp_path / "box")]) == 0
    assert "sent 0 moment(s)" in capsys.readouterr().out
