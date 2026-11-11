import json
from pathlib import Path

from videobrowser.ratings import ClipRecord, RatingsDB, clip_key, default_path, overlap, register_all


def _record(t_start=10.0, t_end=18.0, theme="overtake", files=(), rid="GX010001", day="2026-08-01"):
    return ClipRecord(
        date_label=day,
        recording_id=rid,
        recording_duration=612.4,
        theme=theme,
        t_start=t_start,
        t_end=t_end,
        utc_start=1785571200.0 + t_start,
        camera="front",
        features={"theme": theme, "riders": 2},
        heuristic_score=1.5,
        files=list(files),
    )


def test_default_path_follows_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOBROWSER_RATINGS", str(tmp_path / "x.sqlite"))
    assert default_path() == tmp_path / "x.sqlite"
    monkeypatch.delenv("VIDEOBROWSER_RATINGS")
    assert default_path() == Path.home() / ".videobrowser" / "ratings.sqlite"


def test_clip_key_is_stable_and_tells_days_and_recordings_apart():
    a = clip_key("2026-08-01", "GX010001", 612.4, 10.2, "overtake")
    assert a == clip_key("2026-08-01", "GX010001", 612.0, 9.8, "overtake")  # rounding
    assert a != clip_key("2026-08-02", "GX010001", 612.4, 10.2, "overtake")  # GoPro names repeat
    assert a != clip_key("2026-08-01", "GX010001", 640.0, 10.2, "overtake")
    assert a != clip_key("2026-08-01", "GX010001", 612.4, 10.2, "duel")


def test_overlap():
    assert overlap((0, 10), (0, 10)) == 1.0
    assert overlap((0, 10), (5, 15)) == 5 / 15
    assert overlap((0, 10), (10, 20)) == 0.0


def test_machine_id_is_made_once(tmp_path):
    path = tmp_path / "r.sqlite"
    first = RatingsDB.open(path)
    first.close()
    second = RatingsDB.open(path)
    assert second.machine_id == first.machine_id and len(first.machine_id) == 32
    second.close()


def test_register_twice_keeps_one_row_and_first_seen(tmp_path):
    db = RatingsDB.open(tmp_path / "r.sqlite")
    key = db.register(_record(files=[("short", tmp_path / "a" / "01.mp4")]))
    first_seen = db.clip(key)["first_seen"]
    again = _record(files=[("short", tmp_path / "a" / "01.mp4")])
    again.features = {"theme": "overtake", "riders": 3}
    assert db.register(again) == key
    row = db.clip(key)
    assert row["first_seen"] == first_seen and json.loads(row["features"])["riders"] == 3
    assert db.conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0] == 1
    assert [r["path"] for r in db.files(key)] == [str(tmp_path / "a" / "01.mp4")]
    db.close()


def test_vote_latest_counts_and_rejects_nonsense(tmp_path):
    db = RatingsDB.open(tmp_path / "r.sqlite")
    key = db.register(_record())
    assert db.latest_vote(key) is None
    db.vote(key, 1, ["great"])
    db.vote(key, -1, ["boring", "boring"])
    latest = db.latest_vote(key)
    assert latest["vote"] == -1 and json.loads(latest["tags"]) == ["boring"]
    assert db.stats() == {"moments": 1, "up": 0, "down": 1}
    db.vote(key, 0)
    assert db.stats() == {"moments": 1, "up": 0, "down": 0}
    for bad in (2, "up"):
        try:
            db.vote(key, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("bad vote accepted")
    try:
        db.vote("nope", 1)
    except KeyError:
        pass
    else:
        raise AssertionError("unknown moment accepted")
    db.close()


def test_rescanned_moment_inherits_the_vote_of_an_overlapping_one(tmp_path):
    db = RatingsDB.open(tmp_path / "r.sqlite")
    old = db.register(_record(10.0, 18.0))
    db.vote(old, 1, ["great"])
    # A rescan moved the window 2 s: a new key, same footage.
    moved = db.register(_record(12.0, 20.0))
    assert moved != old
    inherited = db.latest_vote(moved)
    assert inherited["vote"] == 1 and inherited["inherited_from"] == old
    assert json.loads(inherited["tags"]) == ["great"]
    # Too little overlap, another recording or another day: no vote.
    assert db.latest_vote(db.register(_record(16.0, 26.0))) is None
    assert db.latest_vote(db.register(_record(12.0, 20.0, rid="GX010002"))) is None
    assert db.latest_vote(db.register(_record(12.0, 20.0, day="2026-08-02"))) is None
    db.close()


def test_a_vote_reaches_the_same_footage_in_the_other_outputs(tmp_path):
    db = RatingsDB.open(tmp_path / "r.sqlite")
    moment = db.register(_record(10.0, 18.0, theme="overtake"))
    vertical = db.register(_record(9.0, 19.0, theme="clip"))
    elsewhere = db.register(_record(100.0, 110.0, theme="clip"))
    db.vote(moment, -1, ["paddock"])
    assert db.latest_vote(vertical)["vote"] == -1
    assert db.latest_vote(vertical)["inherited_from"] == moment
    assert db.latest_vote(elsewhere) is None
    # The latest vote wins, wherever it was cast.
    db.vote(vertical, 1)
    assert db.latest_vote(moment)["vote"] == 1
    db.vote(moment, -1, spread=False)
    assert db.latest_vote(vertical)["vote"] == 1
    db.close()


def test_forget_files_only_drops_this_folder_and_prefix(tmp_path):
    db = RatingsDB.open(tmp_path / "r.sqlite")
    day = tmp_path / "vertical" / "2026-08-01"
    a = db.register(_record(10.0, 18.0, files=[("vertical", day / "GX010001_00010_x.mp4")]))
    b = db.register(_record(10.0, 18.0, rid="GX010002", files=[("vertical", day / "GX010002_00010_x.mp4")]))
    c = db.register(_record(40.0, 48.0, theme="duel", files=[("short", day / "GX010001_00040_x.mp4")]))
    db.forget_files("vertical", day, "GX010001_")
    assert db.files(a) == [] and len(db.files(b)) == 1 and len(db.files(c)) == 1
    assert db.key_for_path(day / "GX010002_00010_x.mp4") == b
    db.close()


def test_register_all_never_raises(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    assert register_all([_record()], path=blocker / "r.sqlite") == 0
    assert "ratings: not saved" in capsys.readouterr().out
    assert register_all([_record(), _record(30.0, 40.0)], path=tmp_path / "r.sqlite") == 2


def test_opening_a_file_from_before_the_rating_page_adds_why(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE clips (clip_key TEXT PRIMARY KEY, date_label TEXT NOT NULL, recording_id TEXT NOT NULL,"
        " camera TEXT, theme TEXT NOT NULL, t_start REAL NOT NULL, t_end REAL NOT NULL, utc_start REAL NOT NULL,"
        " features TEXT NOT NULL, heuristic_score REAL, model_score REAL, model_version INTEGER,"
        " first_seen TEXT NOT NULL, last_seen TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()
    db = RatingsDB.open(path)
    record = _record()
    record.why = ["2 rider(s) in view"]
    key = db.register(record)
    assert json.loads(db.clip(key)["why"]) == ["2 rider(s) in view"]
    db.close()


def test_tagged_lists_moments_whose_latest_vote_carries_the_tag(tmp_path):
    from videobrowser.ratings import NOT_TOGETHER, tagged_spans

    db = RatingsDB.open(tmp_path / "r.sqlite")
    a = db.register(_record(10, 18))
    b = db.register(_record(100, 108))
    db.vote(a, 0, [NOT_TOGETHER], spread=False)
    db.vote(b, 0, [NOT_TOGETHER], spread=False)
    db.vote(b, 1, ["great"], spread=False)  # untagged again
    assert db.tagged(NOT_TOGETHER) == [("2026-08-01", "GX010001", 10.0, 18.0)]
    db.close()
    assert tagged_spans(NOT_TOGETHER, tmp_path / "r.sqlite") == [("2026-08-01", "GX010001", 10.0, 18.0)]
    assert tagged_spans(NOT_TOGETHER, tmp_path / "missing.sqlite") == []
