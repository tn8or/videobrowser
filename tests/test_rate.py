import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from videobrowser.cli import main
from videobrowser.rate import list_moments, make_server
from videobrowser.ratings import ClipRecord, RatingsDB


def _record(tmp_path, t_start, theme, files, rid="GX010001", why=()):
    return ClipRecord(
        date_label="2026-08-01", recording_id=rid, recording_duration=600.0, theme=theme,
        t_start=t_start, t_end=t_start + 8, utc_start=1785571200.0 + t_start, camera="front+rear",
        features={"riders": 2}, heuristic_score=t_start / 10, files=files, why=list(why),
    )


@pytest.fixture
def setup(tmp_path):
    db_path = Path(os.environ["VIDEOBROWSER_RATINGS"])
    out = tmp_path / "out"
    short = out / "shorts" / "2026-08-01" / "action-01"
    short.mkdir(parents=True)
    vdir = out / "vertical" / "2026-08-01"
    vdir.mkdir(parents=True)
    front, rear = short / "01a_front.mp4", short / "01b_rear.mp4"
    front.write_bytes(bytes(range(256)) * 4)
    rear.write_bytes(b"rear")
    vertical = vdir / "GX010001_00010_x.mp4"
    vertical.write_bytes(b"vertical")
    db = RatingsDB.open(db_path)
    moment = db.register(_record(tmp_path, 10.0, "overtake",
                                 [("short", front), ("short", rear), ("vertical", vertical)],
                                 why=["2 rider(s) in view", "GPS 92 km/h"]))
    clip = db.register(_record(tmp_path, 9.0, "clip", [("vertical", vertical)]))
    gone = db.register(_record(tmp_path, 200.0, "duel", [("short", short / "missing.mp4")]))
    db.close()
    return db_path, out, {"moment": moment, "clip": clip, "gone": gone, "front": front}


def test_list_moments_shows_existing_files_best_kind_first(setup):
    db_path, out, keys = setup
    db = RatingsDB.open(db_path)
    moments = list_moments(db)
    assert [m["key"] for m in moments] == [keys["clip"], keys["moment"]]  # time order; missing file left out
    m = moments[1]
    assert [f["kind"] for f in m["files"]] == ["short", "short"]  # the short's pieces beat the vertical
    assert m["why"] == ["2 rider(s) in view", "GPS 92 km/h"]
    assert [x["key"] for x in list_moments(db, order="score")] == [keys["moment"], keys["clip"]]
    assert [x["key"] for x in list_moments(db, theme="overtake")] == [keys["moment"]]
    assert list_moments(db, day="2026-08-02") == []
    assert list_moments(db, under=out / "vertical") and all(
        f["kind"] == "vertical" for x in list_moments(db, under=out / "vertical") for f in x["files"]
    )
    db.vote(keys["moment"], 1)  # also reaches the overlapping vertical clip
    assert list_moments(db, unrated=True) == []
    db.close()


@pytest.fixture
def server(setup):
    db_path, out, keys = setup
    srv = make_server(port=0, db_path=db_path, tags=["wheelie"])
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", keys
    srv.shutdown()
    srv.server_close()


def _get(url, headers=None):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {})) as r:
        return r.status, dict(r.headers), r.read()


def _post(url, data, content_type="application/json"):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), method="POST",
                                 headers={"Content-Type": content_type})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_server_page_moments_files_and_votes(server):
    base, keys = server
    status, headers, body = _get(base + "/")
    assert status == 200 and b"Rate moments" in body and b"http" not in body.split(b"<script>")[1]  # offline
    facets = json.loads(_get(base + "/api/facets")[2])
    assert facets["days"] == ["2026-08-01"] and "wheelie" in facets["tags"] and "great" in facets["tags"]
    moments = json.loads(_get(base + "/api/moments?unrated=1&order=score")[2])
    assert moments[0]["key"] == keys["moment"]
    # Files by number, with byte ranges for Safari and seeking.
    status, headers, body = _get(f"{base}/file/{keys['moment']}/0", {"Range": "bytes=10-19"})
    assert status == 206 and body == bytes(range(10, 20)) and headers["Content-Range"] == "bytes 10-19/1024"
    assert _get(f"{base}/file/{keys['moment']}/0")[2] == keys["front"].read_bytes()
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(f"{base}/file/{keys['moment']}/9")
    assert err.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(f"{base}/file/{keys['gone']}/0")
    assert err.value.code == 404
    # A vote with tags reaches the same footage in the other outputs.
    result = _post(base + "/api/vote", {"key": keys["moment"], "vote": 1, "tags": ["great", "wheelie"]})
    assert result["keys"][0] == keys["moment"] and keys["clip"] in result["keys"]
    assert result["stats"]["up"] == 2
    assert json.loads(_get(base + "/api/moments?unrated=1")[2]) == []
    db = RatingsDB.open(Path(os.environ["VIDEOBROWSER_RATINGS"]))
    assert json.loads(db.latest_vote(keys["moment"])["tags"]) == ["great", "wheelie"]
    db.close()


def test_server_refuses_forms_bad_votes_and_foreign_hosts(server):
    base, keys = server
    for data, ctype, code in (
        ({"key": keys["moment"], "vote": 1}, "application/x-www-form-urlencoded", 415),
        ({"key": keys["moment"], "vote": 5}, "application/json", 400),
        ({"key": "nope", "vote": 1}, "application/json", 404),
        ({"vote": 1}, "application/json", 400),
    ):
        with pytest.raises(urllib.error.HTTPError) as err:
            _post(base + "/api/vote", data, ctype)
        assert err.value.code == code
    with pytest.raises(urllib.error.HTTPError) as err:
        _get(base + "/api/moments", {"Host": "evil.example:8765"})
    assert err.value.code == 403


def test_vote_and_stats_commands(setup, capsys):
    db_path, out, keys = setup
    assert main(["vote", str(keys["front"]), "down", "--tag", "boring"]) == 0
    assert "down: 01a_front.mp4 (and 1 clip(s) of the same footage)" in capsys.readouterr().out
    db = RatingsDB.open(db_path)
    latest = db.latest_vote(keys["moment"])
    assert latest["vote"] == -1 and json.loads(latest["tags"]) == ["boring"]
    db.close()
    assert main(["ratings", "stats"]) == 0
    assert "3 moments, 0 up, 2 down" in capsys.readouterr().out
    assert main(["vote", str(out / "nothing.mp4"), "up"]) == 1


def _m(key, score, model=None):
    return {"key": key, "score": score, "model_score": model, "day": "2026-08-01", "utc_start": 0.0}


def test_draw_round_takes_high_low_and_middle():
    from videobrowser.rate import draw_round

    moments = [_m(f"k{n:02d}", float(n)) for n in range(30)]
    drawn = draw_round(moments, each=5)
    assert [m["key"] for m in drawn] == (
        ["k29", "k28", "k27", "k26", "k25"]
        + ["k04", "k03", "k02", "k01", "k00"]
        + ["k16", "k15", "k14", "k13", "k12"]
    )
    assert [m["bucket"] for m in drawn] == ["high"] * 5 + ["low"] * 5 + ["medium"] * 5
    # The learned score wins over the hand-tuned one once there is one.
    flipped = [_m("a", 9.0, model=0.1), _m("b", 1.0, model=0.9)]
    assert [m["key"] for m in draw_round(flipped, each=1)] == ["b", "a"]
    # Fewer than a round: each moment once.
    few = draw_round([_m(f"k{n}", float(n)) for n in range(7)], each=5)
    assert sorted(m["key"] for m in few) == [f"k{n}" for n in range(7)]


def test_server_rounds_and_reweigh(server):
    base, keys = server
    drawn = json.loads(_get(base + "/api/moments")[2])  # rounds by default, unrated only
    assert [m["bucket"] for m in drawn] == ["high", "low"]
    assert [m["key"] for m in drawn] == [keys["moment"], keys["clip"]]
    result = _post(base + "/api/reweigh", {})
    assert result["ok"] and result["done"]
    _post(base + "/api/vote", {"key": keys["moment"], "vote": -1, "tags": []})
    assert json.loads(_get(base + "/api/moments?order=rounds")[2]) == []
