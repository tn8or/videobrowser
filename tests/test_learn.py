import json

import numpy as np

from videobrowser import learn
from videobrowser.ratings import ClipRecord, RatingsDB, register_all


def _rec(n, riders, theme="action", day="2026-08-01"):
    return ClipRecord(
        date_label=day, recording_id=f"GX{n:06d}", recording_duration=600.0, theme=theme,
        t_start=10.0, t_end=18.0, utc_start=1785571200.0 + n, camera="front",
        features={"theme": theme, "riders": riders, "kerb_share": 0.1, "cameras": ["front"],
                  "event_types": ["nearby_rider"], "kneepuck": False},
        heuristic_score=1.0,
    )


def _voted_db(tmp_path, n=40):
    """Up when 3+ riders in view, down otherwise."""
    db = RatingsDB.open(tmp_path / "r.sqlite")
    rng = np.random.default_rng(1)
    for i in range(n):
        riders = int(rng.integers(0, 6))
        key = db.register(_rec(i, riders))
        db.vote(key, 1 if riders >= 3 else -1, ["great"] if riders == 5 else [])
    return db


def test_vectorize():
    vec = learn.vectorize({"theme": "duel", "riders": 2, "kneepuck": True, "pace": "track",
                           "cameras": ["front", "rear"], "event_types": ["rider_pass"], "speed_kmh": None})
    assert vec["riders"] == 2 and vec["kneepuck"] == 1 and vec["theme=duel"] == 1
    assert vec["camera=rear"] == 1 and vec["event=rider_pass"] == 1 and vec["pace=track"] == 1
    assert vec["missing:speed_kmh"] == 1 and "speed_kmh" not in vec and "red_lamp" not in vec


def test_blend_weight():
    assert learn.blend_weight(10) == 0.0 and learn.blend_weight(30) == 0.0
    assert learn.blend_weight(115) == 0.5 and learn.blend_weight(500) == 1.0


def test_train_learns_from_votes_scores_everything_and_explains(tmp_path):
    db = _voted_db(tmp_path)
    result = learn.train(db)
    assert result["version"] == 1 and result["n"] == 40
    assert result["cv_accuracy"] >= 0.8 and result["cv_auc"] >= 0.9
    version, model, metrics = learn.current(db)
    assert version == 1 and model.predict({"riders": 5}) > 0.8 > 0.2 > model.predict({"riders": 0})
    scores = [r[0] for r in db.conn.execute("SELECT model_score FROM clips")]
    assert all(s is not None for s in scores)
    assert "riders in view" in learn.explain(model)[0]
    # A second model, then back to the first.
    db.vote(db.register(_rec(99, 0)), 1)
    assert learn.train(db)["version"] == 2
    learn.use(db, 1)
    assert learn.current(db)[0] == 1
    assert {r[0] for r in db.conn.execute("SELECT model_version FROM clips")} == {1}
    db.close()


def test_train_needs_enough_votes_both_ways_and_skips_spread_votes(tmp_path):
    db = RatingsDB.open(tmp_path / "r.sqlite")
    keys = [db.register(_rec(i, i)) for i in range(8)]
    for k in keys[:5]:
        db.vote(k, 1)
    assert learn.train(db) is None  # all up
    db.vote(keys[5], -1)
    assert learn.train(db)["n"] == 6
    # The same footage in another output got the vote too, but counts once.
    other = db.register(ClipRecord("2026-08-01", "GX000000", 600.0, "clip", 9.0, 19.0, 0.0, features={}))
    assert db.latest_vote(other) is not None  # inherited from keys[0]
    feats, y, w = learn.training_set(db)
    assert len(y) == 6
    db.close()


def test_new_moments_get_the_model_score_and_the_scorer_blends(tmp_path):
    db = _voted_db(tmp_path)
    learn.train(db)
    db.close()
    register_all([_rec(500, 5, day="2026-09-01")], path=tmp_path / "r.sqlite")
    db = RatingsDB.open(tmp_path / "r.sqlite")
    row = db.conn.execute("SELECT model_score, model_version FROM clips WHERE date_label = '2026-09-01'").fetchone()
    assert row["model_score"] > 0.8 and row["model_version"] == 1
    db.close()
    scorer = learn.load_scorer(tmp_path / "r.sqlite")
    assert scorer.version == 1 and scorer.weight == learn.blend_weight(40)
    assert scorer.blend(1.0, 0.0) == scorer.weight
    assert learn.load_scorer(tmp_path / "none.sqlite") is None


def test_shorts_rank_by_the_blend_but_own_crashes_stay_first(tmp_path):
    from videobrowser.shorts import learned_scores
    from videobrowser.themes import Moment, Segment

    model = learn.fit([{"riders": r} for r in range(6)], np.array([0, 0, 0, 1, 1, 1.0]))
    scorer = learn.Scorer(model, 3, 1.0)

    class Day:  # what moment_features reads
        events, action, riding, vehicles, motion, clocks, from_racebox, bridges = [], {}, {}, {}, {}, {}, set(), {}

        def facing(self, rid):
            return "front"

    import videobrowser.shorts as shorts

    calm = Moment("action", 0, 8, 5.0, [Segment("A", 0, 8)])
    busy = Moment("action", 100, 108, 1.0, [Segment("A", 100, 108)])
    crash = Moment("crash", 200, 208, 10.0, [Segment("A", 200, 208)])
    feats = {id(calm): {"riders": 0}, id(busy): {"riders": 5}, id(crash): {"riders": 0}}
    orig = shorts.moment_features
    shorts.moment_features = lambda m, day, min_speed=0: feats[id(m)]
    try:
        known = learned_scores([calm, busy, crash], Day(), scorer, 0)
    finally:
        shorts.moment_features = orig
    assert busy.score > calm.score and crash.score == 10.0
    assert known[id(calm)][1] == 5.0 and known[id(busy)][2] > 0.5
    assert any("learned score" in w for w in busy.why)


def test_train_command(tmp_path, monkeypatch, capsys):
    from videobrowser.cli import main

    monkeypatch.setenv("VIDEOBROWSER_RATINGS", str(tmp_path / "r.sqlite"))
    assert main(["train"]) == 0
    assert "not enough votes" in capsys.readouterr().out
    _voted_db(tmp_path).close()
    assert main(["train", "--explain"]) == 0
    out = capsys.readouterr().out
    assert "model v1 from 40 votes" in out and "riders in view" in out
    assert main(["train", "--list"]) == 0 and "* v1" in capsys.readouterr().out
    assert main(["train", "--use", "7"]) == 1


def test_gate_min_score_top_explore_and_exemptions():
    items = [(f"k{n:02d}", n / 100, "action") for n in range(100)]
    items += [("crash", 0.01, "crash"), ("unscored", None, "action")]
    off = learn.gate(items)
    assert set(off.values()) == {None}
    verdict = learn.gate(items, min_score=0.3, explore=0.0)
    assert verdict["k29"].startswith("left out: model score 0.29 < 0.3") and verdict["k30"] is None
    assert verdict["crash"] is None and verdict["unscored"] is None
    top = learn.gate(items, top=20, explore=0.0)
    kept = [k for k, v in top.items() if v is None and k.startswith("k")]
    assert sorted(kept) == [f"k{n}" for n in range(80, 100)]
    assert "not in the top 20%" in top["k10"]
    # Exploration keeps a stable share of the left-out ones.
    a = learn.gate(items, min_score=0.5, explore=0.2)
    b = learn.gate(items, min_score=0.5, explore=0.2)
    explored = [k for k, v in a.items() if v and v.startswith("explore:")]
    assert a == b and 4 <= len(explored) <= 18
    assert all("kept to learn from" in a[k] for k in explored)


def test_shorts_skip_low_scores_into_left_out(tmp_path):
    from videobrowser.shorts import skip_low_scores
    from videobrowser.themes import Moment, Segment
    from videobrowser.sync import CameraClock

    class Day:
        date_label = "2026-08-01"
        clocks = {"A": CameraClock("A", 0.0, 600.0, [])}

        def facing(self, rid):
            return "front"

    low = Moment("action", 0, 8, 1.0, [Segment("A", 0, 8)])
    high = Moment("action", 100, 108, 1.0, [Segment("A", 100, 108)])
    crash = Moment("crash", 200, 208, 10.0, [Segment("A", 200, 208)])
    known = {id(low): ({}, 1.0, 0.1), id(high): ({}, 1.0, 0.9), id(crash): ({}, 10.0, 0.05)}
    left_out = []
    kept = skip_low_scores([low, high, crash], Day(), known, left_out, 0.5, 0, 0.0)
    assert kept == [high, crash] and left_out == [low]
    assert low.why[-1].startswith("left out: model score 0.10")
    # Without a learned score nothing is skipped.
    plain = {k: (f, h, None) for k, (f, h, _) in known.items()}
    assert skip_low_scores([low, high], Day(), plain, [], 0.5, 0, 0.0) == [low, high]
