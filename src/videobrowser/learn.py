"""The learned score: what your votes say about the detector features.

``videobrowser train`` fits an L2-regularised logistic regression (plain
numpy) on the features snapshot of every moment you voted up or down
(``ratings.py``), and stores it, versioned, in the ratings database. The
model's P(up) is written to every moment as ``model_score``.

Until there are enough votes the hand-tuned score still leads: ranks blend
the two, ``blend_weight`` going from 0 at ``BLEND_FROM`` votes to 1 at
``BLEND_FULL`` (``Scorer.blend``). The model is also only trusted as far as
it predicts a day it has not seen: the held-out AUC is worked out with whole
days held out and only each day's moments compared with each other (clips
from one day look alike and get the same vote, so random folds flatter it,
and a day's own mix of ups and downs would otherwise look like signal), and the weight is scaled from 0 at AUC
``AUC_FLOOR`` to full at ``AUC_FULL``. With fewer than ``MIN_DAYS`` days
voted on there is no honest estimate and the model counts 0%.

Votes a moment only got from another one (the same footage in another
output, or a rescan's predecessor) are left out of training, so one piece of
footage counts once. A "great" tag counts the vote twice.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from videobrowser.ratings import RatingsDB

BLEND_FROM = 30
BLEND_FULL = 200
MIN_VOTES = 6
L2 = 10.0
STEPS = 800
FOLDS = 5
MIN_DAYS = 3
AUC_FLOOR = 0.5
AUC_FULL = 0.8
# A reason given with a down vote counts for more than a bare thumb.
TAG_WEIGHT = {"great": 2.0, "paddock": 2.0, "boring": 1.5, "bad picture": 1.5, "crash seen": 2.0}
# What the model sees. Chosen by a rule fixed beforehand, not by trying
# features against the held-out score: a number must be there for at least
# half the voted moments (``HELD_BACK`` are below that for now, mostly the
# GoPro IMU ones that only some days have), and nothing that mostly tells the
# day or the camera setup apart (hour, cameras, GoPro GPS, "no X stored"
# flags, which stood in for "no GPS" and so for the day). On the 148 votes
# so far that rule gave an AUC on unseen days of ~0.74, against ~0.55-0.63
# with everything; see README. Put the held back ones in when enough votes
# carry them.
NUMERIC = (
    "riders", "nearest_rider", "action_density", "vehicle_share", "motion", "duration",
    "speed_kmh", "speed_max", "speed_min", "speed_change", "turn_rate_max", "cornering_share",
    "rider_share", "group_share", "closing",
)
HELD_BACK = ("kerb_share", "pieces", "hour", "riding_share", "brake_g", "accel_g", "lean_max")
_NUMERIC = frozenset(NUMERIC)
FLAGS: tuple[str, ...] = ()  # was ("kneepuck", "red_lamp", "gopro_gps")
USE_MISSING_FLAGS = False
USE_CATEGORIES = False  # theme, pace, camera, event type
# Plain words for ``--explain``.
NAMES = {
    "riders": "riders in view", "kerb_share": "kerb close by", "nearest_rider": "nearest rider's size",
    "action_density": "action per second", "speed_kmh": "speed", "riding_share": "riding on the bike",
    "vehicle_share": "car in view", "motion": "scene motion", "duration": "length", "pieces": "cameras in it",
    "hour": "time of day", "speed_max": "top speed", "speed_min": "lowest speed",
    "speed_change": "speeding up (vs slowing)", "brake_g": "hard braking", "accel_g": "hard acceleration",
    "turn_rate_max": "sharpest turn", "cornering_share": "time in corners", "lean_max": "deepest lean",
    "rider_share": "time with a rider in view", "group_share": "time with a group in view",
    "closing": "closing in on a rider", "kneepuck": "knee down", "red_lamp": "red signal lamp", "gopro_gps": "GoPro GPS",
}


def vectorize(features: dict) -> dict[str, float]:
    """The model's inputs from a features snapshot: numbers as they are (a
    missing one gets a ``missing:`` flag instead), yes/no as 0/1, and one
    flag per theme, pace, camera and event type."""
    out: dict[str, float] = {}
    for name in NUMERIC:
        value = features.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            out[name] = float(value)
        elif USE_MISSING_FLAGS:
            out[f"missing:{name}"] = 1.0
    for name in FLAGS:
        if features.get(name):
            out[name] = 1.0
    if not USE_CATEGORIES:
        return out
    if features.get("theme"):
        out[f"theme={features['theme']}"] = 1.0
    if features.get("pace"):
        out[f"pace={features['pace']}"] = 1.0
    for cam in features.get("cameras", []) or []:
        out[f"camera={cam}"] = 1.0
    for kind in features.get("event_types", []) or []:
        out[f"event={kind}"] = 1.0
    return out


def blend_weight(n_votes: int, metrics: dict | None = None) -> float:
    """How much the learned score counts against the hand-tuned one: more
    with more votes, and only as far as it predicts days it has not seen.
    *metrics*: the model's stored metrics; without a day-held-out AUC in
    them (an older model, or fewer than ``MIN_DAYS`` days) it counts 0."""
    ramp = min(1.0, max(0.0, (n_votes - BLEND_FROM) / (BLEND_FULL - BLEND_FROM)))
    if metrics is None:
        return float(ramp)
    auc = metrics.get("cv_auc") if metrics.get("cv_by") == "day" else None
    if auc is None:
        return 0.0
    return float(ramp * min(1.0, max(0.0, (auc - AUC_FLOOR) / (AUC_FULL - AUC_FLOOR))))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = L2) -> tuple[np.ndarray, float]:
    """Logistic regression by gradient descent; *x* standardized."""
    n, d = x.shape
    coef = np.zeros(d)
    bias = 0.0
    total = w.sum()
    lr = 0.5
    for _ in range(STEPS):
        p = _sigmoid(x @ coef + bias)
        err = (p - y) * w
        coef -= lr * (x.T @ err / total + l2 * coef / total)
        bias -= lr * err.sum() / total
    return coef, bias


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return None
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def _pair_wins(y: np.ndarray, p: np.ndarray) -> tuple[float, int]:
    """(wins, pairs): of the up/down pairs in *y*, how many *p* orders right
    (ties half)."""
    pos, neg = p[y == 1], p[y == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins), len(pos) * len(neg)


def _within_day_auc(y: np.ndarray, p: np.ndarray, days: np.ndarray) -> float | None:
    """AUC over the up/down pairs *within* each day, pooled. A day's own mix
    of ups and downs, or a model that only knows the average day, can't
    move it: a constant score gets exactly 0.5. That is also how the score
    is used, ranking a day's moments against each other."""
    wins = pairs = 0
    for d in set(days.tolist()):
        i = days == d
        w_, n_ = _pair_wins(y[i], p[i])
        wins, pairs = wins + w_, pairs + n_
    return wins / pairs if pairs else None


@dataclass
class Model:
    names: list[str]
    coef: np.ndarray
    bias: float
    mean: np.ndarray
    std: np.ndarray

    def predict(self, features: dict) -> float:
        vec = vectorize(features)
        # A number the moment lacks (no GPS, say) counts as an average one, not
        # as 0, which would make "no GPS" look like a very slow bike.
        x = np.array([vec.get(n, self.mean[i] if n in _NUMERIC else 0.0) for i, n in enumerate(self.names)])
        return float(_sigmoid(np.array([((x - self.mean) / self.std) @ self.coef + self.bias]))[0])

    def to_json(self) -> dict:
        return {
            "names": self.names, "coef": self.coef.tolist(), "bias": self.bias,
            "mean": self.mean.tolist(), "std": self.std.tolist(),
        }

    @classmethod
    def from_json(cls, data: dict) -> Model:
        return cls(
            list(data["names"]), np.array(data["coef"], dtype=float), float(data["bias"]),
            np.array(data["mean"], dtype=float), np.array(data["std"], dtype=float),
        )


def training_set(db: RatingsDB) -> tuple[list[dict], np.ndarray, np.ndarray, list[str]]:
    """(feature snapshots, labels 1/0, sample weights, day of each) of every
    moment whose latest vote is its own up or down."""
    rows = db.conn.execute(
        """
        SELECT c.features, c.date_label, v.vote, v.tags FROM clips c
        JOIN (
            SELECT clip_key, vote, tags, inherited_from, ROW_NUMBER() OVER (
                PARTITION BY clip_key ORDER BY voted_at DESC, id DESC
            ) AS n FROM votes
        ) v ON v.clip_key = c.clip_key AND v.n = 1
        WHERE v.vote != 0 AND v.inherited_from IS NULL
        """
    ).fetchall()
    feats, labels, weights, days = [], [], [], []
    for row in rows:
        feats.append(json.loads(row["features"] or "{}"))
        days.append(row["date_label"])
        labels.append(1.0 if row["vote"] > 0 else 0.0)
        tags = json.loads(row["tags"] or "[]")
        weights.append(max([1.0, *(TAG_WEIGHT.get(t, 1.0) for t in tags)]))
    return feats, np.array(labels), np.array(weights), days


def fit(feats: list[dict], y: np.ndarray, w: np.ndarray | None = None) -> Model:
    w = np.ones(len(y)) if w is None else w
    vecs = [vectorize(f) for f in feats]
    names = sorted({k for v in vecs for k in v})
    # Numbers a moment lacks are NaN here: left out of the mean and spread,
    # then filled with the mean (0 once standardised).
    x = np.array(
        [[v.get(n, np.nan if n in _NUMERIC else 0.0) for n in names] for v in vecs], dtype=float
    ) if names else np.zeros((len(vecs), 0))
    if len(x):
        seen = ~np.isnan(x)
        count = seen.sum(axis=0)
        total = np.where(seen, x, 0.0).sum(axis=0)
        mean = np.divide(total, count, out=np.zeros(len(names)), where=count > 0)
        x = np.where(seen, x, mean)
        std = np.array([x[seen[:, j], j].std() if count[j] else 1.0 for j in range(len(names))])
    else:
        mean, std = np.zeros(len(names)), np.ones(len(names))
    std = np.where(std > 1e-9, std, 1.0)
    coef, bias = _fit((x - mean) / std, y, w)
    return Model(names, coef, bias, mean, std)


def _day_folds(days: list[str], folds: int) -> list[np.ndarray]:
    """Index arrays, one per fold, each day wholly in one fold (biggest days
    first into the emptiest fold)."""
    by_day: dict[str, list[int]] = {}
    for i, d in enumerate(days):
        by_day.setdefault(d, []).append(i)
    out: list[list[int]] = [[] for _ in range(min(folds, len(by_day)))]
    for idx in sorted(by_day.values(), key=len, reverse=True):
        min(out, key=len).extend(idx)
    return [np.array(sorted(f)) for f in out]


def cross_validate(
    feats: list[dict], y: np.ndarray, w: np.ndarray, days: list[str] | None = None, folds: int = FOLDS
) -> dict:
    """Accuracy and AUC on held-out votes, None when too few. With *days*
    (and at least ``MIN_DAYS`` of them) whole days are held out, so the model
    is tested on days it has not seen (``cv_by`` "day"); otherwise random
    folds (``cv_by`` "random"), which flatter it."""
    n = len(y)
    by_day = days is not None and len(set(days)) >= MIN_DAYS
    meta = {"cv_by": "day" if by_day else "random"}
    if n < folds * 2 or len(set(y.tolist())) < 2:
        return {"cv_accuracy": None, "cv_auc": None, **meta}
    if by_day:
        tests = _day_folds(days, folds)
    else:
        order = np.random.default_rng(0).permutation(n)
        tests = [order[k::folds] for k in range(folds)]
    preds = np.full(n, np.nan)
    for test in tests:
        train = np.setdiff1d(np.arange(n), test)
        if len(set(y[train].tolist())) < 2:
            continue
        model = fit([feats[i] for i in train], y[train], w[train])
        preds[test] = [model.predict(feats[i]) for i in test]
    ok = ~np.isnan(preds)
    if not ok.any():
        return {"cv_accuracy": None, "cv_auc": None, **meta}
    auc = _within_day_auc(y[ok], preds[ok], np.array(days)[ok]) if by_day else _auc(y[ok], preds[ok])
    return {
        "cv_accuracy": round(float(((preds[ok] >= 0.5) == (y[ok] == 1)).mean()), 3),
        "cv_auc": None if auc is None else round(auc, 3),
        **meta,
    }


def train(db: RatingsDB) -> dict | None:
    """Fit a model on the votes, store it as a new version, make it the one
    in use and score every moment with it. None with too few votes (or all
    one way)."""
    feats, y, w, days = training_set(db)
    if len(y) < MIN_VOTES or len(set(y.tolist())) < 2:
        return None
    model = fit(feats, y, w)
    metrics = {"n": int(len(y)), "up": int(y.sum()), "down": int(len(y) - y.sum()), **cross_validate(feats, y, w, days)}
    cur = db.conn.execute(
        "INSERT INTO models(trained_at, n_votes, weights, metrics) VALUES (?, ?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), int(len(y)),
         json.dumps(model.to_json()), json.dumps(metrics)),
    )
    version = int(cur.lastrowid)
    use(db, version)
    return {"version": version, **metrics}


def use(db: RatingsDB, version: int) -> None:
    """Make model *version* the one in use and score every moment with it."""
    row = db.conn.execute("SELECT weights FROM models WHERE version = ?", (version,)).fetchone()
    if row is None:
        raise KeyError(version)
    db.conn.execute(
        "INSERT INTO meta(key, value) VALUES ('model_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (str(version),),
    )
    model = Model.from_json(json.loads(row["weights"]))
    updates = [
        (model.predict(json.loads(r["features"] or "{}")), version, r["clip_key"])
        for r in db.conn.execute("SELECT clip_key, features FROM clips")
    ]
    db.conn.executemany("UPDATE clips SET model_score = ?, model_version = ? WHERE clip_key = ?", updates)
    db.conn.commit()


def current(db: RatingsDB) -> tuple[int, Model, dict] | None:
    """(version, model, metrics) in use, None before the first training."""
    meta = db.conn.execute("SELECT value FROM meta WHERE key = 'model_version'").fetchone()
    if meta is None:
        return None
    row = db.conn.execute(
        "SELECT version, weights, metrics FROM models WHERE version = ?", (int(meta["value"]),)
    ).fetchone()
    if row is None:
        return None
    return int(row["version"]), Model.from_json(json.loads(row["weights"])), json.loads(row["metrics"])


def explain(model: Model, top: int = 12) -> list[str]:
    """The strongest weights in plain words: + pulls a moment up, - down,
    per standard deviation of the input."""
    lines = []
    for i in np.argsort(-np.abs(model.coef))[:top]:
        name = model.names[i]
        if abs(model.coef[i]) < 1e-3:
            continue
        lines.append(f"{model.coef[i]:+.2f}  {_plain(name)}")
    return lines


def _plain(name: str) -> str:
    if name.startswith("missing:"):
        return f"no {NAMES.get(name[8:], name[8:])} stored"
    if "=" in name:
        kind, value = name.split("=", 1)
        return {"theme": "theme", "pace": "pace", "camera": "camera", "event": "event"}.get(kind, kind) + f" {value}"
    return NAMES.get(name, name)


@dataclass
class Scorer:
    """The model in use and how much it counts, for ranking moments."""

    model: Model
    version: int
    weight: float

    def blend(self, model_score: float, heuristic_rank: float) -> float:
        """*heuristic_rank*: 0..1, the moment's place among the day's by the
        hand-tuned score."""
        return self.weight * model_score + (1.0 - self.weight) * heuristic_rank


def load_scorer(path=None) -> Scorer | None:
    """The scorer from the ratings database, None without a model (or a
    readable database)."""
    import sqlite3

    try:
        db = RatingsDB.open(path)
    except (sqlite3.Error, OSError):
        return None
    try:
        found = current(db)
        if found is None:
            return None
        version, model, metrics = found
        return Scorer(model, version, blend_weight(int(metrics.get("n", 0)), metrics))
    except (sqlite3.Error, ValueError, KeyError):
        return None
    finally:
        db.close()


# --- skipping low scorers (--min-score / --top) ----------------------------

# Themes about our own bike are never skipped.
NEVER_SKIP = frozenset({"crash", "own_off_track"})
DEFAULT_EXPLORE = 0.1


def _explore_draw(key: str) -> float:
    """A stable 0..1 number per moment, so the same moments are explored on
    every run."""
    import hashlib

    return int(hashlib.sha1(f"explore|{key}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def gate(
    items: list[tuple[str, float | None, str]],
    min_score: float = 0.0,
    top: float = 0.0,
    explore: float = DEFAULT_EXPLORE,
) -> dict[str, str | None]:
    """Which of *items* ((key, model score or None, theme), one day) to skip.

    Returns key -> None to keep, "explore: ..." to keep although below the
    bar (so the model keeps seeing what it rates low), or "left out: ..."
    to skip. *min_score*: skip a model score below it (0..1, 0 = off).
    *top*: keep only the best *top* percent by model score (0 = off).
    Moments without a model score, and own crashes and off-track moments,
    are always kept.
    """
    out: dict[str, str | None] = {key: None for key, _, _ in items}
    scored = [(key, p) for key, p, theme in items if p is not None and theme not in NEVER_SKIP]
    if not scored or (min_score <= 0 and top <= 0):
        return out
    best_first = sorted(scored, key=lambda kp: -kp[1])
    keep_n = math.ceil(len(best_first) * min(top, 100.0) / 100.0) if top > 0 else len(best_first)
    for rank, (key, p) in enumerate(best_first):
        reason = None
        if min_score > 0 and p < min_score:
            reason = f"model score {p:.2f} < {min_score:g}"
        elif rank >= keep_n:
            reason = f"model score {p:.2f} not in the top {top:g}%"
        if reason is None:
            continue
        out[key] = f"explore: {reason}, kept to learn from" if _explore_draw(key) < explore else f"left out: {reason}"
    return out
