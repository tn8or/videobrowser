"""The learned score: what your votes say about the detector features.

``videobrowser train`` fits an L2-regularised logistic regression (plain
numpy) on the features snapshot of every moment you voted up or down
(``ratings.py``), and stores it, versioned, in the ratings database. The
model's P(up) is written to every moment as ``model_score``.

Until there are enough votes the hand-tuned score still leads: ranks blend
the two, ``blend_weight`` going from 0 at ``BLEND_FROM`` votes to 1 at
``BLEND_FULL`` (``Scorer.blend``).

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
L2 = 1.0
STEPS = 800
FOLDS = 5
TAG_WEIGHT = {"great": 2.0}
NUMERIC = (
    "riders", "kerb_share", "nearest_rider", "action_density", "speed_kmh", "riding_share",
    "vehicle_share", "motion", "duration", "pieces", "hour",
)
FLAGS = ("kneepuck", "red_lamp", "gopro_gps")
# Plain words for ``--explain``.
NAMES = {
    "riders": "riders in view", "kerb_share": "kerb close by", "nearest_rider": "nearest rider's size",
    "action_density": "action per second", "speed_kmh": "speed", "riding_share": "riding on the bike",
    "vehicle_share": "car in view", "motion": "scene motion", "duration": "length", "pieces": "cameras in it",
    "hour": "time of day", "kneepuck": "knee down", "red_lamp": "red signal lamp", "gopro_gps": "GoPro GPS",
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
        else:
            out[f"missing:{name}"] = 1.0
    for name in FLAGS:
        if features.get(name):
            out[name] = 1.0
    if features.get("theme"):
        out[f"theme={features['theme']}"] = 1.0
    if features.get("pace"):
        out[f"pace={features['pace']}"] = 1.0
    for cam in features.get("cameras", []) or []:
        out[f"camera={cam}"] = 1.0
    for kind in features.get("event_types", []) or []:
        out[f"event={kind}"] = 1.0
    return out


def blend_weight(n_votes: int) -> float:
    """How much the learned score counts against the hand-tuned one."""
    return float(min(1.0, max(0.0, (n_votes - BLEND_FROM) / (BLEND_FULL - BLEND_FROM))))


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


@dataclass
class Model:
    names: list[str]
    coef: np.ndarray
    bias: float
    mean: np.ndarray
    std: np.ndarray

    def predict(self, features: dict) -> float:
        vec = vectorize(features)
        x = np.array([vec.get(n, 0.0) for n in self.names])
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


def training_set(db: RatingsDB) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """(feature snapshots, labels 1/0, sample weights) of every moment whose
    latest vote is its own up or down."""
    rows = db.conn.execute(
        """
        SELECT c.features, v.vote, v.tags FROM clips c
        JOIN (
            SELECT clip_key, vote, tags, inherited_from, ROW_NUMBER() OVER (
                PARTITION BY clip_key ORDER BY voted_at DESC, id DESC
            ) AS n FROM votes
        ) v ON v.clip_key = c.clip_key AND v.n = 1
        WHERE v.vote != 0 AND v.inherited_from IS NULL
        """
    ).fetchall()
    feats, labels, weights = [], [], []
    for row in rows:
        feats.append(json.loads(row["features"] or "{}"))
        labels.append(1.0 if row["vote"] > 0 else 0.0)
        tags = json.loads(row["tags"] or "[]")
        weights.append(max([1.0, *(TAG_WEIGHT.get(t, 1.0) for t in tags)]))
    return feats, np.array(labels), np.array(weights)


def fit(feats: list[dict], y: np.ndarray, w: np.ndarray | None = None) -> Model:
    w = np.ones(len(y)) if w is None else w
    vecs = [vectorize(f) for f in feats]
    names = sorted({k for v in vecs for k in v})
    x = np.array([[v.get(n, 0.0) for n in names] for v in vecs]) if names else np.zeros((len(vecs), 0))
    mean = x.mean(axis=0) if len(x) else np.zeros(len(names))
    std = x.std(axis=0) if len(x) else np.ones(len(names))
    std = np.where(std > 1e-9, std, 1.0)
    coef, bias = _fit((x - mean) / std, y, w)
    return Model(names, coef, bias, mean, std)


def cross_validate(feats: list[dict], y: np.ndarray, w: np.ndarray, folds: int = FOLDS) -> dict:
    """Accuracy and AUC on held-out votes, None when too few."""
    n = len(y)
    if n < folds * 2 or len(set(y.tolist())) < 2:
        return {"cv_accuracy": None, "cv_auc": None}
    order = np.random.default_rng(0).permutation(n)
    preds = np.zeros(n)
    for k in range(folds):
        test = order[k::folds]
        train = np.setdiff1d(order, test)
        model = fit([feats[i] for i in train], y[train], w[train])
        preds[test] = [model.predict(feats[i]) for i in test]
    return {
        "cv_accuracy": round(float(((preds >= 0.5) == (y == 1)).mean()), 3),
        "cv_auc": None if (auc := _auc(y, preds)) is None else round(auc, 3),
    }


def train(db: RatingsDB) -> dict | None:
    """Fit a model on the votes, store it as a new version, make it the one
    in use and score every moment with it. None with too few votes (or all
    one way)."""
    feats, y, w = training_set(db)
    if len(y) < MIN_VOTES or len(set(y.tolist())) < 2:
        return None
    model = fit(feats, y, w)
    metrics = {"n": int(len(y)), "up": int(y.sum()), "down": int(len(y) - y.sum()), **cross_validate(feats, y, w)}
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
        return Scorer(model, version, blend_weight(int(metrics.get("n", 0))))
    except (sqlite3.Error, ValueError, KeyError):
        return None
    finally:
        db.close()
