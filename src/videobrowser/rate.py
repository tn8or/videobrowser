"""``videobrowser rate``: a local page for voting on the cut moments.

A small web server on this machine (``127.0.0.1`` by default) serves one
page plus the moments' video files straight from disk, and opens the
browser. It needs no network and loads nothing from the internet. Votes and
tags go to the ratings database (``ratings.py``).

On the page: ↑/K up, ↓/J down, 0 clears, →/space next, ← back, 1-9 toggle
the quick tags, which can also be clicked with the mouse. The tag "not
together" says a moment's cameras do not belong together (wrong sync), and
the next ``shorts`` run leaves the other camera out. By default it
goes in rounds of 15 unrated moments drawn at random across the score, one
from each fifteenth of it, in random order; then the score is re-weighed
from the votes (``learn.py``) and the next round drawn. Stopping the server re-weighs once more.

Only files the ratings database lists for this machine are served, and only
by their place in a moment's file list, so the page cannot be used to read
other files. POSTs must be JSON, which a page on another site cannot send
here without the browser asking first.
"""

from __future__ import annotations

import json
import mimetypes
import random
import re
import sqlite3
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from videobrowser.ratings import NOT_TOGETHER, RatingsDB

DEFAULT_PORT = 8765
# ``NOT_TOGETHER``: the moment's cameras do not show the same seconds (wrong
# sync); the next ``shorts`` run leaves the other camera out. "wrong theme":
# the detector named it wrong (e.g. catching up while the others pull away),
# which the up/down vote on the clip itself says nothing about.
DEFAULT_TAGS = ("great", "boring", "paddock", "wrong camera", "bad picture", NOT_TOGETHER, "wrong theme")
# Which files show a moment, best first: a short's pieces show every camera.
KIND_ORDER = ("short", "vertical", "horizontal")
ORDERS = ("rounds", "time", "score", "uncertain")
# A round: this many unrated moments, drawn at random across the score
# (one from each equal band of it) and shown in random order. After each
# round the score is re-weighed (``reweigh``) and the next round drawn with it.
ROUND_SIZE = 15
# Two moments on one recording show the same footage when they overlap by
# at least this share of the shorter one. Lower than ``themes.dedupe``'s
# half: that keeps two 6 s action windows 3 s apart, and a kneepuck at the
# file's start beside the next one, which are the same pictures to rate.
SAME_FOOTAGE_OVERLAP = 0.3
CHUNK = 1 << 20


def list_moments(
    db: RatingsDB,
    day: str | None = None,
    theme: str | None = None,
    unrated: bool = False,
    order: str = "time",
    under: Path | None = None,
) -> list[dict]:
    """Moments with at least one file on this machine that still exists,
    each with its latest vote, tags and the files to show."""
    rows = db.conn.execute(
        """
        SELECT c.*, v.vote AS vote, v.tags AS tags FROM clips c
        LEFT JOIN (
            SELECT clip_key, vote, tags, ROW_NUMBER() OVER (
                PARTITION BY clip_key ORDER BY voted_at DESC, id DESC
            ) AS n FROM votes
        ) v ON v.clip_key = c.clip_key AND v.n = 1
        """
    ).fetchall()
    files: dict[str, list[tuple[str, str, float | None]]] = {}
    for f in db.conn.execute(
        "SELECT clip_key, kind, path, start FROM clip_files WHERE machine_id = ? ORDER BY path",
        (db.machine_id,),
    ):
        files.setdefault(f["clip_key"], []).append((f["kind"], f["path"], f["start"]))
    under = under.resolve() if under is not None else None
    out = []
    for row in rows:
        if day and row["date_label"] != day:
            continue
        vote = row["vote"] or 0
        shown = _shown_files(files.get(row["clip_key"], []), under)
        if not shown:
            continue
        out.append(
            {
                "_same": {_identity(f[1]) for _, f in shown},
                "key": row["clip_key"],
                "day": row["date_label"],
                "recording": row["recording_id"],
                "theme": row["theme"],
                "camera": row["camera"],
                "t_start": row["t_start"],
                "t_end": row["t_end"],
                "utc_start": row["utc_start"],
                "score": row["heuristic_score"],
                "model_score": row["model_score"],
                "why": json.loads(row["why"] or "[]") if "why" in row.keys() else [],
                "features": json.loads(row["features"] or "{}"),
                "vote": vote,
                "tags": json.loads(row["tags"] or "[]"),
                "explore": any(str(w).startswith("explore:") for w in json.loads(row["why"] or "[]")),
                "files": [
                    {"n": i, "kind": kind, "name": Path(path).name, "start": start}
                    for i, (kind, path, start) in shown
                ],
            }
        )
    # Grouped before the theme and unrated filters: a group counts as rated
    # once any of its moments is, whichever way that vote came in.
    out = group_same_footage(out)
    if theme:
        out = [m for m in out if theme in m["themes"]]
    if unrated:
        out = [m for m in out if m["vote"] == 0]
    if order == "score":
        out.sort(key=lambda m: (-(m["score"] or 0.0), m["day"], m["utc_start"]))
    elif order == "uncertain":
        # Moments the model is least sure of teach it most; without a model
        # yet, plain time order.
        out.sort(
            key=lambda m: (
                abs(m["model_score"] - 0.5) if m["model_score"] is not None else 1.0,
                m["day"],
                m["utc_start"],
            )
        )
    else:
        out.sort(key=lambda m: (m["day"], m["utc_start"], m["theme"]))
    return out


def _identity(path: str) -> tuple[int, int]:
    """A file's size and mtime: the same for every copy ``copy2`` made of
    one scan clip."""
    st = Path(path).stat()
    return (st.st_size, st.st_mtime_ns)


def _same_footage(a: dict, b: dict) -> bool:
    """Whether *a* and *b* show the same footage: they share a file (the
    same scan clip copied into two shorts), or they lie on the same
    recording and overlap by ``SAME_FOOTAGE_OVERLAP`` of the shorter one."""
    if a["day"] != b["day"]:
        return False
    if a["_same"] & b["_same"]:
        return True
    if a["recording"] != b["recording"]:
        return False
    inter = min(a["t_end"], b["t_end"]) - max(a["t_start"], b["t_start"])
    return inter > 0 and inter >= SAME_FOOTAGE_OVERLAP * min(a["t_end"] - a["t_start"], b["t_end"] - b["t_start"])


def group_same_footage(moments: list[dict]) -> list[dict]:
    """One entry per piece of footage. A moment can be cut under several
    themes (a kneepuck inside a duel), as a short and as vertical and
    landscape clips, and a short copies the whole scan clip for each moment
    in it, so several moments a few seconds apart show the same file. The
    page shows each such group once: the member with the most cameras, then
    the best score, with the others' keys in ``also`` (a vote goes to all of
    them) and every theme in ``themes``."""
    parent = list(range(len(moments)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_day: dict[str, list[int]] = {}
    for i, m in enumerate(moments):
        by_day.setdefault(m["day"], []).append(i)
    for idx in by_day.values():
        for n, i in enumerate(idx):
            for j in idx[n + 1 :]:
                if find(i) != find(j) and _same_footage(moments[i], moments[j]):
                    parent[find(j)] = find(i)
    groups: dict[int, list[dict]] = {}
    for i, m in enumerate(moments):
        groups.setdefault(find(i), []).append(m)
    out = []
    for members in groups.values():
        best = max(members, key=lambda m: (len(m["files"]), _score(m), -m["utc_start"]))
        rest = sorted((m for m in members if m is not best), key=lambda m: m["utc_start"])
        shown = {k: v for k, v in best.items() if k != "_same"}
        shown["also"] = [m["key"] for m in rest]
        shown["themes"] = list(dict.fromkeys([best["theme"], *(m["theme"] for m in rest)]))
        shown["also_at"] = [f"{m['theme']} @ {m['t_start']:.0f} s" for m in rest]
        if shown["vote"] == 0:  # a vote on any of them is a vote on the footage
            voted = next((m for m in rest if m["vote"] != 0), None)
            if voted is not None:
                shown["vote"], shown["tags"] = voted["vote"], voted["tags"]
        out.append(shown)
    return out


def same_footage_keys(db: RatingsDB, key: str, under: Path | None = None) -> list[str]:
    """The other moments ``list_moments`` shows as the same footage as
    *key* (with the same *under* as the page), worked out here and not
    taken from the page."""
    for m in list_moments(db, under=under):
        if key == m["key"] or key in m["also"]:
            return [k for k in (m["key"], *m["also"]) if k != key]
    return []


def _score(m: dict) -> float:
    """What the pipeline thinks of a moment: the learned score once there is
    one, else the hand-tuned one."""
    if m["model_score"] is not None:
        return float(m["model_score"])
    return float(m["score"] or 0.0)


def draw_round(
    moments: list[dict], size: int = ROUND_SIZE, rng: random.Random | None = None
) -> list[dict]:
    """The next round from *moments* (the unrated ones): the score's order
    cut into *size* equal bands, one moment at random from each, shuffled.
    So a round spans the whole score without picking the same kind of
    moments each time. Each is marked with its ``bucket``: high, medium or
    low, the third of the score it came from."""
    rng = rng or random.Random()
    ranked = sorted(moments, key=lambda m: (-_score(m), m["day"], m["utc_start"], m["key"]))
    n = len(ranked)
    if not n:
        return []
    picks = range(n) if n <= size else (
        rng.randrange(b * n // size, (b + 1) * n // size) for b in range(size)
    )
    def bucket(i: int) -> str:
        at = i / (n - 1) if n > 1 else 0.0
        return "high" if at < 1 / 3 else "low" if at > 2 / 3 else "medium"

    out = [{**ranked[i], "bucket": bucket(i)} for i in picks]
    rng.shuffle(out)
    return out


def reweigh(db: RatingsDB) -> str:
    """Re-weigh the score from the votes so far (``learn.train``), between
    rounds; returns what was done."""
    from videobrowser.learn import MIN_VOTES, blend_weight, train

    result = train(db)
    if result is None:
        return f"not enough votes to learn from yet (needs {MIN_VOTES}, some up and some down)"
    auc = f", held-out AUC {result['cv_auc']:.2f}" if result.get("cv_auc") is not None else ""
    return (
        f"model v{result['version']} from {result['n']} votes ({result['up']} up, {result['down']} down{auc}); "
        f"counts {blend_weight(result['n']):.0%} against the hand-tuned score"
    )


def _shown_files(files: list[tuple], under: Path | None) -> list[tuple[int, tuple]]:
    """(index into *files*, (kind, path)) of the files to show: those of the
    best kind present that still exist (and lie under *under*)."""
    present = [
        (i, f)
        for i, f in enumerate(files)
        if Path(f[1]).is_file() and (under is None or under in Path(f[1]).resolve().parents)
    ]
    for kind in KIND_ORDER:
        picked = [(i, f) for i, f in present if f[0] == kind]
        if picked:
            return _distinct(picked)
    return _distinct(present)


def _distinct(files: list[tuple[int, tuple]]) -> list[tuple[int, tuple]]:
    """*files* without copies of the same clip: a moment that lands in two
    shorts (an action short and a theme short) has the same scan clip copied
    into both, with the same size and, from ``copy2``, the same mtime."""
    seen: set[tuple[int, int]] = set()
    out = []
    for i, f in files:
        same = _identity(f[1])
        if same in seen:
            continue
        seen.add(same)
        out.append((i, f))
    return out


def moment_file(db: RatingsDB, key: str, n: int) -> Path | None:
    """The *n*-th of this machine's files for *key* (as ``list_moments``
    numbers them), None when there is no such file."""
    rows = db.conn.execute(
        "SELECT path FROM clip_files WHERE clip_key = ? AND machine_id = ? ORDER BY path",
        (key, db.machine_id),
    ).fetchall()
    if not 0 <= n < len(rows):
        return None
    path = Path(rows[n]["path"])
    return path if path.is_file() else None


def facets(db: RatingsDB, tags: list[str]) -> dict:
    used: set[str] = set()
    for (raw,) in db.conn.execute("SELECT DISTINCT tags FROM votes"):
        used.update(json.loads(raw or "[]"))
    return {
        "days": [r[0] for r in db.conn.execute("SELECT DISTINCT date_label FROM clips ORDER BY 1")],
        "themes": [r[0] for r in db.conn.execute("SELECT DISTINCT theme FROM clips ORDER BY 1")],
        "tags": list(dict.fromkeys([*tags, *sorted(used)])),
        "stats": db.stats(),
    }


class _Handler(BaseHTTPRequestHandler):
    server_version = "videobrowser-rate"
    db_path: Path | None = None
    tags: list[str] = list(DEFAULT_TAGS)
    under: Path | None = None
    local_only = True

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def _db(self) -> RatingsDB:
        return RatingsDB.open(self.db_path)

    def _send_json(self, data, status=HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _host_ok(self) -> bool:
        """Bound to this machine only: refuse other Host names, so a web
        page cannot reach the server through a name it controls."""
        if not self.local_only:
            return True
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in ("127.0.0.1", "localhost", "::1")

    def do_GET(self):  # noqa: N802
        if not self._host_ok():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif url.path == "/api/facets":
                db = self._db()
                try:
                    self._send_json(facets(db, self.tags))
                finally:
                    db.close()
            elif url.path == "/api/moments":
                db = self._db()
                try:
                    order = q.get("order", "rounds")
                    if order not in ORDERS:
                        order = "rounds"
                    found = list_moments(
                        db,
                        day=q.get("day") or None,
                        theme=q.get("theme") or None,
                        unrated=order == "rounds" or q.get("unrated") == "1",
                        order="time" if order == "rounds" else order,
                        under=self.under,
                    )
                    self._send_json(draw_round(found) if order == "rounds" else found)
                finally:
                    db.close()
            elif m := re.fullmatch(r"/file/([0-9a-f]{1,64})/(\d+)", url.path):
                db = self._db()
                try:
                    path = moment_file(db, m.group(1), int(m.group(2)))
                finally:
                    db.close()
                if path is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                else:
                    self._send_file(path)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except sqlite3.Error as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):  # noqa: N802
        if not self._host_ok():
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        route = urlparse(self.path).path
        if route not in ("/api/vote", "/api/reweigh"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            self.send_error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
            return
        if route == "/api/reweigh":
            db = self._db()
            try:
                self._send_json({"ok": True, "done": reweigh(db), "stats": db.stats()})
            except sqlite3.Error as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            finally:
                db.close()
            return
        try:
            length = min(int(self.headers.get("Content-Length") or 0), 65536)
            data = json.loads(self.rfile.read(length) or b"{}")
            key, vote = str(data["key"]), int(data["vote"])
            tags = [str(t)[:40] for t in data.get("tags", [])][:20]
        except (ValueError, KeyError, TypeError):
            self._send_json({"error": "need key, vote and tags"}, HTTPStatus.BAD_REQUEST)
            return
        db = self._db()
        try:
            keys = db.vote(key, vote, tags, also=same_footage_keys(db, key, self.under))
            self._send_json({"ok": True, "keys": keys, "stats": db.stats()})
        except KeyError:
            self._send_json({"error": "no such moment"}, HTTPStatus.NOT_FOUND)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except sqlite3.Error as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
        finally:
            db.close()

    def _send_file(self, path: Path) -> None:
        """The file, with byte ranges: Safari will not play a video
        without them, and seeking needs them everywhere."""
        size = path.stat().st_size
        start, end = 0, size - 1
        ranged = False
        header = self.headers.get("Range")
        if header and (m := re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())):
            if m.group(1):
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
            elif m.group(2):  # the last N bytes
                start = max(0, size - int(m.group(2)))
            ranged = True
            if start > end or start >= size:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(HTTPStatus.PARTIAL_CONTENT if ranged else HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if ranged:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(CHUNK, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except ConnectionError:
                # The browser dropped the stream (next clip, a seek): nothing to report.
                # Covers WinError 10053, ConnectionResetError and BrokenPipeError.
                self.close_connection = True


def make_server(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    db_path: Path | None = None,
    tags: list[str] | None = None,
    under: Path | None = None,
) -> ThreadingHTTPServer:
    """The rating server, not yet serving. Port 0 picks a free one; a busy
    *port* falls back to a free one."""
    handler = type(
        "RateHandler",
        (_Handler,),
        {
            "db_path": db_path,
            "tags": list(dict.fromkeys([*DEFAULT_TAGS, *(tags or [])])),
            "under": under,
            "local_only": host in ("127.0.0.1", "localhost", "::1"),
        },
    )
    RatingsDB.open(db_path).close()  # create the file before the first request
    try:
        return ThreadingHTTPServer((host, port), handler)
    except OSError:
        if port == 0:
            raise
        return ThreadingHTTPServer((host, 0), handler)


def serve(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    db_path: Path | None = None,
    tags: list[str] | None = None,
    under: Path | None = None,
    open_browser: bool = True,
) -> None:
    from videobrowser.sync_ratings import sync_if_configured

    sync_if_configured(db_path)
    server = make_server(host, port, db_path, tags, under)
    shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    url = f"http://{shown}:{server.server_address[1]}/"
    print(f"rate: {url}  (Ctrl+C stops)")
    if open_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        sync_if_configured(db_path)
        db = RatingsDB.open(db_path)
        try:
            print(f"rate: {reweigh(db)}")
        finally:
            db.close()


PAGE = r"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Rate moments</title>
<style>
:root{--bg:#111;--panel:#1c1c1c;--fg:#eee;--dim:#999;--line:#333;--up:#2e9d57;--down:#c8443a;--accent:#d9a42b}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
header{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line)}
header h1{font-size:16px;margin:0 12px 0 0}
select,button,label{font:inherit;color:var(--fg)}
select,button{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:5px 9px}
button{cursor:pointer}
#stats{margin-left:auto;color:var(--dim)}
main{display:grid;grid-template-columns:1fr 320px;gap:16px;padding:16px}
@media (max-width:900px){main{grid-template-columns:1fr}}
#videos{display:flex;gap:10px;justify-content:center;align-items:flex-start;min-height:60vh}
video{max-height:78vh;max-width:100%;background:#000;border-radius:8px}
#videos.many video{max-width:calc(50% - 5px)}
aside{background:var(--panel);border-radius:8px;padding:14px}
.big{display:flex;gap:8px;margin:10px 0}
.big button{flex:1;font-size:18px;padding:10px}
#up.on{background:var(--up);border-color:var(--up)}
#down.on{background:var(--down);border-color:var(--down)}
#tags{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}
#tags button.on{background:var(--accent);border-color:var(--accent);color:#111}
.dim{color:var(--dim)}
ul{padding-left:18px;margin:6px 0}
kbd{background:#000;border:1px solid var(--line);border-radius:4px;padding:0 4px;font-size:12px}
#empty,#roundend{padding:40px;text-align:center;color:var(--dim)}
#roundend button{font-size:18px;padding:10px 16px}
.bucket{border-radius:4px;padding:1px 6px;font-size:12px;color:#111;background:var(--accent)}
</style>
<header>
  <h1>Rate moments</h1>
  <select id="day"><option value="">All days</option></select>
  <select id="theme"><option value="">All themes</option></select>
  <select id="order">
    <option value="rounds">Rounds: 15 at random across the score</option>
    <option value="time">Time order</option>
    <option value="score">Best score first</option>
    <option value="uncertain">Most uncertain first</option>
  </select>
  <label><input type="checkbox" id="unrated" checked> Unrated only</label>
  <span id="stats"></span>
</header>
<main>
  <div><div id="videos"></div><div id="empty" hidden>Nothing to rate here.</div>
    <div id="roundend" hidden><p>Round done.</p><button id="again">Re-weigh and draw the next round →</button>
      <p class="dim" id="reweighed"></p></div></div>
  <aside>
    <div><b id="title"></b> <span class="dim" id="pos"></span></div>
    <div class="dim" id="sub"></div>
    <div class="big">
      <button id="down" title="Down (↓ or J)">👎 Down</button>
      <button id="up" title="Up (↑ or K)">👍 Up</button>
    </div>
    <div class="big">
      <button id="prev" title="Back (←)">← Back</button>
      <button id="next" title="Next (→ or space)">Next →</button>
    </div>
    <div id="tags"></div>
    <div class="dim">Why it was picked</div>
    <ul id="why"></ul>
    <p class="dim"><kbd>↑</kbd>/<kbd>K</kbd> up · <kbd>↓</kbd>/<kbd>J</kbd> down · <kbd>0</kbd> clear ·
      <kbd>→</kbd>/<kbd>space</kbd> next · <kbd>←</kbd> back · <kbd>1</kbd>-<kbd>9</kbd> tags</p>
    <p class="dim" id="session"></p>
  </aside>
</main>
<script>
const $ = id => document.getElementById(id);
let list = [], i = 0, tags = [], sessionVotes = 0, round = 0;
const rounds = () => $('order').value === 'rounds';

async function getJSON(url) { const r = await fetch(url); return r.json(); }

async function init() {
  const f = await getJSON('/api/facets');
  for (const d of f.days) $('day').add(new Option(d, d));
  for (const t of f.themes) $('theme').add(new Option(t, t));
  tags = f.tags;
  showStats(f.stats);
  for (const id of ['day', 'theme', 'order', 'unrated']) $(id).onchange = load;
  $('up').onclick = () => vote(1);
  $('down').onclick = () => vote(-1);
  $('next').onclick = () => go(1);
  $('prev').onclick = () => go(-1);
  $('again').onclick = nextRound;
  await load();
}

async function load() {
  const p = new URLSearchParams({day: $('day').value, theme: $('theme').value,
    order: $('order').value, unrated: $('unrated').checked ? '1' : ''});
  list = await getJSON('/api/moments?' + p);
  i = 0;
  $('unrated').disabled = rounds();
  $('roundend').hidden = true;
  if (rounds()) round++;
  show();
}

async function nextRound() {
  $('again').disabled = true;
  const r = await fetch('/api/reweigh', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}'});
  const data = await r.json();
  if (data.stats) showStats(data.stats);
  $('reweighed').textContent = data.done || '';
  $('again').disabled = false;
  await load();
}

function endOfRound() {
  $('videos').innerHTML = '';
  $('roundend').hidden = false;
}

function showStats(s) {
  $('stats').textContent = `${s.moments} moments · ${s.up} up · ${s.down} down`;
}

// A moment's cameras filmed the same seconds from different sides (an
// overtake: the rear camera sees the rider come, the front one sees them go).
// With each file's start on the day clock they play in step on one clock that
// loops over all of them; a camera shows its first or last frame while the
// others run. Seeking one video moves the shared clock.
let syncRaf = 0;
function syncPlay(videos, starts) {
  cancelAnimationFrame(syncRaf);
  const base = Math.min(...starts);
  const lag = starts.map(s => s - base);
  let t0 = performance.now(), ours = 0;
  const span = () => Math.max(...videos.map((v, k) => lag[k] + (v.duration || 0)));
  videos.forEach((v, k) => {
    v.loop = false;
    v.addEventListener('seeking', () => {
      if (ours > 0) { ours--; return; }
      t0 = performance.now() - (v.currentTime + lag[k]) * 1000;  // the viewer scrubbed
    });
  });
  const set = (v, t) => { ours++; v.currentTime = t; };
  const tick = () => {
    const total = span();
    if (total > 0) {
      const clock = ((performance.now() - t0) / 1000) % total;
      videos.forEach((v, k) => {
        const local = clock - lag[k], dur = v.duration || 0;
        if (local >= 0 && local < dur) {
          if (Math.abs(v.currentTime - local) > 0.3) set(v, local);
          // A clip reaches its end up to the drift allowance (0.3 s) before
          // the shared clock does. play() on an ended video seeks to 0, and
          // that seek is not one of ours, so it is read as the viewer
          // scrubbing: the clock jumps back and every camera restarts. Hold
          // the last frame instead; cameras still inside their span keep going.
          else if (!v.ended && v.paused) v.play().catch(() => { v.muted = true; v.play().catch(() => {}); });
        } else {
          if (!v.paused) v.pause();
          const edge = local < 0 ? 0 : Math.max(0, dur - 0.05);
          if (Math.abs(v.currentTime - edge) > 0.1) set(v, edge);
        }
      });
    }
    syncRaf = requestAnimationFrame(tick);
  };
  syncRaf = requestAnimationFrame(tick);
}

function show() {
  cancelAnimationFrame(syncRaf);
  const box = $('videos');
  box.innerHTML = '';
  $('empty').hidden = list.length > 0;
  if (!list.length) { $('title').textContent = ''; $('pos').textContent = ''; $('sub').textContent = '';
    $('why').innerHTML = ''; $('tags').innerHTML = ''; return; }
  const m = list[i];
  box.classList.toggle('many', m.files.length > 1);
  for (const f of m.files) {
    const v = document.createElement('video');
    v.src = `/file/${m.key}/${f.n}`; v.autoplay = true; v.loop = true; v.muted = m.files.length > 1 && f !== m.files[0];
    v.controls = true; v.playsInline = true; v.title = f.name;
    box.appendChild(v);
    // Browsers may refuse to autoplay with sound before the first key press.
    v.play().catch(() => { v.muted = true; v.play().catch(() => {}); });
  }
  const starts = m.files.map(f => f.start);
  if (m.files.length > 1 && starts.every(s => s != null) && new Set(starts).size > 1)
    syncPlay([...box.querySelectorAll('video')], starts);
  $('title').innerHTML = '';
  if (m.bucket) { const b = document.createElement('span'); b.className = 'bucket';
    b.textContent = `${m.bucket}`; $('title').append(b, ' '); }
  if (m.explore) { const b = document.createElement('span'); b.className = 'bucket';
    b.title = 'Kept although the learned score rates it low, so it can learn from your vote';
    b.textContent = 'explore'; $('title').append(b, ' '); }
  $('title').append(`${(m.themes || [m.theme]).join(' + ')} · ${m.camera || ''}`);
  $('pos').textContent = (rounds() ? `round ${round}: ` : '') + `${i + 1} of ${list.length}`;
  const score = m.model_score != null ? `model ${(m.model_score * 100).toFixed(0)}%` :
    (m.score != null ? `score ${m.score.toFixed(2)}` : '');
  $('sub').textContent = `${m.day} · ${m.recording} @ ${m.t_start.toFixed(0)} s · ${score}` +
    (m.also_at && m.also_at.length ? ` · same footage: ${m.also_at.join(', ')}` : '');
  $('why').innerHTML = '';
  for (const w of m.why) { const li = document.createElement('li'); li.textContent = w; $('why').appendChild(li); }
  if (!m.why.length) $('why').innerHTML = '<li class="dim">not recorded</li>';
  $('up').classList.toggle('on', m.vote > 0);
  $('down').classList.toggle('on', m.vote < 0);
  $('tags').innerHTML = '';
  tags.forEach((t, n) => {
    const b = document.createElement('button');
    b.textContent = (n < 9 ? `${n + 1} ` : '') + t;
    b.classList.toggle('on', m.tags.includes(t));
    if (t === 'not together') b.title = "The cameras don't show the same seconds: the next shorts run leaves the other camera out";
    b.onclick = () => toggleTag(t);
    $('tags').appendChild(b);
  });
}

async function save(m) {
  const r = await fetch('/api/vote', {method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({key: m.key, vote: m.vote, tags: m.tags})});
  const data = await r.json();
  if (data.stats) showStats(data.stats);
  // The same footage in the other outputs got the vote too: with "unrated
  // only", those need no rating any more.
  const also = new Set((data.keys || []).slice(1));
  if (also.size) {
    for (const o of list) if (also.has(o.key)) { o.vote = m.vote; o.tags = [...m.tags]; }
    if ($('unrated').checked || rounds()) {
      const cur = list[i];
      list = list.filter(o => o === cur || !also.has(o.key));
      i = list.indexOf(cur);
    }
  }
}

async function vote(v) {
  if (!list.length) return;
  const m = list[i];
  m.vote = m.vote === v ? 0 : v;
  sessionVotes++;
  $('session').textContent = `${sessionVotes} vote(s) this session`;
  await save(m);
  show();
  if (m.vote !== 0) setTimeout(() => go(1), 250);
}

async function toggleTag(t) {
  if (!list.length) return;
  const m = list[i];
  m.tags = m.tags.includes(t) ? m.tags.filter(x => x !== t) : [...m.tags, t];
  await save(m);
  show();
}

function go(d) {
  if (!list.length) return;
  if (rounds() && d > 0 && i === list.length - 1) { endOfRound(); return; }
  $('roundend').hidden = true;
  i = Math.max(0, Math.min(list.length - 1, i + d));
  show();
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT' || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key;
  if (!$('roundend').hidden && (k === 'Enter' || k === 'ArrowRight' || k === ' ')) { e.preventDefault(); nextRound(); return; }
  if (k === 'ArrowUp' || k === 'k') vote(1);
  else if (k === 'ArrowDown' || k === 'j') vote(-1);
  else if (k === '0' && list.length) { list[i].vote = 0; save(list[i]).then(show); }
  else if (k === 'ArrowRight' || k === ' ') go(1);
  else if (k === 'ArrowLeft') go(-1);
  else if (/^[1-9]$/.test(k) && tags[+k - 1]) toggleTag(tags[+k - 1]);
  else return;
  e.preventDefault();
});

init();
</script>
</html>
"""
