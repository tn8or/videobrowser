from datetime import datetime, timezone
from pathlib import Path

from videobrowser.overlay_bridge import ensure_on_path


def _stats():
    ensure_on_path()
    from overlay import stats as stats_mod

    stats_mod._remote_dev_cache.clear()
    return stats_mod


def test_local_tmp_is_not_remote(tmp_path: Path):
    assert _stats().is_remote_filesystem(tmp_path) is False


def test_smbfs_mount_is_remote(tmp_path: Path, monkeypatch):
    stats = _stats()
    share = str(tmp_path.resolve())
    listing = (
        "/dev/disk3s1s1 on / (apfs, sealed, local)\n"
        f"//host/share on {share} (smbfs, nodev, nosuid, mounted by tommy)\n"
    )

    class _Result:
        returncode = 0
        stdout = listing

    monkeypatch.setattr(stats.subprocess, "run", lambda *a, **k: _Result())
    assert stats.is_remote_filesystem(tmp_path / "stats.sqlite") is True


def test_local_store_uses_wal(tmp_path: Path):
    stats = _stats()
    path = tmp_path / "stats.sqlite"
    store = stats.StatsStore(path)
    try:
        mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
        store.create_session("padborg", datetime.now(timezone.utc), "x.vbo")
    finally:
        store.close()
    assert path.exists()


def test_remote_store_uses_local_delete_journal_and_syncs_back(tmp_path: Path, monkeypatch):
    stats = _stats()
    remote = tmp_path / "share" / "stats.sqlite"
    remote.parent.mkdir()
    monkeypatch.setattr(stats, "is_remote_filesystem", lambda path: True)

    started = datetime(2026, 8, 2, 10, 0, tzinfo=timezone.utc)
    with stats.StatsStore(remote) as store:
        assert store.db_path != store.requested_path
        mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "delete"
        store.create_session("padborg", started, "x.vbo")

    assert remote.exists()
    assert not Path(f"{remote}-wal").exists()
    assert not Path(f"{remote}-shm").exists()

    monkeypatch.setattr(stats, "is_remote_filesystem", lambda path: False)
    with stats.StatsStore(remote) as store:
        assert store.session_exists("padborg", started)


def test_remote_open_replays_existing_wal(tmp_path: Path, monkeypatch):
    stats = _stats()
    remote = tmp_path / "share" / "stats.sqlite"
    remote.parent.mkdir()
    started = datetime(2026, 8, 2, 11, 0, tzinfo=timezone.utc)

    with stats.StatsStore(remote) as store:
        store.create_session("padborg", started, "x.vbo")

    monkeypatch.setattr(stats, "is_remote_filesystem", lambda path: True)
    with stats.StatsStore(remote) as store:
        assert store.session_exists("padborg", started)



def test_local_tmp_is_not_remote(tmp_path: Path):
    assert _stats().is_remote_filesystem(tmp_path) is False


def test_local_store_uses_wal(tmp_path: Path):
    stats = _stats()
    path = tmp_path / "stats.sqlite"
    store = stats.StatsStore(path)
    try:
        mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
        store.create_session("padborg", datetime.now(timezone.utc), "x.vbo")
    finally:
        store.close()
    assert path.exists()


def test_remote_store_uses_local_delete_journal_and_syncs_back(tmp_path: Path, monkeypatch):
    stats = _stats()
    remote = tmp_path / "share" / "stats.sqlite"
    remote.parent.mkdir()
    monkeypatch.setattr(stats, "is_remote_filesystem", lambda path: True)

    started = datetime(2026, 8, 2, 10, 0, tzinfo=timezone.utc)
    with stats.StatsStore(remote) as store:
        assert store.db_path != store.requested_path
        mode = store.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "delete"
        store.create_session("padborg", started, "x.vbo")

    assert remote.exists()
    assert not Path(f"{remote}-wal").exists()
    assert not Path(f"{remote}-shm").exists()

    monkeypatch.setattr(stats, "is_remote_filesystem", lambda path: False)
    with stats.StatsStore(remote) as store:
        assert store.session_exists("padborg", started)


def test_remote_open_replays_existing_wal(tmp_path: Path, monkeypatch):
    stats = _stats()
    remote = tmp_path / "share" / "stats.sqlite"
    remote.parent.mkdir()
    started = datetime(2026, 8, 2, 11, 0, tzinfo=timezone.utc)

    with stats.StatsStore(remote) as store:
        store.create_session("padborg", started, "x.vbo")

    monkeypatch.setattr(stats, "is_remote_filesystem", lambda path: True)
    with stats.StatsStore(remote) as store:
        assert store.session_exists("padborg", started)
