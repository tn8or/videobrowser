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


def test_windows_local_drive_is_not_remote(tmp_path: Path, monkeypatch):
    stats = _stats()
    monkeypatch.setattr(stats.sys, "platform", "win32")
    monkeypatch.setattr(stats, "_windows_is_remote", lambda path: False)
    assert stats.is_remote_filesystem(tmp_path / "stats.sqlite") is False


def test_windows_unc_path_is_remote(monkeypatch):
    stats = _stats()
    monkeypatch.setattr(stats.os.path, "abspath", lambda p: str(p))
    assert stats._windows_is_remote(Path("//nas/share/stats.sqlite")) is True


def test_remote_stores_get_separate_working_copies(tmp_path: Path, monkeypatch):
    """Parallel renderers each open the stats file; one must not clobber another."""
    stats = _stats()
    remote = tmp_path / "share" / "stats.sqlite"
    remote.parent.mkdir()
    started = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)
    with stats.StatsStore(remote) as store:
        store.create_session("padborg", started, "x.vbo")
    before = remote.stat().st_mtime_ns

    monkeypatch.setattr(stats, "is_remote_filesystem", lambda path: True)
    a = stats.StatsStore(remote)
    b = stats.StatsStore(remote)
    try:
        assert a.db_path != b.db_path
        a.close()
        assert b.session_exists("padborg", started)
    finally:
        b.close()
    # Read-only stores do not copy back over the network file.
    assert remote.stat().st_mtime_ns == before
    assert not a.db_path.exists() and not b.db_path.exists()
