from pathlib import Path

from videobrowser.racebox import fetch


def test_migrate_legacy_auth_copies_once(tmp_path: Path):
    legacy = tmp_path / "racebox" / "fetcher" / "auth_state.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"cookies": []}')
    new = tmp_path / "home" / ".videobrowser" / "racebox_auth.json"

    fetch.migrate_legacy_auth(new, legacy)
    assert new.read_text() == '{"cookies": []}'

    # An existing login is never overwritten by the old one.
    new.write_text('{"cookies": [1]}')
    fetch.migrate_legacy_auth(new, legacy)
    assert new.read_text() == '{"cookies": [1]}'


def test_migrate_legacy_auth_without_legacy_file(tmp_path: Path):
    new = tmp_path / "racebox_auth.json"
    fetch.migrate_legacy_auth(new, tmp_path / "missing.json")
    assert not new.exists()


def test_fetch_sessions_skips_without_login(tmp_path: Path, monkeypatch, capsys):
    import pytest

    pytest.importorskip("playwright")
    import videobrowser.racebox_stage as rbs

    monkeypatch.setattr(fetch, "AUTH_STATE_PATH", tmp_path / "racebox_auth.json")
    monkeypatch.setattr(fetch, "LEGACY_AUTH_STATE_PATHS", (tmp_path / "missing.json",))
    rbs.fetch_sessions(tmp_path / "sessions")
    assert "videobrowser racebox-login" in capsys.readouterr().out
    assert not (tmp_path / "sessions").exists()


def test_racebox_login_command(monkeypatch):
    from videobrowser import cli

    calls = []
    monkeypatch.setattr(fetch, "interactive_login", lambda: calls.append(True))
    assert cli.main(["racebox-login"]) == 0
    assert calls == [True]
