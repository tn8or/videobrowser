import pytest


@pytest.fixture(autouse=True)
def _ratings_in_tmp(tmp_path, monkeypatch):
    """Keep every test's ratings database and config out of the real home folder."""
    monkeypatch.setenv("VIDEOBROWSER_RATINGS", str(tmp_path / "ratings.sqlite"))
    monkeypatch.setenv("VIDEOBROWSER_CONFIG", str(tmp_path / "config.toml"))
