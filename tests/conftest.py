import pytest


@pytest.fixture(autouse=True)
def _ratings_in_tmp(tmp_path, monkeypatch):
    """Keep every test's ratings database out of the real home folder."""
    monkeypatch.setenv("VIDEOBROWSER_RATINGS", str(tmp_path / "ratings.sqlite"))
