import sys
import types

from videobrowser.pipeline import pick_device


def _fake_torch(monkeypatch, cuda: bool, mps: bool) -> None:
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_auto_prefers_cuda(monkeypatch):
    _fake_torch(monkeypatch, cuda=True, mps=False)
    assert pick_device("auto") == "cuda"


def test_auto_uses_mps_on_mac(monkeypatch):
    _fake_torch(monkeypatch, cuda=False, mps=True)
    assert pick_device("auto") == "mps"


def test_auto_falls_back_to_cpu(monkeypatch):
    _fake_torch(monkeypatch, cuda=False, mps=False)
    assert pick_device("auto") == "cpu"


def test_explicit_device_wins(monkeypatch):
    _fake_torch(monkeypatch, cuda=True, mps=False)
    assert pick_device("cpu") == "cpu"
