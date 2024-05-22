import pytest


@pytest.fixture
def force_legacy_torch_compat_api(monkeypatch):
    monkeypatch.setenv("USE_TORCH", "1")
