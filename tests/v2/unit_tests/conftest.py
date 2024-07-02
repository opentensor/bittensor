import pytest
from aioresponses import aioresponses


@pytest.fixture
def force_legacy_torch_compat_api(monkeypatch):
    monkeypatch.setenv("USE_TORCH", "1")


@pytest.fixture
def mock_aioresponse():
    with aioresponses() as m:
        yield m
