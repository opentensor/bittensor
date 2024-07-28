import pytest

from bittensor.utils.registration import LazyLoadedTorch


class MockBittensorLogging:
    def __init__(self):
        self.messages = []

    def error(self, message):
        self.messages.append(message)


@pytest.fixture
def mock_bittensor_logging(monkeypatch):
    mock_logger = MockBittensorLogging()
    monkeypatch.setattr("bittensor.logging", mock_logger)
    return mock_logger


def test_lazy_loaded_torch__torch_installed(monkeypatch, mock_bittensor_logging):
    import torch

    lazy_torch = LazyLoadedTorch()

    assert bool(torch) is True

    assert lazy_torch.nn is torch.nn
    with pytest.raises(AttributeError):
        lazy_torch.no_such_thing


def test_lazy_loaded_torch__no_torch(monkeypatch, mock_bittensor_logging):
    monkeypatch.setattr("bittensor.utils.registration._get_real_torch", lambda: None)

    torch = LazyLoadedTorch()

    assert not torch

    with pytest.raises(ImportError):
        torch.some_attribute

    # Check if the error message is logged correctly
    assert len(mock_bittensor_logging.messages) == 1
    assert "This command requires torch." in mock_bittensor_logging.messages[0]
