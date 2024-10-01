# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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
    monkeypatch.setattr("bittensor.utils.registration.logging", mock_logger)
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
