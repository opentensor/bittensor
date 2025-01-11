from unittest.mock import MagicMock

import pytest
from async_substrate_interface import SubstrateInterface
from scalecodec.types import GenericExtrinsic

from bittensor.core.extrinsics import utils
from bittensor.core.subtensor import Subtensor


@pytest.fixture
def mock_subtensor():
    mock_subtensor = MagicMock(autospec=Subtensor)
    mock_substrate = MagicMock(autospec=SubstrateInterface)
    mock_subtensor.substrate = mock_substrate
    yield mock_subtensor


@pytest.fixture
def starting_block():
    yield {"header": {"number": 1, "hash": "0x0100"}}


def test_submit_extrinsic_success(mock_subtensor):
    mock_subtensor.substrate.submit_extrinsic.return_value = True
    mock_extrinsic = MagicMock(autospec=GenericExtrinsic)
    result = utils.submit_extrinsic(mock_subtensor, mock_extrinsic, True, True)
    assert result is True
