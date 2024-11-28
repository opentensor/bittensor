import time
from unittest.mock import MagicMock, patch

import pytest
from substrateinterface.base import (
    SubstrateInterface,
    GenericExtrinsic,
    SubstrateRequestException,
)

from bittensor.core.extrinsics import utils


def test_submit_extrinsic_timeout():
    timeout = 3

    def wait(extrinsic, wait_for_inclusion, wait_for_finalization):
        time.sleep(timeout + 1)
        return True

    mock_substrate = MagicMock(autospec=SubstrateInterface)
    mock_substrate.submit_extrinsic = wait
    mock_extrinsic = MagicMock(autospec=GenericExtrinsic)
    with patch.object(utils, "EXTRINSIC_SUBMISSION_TIMEOUT", timeout):
        with pytest.raises(SubstrateRequestException):
            utils.submit_extrinsic(mock_substrate, mock_extrinsic, True, True)


def test_submit_extrinsic_success():
    mock_substrate = MagicMock(autospec=SubstrateInterface)
    mock_substrate.submit_extrinsic.return_value = True
    mock_extrinsic = MagicMock(autospec=GenericExtrinsic)
    result = utils.submit_extrinsic(mock_substrate, mock_extrinsic, True, True)
    assert result is True
