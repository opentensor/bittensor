import time
from unittest.mock import MagicMock, patch
import importlib
import pytest
from scalecodec.types import GenericExtrinsic
from substrateinterface.base import SubstrateInterface, ExtrinsicReceipt
from substrateinterface.exceptions import ExtrinsicNotFound, SubstrateRequestException

from bittensor.core.extrinsics import utils
from bittensor.core.subtensor import Subtensor


@pytest.fixture
def set_extrinsics_timeout_env(monkeypatch):
    monkeypatch.setenv("EXTRINSIC_SUBMISSION_TIMEOUT", "1")


@pytest.fixture
def mock_subtensor():
    mock_subtensor = MagicMock(autospec=Subtensor)
    mock_substrate = MagicMock(autospec=SubstrateInterface)
    mock_subtensor.substrate = mock_substrate
    yield mock_subtensor


@pytest.fixture
def starting_block():
    yield {"header": {"number": 1, "hash": "0x0100"}}


def test_submit_extrinsic_timeout(mock_subtensor):
    timeout = 1

    def wait(extrinsic, wait_for_inclusion, wait_for_finalization):
        time.sleep(timeout + 0.01)
        return True

    mock_subtensor.substrate.submit_extrinsic = wait
    mock_extrinsic = MagicMock(autospec=GenericExtrinsic)
    with patch.object(utils, "EXTRINSIC_SUBMISSION_TIMEOUT", timeout):
        with pytest.raises(SubstrateRequestException):
            utils.submit_extrinsic(mock_subtensor, mock_extrinsic, True, True)


def test_submit_extrinsic_success(mock_subtensor):
    mock_subtensor.substrate.submit_extrinsic.return_value = True
    mock_extrinsic = MagicMock(autospec=GenericExtrinsic)
    result = utils.submit_extrinsic(mock_subtensor, mock_extrinsic, True, True)
    assert result is True


def test_submit_extrinsic_timeout_env(set_extrinsics_timeout_env, mock_subtensor):
    importlib.reload(utils)
    timeout = utils.EXTRINSIC_SUBMISSION_TIMEOUT
    assert timeout < 5  # should be less than 5 seconds as taken from test env var

    def wait(extrinsic, wait_for_inclusion, wait_for_finalization):
        time.sleep(timeout + 1)
        return True

    mock_subtensor.substrate.submit_extrinsic = wait
    mock_extrinsic = MagicMock(autospec=GenericExtrinsic)
    with pytest.raises(SubstrateRequestException):
        utils.submit_extrinsic(mock_subtensor, mock_extrinsic, True, True)


def test_submit_extrinsic_success_env(set_extrinsics_timeout_env, mock_subtensor):
    importlib.reload(utils)
    mock_subtensor.substrate.submit_extrinsic.return_value = True
    mock_extrinsic = MagicMock(autospec=GenericExtrinsic)
    result = utils.submit_extrinsic(mock_subtensor, mock_extrinsic, True, True)
    assert result is True


def test_submit_extrinsic_timeout_env_float(monkeypatch, mock_subtensor):
    monkeypatch.setenv("EXTRINSIC_SUBMISSION_TIMEOUT", "1.45")  # use float

    importlib.reload(utils)
    timeout = utils.EXTRINSIC_SUBMISSION_TIMEOUT

    assert timeout == 1.45  # parsed correctly

    def wait(extrinsic, wait_for_inclusion, wait_for_finalization):
        time.sleep(timeout + 0.3)  # sleep longer by float
        return True

    mock_subtensor.substrate.submit_extrinsic = wait
    mock_extrinsic = MagicMock(autospec=GenericExtrinsic)
    with pytest.raises(SubstrateRequestException):
        utils.submit_extrinsic(mock_subtensor, mock_extrinsic, True, True)


def test_import_timeout_env_parse(monkeypatch):
    # int
    monkeypatch.setenv("EXTRINSIC_SUBMISSION_TIMEOUT", "1")
    importlib.reload(utils)
    assert utils.EXTRINSIC_SUBMISSION_TIMEOUT == 1  # parsed correctly

    # float
    monkeypatch.setenv("EXTRINSIC_SUBMISSION_TIMEOUT", "1.45")  # use float
    importlib.reload(utils)
    assert utils.EXTRINSIC_SUBMISSION_TIMEOUT == 1.45  # parsed correctly

    # invalid
    monkeypatch.setenv("EXTRINSIC_SUBMISSION_TIMEOUT", "not_an_int")
    with pytest.raises(ValueError) as e:
        importlib.reload(utils)
    assert "must be a float" in str(e.value)

    # negative
    monkeypatch.setenv("EXTRINSIC_SUBMISSION_TIMEOUT", "-1")
    with pytest.raises(ValueError) as e:
        importlib.reload(utils)
    assert "cannot be negative" in str(e.value)

    # default (not checking exact value, just that it's a value)
    monkeypatch.delenv("EXTRINSIC_SUBMISSION_TIMEOUT")
    importlib.reload(utils)
    assert isinstance(utils.EXTRINSIC_SUBMISSION_TIMEOUT, float)  # has a default value
    assert utils.EXTRINSIC_SUBMISSION_TIMEOUT > 0  # is positive
