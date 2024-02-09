import pytest
from unittest.mock import MagicMock, patch
from bittensor.subtensor import subtensor as Subtensor
from bittensor.wallet import wallet as Wallet
from bittensor.extrinsics.serving import serve_extrinsic


@pytest.fixture
def mock_subtensor():
    mock_subtensor = MagicMock(spec=Subtensor)
    mock_subtensor.network = "test_network"
    return mock_subtensor


@pytest.fixture
def mock_wallet():
    wallet = MagicMock(spec=Wallet)
    wallet.hotkey.ss58_address = "hotkey_address"
    wallet.coldkeypub.ss58_address = "coldkey_address"
    return wallet


@pytest.mark.parametrize(
    "ip,port,protocol,netuid,placeholder1,placeholder2,wait_for_inclusion,wait_for_finalization,prompt,expected,test_id,",
    [
        (
            "192.168.1.1",
            9221,
            1,
            0,
            0,
            0,
            False,
            True,
            False,
            True,
            "happy-path-no-wait",
        ),
        (
            "192.168.1.2",
            9222,
            2,
            1,
            1,
            1,
            True,
            False,
            False,
            True,
            "happy-path-wait-for-inclusion",
        ),
        (
            "192.168.1.3",
            9223,
            3,
            2,
            2,
            2,
            False,
            True,
            True,
            True,
            "happy-path-wait-for-finalization-and-prompt",
        ),
    ],
    ids=[
        "happy-path-no-wait",
        "happy-path-wait-for-inclusion",
        "happy-path-wait-for-finalization-and-prompt",
    ],
)
def test_serve_extrinsic_happy_path(
    mock_subtensor,
    mock_wallet,
    ip,
    port,
    protocol,
    netuid,
    placeholder1,
    placeholder2,
    wait_for_inclusion,
    wait_for_finalization,
    prompt,
    expected,
    test_id,
):
    # Arrange
    mock_subtensor._do_serve_axon.return_value = (True, "")
    with patch("bittensor.extrinsics.serving.Confirm.ask", return_value=True):
        # Act
        result = serve_extrinsic(
            mock_subtensor,
            mock_wallet,
            ip,
            port,
            protocol,
            netuid,
            placeholder1,
            placeholder2,
            wait_for_inclusion,
            wait_for_finalization,
            prompt,
        )

        # Assert
        assert result == expected, f"Test ID: {test_id}"


# Various edge cases
@pytest.mark.parametrize(
    "ip,port,protocol,netuid,placeholder1,placeholder2,wait_for_inclusion,wait_for_finalization,prompt,expected,test_id,",
    [
        (
            "192.168.1.4",
            9224,
            4,
            3,
            3,
            3,
            True,
            True,
            False,
            True,
            "edge_case_max_values",
        ),
    ],
    ids=["edge-case-max-values"],
)
def test_serve_extrinsic_edge_cases(
    mock_subtensor,
    mock_wallet,
    ip,
    port,
    protocol,
    netuid,
    placeholder1,
    placeholder2,
    wait_for_inclusion,
    wait_for_finalization,
    prompt,
    expected,
    test_id,
):
    # Arrange
    mock_subtensor._do_serve_axon.return_value = (True, "")
    with patch("bittensor.extrinsics.serving.Confirm.ask", return_value=True):
        # Act
        result = serve_extrinsic(
            mock_subtensor,
            mock_wallet,
            ip,
            port,
            protocol,
            netuid,
            placeholder1,
            placeholder2,
            wait_for_inclusion,
            wait_for_finalization,
            prompt,
        )

        # Assert
        assert result == expected


# Various error cases
@pytest.mark.parametrize(
    "ip,port,protocol,netuid,placeholder1,placeholder2,wait_for_inclusion,wait_for_finalization,prompt,expected_error_message,test_id,",
    [
        (
            "192.168.1.5",
            9225,
            5,
            4,
            4,
            4,
            True,
            True,
            False,
            False,
            "error-case-failed-serve",
        ),
    ],
    ids=["error-case-failed-serve"],
)
def test_serve_extrinsic_error_cases(
    mock_subtensor,
    mock_wallet,
    ip,
    port,
    protocol,
    netuid,
    placeholder1,
    placeholder2,
    wait_for_inclusion,
    wait_for_finalization,
    prompt,
    expected_error_message,
    test_id,
):
    # Arrange
    mock_subtensor._do_serve_axon.return_value = (False, "Error serving axon")
    with patch("bittensor.extrinsics.serving.Confirm.ask", return_value=True):
        # Act
        result = serve_extrinsic(
            mock_subtensor,
            mock_wallet,
            ip,
            port,
            protocol,
            netuid,
            placeholder1,
            placeholder2,
            wait_for_inclusion,
            wait_for_finalization,
            prompt,
        )

        # Assert
        assert result == expected_error_message
