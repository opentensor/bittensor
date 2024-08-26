import pytest

from unittest.mock import MagicMock, patch
from bittensor.subtensor import Subtensor
from bittensor.wallet import wallet as Wallet
from bittensor.axon import axon as Axon
from bittensor.extrinsics.serving import (
    serve_extrinsic,
    publish_metadata,
    serve_axon_extrinsic,
)


@pytest.fixture
def mock_subtensor():
    mock_subtensor = MagicMock(spec=Subtensor)
    mock_subtensor.network = "test_network"
    mock_subtensor.substrate = MagicMock()
    return mock_subtensor


@pytest.fixture
def mock_wallet():
    wallet = MagicMock(spec=Wallet)
    wallet.hotkey.ss58_address = "hotkey_address"
    wallet.coldkeypub.ss58_address = "coldkey_address"
    return wallet


@pytest.fixture
def mock_axon(mock_wallet):
    axon = MagicMock(spec=Axon)
    axon.wallet = mock_wallet()
    axon.external_port = 9221
    return axon


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
        assert result == expected, f"Test ID: {test_id}"


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
        assert result == expected_error_message, f"Test ID: {test_id}"


@pytest.mark.parametrize(
    "netuid, wait_for_inclusion, wait_for_finalization, prompt, external_ip, external_ip_success, serve_success, expected_result, test_id",
    [
        # Happy path test
        (1, False, True, False, "192.168.1.1", True, True, True, "happy-ext-ip"),
        (1, False, True, True, None, True, True, True, "happy-net-external-ip"),
        # Edge cases
        (1, True, True, False, "192.168.1.1", True, True, True, "edge-case-wait"),
        # Error cases
        (1, False, True, False, None, False, True, False, "error-fetching-external-ip"),
        (
            1,
            False,
            True,
            False,
            "192.168.1.1",
            True,
            False,
            False,
            "error-serving-axon",
        ),
    ],
    ids=[
        "happy-axon-external-ip",
        "happy-net-external-ip",
        "edge-case-wait",
        "error-fetching-external-ip",
        "error-serving-axon",
    ],
)
def test_serve_axon_extrinsic(
    mock_subtensor,
    mock_axon,
    netuid,
    wait_for_inclusion,
    wait_for_finalization,
    prompt,
    external_ip,
    external_ip_success,
    serve_success,
    expected_result,
    test_id,
):
    mock_axon.external_ip = external_ip
    # Arrange
    with patch(
        "bittensor.utils.networking.get_external_ip",
        side_effect=Exception("Failed to fetch IP")
        if not external_ip_success
        else MagicMock(return_value="192.168.1.1"),
    ), patch.object(mock_subtensor, "serve", return_value=serve_success):
        # Act
        if not external_ip_success:
            with pytest.raises(RuntimeError):
                result = serve_axon_extrinsic(
                    mock_subtensor,
                    netuid,
                    mock_axon,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                    prompt=prompt,
                )
        else:
            result = serve_axon_extrinsic(
                mock_subtensor,
                netuid,
                mock_axon,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
                prompt=prompt,
            )
            # Assert
            assert result == expected_result, f"Test ID: {test_id}"


@pytest.mark.parametrize(
    "wait_for_inclusion, wait_for_finalization, net_uid, type_u, data, response_success, expected_result, test_id",
    [
        (
            True,
            True,
            1,
            "Sha256",
            b"mock_bytes_data",
            True,
            True,
            "happy-path-wait",
        ),
        (
            False,
            False,
            1,
            "Sha256",
            b"mock_bytes_data",
            True,
            True,
            "happy-path-no-wait",
        ),
    ],
    ids=["happy-path-wait", "happy-path-no-wait"],
)
def test_publish_metadata(
    mock_subtensor,
    mock_wallet,
    wait_for_inclusion,
    wait_for_finalization,
    net_uid,
    type_u,
    data,
    response_success,
    expected_result,
    test_id,
):
    # Arrange
    with patch.object(mock_subtensor.substrate, "compose_call"), patch.object(
        mock_subtensor.substrate, "create_signed_extrinsic"
    ), patch.object(
        mock_subtensor.substrate,
        "submit_extrinsic",
        return_value=MagicMock(
            is_success=response_success,
            process_events=MagicMock(),
            error_message="error",
        ),
    ):
        # Act
        result = publish_metadata(
            subtensor=mock_subtensor,
            wallet=mock_wallet,
            netuid=net_uid,
            data_type=type_u,
            data=data,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )
        # Assert
        assert result == expected_result, f"Test ID: {test_id}"
