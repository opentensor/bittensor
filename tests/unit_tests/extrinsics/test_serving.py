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

from unittest.mock import MagicMock, patch

import pytest
from bittensor_wallet import Wallet

from bittensor.core.axon import Axon
from bittensor.core.subtensor import Subtensor
from bittensor.core.extrinsics import serving


@pytest.fixture
def mock_subtensor(mocker):
    mock_subtensor = mocker.MagicMock(spec=Subtensor)
    mock_subtensor.network = "test_network"
    mock_subtensor.substrate = mocker.MagicMock()
    return mock_subtensor


@pytest.fixture
def mock_wallet(mocker):
    wallet = mocker.MagicMock(spec=Wallet)
    wallet.hotkey.ss58_address = "hotkey_address"
    wallet.coldkeypub.ss58_address = "coldkey_address"
    return wallet


@pytest.fixture
def mock_axon(mock_wallet, mocker):
    axon = mocker.MagicMock(spec=Axon)
    axon.wallet = mock_wallet()
    axon.external_port = 9221
    return axon


@pytest.mark.parametrize(
    "ip,port,protocol,netuid,placeholder1,placeholder2,wait_for_inclusion,wait_for_finalization,expected,test_id,",
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
            "happy-path-wait-for-finalization",
        ),
    ],
    ids=[
        "happy-path-no-wait",
        "happy-path-wait-for-inclusion",
        "happy-path-wait-for-finalization",
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
    expected,
    test_id,
    mocker,
):
    # Arrange
    serving.do_serve_axon = mocker.MagicMock(return_value=(True, ""))
    # Act
    result = serving.serve_extrinsic(
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
    )

    # Assert
    assert result == expected, f"Test ID: {test_id}"


# Various edge cases
@pytest.mark.parametrize(
    "ip,port,protocol,netuid,placeholder1,placeholder2,wait_for_inclusion,wait_for_finalization,expected,test_id,",
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
    expected,
    test_id,
    mocker,
):
    # Arrange
    serving.do_serve_axon = mocker.MagicMock(return_value=(True, ""))
    # Act
    result = serving.serve_extrinsic(
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
    )

    # Assert
    assert result == expected, f"Test ID: {test_id}"


# Various error cases
@pytest.mark.parametrize(
    "ip,port,protocol,netuid,placeholder1,placeholder2,wait_for_inclusion,wait_for_finalization,expected_error_message,test_id,",
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
    expected_error_message,
    test_id,
    mocker,
):
    # Arrange
    serving.do_serve_axon = mocker.MagicMock(return_value=(False, "Error serving axon"))
    # Act
    result = serving.serve_extrinsic(
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
    )

    # Assert
    assert result == expected_error_message, f"Test ID: {test_id}"


@pytest.mark.parametrize(
    "netuid, wait_for_inclusion, wait_for_finalization, external_ip, external_ip_success, serve_success, expected_result, test_id",
    [
        # Happy path test
        (1, False, True, "192.168.1.1", True, True, True, "happy-ext-ip"),
        (1, False, True, None, True, True, True, "happy-net-external-ip"),
        # Edge cases
        (1, True, True, "192.168.1.1", True, True, True, "edge-case-wait"),
        # Error cases
        (1, False, True, None, False, True, False, "error-fetching-external-ip"),
        (
            1,
            False,
            True,
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
    external_ip,
    external_ip_success,
    serve_success,
    expected_result,
    test_id,
    mocker,
):
    mock_axon.external_ip = external_ip
    # Arrange
    with patch(
        "bittensor.utils.networking.get_external_ip",
        side_effect=Exception("Failed to fetch IP")
        if not external_ip_success
        else MagicMock(return_value="192.168.1.1"),
    ):
        serving.do_serve_axon = mocker.MagicMock(return_value=(serve_success, ""))
        # Act
        if not external_ip_success:
            with pytest.raises(ConnectionError):
                serving.serve_axon_extrinsic(
                    mock_subtensor,
                    netuid,
                    mock_axon,
                    wait_for_inclusion=wait_for_inclusion,
                    wait_for_finalization=wait_for_finalization,
                )
        else:
            result = serving.serve_axon_extrinsic(
                mock_subtensor,
                netuid,
                mock_axon,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
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
    with (
        patch.object(mock_subtensor.substrate, "compose_call"),
        patch.object(mock_subtensor.substrate, "create_signed_extrinsic"),
        patch.object(
            mock_subtensor.substrate,
            "submit_extrinsic",
            return_value=MagicMock(
                is_success=response_success,
                process_events=MagicMock(),
                error_message="error",
            ),
        ),
    ):
        # Act
        result = serving.publish_metadata(
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
