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

from bittensor.core.extrinsics.prometheus import (
    prometheus_extrinsic,
)
from bittensor.core.subtensor import Subtensor
from bittensor.core.settings import version_as_int


# Mocking the bittensor and networking modules
@pytest.fixture
def mock_bittensor():
    with patch("bittensor.core.subtensor.Subtensor") as mock:
        yield mock


@pytest.fixture
def mock_wallet():
    with patch("bittensor_wallet.Wallet") as mock:
        yield mock


@pytest.fixture
def mock_net():
    with patch("bittensor.utils.networking") as mock:
        yield mock


@pytest.mark.parametrize(
    "ip, port, netuid, wait_for_inclusion, wait_for_finalization, expected_result, test_id",
    [
        (None, 9221, 0, False, True, True, "happy-path-default-ip"),
        ("192.168.0.1", 9221, 0, False, True, True, "happy-path-custom-ip"),
        (None, 9221, 0, True, False, True, "happy-path-wait-for-inclusion"),
        (None, 9221, 0, False, False, True, "happy-path-no-waiting"),
    ],
)
def test_prometheus_extrinsic_happy_path(
    mock_bittensor,
    mock_wallet,
    mock_net,
    ip,
    port,
    netuid,
    wait_for_inclusion,
    wait_for_finalization,
    expected_result,
    test_id,
):
    # Arrange
    subtensor = MagicMock(spec=Subtensor)
    subtensor.network = "test_network"
    subtensor.substrate = MagicMock()
    wallet = MagicMock(spec=Wallet)
    mock_net.get_external_ip.return_value = "192.168.0.1"
    mock_net.ip_to_int.return_value = 3232235521  # IP in integer form
    mock_net.ip_version.return_value = 4
    neuron = MagicMock()
    neuron.is_null = False
    neuron.prometheus_info.version = version_as_int
    neuron.prometheus_info.ip = 3232235521
    neuron.prometheus_info.port = port
    neuron.prometheus_info.ip_type = 4
    subtensor.get_neuron_for_pubkey_and_subnet.return_value = neuron
    subtensor._do_serve_prometheus.return_value = (True, None)

    # Act
    result = prometheus_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        ip=ip,
        port=port,
        netuid=netuid,
        wait_for_inclusion=wait_for_inclusion,
        wait_for_finalization=wait_for_finalization,
    )

    # Assert
    assert result == expected_result, f"Test ID: {test_id}"


# Edge cases
@pytest.mark.parametrize(
    "ip, port, netuid, test_id",
    [
        ("0.0.0.0", 0, 0, "edge-case-min-values"),
        ("255.255.255.255", 65535, 2147483647, "edge-case-max-values"),
    ],
)
def test_prometheus_extrinsic_edge_cases(
    mock_bittensor, mock_wallet, mock_net, ip, port, netuid, test_id
):
    # Arrange
    subtensor = MagicMock(spec=Subtensor)
    subtensor.network = "test_network"
    subtensor.substrate = MagicMock()
    wallet = MagicMock(spec=Wallet)
    mock_net.get_external_ip.return_value = ip
    mock_net.ip_to_int.return_value = 3232235521  # IP in integer form
    mock_net.ip_version.return_value = 4
    neuron = MagicMock()
    neuron.is_null = True
    subtensor.get_neuron_for_pubkey_and_subnet.return_value = neuron
    subtensor._do_serve_prometheus.return_value = (True, None)

    # Act
    result = prometheus_extrinsic(
        subtensor=subtensor,
        wallet=wallet,
        ip=ip,
        port=port,
        netuid=netuid,
        wait_for_inclusion=False,
        wait_for_finalization=True,
    )

    # Assert
    assert result is True, f"Test ID: {test_id}"


# Error cases
def test_prometheus_extrinsic_error_cases(mock_bittensor, mock_wallet, mocker):
    # Arrange
    subtensor = MagicMock(spec=Subtensor)
    subtensor.network = "test_network"
    subtensor.substrate = MagicMock()
    subtensor.substrate.websocket.sock.getsockopt.return_value = 0
    wallet = MagicMock(spec=Wallet)
    neuron = MagicMock()
    neuron.is_null = True
    subtensor.get_neuron_for_pubkey_and_subnet.return_value = neuron
    subtensor._do_serve_prometheus.return_value = (True,)

    with mocker.patch(
        "bittensor.utils.networking.get_external_ip", side_effect=RuntimeError
    ):
        # Act & Assert
        with pytest.raises(RuntimeError):
            prometheus_extrinsic(
                subtensor=subtensor,
                wallet=wallet,
                ip=None,
                port=9221,
                netuid=1,
                wait_for_inclusion=False,
                wait_for_finalization=True,
            )
