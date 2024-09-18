import pytest
from unittest.mock import MagicMock, patch
import bittensor
from bittensor.subtensor import Subtensor
from bittensor.wallet import wallet as Wallet
from bittensor.extrinsics.prometheus import prometheus_extrinsic


# Mocking the bittensor and networking modules
@pytest.fixture
def mock_bittensor():
    with patch("bittensor.subtensor") as mock:
        yield mock


@pytest.fixture
def mock_wallet():
    with patch("bittensor.wallet") as mock:
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
    wallet = MagicMock(spec=Wallet)
    mock_net.get_external_ip.return_value = "192.168.0.1"
    mock_net.ip_to_int.return_value = 3232235521  # IP in integer form
    mock_net.ip_version.return_value = 4
    neuron = MagicMock()
    neuron.is_null = False
    neuron.prometheus_info.version = bittensor.__version_as_int__
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
    assert result == True, f"Test ID: {test_id}"


# Error cases
@pytest.mark.parametrize(
    "ip, port, netuid, exception, test_id",
    [
        (
            None,
            9221,
            0,
            RuntimeError("Unable to attain your external ip."),
            "error-case-no-external-ip",
        ),
    ],
)
def test_prometheus_extrinsic_error_cases(
    mock_bittensor, mock_wallet, mock_net, ip, port, netuid, exception, test_id
):
    # Arrange
    subtensor = MagicMock(spec=Subtensor)
    subtensor.network = "test_network"
    wallet = MagicMock(spec=Wallet)
    mock_net.get_external_ip.side_effect = exception
    neuron = MagicMock()
    neuron.is_null = True
    subtensor.get_neuron_for_pubkey_and_subnet.return_value = neuron
    subtensor._do_serve_prometheus.return_value = (True,)

    # Act & Assert
    with pytest.raises(ValueError):
        prometheus_extrinsic(
            subtensor=subtensor,
            wallet=wallet,
            ip=ip,
            port=port,
            netuid=netuid,
            wait_for_inclusion=False,
            wait_for_finalization=True,
        )
