# Standard Lib
from copy import deepcopy
from unittest.mock import MagicMock, patch

# Pytest
import pytest

# Bittensor
import bittensor
from bittensor.commands.overview import OverviewCommand
from tests.unit_tests.factories.neuron_factory import NeuronInfoLiteFactory


@pytest.fixture
def mock_subtensor():
    mock = MagicMock()
    mock.get_balance = MagicMock(return_value=100)
    return mock


def fake_config(**kwargs):
    config = deepcopy(construct_config())
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def construct_config():
    parser = bittensor.cli.__create_parser__()
    defaults = bittensor.config(parser=parser, args=[])
    # Parse commands and subcommands
    for command in bittensor.ALL_COMMANDS:
        if (
            command in bittensor.ALL_COMMANDS
            and "commands" in bittensor.ALL_COMMANDS[command]
        ):
            for subcommand in bittensor.ALL_COMMANDS[command]["commands"]:
                defaults.merge(
                    bittensor.config(parser=parser, args=[command, subcommand])
                )
        else:
            defaults.merge(bittensor.config(parser=parser, args=[command]))

    defaults.netuid = 1
    # Always use mock subtensor.
    defaults.subtensor.network = "finney"
    # Skip version checking.
    defaults.no_version_checking = True

    return defaults


@pytest.fixture
def mock_wallet():
    mock = MagicMock()
    mock.coldkeypub_file.exists_on_device = MagicMock(return_value=True)
    mock.coldkeypub_file.is_encrypted = MagicMock(return_value=False)
    mock.coldkeypub.ss58_address = "fake_address"
    return mock


class MockHotkey:
    def __init__(self, hotkey_str):
        self.hotkey_str = hotkey_str


class MockCli:
    def __init__(self, config):
        self.config = config


@pytest.mark.parametrize(
    "config_all, exists_on_device, is_encrypted, expected_balance, test_id",
    [
        (True, True, False, 100, "happy_path_all_wallets"),
        (False, True, False, 100, "happy_path_single_wallet"),
        (True, False, False, 0, "edge_case_no_wallets_found"),
        (True, True, True, 0, "edge_case_encrypted_wallet"),
    ],
)
def test_get_total_balance(
    mock_subtensor,
    mock_wallet,
    config_all,
    exists_on_device,
    is_encrypted,
    expected_balance,
    test_id,
):
    # Arrange
    cli = MockCli(fake_config(all=config_all))
    mock_wallet.coldkeypub_file.exists_on_device.return_value = exists_on_device
    mock_wallet.coldkeypub_file.is_encrypted.return_value = is_encrypted

    with patch(
        "bittensor.wallet", return_value=mock_wallet
    ) as mock_wallet_constructor, patch(
        "bittensor.commands.overview.get_coldkey_wallets_for_path",
        return_value=[mock_wallet] if config_all else [],
    ), patch(
        "bittensor.commands.overview.get_all_wallets_for_path",
        return_value=[mock_wallet],
    ), patch(
        "bittensor.commands.overview.get_hotkey_wallets_for_wallet",
        return_value=[mock_wallet],
    ):
        # Act
        result_hotkeys, result_balance = OverviewCommand._get_total_balance(
            0, mock_subtensor, cli
        )

        # Assert
        assert result_balance == expected_balance, f"Test ID: {test_id}"
        assert all(
            isinstance(hotkey, MagicMock) for hotkey in result_hotkeys
        ), f"Test ID: {test_id}"


@pytest.mark.parametrize(
    "config, all_hotkeys, expected_result, test_id",
    [
        # Happy path tests
        (
            {"all_hotkeys": False, "hotkeys": ["abc123", "xyz456"]},
            [MockHotkey("abc123"), MockHotkey("xyz456"), MockHotkey("mno567")],
            ["abc123", "xyz456"],
            "test_happy_path_included",
        ),
        (
            {"all_hotkeys": True, "hotkeys": ["abc123", "xyz456"]},
            [MockHotkey("abc123"), MockHotkey("xyz456"), MockHotkey("mno567")],
            ["mno567"],
            "test_happy_path_excluded",
        ),
        # Edge cases
        (
            {"all_hotkeys": False, "hotkeys": []},
            [MockHotkey("abc123"), MockHotkey("xyz456")],
            [],
            "test_edge_no_hotkeys_specified",
        ),
        (
            {"all_hotkeys": True, "hotkeys": []},
            [MockHotkey("abc123"), MockHotkey("xyz456")],
            ["abc123", "xyz456"],
            "test_edge_all_hotkeys_excluded",
        ),
        (
            {"all_hotkeys": False, "hotkeys": ["abc123", "xyz456"]},
            [],
            [],
            "test_edge_no_hotkeys_available",
        ),
        (
            {"all_hotkeys": True, "hotkeys": ["abc123", "xyz456"]},
            [],
            [],
            "test_edge_no_hotkeys_available_excluded",
        ),
    ],
)
def test_get_hotkeys(config, all_hotkeys, expected_result, test_id):
    # Arrange
    cli = MockCli(
        fake_config(
            hotkeys=config.get("hotkeys"), all_hotkeys=config.get("all_hotkeys")
        )
    )

    # Act
    result = OverviewCommand._get_hotkeys(cli, all_hotkeys)

    # Assert
    assert [
        hotkey.hotkey_str for hotkey in result
    ] == expected_result, f"Failed {test_id}"


def test_get_hotkeys_error():
    # Arrange
    cli = MockCli(fake_config(hotkeys=["abc123", "xyz456"], all_hotkeys=False))
    all_hotkeys = None

    # Act
    with pytest.raises(TypeError):
        OverviewCommand._get_hotkeys(cli, all_hotkeys)


@pytest.fixture
def neuron_info():
    return [
        (1, [NeuronInfoLiteFactory(netuid=1)], None),
        (2, [NeuronInfoLiteFactory(netuid=2)], None),
    ]


@pytest.fixture
def neurons_dict():
    return {
        "1": [NeuronInfoLiteFactory(netuid=1)],
        "2": [NeuronInfoLiteFactory(netuid=2)],
    }


@pytest.fixture
def netuids_list():
    return [1, 2]


# Test cases
@pytest.mark.parametrize(
    "test_id, results, expected_neurons, expected_netuids",
    [
        # Test ID: 01 - Happy path, all neurons processed correctly
        (
            "01",
            [
                (1, [NeuronInfoLiteFactory(netuid=1)], None),
                (2, [NeuronInfoLiteFactory(netuid=2)], None),
            ],
            {
                "1": [NeuronInfoLiteFactory(netuid=1)],
                "2": [NeuronInfoLiteFactory(netuid=2)],
            },
            [1, 2],
        ),
        # Test ID: 02 - Error message present, should skip processing for that netuid
        (
            "02",
            [
                (1, [NeuronInfoLiteFactory(netuid=1)], None),
                (2, [], "Error fetching data"),
            ],
            {"1": [NeuronInfoLiteFactory()]},
            [1],
        ),
        # Test ID: 03 - No neurons found for a netuid, should remove the netuid
        (
            "03",
            [(1, [NeuronInfoLiteFactory()], None), (2, [], None)],
            {"1": [NeuronInfoLiteFactory()]},
            [1],
        ),
        # Test ID: 04 - Mixed conditions
        (
            "04",
            [
                (1, [NeuronInfoLiteFactory(netuid=1)], None),
                (2, [], None),
            ],
            {"1": [NeuronInfoLiteFactory()]},
            [1],
        ),
    ],
)
def test_process_neuron_results(
    test_id, results, expected_neurons, expected_netuids, neurons_dict, netuids_list
):
    # Act
    actual_neurons = OverviewCommand._process_neuron_results(
        results, neurons_dict, netuids_list
    )

    # Assert
    assert actual_neurons.keys() == expected_neurons.keys(), f"Failed test {test_id}"
    assert netuids_list == expected_netuids, f"Failed test {test_id}"
