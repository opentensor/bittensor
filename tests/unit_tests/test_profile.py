# Standard Library
from copy import deepcopy
from unittest.mock import patch, MagicMock, mock_open

# 3rd Party
import pytest

# Bittensor
from bittensor.commands.profile import ProfileCommand
from bittensor import config as bittensor_config


class MockDefaults:
    profile = {
        "name": "default",
        "path": "~/.bittensor/profiles/",
    }


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (
            {
                "name": "Alice",
                "coldkey": "ckey",
                "hotkey": "hkey",
                "subtensor_network": "mainnet",
                "active_netuid": "0",
            },
            "Alice",
        ),
        (
            {
                "name": "Bob",
                "coldkey": "bckey",
                "hotkey": "bhkey",
                "subtensor_network": "test",
                "active_netuid": "1",
            },
            "Bob",
        ),
    ],
    ids=["happy-path-Alice", "happy-path-Bob"],
)
@patch("bittensor.commands.profile.Prompt")
@patch("bittensor.commands.profile.ProfileCommand._write_profile")
@patch("bittensor.commands.profile.defaults", MockDefaults)
def test_run(mock_write_profile, mock_prompt, test_input, expected):
    # Arrange
    mock_cli = MagicMock()
    mock_cli.config = deepcopy(bittensor_config())
    mock_cli.config.path = "/fake/path/"
    for attr, value in test_input.items():
        setattr(mock_cli.config, attr, value)
    mock_prompt.ask.side_effect = lambda x, default="": test_input.get(
        x.split()[-1], ""
    )

    # Act
    ProfileCommand.run(mock_cli)

    # Assert
    mock_write_profile.assert_called_once()
    assert getattr(mock_cli.config, "name") == expected


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            bittensor_config(),
            True,
        ),
        # Edge cases
        (
            None,
            False,
        ),
    ],
)
def test_check_config(test_input, expected):
    # Arrange - In this case, all inputs are provided via test parameters, so we omit the Arrange section.

    # Act
    result = ProfileCommand.check_config(test_input)

    # Assert
    assert result == expected


def test_write_profile():
    # Create a mock config object with the necessary attributes
    mock_config = type(
        "Config",
        (object,),
        {
            "path": "~/.bittensor/profiles/",
            "name": "test_profile",
            "coldkey": "xyz123",
            "hotkey": "abc789",
            "subtensor_network": "finney",
            "active_netuid": "123",
        },
    )()

    # Setup the mock for os.makedirs and open
    with patch("os.makedirs") as mock_makedirs, patch(
        "os.path.expanduser", return_value="/.bittensor/profiles/"
    ), patch("builtins.open", mock_open()) as mock_file:
        # Call the function with the mock config
        ProfileCommand._write_profile(mock_config)

        # Assert that makedirs was called correctly
        mock_makedirs.assert_called_once_with("/.bittensor/profiles/", exist_ok=True)

        # Assert that open was called correctly; construct the expected file path and contents
        expected_path = "/.bittensor/profiles/test_profile"
        expected_contents = (
            f"name = {mock_config.name}\n"
            f"coldkey = {mock_config.coldkey}\n"
            f"hotkey = {mock_config.hotkey}\n"
            f"subtensor_network = {mock_config.subtensor_network}\n"
            f"active_netuid = {mock_config.active_netuid}\n"
        )

        # Assert the open function was called correctly and the right contents were written
        mock_file.assert_called_once_with(expected_path, "w+")
        mock_file().write.assert_called_once_with(expected_contents)
