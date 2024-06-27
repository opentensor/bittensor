# Standard Library
import os
from unittest import mock
from unittest.mock import patch, MagicMock, mock_open

# 3rd Party
import pytest
from rich.table import Table

# Bittensor
from bittensor.commands.profile import (
    ProfileCommand,
    ProfileListCommand,
    ProfileShowCommand,
    ProfileDeleteCommand,
    ProfileSetValueCommand,
    ProfileDeleteValueCommand,
    ProfileSetCommand,
    list_profiles,
    get_profile_file_path,
    open_profile,
)
from bittensor import config as bittensor_config


class MockDefaults:
    profile = {
        "name": "default",
        "path": "~/.bittensor/profiles/",
    }


@pytest.fixture
def mock_cli():
    mock_cli = MagicMock()
    mock_cli.config = bittensor_config()
    mock_cli.config.profile = MagicMock()
    mock_cli.config.profile.path = "~/.bittensor/profiles/"
    mock_cli.config.profile.name = "default"
    mock_cli.config.is_set = MagicMock(return_value=True)
    mock_cli.config.no_prompt = False
    return mock_cli


def test_list_profiles_success():
    path = "/fake/path/"
    with patch("os.listdir", return_value=["profile1.yaml", "profile2.yml"]), patch(
            "os.makedirs"
    ) as mock_makedirs:
        profiles = list_profiles(path)
        assert profiles == ["profile1.yaml", "profile2.yml"]
        mock_makedirs.assert_called_once_with(path, exist_ok=True)


def test_list_profiles_no_profiles():
    path = "/fake/path/"
    with patch("os.listdir", return_value=[]), patch("os.makedirs"):
        profiles = list_profiles(path)
        assert profiles == []


def test_list_profiles_os_error():
    path = "/fake/path/"
    with patch("os.makedirs", side_effect=OSError("Error")), patch("os.listdir"):
        profiles = list_profiles(path)
        assert profiles == []


def test_get_profile_file_path():
    path = "/fake/path/"
    profile_name = "profile1"
    with patch("bittensor.commands.profile.os.listdir", return_value=["profile1.yaml", "profile2.yml"]), patch(
            "bittensor.commands.profile.list_profiles", return_value=["profile1.yaml", "profile2.yml"]
    ):
        profile_path = get_profile_file_path(path, profile_name)
        expected_path = os.path.join(path, "profile1.yaml")
        assert profile_path == expected_path


def test_get_profile_file_path_not_found():
    path = "/fake/path/"
    with patch("os.listdir", return_value=["profile1.yaml", "profile2.yml"]):
        profile_path = get_profile_file_path(path, "profile3")
        assert profile_path is None


@patch("bittensor.commands.profile.Prompt.ask", return_value="test_profile")
def test_open_profile(mock_prompt, mock_cli):
    mock_cli.config.profile.name = "test_profile"
    with patch(
            "builtins.open", mock_open(read_data="profile:\n  name: test_profile")
    ), patch("os.listdir", return_value=["test_profile.yaml"]):
        config, profile_path, contents = open_profile(mock_cli, "show")
        assert config.profile.name == "test_profile.yaml"
        assert contents == {"profile": {"name": "test_profile"}}


@patch("bittensor.commands.profile.Prompt.ask", return_value="test_profile")
def test_open_profile_not_found(mock_prompt, mock_cli):
    mock_cli.config.profile.name = "test_profile"
    with patch("os.listdir", return_value=["test_profile.yaml"]):
        config, profile_path, contents = open_profile(mock_cli, "show")
        assert config is None
        assert profile_path is None
        assert contents is None


def test_run_profile_command(mock_cli):
    with patch("bittensor.subtensor"), patch.object(ProfileCommand, "_run") as mock_run:
        ProfileCommand.run(mock_cli)
        mock_run.assert_called_once()


def test_profile_command_write_profile():
    mock_config = bittensor_config()
    mock_config.profile = MagicMock()
    mock_config.profile.path = "~/.bittensor/profiles/"
    mock_config.profile.name = "test_profile"
    mock_config.subtensor = MagicMock()
    mock_config.subtensor.network = "testnet"
    mock_config.subtensor.chain_endpoint = "endpoint"
    mock_config.wallet = MagicMock()
    mock_config.wallet.name = "wallet_name"
    mock_config.wallet.hotkey = "hotkey"
    mock_config.wallet.path = "/wallet/path/"
    with patch("os.makedirs"), patch(
            "os.path.expanduser", return_value="/.bittensor/profiles/"
    ), patch("builtins.open", mock_open()) as mock_file:
        ProfileCommand._write_profile(mock_config)
        mock_file.assert_called_once_with("/.bittensor/profiles/test_profile.yml", "w+")


@patch("bittensor.commands.profile.Prompt.ask", return_value="test_profile")
def test_profile_list_command(mock_prompt, mock_cli):
    mock_cli.config.profile.path = "/fake/path/"
    with patch("os.listdir", return_value=["test_profile.yaml"]), patch(
            "builtins.open",
            mock_open(read_data="profile:\n  name: test_profile\nsubtensor:\n  network: mainnet\nnetuid: 0")
    ), patch("bittensor.__console__.print"):
        ProfileListCommand.run(mock_cli)


@patch("bittensor.commands.profile.Prompt.ask", return_value="test_profile")
def test_profile_show_command(mock_prompt, mock_cli):
    mock_cli.config.profile.name = "test_profile"
    with patch(
            "builtins.open",
            mock_open(read_data="profile:\n  name: test_profile\nsubtensor:\n  network: mainnet\nnetuid: 0")
    ), patch("os.listdir", return_value=["test_profile.yaml"]), patch(
        "bittensor.__console__.print"
    ):
        ProfileShowCommand.run(mock_cli)


@patch("bittensor.commands.profile.Prompt.ask", return_value="test_profile")
def test_profile_delete_command(mock_prompt, mock_cli):
    mock_cli.config.profile.name = "test_profile"
    mock_cli.config.profile.path = "/fake/path/"
    profile_path = "/fake/path/test_profile.yaml"

    with patch("os.remove") as mock_remove, patch(
            "bittensor.commands.profile.get_profile_path_from_config",
            return_value=(mock_cli.config, profile_path)
    ), patch("bittensor.commands.profile.os.path.exists", return_value=True), patch(
        "bittensor.commands.profile.os.listdir", return_value=["test_profile.yaml"]
    ), patch(
        "builtins.open", mock_open(read_data="profile:\n  name: test_profile")
    ):
        ProfileDeleteCommand.run(mock_cli)
        mock_remove.assert_called_once_with("/fake/path/test_profile.yaml")


@patch("bittensor.commands.profile.Prompt.ask", return_value="test_profile")
def test_profile_set_value_command(mock_prompt, mock_cli):
    mock_cli.config.args = ["profile.name=new_name"]
    with patch(
            "builtins.open", mock_open(read_data="profile:\n  name: test_profile")
    ), patch("os.listdir", return_value=["test_profile.yaml"]), patch(
        "bittensor.__console__.print"
    ):
        ProfileSetValueCommand.run(mock_cli)


@patch("bittensor.commands.profile.Prompt.ask", return_value="test_profile")
def test_profile_delete_value_command(mock_prompt, mock_cli):
    mock_cli.config.args = ["profile.name"]
    with patch(
            "builtins.open", mock_open(read_data="profile:\n  name: test_profile")
    ), patch("os.listdir", return_value=["test_profile.yaml"]), patch(
        "bittensor.__console__.print"
    ):
        ProfileDeleteValueCommand.run(mock_cli)


@patch("bittensor.commands.profile.Prompt.ask", return_value="test_profile")
def test_profile_show_command_format(mock_prompt, mock_cli):
    mock_cli.config.profile.name = "test_profile"
    with patch(
            "builtins.open",
            mock_open(read_data="profile:\n  name: test_profile\nsubtensor:\n  network: mainnet\nnetuid: 0")
    ), patch("os.listdir", return_value=["test_profile.yaml"]), patch(
        "bittensor.__console__.print"
    ) as mock_print:
        ProfileShowCommand.run(mock_cli)

        printed_table = mock_print.call_args[0][0]
        assert isinstance(printed_table, Table)

        assert printed_table.title == "[white]Profile [bold white]test_profile.yaml"

        columns = [column.header for column in printed_table.columns]
        assert columns == ["[overline white]PARAMETERS", "[overline white]VALUES"]

        param_cells = printed_table.columns[0]._cells
        value_cells = printed_table.columns[1]._cells
        rows = list(zip(param_cells, value_cells))

        expected_rows = [
            ("  [bold white]profile.name", "[green]test_profile"),
            ("  [bold white]subtensor.network", "[green]mainnet"),
            ("  [bold white]netuid", "[green]0"),
        ]
        assert rows == expected_rows


@patch("bittensor.commands.profile.Prompt.ask", return_value="test_profile")
def test_profile_list_command_format(mock_prompt, mock_cli):
    mock_cli.config.profile.path = "/fake/path/"
    with patch("os.listdir", return_value=["test_profile.yaml"]), patch(
            "builtins.open",
            mock_open(read_data="profile:\n  name: test_profile\nsubtensor:\n  network: mainnet\nnetuid: 0")
    ), patch("bittensor.__console__.print") as mock_print, patch("os.makedirs", return_value=True):
        ProfileListCommand.run(mock_cli)

        printed_table = mock_print.call_args[0][0]
        assert isinstance(printed_table, Table)

        assert printed_table.title == "[white]Profiles"

        columns = [column.header for column in printed_table.columns]
        assert columns == ["A", "Name", "Network", "Netuid"]

        a_cells = printed_table.columns[0]._cells
        name_cells = printed_table.columns[1]._cells
        network_cells = printed_table.columns[2]._cells
        netuid_cells = printed_table.columns[3]._cells
        rows = list(zip(a_cells, name_cells, network_cells, netuid_cells))

        expected_rows = [
            (" ", "test_profile", "mainnet", "0"),
        ]
        assert rows == expected_rows


@patch("bittensor.commands.profile.Prompt.ask", return_value="test_profile")
def test_profile_set_command(mock_prompt, mock_cli):
    mock_cli.config.profile.name = "test_profile"
    with patch(
            "builtins.open", mock_open(read_data="profile:\n  name: test_profile")
    ), patch("os.listdir", return_value=["test_profile.yaml"]), patch(
        "bittensor.__console__.print"
    ), patch(
        "os.path.exists", return_value=True
    ), patch(
        "yaml.safe_load", return_value={"profile": {"active": "old_profile"}}
    ), patch(
        "yaml.safe_dump"
    ) as mock_safe_dump:
        ProfileSetCommand.run(mock_cli)

        expected_config = {"profile": {"active": "test_profile"}}
        mock_safe_dump.assert_called_once_with(expected_config, mock.ANY)


@patch("bittensor.commands.profile.Prompt.ask", return_value="test_profile")
def test_profile_set_command_no_profile(mock_prompt, mock_cli):
    mock_cli.config.profile.name = "test_profile"
    with patch(
            "os.listdir", return_value=[]
    ), patch("bittensor.__console__.print") as mock_print:
        ProfileSetCommand.run(mock_cli)
        mock_print.assert_called_once()
