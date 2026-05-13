import argparse
import sys
import subprocess
import pytest
import os
import bittensor as bt
from bittensor.core.config import InvalidConfigFile


def _config_call():
    """Create a config object from the bt cli args."""
    parser = argparse.ArgumentParser()
    bt.Axon.add_args(parser)
    bt.Subtensor.add_args(parser)
    bt.AsyncSubtensor.add_args(parser)
    bt.Wallet.add_args(parser)
    bt.logging.add_args(parser)
    bt.PriorityThreadPoolExecutor.add_args(parser)
    config = bt.Config(parser)
    return config


TEST_ARGS = [
    "bittensor",
    "--config",
    "path/to/config.yaml",
    "--strict",
    "--no_version_checking",
]


def test_bittensor_cli_parser_enabled(monkeypatch):
    """Tests that the bt cli args are processed."""

    monkeypatch.setenv("BT_NO_PARSE_CLI_ARGS", "false")
    monkeypatch.setattr(sys, "argv", TEST_ARGS)

    with pytest.raises(InvalidConfigFile) as error:
        _config_call()

    assert "No such file or directory" in str(error)


def test_bittensor_cli_parser_disabled(monkeypatch):
    """Tests that the bt cli args are not processed."""
    monkeypatch.setattr(sys, "argv", TEST_ARGS)

    config = _config_call()

    assert config.config is False
    assert config.strict is False
    assert config.no_version_checking is False


def test_import_does_not_hijack_help():
    """Importing bittensor should not intercept --help."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import bittensor; import argparse; "
            "p = argparse.ArgumentParser('user_script'); "
            "p.add_argument('--my-arg', default=1); "
            "p.parse_args()",
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--my-arg" in result.stdout
    assert "--logging" not in result.stdout
