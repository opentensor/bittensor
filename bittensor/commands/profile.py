# Standard Library
import argparse
import os

# 3rd Party
from rich.prompt import Prompt

# Bittensor
import bittensor

# Local
from . import defaults


PROFILE_ATTRIBUTES = [
    "name",
    "path",
    "coldkey",
    "hotkey",
    "subtensor_network",
    "active_netuid",
]


class ProfileCommand:
    """
    Executes the ``create`` command.

    This class provides functionality to create a profile by prompting the user to enter various attributes.
    The entered attributes are then written to a profile file.

    """

    @staticmethod
    def run(cli):
        ProfileCommand._run(cli)

    @staticmethod
    def _run(cli: "bittensor.cli"):
        config = cli.config.copy()
        for attribute in PROFILE_ATTRIBUTES:
            if defaults.profile.get(attribute):
                setattr(
                    config,
                    attribute,
                    Prompt.ask(
                        f"Enter {attribute}", default=defaults.profile[attribute]
                    ),
                )
            else:
                setattr(
                    config,
                    attribute,
                    Prompt.ask(f"Enter {attribute}"),
                )

        ProfileCommand._write_profile(config)

    @staticmethod
    def _write_profile(config: "bittensor.config"):
        path = os.path.expanduser(config.path)

        os.makedirs(path, exist_ok=True)

        profile = (
            f"name = {config.name}\n"
            f"coldkey = {config.coldkey}\n"
            f"hotkey = {config.hotkey}\n"
            f"subtensor_network = {config.subtensor_network}\n"
            f"active_netuid = {config.active_netuid}\n"
        )
        with open(f"{path}{config.name}", "w+") as f:
            f.write(profile)

    @staticmethod
    def check_config(config: "bittensor.config"):
        return config is not None

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        list_parser = parser.add_parser("create", help="""Create profile""")
        list_parser.set_defaults(func=ProfileCommand.run)
