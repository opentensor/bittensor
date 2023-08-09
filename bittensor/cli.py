# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import sys
import argparse
import bittensor
from typing import List, Optional
from .commands import *

# Create a console instance for CLI display.
console = bittensor.__console__

COMMANDS = {
    "transfer": TransferCommand,
    "register": RegisterCommand,
    "unstake": UnStakeCommand,
    "stake": StakeCommand,
    "overview": OverviewCommand,
    "list": ListCommand,
    "new_coldkey": NewColdkeyCommand,
    "new_hotkey": NewHotkeyCommand,
    "regen_coldkey": RegenColdkeyCommand,
    "regen_coldkeypub": RegenColdkeypubCommand,
    "regen_hotkey": RegenHotkeyCommand,
    "metagraph": MetagraphCommand,
    "inspect": InspectCommand,
    "update": UpdateCommand,
    "nominate": NominateCommand,
    "delegate": DelegateStakeCommand,
    "undelegate": DelegateUnstakeCommand,
    "my_delegates": MyDelegatesCommand,
    "list_delegates": ListDelegatesCommand,
    "list_subnets": ListSubnetsCommand,
    "recycle_register": RecycleRegisterCommand,
    "senate": SenateCommand,
    "proposals": ProposalsCommand,
    "proposal_votes": ShowVotesCommand,
    "senate_register": SenateRegisterCommand,
    "senate_leave": SenateLeaveCommand,
    "senate_vote": VoteCommand,
    "register_subnet": RegisterSubnetworkCommand,
    "run_faucet": RunFaucetCommand,
    'get_subnet_burn_cost': GetSubnetBurnCostCommand,
}


class cli:
    """
    Implementation of the Command Line Interface (CLI) class for the Bittensor protocol.
    This class handles operations like key management (hotkey and coldkey) and token transfer.
    """

    def __init__(
        self,
        config: Optional["bittensor.config"] = None,
        args: Optional[List[str]] = None,
    ):
        """
        Initializes a bittensor.CLI object.

        Args:
            config (bittensor.config, optional): The configuration settings for the CLI.
            args (List[str], optional): List of command line arguments.
        """
        # Turns on console for cli.
        bittensor.turn_console_on()

        # If no config is provided, create a new one from args.
        if config == None:
            config = cli.create_config(args)

        self.config = config

        # Check if the config is valid.
        cli.check_config(self.config)

        # If no_version_checking is not set or set as False in the config, version checking is done.
        if not self.config.get("no_version_checking", d=True):
            try:
                bittensor.utils.version_checking()
            except:
                # If version checking fails, inform user with an exception.
                raise RuntimeError(
                    "To avoid internet-based version checking, pass --no_version_checking while running the CLI."
                )

    @staticmethod
    def __create_parser__() -> "argparse.ArgumentParser":
        """
        Creates the argument parser for the Bittensor CLI.

        Returns:
            argparse.ArgumentParser: An argument parser object for Bittensor CLI.
        """
        # Define the basic argument parser.
        parser = argparse.ArgumentParser(
            description=f"bittensor cli v{bittensor.__version__}",
            usage="btcli <command> <command args>",
            add_help=True,
        )
        # Add arguments for each sub-command.
        cmd_parsers = parser.add_subparsers(dest="command")
        # Add argument parsers for all available commands.
        for command in COMMANDS.values():
            command.add_args(cmd_parsers)
        return parser

    @staticmethod
    def create_config(args: List[str]) -> "bittensor.config":
        """
        From the argument parser, add config to bittensor.executor and local config

        Args:
            args (List[str]): List of command line arguments.

        Returns:
            bittensor.config: The configuration object for Bittensor CLI.
        """
        parser = cli.__create_parser__()

        # If no arguments are passed, print help text and exit the program.
        if len(args) == 0:
            parser.print_help()
            sys.exit()

        return bittensor.config(parser, args=args)

    @staticmethod
    def check_config(config: "bittensor.config"):
        """
        Checks if the essential configuration exists under different command

        Args:
            config (bittensor.config): The configuration settings for the CLI.
        """
        # Check if command exists, if so, run the corresponding check_config.
        # If command doesn't exist, inform user and exit the program.
        if config.command in COMMANDS:
            COMMANDS[config.command].check_config(config)
        else:
            console.print(f":cross_mark:[red]Unknown command: {config.command}[/red]")
            sys.exit()

    def run(self):
        """
        Executes the command from the configuration.
        """
        # Check if command exists, if so, run the corresponding method.
        # If command doesn't exist, inform user and exit the program.
        if self.config.command in COMMANDS:
            COMMANDS[self.config.command].run(self)
        else:
            console.print(
                f":cross_mark:[red]Unknown command: {self.config.command}[/red]"
            )
            sys.exit()
