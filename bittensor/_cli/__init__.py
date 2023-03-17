"""
Create and init the CLI class, which handles the coldkey, hotkey and money transfer 
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

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
from . import cli_impl
from .commands import *
from typing import List, Optional
from .naka_cli_impl import CLI as naka_CLI
console = bittensor.__console__

# Turn off rich console locals trace.
from rich.traceback import install
install(show_locals=False)

class cli:
    """
    Create and init the CLI class, which handles the coldkey, hotkey and tao transfer 
    """
    def __new__(
            cls,
            config: Optional['bittensor.Config'] = None,
            args: Optional[List[str]] = None, 
        ) -> 'bittensor.CLI':
        r""" Creates a new bittensor.cli from passed arguments.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.cli.config()
                args (`List[str]`, `optional`): 
                    The arguments to parse from the command line.
        """
        if config == None: 
            config = cli.config(args)
        cli.check_config( config )
        if config.subtensor:
            network = config.subtensor.get('network', bittensor.defaults.subtensor.network)

        if network == 'nakamoto':
            # Use nakamoto version of the CLI
            return naka_CLI(config=config)
        else:
            return cli_impl.CLI( config = config)
        
    @staticmethod
    def __create_parser__() -> 'argparse.ArgumentParser':
        """ Creates the argument parser for the bittensor cli.
        """
        parser = argparse.ArgumentParser(
            description=f"bittensor cli v{bittensor.__version__}",
            usage="btcli <command> <command args>",
            add_help=True)

        cmd_parsers = parser.add_subparsers(dest='command')
        RunCommand.add_args( cmd_parsers )
        HelpCommand.add_args( cmd_parsers ) 
        ListCommand.add_args( cmd_parsers )
        StakeCommand.add_args( cmd_parsers )
        UpdateCommand.add_args( cmd_parsers )
        InspectCommand.add_args( cmd_parsers ) 
        WeightsCommand.add_args( cmd_parsers )
        UnStakeCommand.add_args( cmd_parsers )
        OverviewCommand.add_args( cmd_parsers )
        RegisterCommand.add_args( cmd_parsers )
        TransferCommand.add_args( cmd_parsers )
        NominateCommand.add_args( cmd_parsers )
        NewHotkeyCommand.add_args( cmd_parsers )
        MetagraphCommand.add_args( cmd_parsers )
        NewColdkeyCommand.add_args( cmd_parsers )
        MyDelegatesCommand.add_args( cmd_parsers )
        ListSubnetsCommand.add_args( cmd_parsers )
        RegenHotkeyCommand.add_args( cmd_parsers )
        RegenColdkeyCommand.add_args( cmd_parsers )
        DelegateStakeCommand.add_args( cmd_parsers )
        DelegateUnstakeCommand.add_args( cmd_parsers )
        ListDelegatesCommand.add_args( cmd_parsers )
        RegenColdkeypubCommand.add_args( cmd_parsers )
        RecycleRegisterCommand.add_args( cmd_parsers )

        return parser

    @staticmethod   
    def config(args: List[str]) -> 'bittensor.config':
        """ From the argument parser, add config to bittensor.executor and local config 
            Return: bittensor.config object
        """
        parser = cli.__create_parser__()

        # If no arguments are passed, print help text.
        if len(args) == 0:
            parser.print_help()
            sys.exit()

        return bittensor.config( parser, args=args )

    @staticmethod   
    def check_config (config: 'bittensor.Config'):
        """ Check if the essential config exist under different command
        """
        if config.command == "run":
            RunCommand.check_config( config )
        elif config.command == "transfer":
            TransferCommand.check_config( config )
        elif config.command == "register":
            RegisterCommand.check_config( config )
        elif config.command == "unstake":
            UnStakeCommand.check_config( config )
        elif config.command == "stake":
            StakeCommand.check_config( config )
        elif config.command == "overview":
            OverviewCommand.check_config( config )
        elif config.command == "new_coldkey":
            NewColdkeyCommand.check_config( config )
        elif config.command == "new_hotkey":
            NewHotkeyCommand.check_config( config )
        elif config.command == "regen_coldkey":
            RegenColdkeyCommand.check_config( config )
        elif config.command == "regen_coldkeypub":
            RegenColdkeypubCommand.check_config( config )
        elif config.command == "regen_hotkey":
            RegenHotkeyCommand.check_config( config )
        elif config.command == "metagraph":
            MetagraphCommand.check_config( config )
        elif config.command == "weights":
            WeightsCommand.check_config( config )
        elif config.command == "list":
            ListCommand.check_config( config )
        elif config.command == "inspect":
            InspectCommand.check_config( config )
        elif config.command == "help":
            HelpCommand.check_config( config )
        elif config.command == "update":
            UpdateCommand.check_config( config )
        elif config.command == "nominate":
            NominateCommand.check_config( config )
        elif config.command == "list_delegates":
            ListDelegatesCommand.check_config( config )
        elif config.command == "list_subnets":
            ListSubnetsCommand.check_config( config )
        elif config.command == "delegate":
            DelegateStakeCommand.check_config( config )
        elif config.command == "undelegate":
            DelegateUnstakeCommand.check_config( config )
        elif config.command == "my_delegates":
            MyDelegatesCommand.check_config( config )
        elif config.command == "recycle_register":
            RecycleRegisterCommand.check_config( config )
        else:
            console.print(":cross_mark:[red]Unknown command: {}[/red]".format(config.command))
            sys.exit()

                