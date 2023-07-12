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
console = bittensor.__console__

class cli:
    """
    Implementation of the CLI class, which handles the coldkey, hotkey and money transfer
    """
    def __init__(
            self, 
            config: Optional['bittensor.config'] = None,
            args: Optional[List[str]] = None,
        ):
        r""" Initialized a bittensor.CLI object.
            Args:
                config (:obj:`bittensor.Config`, `required`):
                    bittensor.cli.config()
        """
        if config == None:
            self.config = cli.config( args )
        cli.check_config( self.config )

        # (d)efaults to True if config.no_version_checking is not set.
        if not self.config.get("no_version_checking", d=True):
            try:
                bittensor.utils.version_checking()
            except:
                raise RuntimeError("To avoid internet based version checking pass --no_version_checking while running the CLI.")

    @staticmethod
    def __create_parser__() -> 'argparse.ArgumentParser':
        """ Creates the argument parser for the bittensor cli.
        """
        parser = argparse.ArgumentParser(
            description=f"bittensor cli v{bittensor.__version__}",
            usage="btcli <command> <command args>",
            add_help=True)

        cmd_parsers = parser.add_subparsers(dest='command')
        ListCommand.add_args( cmd_parsers )
        StakeCommand.add_args( cmd_parsers )
        UpdateCommand.add_args( cmd_parsers )
        InspectCommand.add_args( cmd_parsers )
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
        SenateCommand.add_args( cmd_parsers )
        ProposalsCommand.add_args( cmd_parsers )
        ShowVotesCommand.add_args( cmd_parsers )
        SenateRegisterCommand.add_args( cmd_parsers )
        SenateLeaveCommand.add_args( cmd_parsers )
        VoteCommand.add_args( cmd_parsers )


        return parser

    @staticmethod
    def config(args: List[str]) -> 'bittensor.config':
        """ From the argument parser, add config to bittensor.executor and local config
            Return: bittensor.config object
        """
        parser = cli.__create_parser__()

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
        if config.command == "transfer":
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
        elif config.command == "list":
            ListCommand.check_config( config )
        elif config.command == "inspect":
            InspectCommand.check_config( config )
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
        elif config.command == "senate":
            SenateCommand.check_config( config )
        elif config.command == "proposals":
            ProposalsCommand.check_config( config )
        elif config.command == "proposal_votes":
            ShowVotesCommand.check_config( config )
        elif config.command == "senate_register":
            SenateRegisterCommand.check_config( config )
        elif config.command == "senate_leave":
            SenateLeaveCommand.check_config( config )
        elif config.command == "senate_vote":
            VoteCommand.check_config( config )
        else:
            console.print(":cross_mark:[red]Unknown command: {}[/red]".format(config.command))
            sys.exit()

    def run ( self ):
        """ Execute the command from config
        """
        if self.config.command == "transfer":
            TransferCommand.run( self )
        elif self.config.command == "register":
            RegisterCommand.run( self )
        elif self.config.command == "unstake":
            UnStakeCommand.run( self )
        elif self.config.command == "stake":
            StakeCommand.run( self )
        elif self.config.command == "overview":
            OverviewCommand.run( self )
        elif self.config.command == "list":
            ListCommand.run( self )
        elif self.config.command == "new_coldkey":
            NewColdkeyCommand.run( self )
        elif self.config.command == "new_hotkey":
            NewHotkeyCommand.run( self )
        elif self.config.command == "regen_coldkey":
            RegenColdkeyCommand.run( self )
        elif self.config.command == "regen_coldkeypub":
            RegenColdkeypubCommand.run( self )
        elif self.config.command == "regen_hotkey":
            RegenHotkeyCommand.run( self )
        elif self.config.command == "metagraph":
            MetagraphCommand.run( self )
        elif self.config.command == "inspect":
            InspectCommand.run( self )
        elif self.config.command == 'update':
            UpdateCommand.run( self )
        elif self.config.command == 'nominate':
            NominateCommand.run( self )
        elif self.config.command == 'delegate':
            DelegateStakeCommand.run( self )
        elif self.config.command == 'undelegate':
            DelegateUnstakeCommand.run( self )
        elif self.config.command == 'my_delegates':
            MyDelegatesCommand.run( self )
        elif self.config.command == 'list_delegates':
            ListDelegatesCommand.run( self )
        elif self.config.command == 'list_subnets':
            ListSubnetsCommand.run( self )
        elif self.config.command == 'recycle_register':
            RecycleRegisterCommand.run( self )
        elif self.config.command == "senate":
            SenateCommand.run( self )
        elif self.config.command == "proposals":
            ProposalsCommand.run( self )
        elif self.config.command == "proposal_votes":
            ShowVotesCommand.run( self )
        elif self.config.command == "senate_register":
            SenateRegisterCommand.run( self )
        elif self.config.command == "senate_leave":
            SenateLeaveCommand.run( self )
        elif self.config.command == "senate_vote":
            VoteCommand.run( self )