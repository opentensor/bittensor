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

import argparse
import os
import sys
from typing import List, Optional

import bittensor
import torch
from rich.prompt import Confirm, Prompt, PromptBase

from . import cli_impl
from .commands.stake import StakeCommand
from .commands.unstake import UnStakeCommand

# Turn off rich console locals trace.
from rich.traceback import install
install(show_locals=False)

console = bittensor.__console__

# Remove incredibly large tracebacks.
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
        return cli_impl.CLI( config = config)

    @staticmethod   
    def config(args: List[str]) -> 'bittensor.config':
        """ From the argument parser, add config to bittensor.executor and local config 
            Return: bittensor.config object
        """
        parser = argparse.ArgumentParser(
            description=f"bittensor cli v{bittensor.__version__}",
            usage="btcli <command> <command args>",
            add_help=True)

        cmd_parsers = parser.add_subparsers(dest='command')
        StakeCommand.add_args( cmd_parsers )
        UnStakeCommand.add_args( cmd_parsers )

        overview_parser = cmd_parsers.add_parser(
            'overview', 
            help='''Show registered account overview.'''
        )
        overview_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        overview_parser.add_argument(
            '--all', 
            dest='all', 
            action='store_true', 
            help='''View overview for all wallets.''',
            default=False,
        )
        overview_parser.add_argument(
            '--no_cache', 
            dest='no_cache', 
            action='store_true', 
            help='''Set true to avoid using the cached overview from IPFS.''',
            default=False,
        )
        overview_parser.add_argument(
            '--width', 
            dest='width', 
            action='store',
            type=int, 
            help='''Set the output width of the overview. Defaults to automatic width from terminal.''',
            default=None,
        )
        overview_parser.add_argument(
            '--sort_by', 
            '--wallet.sort_by',
            dest='sort_by',
            required=False,
            action='store',
            default="",
            type=str,
            help='''Sort the hotkeys by the specified column title (e.g. name, uid, axon).'''
        )
        overview_parser.add_argument(
            '--sort_order',
            '--wallet.sort_order',
            dest="sort_order",
            required=False,
            action='store',
            default="ascending",
            type=str,
            help='''Sort the hotkeys in the specified ordering. (ascending/asc or descending/desc/reverse)'''
        )
        overview_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )  
        bittensor.wallet.add_args( overview_parser )
        bittensor.subtensor.add_args( overview_parser )
        
        run_parser = cmd_parsers.add_parser(
            'run', 
            add_help=True,
            help='''Run the miner.'''
        )
        run_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        run_parser.add_argument(
            '--model', 
            type=str, 
            choices= list(bittensor.neurons.__text_neurons__.keys()), 
            default='None', 
            help='''Miners available through bittensor.neurons'''
        )

        run_parser.add_argument(
            '--synapse', 
            type=str, 
            choices= list(bittensor.synapse.__synapses_types__) + ['All'], 
            default='None', 
            help='''Synapses available through bittensor.synapse'''
        )
        run_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )
        bittensor.subtensor.add_args( run_parser )
        bittensor.wallet.add_args( run_parser )

        metagraph_parser = cmd_parsers.add_parser(
            'metagraph', 
            help='''Metagraph commands'''
        )
        metagraph_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        metagraph_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )
        bittensor.subtensor.add_args( metagraph_parser )


        help_parser = cmd_parsers.add_parser(
            'help', 
            add_help=False,
            help='''Displays the help '''
        )
        help_parser.add_argument(
            '--model', 
            type=str, 
            choices= list(bittensor.neurons.__text_neurons__.keys()), 
            default='None', 
        )
        help_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )

        update_parser = cmd_parsers.add_parser(
            'update', 
            add_help=False,
            help='''Update bittensor '''
        )
        update_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to skip prompt from update.''',
            default=False,
        )
        update_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )

        inspect_parser = cmd_parsers.add_parser(
            'inspect', 
            help='''Inspect a wallet (cold, hot) pair'''
        )
        inspect_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        inspect_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )

        bittensor.wallet.add_args( inspect_parser )
        bittensor.subtensor.add_args( inspect_parser )

        query_parser = cmd_parsers.add_parser(
            'query', 
            help='''Query a uid with your current wallet'''
        )
        query_parser.add_argument(
            "-u", '--uids',
            type=list, 
            nargs='+',
            dest='uids', 
            choices=list(range(2000)), 
            help='''Uids to query'''
        )
        query_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        query_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )

        bittensor.wallet.add_args( query_parser )
        bittensor.subtensor.add_args( query_parser )
        bittensor.dendrite.add_args( query_parser )
        bittensor.logging.add_args( query_parser )

        weights_parser = cmd_parsers.add_parser(
            'weights', 
            help='''Show weights from chain.'''
        )
        weights_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        weights_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )

        bittensor.wallet.add_args( weights_parser )
        bittensor.subtensor.add_args( weights_parser )

        set_weights_parser = cmd_parsers.add_parser(
            'set_weights', 
            help='''Setting weights on the chain.'''
        )
        set_weights_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        set_weights_parser.add_argument ("--uids", type=int, required=False, nargs='*', action='store', help="Uids to set.")
        set_weights_parser.add_argument ("--weights", type=float, required=False, nargs='*', action='store', help="Weights to set.")
        
        set_weights_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )
        bittensor.wallet.add_args( set_weights_parser )
        bittensor.subtensor.add_args( set_weights_parser )

        list_parser = cmd_parsers.add_parser(
            'list', 
            help='''List wallets'''
        )
        list_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        list_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )

        bittensor.wallet.add_args( list_parser )

        transfer_parser = cmd_parsers.add_parser(
            'transfer', 
            help='''Transfer Tao between accounts.'''
        )
        transfer_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )

        register_parser = cmd_parsers.add_parser(
            'register', 
            help='''Register a wallet to a network.'''
        )
        register_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )

        regen_coldkey_parser = cmd_parsers.add_parser(
            'regen_coldkey',
            help='''Regenerates a coldkey from a passed value'''
        )
        regen_coldkey_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )

        regen_coldkeypub_parser = cmd_parsers.add_parser(
            'regen_coldkeypub',
            help='''Regenerates a coldkeypub from the public part of the coldkey.'''
        )
        regen_coldkeypub_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )

        regen_hotkey_parser = cmd_parsers.add_parser(
            'regen_hotkey',
            help='''Regenerates a hotkey from a passed mnemonic'''
        )
        regen_hotkey_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )

        new_coldkey_parser = cmd_parsers.add_parser(
            'new_coldkey', 
            help='''Creates a new coldkey (for containing balance) under the specified path. '''
        )
        new_coldkey_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )

        new_hotkey_parser = cmd_parsers.add_parser(
            'new_hotkey', 
            help='''Creates a new hotkey (for running a miner) under the specified path.'''
        )
        new_hotkey_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )

         
        # Fill arguments for the regen coldkey command.
        regen_coldkey_parser.add_argument(
            "--mnemonic", 
            required=False, 
            nargs="+", 
            help='Mnemonic used to regen your key i.e. horse cart dog ...'
        )
        regen_coldkey_parser.add_argument(
            "--seed", 
            required=False,  
            default=None,
            help='Seed hex string used to regen your key i.e. 0x1234...'
        )
        regen_coldkey_parser.add_argument(
            '--use_password', 
            dest='use_password', 
            action='store_true', 
            help='''Set true to protect the generated bittensor key with a password.''',
            default=True,
        )
        regen_coldkey_parser.add_argument(
            '--no_password', 
            dest='use_password', 
            action='store_false', 
            help='''Set off protects the generated bittensor key with a password.''',
        )
        regen_coldkey_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        regen_coldkey_parser.add_argument(
            '--overwrite_coldkey',
            default=False,
            action='store_false',
            help='''Overwrite the old coldkey with the newly generated coldkey'''
        )
        bittensor.wallet.add_args( regen_coldkey_parser )


        regen_coldkeypub_parser.add_argument(
            "--public_key",
            "--pubkey", 
            dest="public_key_hex",
            required=False,
            default=None, 
            type=str,
            help='The public key (in hex) of the coldkey to regen e.g. 0x1234 ...'
        )
        regen_coldkeypub_parser.add_argument(
            "--ss58_address", 
            "--addr",
            "--ss58",
            dest="ss58_address",
            required=False,  
            default=None,
            type=str,
            help='The ss58 address of the coldkey to regen e.g. 5ABCD ...'
        )
        regen_coldkeypub_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        regen_coldkeypub_parser.add_argument(
            '--overwrite_coldkeypub',
            default=False,
            action='store_true',
            help='''Overwrite the old coldkeypub file with the newly generated coldkeypub'''
        )
        bittensor.wallet.add_args( regen_coldkeypub_parser )


        # Fill arguments for the regen hotkey command.
        regen_hotkey_parser.add_argument(
            "--mnemonic", 
            required=False, 
            nargs="+", 
            help='Mnemonic used to regen your key i.e. horse cart dog ...'
        )
        regen_hotkey_parser.add_argument(
            "--seed", 
            required=False,  
            default=None,
            help='Seed hex string used to regen your key i.e. 0x1234...'
        )
        regen_hotkey_parser.add_argument(
            '--use_password', 
            dest='use_password', 
            action='store_true', 
            help='''Set true to protect the generated bittensor key with a password.''',
            default=False
        )
        regen_hotkey_parser.add_argument(
            '--no_password', 
            dest='no_password', 
            action='store_false', 
            help='''Set off protects the generated bittensor key with a password.'''
        )
        regen_hotkey_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        regen_hotkey_parser.add_argument(
            '--overwrite_hotkey',
            dest='overwrite_hotkey',
            action='store_true',
            default=False,
            help='''Overwrite the old hotkey with the newly generated hotkey'''
        )
        bittensor.wallet.add_args( regen_hotkey_parser )


        # Fill arguments for the new coldkey command.
        new_coldkey_parser.add_argument(
            '--n_words', 
            type=int, 
            choices=[12,15,18,21,24], 
            default=12, 
            help='''The number of words representing the mnemonic. i.e. horse cart dog ... x 24'''
        )
        new_coldkey_parser.add_argument(
            '--use_password', 
            dest='use_password', 
            action='store_true', 
            help='''Set true to protect the generated bittensor key with a password.''',
            default=True,
        )
        new_coldkey_parser.add_argument(
            '--no_password', 
            dest='no_password', 
            action='store_false', 
            help='''Set off protects the generated bittensor key with a password.'''
        )
        new_coldkey_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        new_coldkey_parser.add_argument(
            '--overwrite_coldkey',
            action='store_false',
            default=False,
            help='''Overwrite the old coldkey with the newly generated coldkey'''
        )
        
        bittensor.wallet.add_args( new_coldkey_parser )


        # Fill arguments for the new hotkey command.
        new_hotkey_parser.add_argument(
            '--n_words', 
            type=int, 
            choices=[12,15,18,21,24], 
            default=12, 
            help='''The number of words representing the mnemonic. i.e. horse cart dog ... x 24'''
        )
        new_hotkey_parser.add_argument(
            '--use_password', 
            dest='use_password', 
            action='store_true', 
            help='''Set true to protect the generated bittensor key with a password.''',
            default=False
        )
        new_hotkey_parser.add_argument(
            '--no_password', 
            dest='no_password', 
            action='store_false', 
            help='''Set off protects the generated bittensor key with a password.'''
        )
        new_hotkey_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        new_hotkey_parser.add_argument(
            '--overwrite_hotkey',
            action='store_false',
            default=False,
            help='''Overwrite the old hotkey with the newly generated hotkey'''
        )
        bittensor.wallet.add_args( new_hotkey_parser )



        # Fill arguments for transfer
        transfer_parser.add_argument(
            '--dest', 
            dest="dest", 
            type=str, 
            required=False
        )
        transfer_parser.add_argument(
            '--amount', 
            dest="amount", 
            type=float, 
            required=False
        )
        transfer_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        bittensor.wallet.add_args( transfer_parser )
        bittensor.subtensor.add_args( transfer_parser )


        # Add register args
        register_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        register_parser.add_argument(
            '--netuid',
            type=int,
            help='netuid for subnet to serve this neuron on',
            default=argparse.SUPPRESS,
        )

        bittensor.wallet.add_args( register_parser )
        bittensor.subtensor.add_args( register_parser )

        # Fill run parser.
        run_parser.add_argument(
            '--path', 
            dest="path", 
            default=os.path.expanduser('miners/text/core_server.py'),
            type=str, 
            required=False
        )
        run_parser.add_argument(
            '--netuid',
            type=int,
            help='netuid for subnet to serve this neuron on',
            default=argparse.SUPPRESS,
        )

        become_delegate_parser = cmd_parsers.add_parser(
            'become_delegate', 
            help='''Become a delegate on the network'''
        )
        become_delegate_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        bittensor.wallet.add_args( become_delegate_parser )
        bittensor.subtensor.add_args( become_delegate_parser )

        list_delegates_parser = cmd_parsers.add_parser(
            'list_delegates', 
            help='''List all delegates on the network'''
        )
        list_delegates_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        bittensor.subtensor.add_args( list_delegates_parser )

        list_subnets_parser = cmd_parsers.add_parser(
            'list_subnets', 
            help='''List all subnets on the network'''
        )
        list_subnets_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        bittensor.subtensor.add_args( list_subnets_parser )
        
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
            cli.check_run_config( config )
        elif config.command == "transfer":
            cli.check_transfer_config( config )
        elif config.command == "register":
            cli.check_register_config( config )
        elif config.command == "unstake":
            UnStakeCommand.check_config( config )
        elif config.command == "stake":
            StakeCommand.check_config( config )
        elif config.command == "overview":
            cli.check_overview_config( config )
        elif config.command == "new_coldkey":
            cli.check_new_coldkey_config( config )
        elif config.command == "new_hotkey":
            cli.check_new_hotkey_config( config )
        elif config.command == "regen_coldkey":
            cli.check_regen_coldkey_config( config )
        elif config.command == "regen_coldkeypub":
            cli.check_regen_coldkeypub_config( config )
        elif config.command == "regen_hotkey":
            cli.check_regen_hotkey_config( config )
        elif config.command == "metagraph":
            cli.check_metagraph_config( config )
        elif config.command == "weights":
            cli.check_weights_config( config )
        elif config.command == "set_weights":
            cli.check_set_weights_config( config )
        elif config.command == "inspect":
            cli.check_inspect_config( config )
        elif config.command == "query":
            cli.check_query_config( config )
        elif config.command == "help":
            cli.check_help_config(config)
        elif config.command == "update":
            cli.check_update_config(config)
        elif config.command == "become_delegate":
            cli.check_become_delegate_config(config)
        elif config.command == "list_delegates":
            cli.check_list_delegates_config(config)
        elif config.command == "list_subnets":
            cli.check_list_subnets_config(config)


    def check_list_subnets_config( config: 'bittensor.Config'):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

    def check_list_delegates_config( config: 'bittensor.Config'):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

    def check_become_delegate_config( config: 'bittensor.Config'):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt:
            wallet_hotkey = Prompt.ask("Enter wallet hotkey", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(wallet_hotkey)

        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_coldkey = Prompt.ask("Enter wallet coldkey", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_coldkey)
    

    def check_metagraph_config( config: 'bittensor.Config'):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

        cli.__check_netuid_set( config.metagraph )

    def check_weights_config( config: 'bittensor.Config'):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

        cli.__check_netuid_set( config )

        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            if not Confirm.ask("Show all weights?"):
                wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
                config.wallet.name = str(wallet_name)
                config.all_weights = False
                if not Confirm.ask("Show all hotkeys?"):
                    hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
                    config.wallet.hotkey = str(hotkey)
                    config.all_hotkeys = False
                else:
                    config.all_hotkeys = True
            else:
                config.all_weights = True

    def check_transfer_config( config: 'bittensor.Config'):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        # Get destination.
        if not config.dest:
            dest = Prompt.ask("Enter destination public key: (ss58 or ed2519)")
            if not bittensor.utils.is_valid_bittensor_address_or_public_key( dest ):
                sys.exit()
            else:
                config.dest = str(dest)

        # Get current balance and print to user.
        if not config.no_prompt:
            wallet = bittensor.wallet( config )
            subtensor = bittensor.subtensor( config )
            with bittensor.__console__.status(":satellite: Checking Balance..."):
                account_balance = subtensor.get_balance( wallet.coldkeypub.ss58_address )
                bittensor.__console__.print("Balance: [green]{}[/green]".format(account_balance))
                    
        # Get amount.
        if not config.get('amount'):
            if not config.no_prompt:
                amount = Prompt.ask("Enter TAO amount to transfer")
                try:
                    config.amount = float(amount)
                except ValueError:
                    console.print(":cross_mark:[red] Invalid TAO amount[/red] [bold white]{}[/bold white]".format(amount))
                    sys.exit()
            else:
                console.print(":cross_mark:[red] Invalid TAO amount[/red] [bold white]{}[/bold white]".format(amount))
                sys.exit(1)


    def check_query_config( config: 'bittensor.Config' ):
        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)
                  
        if not config.uids:
            prompt = Prompt.ask("Enter uids to query [i.e. 0 10 1999]", default = 'All')
            if prompt == 'All':
                config.uids = list( range(2000) )
            else:
                try:
                    config.uids = [int(el) for el in prompt.split(' ')]
                except Exception as e:
                    console.print(":cross_mark:[red] Failed to parse uids[/red] [bold white]{}[/bold white], must be space separated list of ints".format(prompt))
                    sys.exit()

    def check_set_weights_config( config: 'bittensor.Config' ):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

        cli.__check_netuid_set( config )

        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

        if not config.uids:
            uids_str = Prompt.ask("Enter uids as list (e.g. 0, 2, 3, 4)")
            config.uids = [int(val) for val in uids_str.split(',')]

        if not config.weights:
            weights_str = Prompt.ask("Enter weights as list (e.g. 0.25, 0.25, 0.25, 0.25)")
            config.weights = [float(val) for val in weights_str.split(',')]

    def check_inspect_config( config: 'bittensor.Config' ):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

        cli.__check_netuid_set( config, allow_none = True )

        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name (optional)", default = None)
            config.wallet.hotkey = hotkey

    def check_stake_config( config: 'bittensor.Config' ):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt and not config.wallet.get('all_hotkeys') and not config.wallet.get('hotkeys'):
            hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)
                    
        # Get amount.
        if not config.get('amount') and not config.get('stake_all') and not config.get('max_stake'):
            if not Confirm.ask("Stake all Tao from account: [bold]'{}'[/bold]?".format(config.wallet.get('name', bittensor.defaults.wallet.name))):
                amount = Prompt.ask("Enter Tao amount to stake")
                try:
                    config.amount = float(amount)
                except ValueError:
                    console.print(":cross_mark:[red]Invalid Tao amount[/red] [bold white]{}[/bold white]".format(amount))
                    sys.exit()
            else:
                config.stake_all = True

    def check_overview_config( config: 'bittensor.Config' ):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

        if config.wallet.get('name') == bittensor.defaults.wallet.name  and not config.no_prompt and not config.all:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

    def _check_for_cuda_reg_config( config: 'bittensor.Config' ) -> None:
        """Checks, when CUDA is available, if the user would like to register with their CUDA device."""
        if torch.cuda.is_available():
            if not config.no_prompt:
                if config.subtensor.register.cuda.get('use_cuda') == None: # flag not set
                    # Ask about cuda registration only if a CUDA device is available.
                    cuda = Confirm.ask("Detected CUDA device, use CUDA for registration?\n")
                    config.subtensor.register.cuda.use_cuda = cuda


                # Only ask about which CUDA device if the user has more than one CUDA device.
                if config.subtensor.register.cuda.use_cuda and config.subtensor.register.cuda.get('dev_id') is None:
                    devices: List[str] = [str(x) for x in range(torch.cuda.device_count())]
                    device_names: List[str] = [torch.cuda.get_device_name(x) for x in range(torch.cuda.device_count())]
                    console.print("Available CUDA devices:")
                    choices_str: str = ""
                    for i, device in enumerate(devices):
                        choices_str += ("  {}: {}\n".format(device, device_names[i]))
                    console.print(choices_str)
                    dev_id = IntListPrompt.ask("Which GPU(s) would you like to use? Please list one, or comma-separated", choices=devices, default='All')
                    if dev_id.lower() == 'all':
                        dev_id = list(range(torch.cuda.device_count()))
                    else:
                        try:
                            # replace the commas with spaces then split over whitespace.,
                            # then strip the whitespace and convert to ints.
                            dev_id = [int(dev_id.strip()) for dev_id in dev_id.replace(',', ' ').split()]
                        except ValueError:
                            console.log(":cross_mark:[red]Invalid GPU device[/red] [bold white]{}[/bold white]\nAvailable CUDA devices:{}".format(dev_id, choices_str))
                            sys.exit(1)
                    config.subtensor.register.cuda.dev_id = dev_id
            else:
                # flag was not set, use default value.
                if config.subtensor.register.cuda.get('use_cuda') is None: 
                    config.subtensor.register.cuda.use_cuda = bittensor.defaults.subtensor.register.cuda.use_cuda

    def check_register_config( config: 'bittensor.Config' ):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

        cli.__check_netuid_set( config )

        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

        if not config.no_prompt:
            cli._check_for_cuda_reg_config(config)
            

    def check_new_coldkey_config( config: 'bittensor.Config' ):
        if config.wallet.get('name') == bittensor.defaults.wallet.name  and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

    def check_new_hotkey_config( config: 'bittensor.Config' ):
        if config.wallet.get('name') == bittensor.defaults.wallet.name  and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

    def check_regen_hotkey_config( config: 'bittensor.Config' ):
        if config.wallet.get('name') == bittensor.defaults.wallet.name  and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)
        
        if config.mnemonic == None and config.seed == None:
            prompt_answer = Prompt.ask("Enter mnemonic or seed")
            if prompt_answer.startswith("0x"):
                config.seed = prompt_answer
            else:
                config.mnemonic = prompt_answer

    def check_regen_coldkey_config( config: 'bittensor.Config' ):
        if config.wallet.get('name') == bittensor.defaults.wallet.name  and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if config.mnemonic == None and config.seed == None:
            prompt_answer = Prompt.ask("Enter mnemonic or seed")
            if prompt_answer.startswith("0x"):
                config.seed = prompt_answer
            else:
                config.mnemonic = prompt_answer

    def check_regen_coldkeypub_config( config: 'bittensor.Config' ):
        if config.wallet.get('name') == bittensor.defaults.wallet.name  and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)
        if config.ss58_address == None and config.public_key_hex == None:
            prompt_answer = Prompt.ask("Enter the ss58_address or the public key in hex")
            if prompt_answer.startswith("0x"):
                config.public_key_hex = prompt_answer
            else:
                config.ss58_address = prompt_answer
        if not bittensor.utils.is_valid_bittensor_address_or_public_key(address = config.ss58_address if config.ss58_address else config.public_key_hex):
            sys.exit(1)

    def check_run_config( config: 'bittensor.Config' ):

        # Check network.
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)
        
        cli.__check_netuid_set( config )

        if config.wallet.get('name') == bittensor.defaults.wallet.name  and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        # Check hotkey.
        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

        # Check Miner
        if config.model == 'None' and not config.no_prompt:
            model = Prompt.ask('Enter miner name', choices = list(bittensor.neurons.__text_neurons__.keys()), default = 'core_server')
            config.model = model

        if 'server' in config.model and config.get('synapse', 'None') == 'None' and not config.no_prompt:
            synapse =  Prompt.ask('Enter synapse', choices = list(bittensor.synapse.__synapses_types__) + ['All'], default = 'All')
            config.synapse = synapse

        # Don't need to ask about registration if they don't want to reregister the wallet.
        if config.wallet.get('reregister', bittensor.defaults.wallet.reregister) and not config.no_prompt:
            cli._check_for_cuda_reg_config(config)

    def __check_netuid_set( config: 'bittensor.Config', allow_none: bool = False ):
        # Make sure netuid is set.
        if config.get('netuid', 'notset') == 'notset':
            if not config.no_prompt:
                netuid = Prompt.ask("Enter netuid", default = str(bittensor.defaults.netuid) if not allow_none else 'None')
                
            else:
                netuid = str(bittensor.defaults.netuid) if not allow_none else 'None'
        else:
            netuid = config.netuid
        
        if isinstance(netuid, str) and netuid.lower() in ['none'] and allow_none:
            config.netuid = None
        else:
            try:
                config.netuid = int(netuid)
            except ValueError:
                raise ValueError('netuid must be an integer or "None" (if applicable)')
                
    def check_help_config( config: 'bittensor.Config'):
        if config.model == 'None':
            model = Prompt.ask('Enter miner name', choices = list(bittensor.neurons.__text_neurons__.keys()), default = 'core_server')
            config.model = model
    
    def check_update_config( config: 'bittensor.Config'):
        if not config.no_prompt:
            answer = Prompt.ask('This will update the local bittensor package', choices = ['Y','N'], default = 'Y')
            config.answer = answer

class IntListPrompt(PromptBase):
    """ Prompt for a list of integers. """
    
    def check_choice( self, value: str ) -> bool:
        assert self.choices is not None
        # check if value is a valid choice or all the values in a list of ints are valid choices
        return value == "All" or \
            value in self.choices or \
            all( val.strip() in self.choices for val in value.replace(',', ' ').split( ))
