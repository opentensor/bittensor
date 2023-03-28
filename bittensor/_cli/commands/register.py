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
from rich.prompt import Prompt, Confirm
from .utils import check_netuid_set, check_for_cuda_reg_config

console = bittensor.__console__

class RegisterCommand:

    @staticmethod
    def run( cli ):
        r""" Register neuron. """
        wallet = bittensor.wallet( config = cli.config )
        subtensor = bittensor.subtensor( config = cli.config )

        # Verify subnet exists
        if not subtensor.subnet_exists( netuid = cli.config.netuid ):
            bittensor.__console__.print(f"[red]Subnet {cli.config.netuid} does not exist[/red]")
            sys.exit(1)

        subtensor.register(
            wallet = wallet,
            netuid = cli.config.netuid,
            prompt = not cli.config.no_prompt,
            TPB = cli.config.subtensor.register.cuda.get('TPB', None),
            update_interval = cli.config.subtensor.register.get('update_interval', None),
            num_processes = cli.config.subtensor.register.get('num_processes', None),
            cuda = cli.config.subtensor.register.cuda.get('use_cuda', bittensor.defaults.subtensor.register.cuda.use_cuda),
            dev_id = cli.config.subtensor.register.cuda.get('dev_id', None),
            output_in_place = cli.config.subtensor.register.get('output_in_place', bittensor.defaults.subtensor.register.output_in_place),
            log_verbose = cli.config.subtensor.register.get('verbose', bittensor.defaults.subtensor.register.verbose),
        )


    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        register_parser = parser.add_parser(
            'register', 
            help='''Register a wallet to a network.'''
        )
        register_parser.add_argument( 
            '--no_version_checking', 
            action='store_true', 
            help='''Set false to stop cli version checking''', 
            default = False 
        )
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

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        check_netuid_set( config, subtensor = bittensor.subtensor( config = config ) )

        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)

        if not config.no_prompt:
            check_for_cuda_reg_config(config)

class RecycleRegisterCommand:

    @staticmethod
    def run( cli ):
        r""" Register neuron by recycling some TAO. """
        wallet = bittensor.wallet( config = cli.config )
        subtensor = bittensor.subtensor( config = cli.config )

        # Verify subnet exists
        if not subtensor.subnet_exists( netuid = cli.config.netuid ):
            bittensor.__console__.print(f"[red]Subnet {cli.config.netuid} does not exist[/red]")
            sys.exit(1)

        # Check current recycle amount
        current_recycle = subtensor.burn( netuid = cli.config.netuid )
        balance = subtensor.get_balance( address = wallet.coldkeypub.ss58_address )

        # Check balance is sufficient
        if balance < current_recycle:
            bittensor.__console__.print(f"[red]Insufficient balance {balance} to register neuron. Current recycle is {current_recycle} TAO[/red]")
            sys.exit(1)

        if not cli.config.no_prompt:
            if Confirm.ask(f"Your balance is: [bold green]{balance}[/bold green]\nThe cost to register by recycle is [bold red]{current_recycle}[/bold red]\nDo you want to continue?", default = False) == False:
                sys.exit(1)
        
        subtensor.burned_register(
            wallet = wallet,
            netuid = cli.config.netuid,
            prompt = not cli.config.no_prompt
        )


    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        recycle_register_parser = parser.add_parser(
            'recycle_register', 
            help='''Register a wallet to a network.'''
        )
        recycle_register_parser.add_argument( 
            '--no_version_checking', 
            action='store_true', 
            help='''Set false to stop cli version checking''', 
            default = False 
        )
        recycle_register_parser.add_argument(
            '--no_prompt', 
            dest='no_prompt', 
            action='store_true', 
            help='''Set true to avoid prompting the user.''',
            default=False,
        )
        recycle_register_parser.add_argument(
            '--netuid',
            type=int,
            help='netuid for subnet to serve this neuron on',
            default=argparse.SUPPRESS,
        )

        bittensor.wallet.add_args( recycle_register_parser )
        bittensor.subtensor.add_args( recycle_register_parser )

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

        check_netuid_set( config, subtensor = bittensor.subtensor( config = config ) )

        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name", default = bittensor.defaults.wallet.hotkey)
            config.wallet.hotkey = str(hotkey)
      