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
from rich.prompt import Prompt
from .utils import check_netuid_set
console = bittensor.__console__

class InspectCommand:
    @staticmethod
    def run (cli):
        r""" Inspect a cold, hot pair.
        """
        wallet = bittensor.wallet(config = cli.config)
        subtensor = bittensor.subtensor( config = cli.config )

        if cli.config.netuid != None:
            # Verify subnet exists
            if not subtensor.subnet_exists( netuid = cli.config.netuid ):
                bittensor.__console__.print(f"[red]Subnet {cli.config.netuid} does not exist[/red]")
                sys.exit(1)

            
        with bittensor.__console__.status(":satellite: Looking up account on: [white]{}[/white] ...".format(cli.config.subtensor.get('network', bittensor.defaults.subtensor.network))):
            
            if cli.config.wallet.get('hotkey', bittensor.defaults.wallet.hotkey) is None:
                # If no hotkey is provided, inspect just the coldkey
                wallet.coldkeypub
                cold_balance = wallet.get_balance( subtensor = subtensor )
                bittensor.__console__.print("\n[bold white]{}[/bold white]:\n  {}[bold white]{}[/bold white]\n {} {}\n".format( wallet, "coldkey:".ljust(15), wallet.coldkeypub.ss58_address, " balance:".ljust(15), cold_balance.__rich__()), highlight=True)

            else:
                wallet.hotkey
                wallet.coldkeypub

                if cli.config.netuid != None:
                    # If a netuid is provided, inspect the hotkey and the neuron
                    dendrite = bittensor.dendrite( wallet = wallet )
                    neuron = subtensor.get_neuron_for_pubkey_and_subnet( ss58_hotkey = wallet.hotkey.ss58_address, netuid = cli.config.netuid )
                    endpoint = bittensor.endpoint.from_neuron( neuron )
                    if neuron.is_null:
                        registered = '[bold white]No[/bold white]'
                        stake = bittensor.Balance.from_tao( 0 )
                        emission = bittensor.Balance.from_rao( 0 )
                        latency = 'N/A'
                    else:
                        registered = '[bold white]Yes[/bold white]'
                        stake = bittensor.Balance.from_tao( neuron.total_stake )
                        emission = bittensor.Balance.from_rao( neuron.emission * 1000000000 )
                        synapses = [bittensor.synapse.TextLastHiddenState()]
                        _, c, t = dendrite.text( endpoints = endpoint, inputs = 'hello world', synapses=synapses)
                        latency = "{}".format((t[0]).tolist()[0]) if (c[0]).tolist()[0] == 1 else 'N/A'

                    cold_balance = wallet.get_balance( subtensor = subtensor )
                    bittensor.__console__.print((
                        "\n[bold white]{}[/bold white]:\n  [bold grey]{}[bold white]{}[/bold white]\n" + \
                        "  {}[bold white]{}[/bold white]\n  {}{}\n  {}{}\n  {}{}\n  {}{}\n  {}{}[/bold grey]"
                    )
                    .format(
                        wallet,
                        "coldkey:".ljust(15),
                        wallet.coldkeypub.ss58_address,
                        "hotkey:".ljust(15),
                        wallet.hotkey.ss58_address,
                        "registered:".ljust(15),
                        registered,
                        "balance:".ljust(15),
                        cold_balance.__rich__(),
                        "stake:".ljust(15),
                        stake.__rich__(),
                        "emission:".ljust(15),
                        emission.__rich_rao__(),
                        "latency:".ljust(15),
                        latency 
                    ), highlight=True)
                else:
                    # Otherwise, print all subnets the hotkey is registered on.
                    # If a netuid is provided, inspect the hotkey and the neuron
                    stake = subtensor.get_stake_for_coldkey_and_hotkey( hotkey_ss58 = wallet.hotkey.ss58_address, coldkey_ss58 = wallet.coldkeypub.ss58_address )
                    if stake == None:
                        # Not registered on any subnets
                        subnets = "[bold white][][/bold white]"
                        stake = bittensor.Balance.from_tao( 0 )
                    else:
                        # Registered on subnets
                        subnets_registered = subtensor.get_netuids_for_hotkey( ss58_hotkey = wallet.hotkey.ss58_address )
                        subnets = f'[bold white]{subnets_registered}[/bold white]'
                        
                        emission = bittensor.Balance.from_rao( 0 )
                        for netuid in subnets_registered:
                            neuron = subtensor.neuron_for_pubkey( ss58_hotkey = wallet.hotkey.ss58_address, netuid = netuid )
                            emission += bittensor.Balance.from_rao( neuron.emission * 1000000000 )

                    cold_balance = wallet.get_balance( subtensor = subtensor )
                    bittensor.__console__.print((
                        "\n[bold white]{}[/bold white]:\n  [bold grey]{}[bold white]{}[/bold white]\n" + \
                        "  {}[bold white]{}[/bold white]\n  {}{}\n  {}{}\n  {}{}\n  {}{}\n  {}{}[/bold grey]"
                    )
                    .format(
                        wallet,
                        "coldkey:".ljust(15),
                        wallet.coldkeypub.ss58_address,
                        "hotkey:".ljust(15),
                        wallet.hotkey.ss58_address,
                        "subnets:".ljust(15),
                        subnets,
                        "balance:".ljust(15),
                        cold_balance.__rich__(),
                        "stake:".ljust(15),
                        stake.__rich__(),
                        "emission:".ljust(15),
                        emission.__rich_rao__(),
                    ), highlight=True)


    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

        check_netuid_set( config, allow_none = True )

        if config.wallet.get('name') == bittensor.defaults.wallet.name and not config.no_prompt:
            wallet_name = Prompt.ask("Enter wallet name", default = bittensor.defaults.wallet.name)
            config.wallet.name = str(wallet_name)

        if config.wallet.get('hotkey') == bittensor.defaults.wallet.hotkey and not config.no_prompt:
            hotkey = Prompt.ask("Enter hotkey name (optional)", default = None)
            config.wallet.hotkey = hotkey

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        inspect_parser = parser.add_parser(
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