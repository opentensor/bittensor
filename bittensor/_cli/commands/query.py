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
console = bittensor.__console__

class QueryCommand:
    @staticmethod
    def run (cli):
        wallet = bittensor.wallet(config = cli.config)
        subtensor = bittensor.subtensor( config = cli.config )

        # Verify subnet exists
        if not subtensor.subnet_exists( netuid = cli.config.netuid ):
            bittensor.__console__.print(f"[red]Subnet {cli.config.netuid} does not exist[/red]")
            sys.exit(1)

        dendrite = bittensor.dendrite( wallet = wallet )
        stats = {}
        for uid in cli.config.uids:
            neuron = subtensor.neuron_for_uid( uid = uid, netuid = cli.config.netuid )
            endpoint = bittensor.endpoint.from_neuron( neuron )
            _, c, t = dendrite.forward_text( endpoints = endpoint, inputs = 'hello world')
            latency = "{}".format(t.tolist()[0]) if c.tolist()[0] == 1 else 'N/A'
            bittensor.__console__.print("\tUid: [bold white]{}[/bold white]\n\tLatency: [bold white]{}[/bold white]\n\tCode: [bold {}]{}[/bold {}]\n\n".format(uid, latency, bittensor.utils.codes.code_to_loguru_color( c.item() ), bittensor.utils.codes.code_to_string( c.item() ), bittensor.utils.codes.code_to_loguru_color( c.item() )), highlight=True)
            stats[uid] = latency
        print (stats)

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
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

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        query_parser = parser.add_parser(
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
