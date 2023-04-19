# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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
import bittensor
from rich.prompt import Prompt
from rich.table import Table
from rich.prompt import Confirm
from .utils import check_netuid_set
console = bittensor.__console__

class WeightsCommand:
    @staticmethod
    def run (cli):
        r""" Prints an weights to screen.
        """
        console = bittensor.__console__
        subtensor = bittensor.subtensor( config = cli.config )
        wallet = bittensor.wallet( config = cli.config )
        metagraph = subtensor.metagraph( netuid = cli.config.get('netuid') )
        metagraph.save()

        table = Table()
        rows = []
        table.add_column("[bold white]uid", style='white', no_wrap=False)
        for uid in metagraph.uids.tolist():
            table.add_column("[bold white]{}".format(uid), style='white', no_wrap=False)
            if cli.config.all_weights:
                rows.append(["[bold white]{}".format(uid) ] + ['{:.3f}'.format(v) for v in metagraph.W[uid].tolist()])
            else:
                if metagraph.coldkeys[uid] == wallet.coldkeypub.ss58_address:
                    if not cli.config.all_hotkeys:
                        if metagraph.hotkeys[uid] == wallet.hotkey.ss58_address:
                            rows.append(["[bold white]{}".format(uid) ] + ['{:.3f}'.format(v) for v in metagraph.W[uid].tolist()])
                    else:
                        rows.append(["[bold white]{}".format(uid) ] + ['{:.3f}'.format(v) for v in metagraph.W[uid].tolist()])

        for row in rows:
            table.add_row(*row)
        table.box = None
        table.pad_edge = False
        table.width = None
        with console.pager():
            console.print(table)

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

        check_netuid_set( config, subtensor = bittensor.subtensor( config = config ) )

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

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        weights_parser = parser.add_parser(
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
