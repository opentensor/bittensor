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

import argparse
import bittensor
from rich.table import Table
from rich.prompt import Prompt
console = bittensor.__console__

class ListDelegatesCommand:

    @staticmethod
    def run( cli ):
        r"""
        List all delegates on the network.
        """
        subtensor = bittensor.subtensor( config = cli.config )
        delegates: bittensor.DelegateInfo = subtensor.get_delegates()

        table = Table(show_footer=True, width=cli.config.get('width', None), pad_edge=False, box=None)
        table.add_column("[overline white]DELEGATE",  str(len(delegates)), footer_style = "overline white", style='bold white')
        table.add_column("[overline white]TAKE", style='white')
        table.add_column("[overline white]OWNER", style='yellow')
        table.add_column("[overline white]NOMINATORS", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]TOTAL STAKE(\u03C4)", justify='right', style='green', no_wrap=True)

        for delegate in delegates:
            table.add_row(
                str(delegate.hotkey_ss58),
                str(delegate.take),
                str(delegate.owner_ss58),
                str(delegate.nominators),
                str(delegate.total_stake),
            )
        bittensor.__console__.print(table)

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        list_delegates_parser = parser.add_parser(
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

    @staticmethod   
    def check_config( config: 'bittensor.Config' ):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)






      