
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
from rich.prompt import Prompt
from rich.table import Table
from .utils import check_netuid_set
console = bittensor.__console__

class MetagraphCommand:
    @staticmethod
    def run (cli):
        r""" Prints an entire metagraph."""
        console = bittensor.__console__
        subtensor = bittensor.subtensor( config = cli.config )
        console.print(":satellite: Syncing with chain: [white]{}[/white] ...".format(cli.config.subtensor.network))
        metagraph = subtensor.metagraph( netuid = cli.config.netuid )
        metagraph.save()
        difficulty = subtensor.difficulty( cli.config.netuid )
        subnet_emission = bittensor.Balance.from_tao(subtensor.get_emission_value_by_subnet(cli.config.netuid))
        total_issuance = bittensor.Balance.from_tao(subtensor.total_issuance())

        TABLE_DATA = [] 
        total_stake = 0.0
        total_rank = 0.0
        total_validator_trust = 0.0
        total_trust = 0.0
        total_consensus = 0.0
        total_incentive = 0.0
        total_dividends = 0.0
        total_emission = 0  
        for uid in metagraph.uids:
            ep = metagraph.endpoint_objs[uid]
            row = [
                str(ep.uid), 
                '{:.5f}'.format( metagraph.total_stake[uid]),
                '{:.5f}'.format( metagraph.ranks[uid]),
                '{:.5f}'.format( metagraph.trust[uid]), 
                '{:.5f}'.format( metagraph.consensus[uid]),
                '{:.5f}'.format( metagraph.incentive[uid]),
                '{:.5f}'.format( metagraph.dividends[uid]),
                '{}'.format( int(metagraph.emission[uid] * 1000000000)),
                '{:.5f}'.format( metagraph.validator_trust[uid]),
                '*' if metagraph.validator_permit[uid] else '',
                str((metagraph.block.item() - metagraph.last_update[uid].item())),
                str( metagraph.active[uid].item() ),
                ep.ip + ':' + str(ep.port) if ep.is_serving else '[yellow]none[/yellow]', 
                ep.hotkey[:10],
                ep.coldkey[:10]
            ]
            total_stake += metagraph.total_stake[uid]
            total_rank += metagraph.ranks[uid]
            total_validator_trust += metagraph.validator_trust[uid]
            total_trust += metagraph.trust[uid]
            total_consensus += metagraph.consensus[uid]
            total_incentive += metagraph.incentive[uid]
            total_dividends += metagraph.dividends[uid]
            total_emission += int(metagraph.emission[uid] * 1000000000)
            TABLE_DATA.append(row)
        total_neurons = len(metagraph.uids)                
        table = Table(show_footer=False)
        table.title = (
            "[white]Metagraph: net: {}:{}, block: {}, N: {}/{}, tau: {}/block, stake: {}, issuance: {}, difficulty: {}".format(subtensor.network, metagraph.netuid, metagraph.block.item(), sum(metagraph.active.tolist()), metagraph.n.item(), bittensor.Balance.from_tao(metagraph.tau.item()), bittensor.Balance.from_tao(total_stake), total_issuance, difficulty )
        )
        table.add_column("[overline white]UID",  str(total_neurons), footer_style = "overline white", style='yellow')
        table.add_column("[overline white]STAKE(\u03C4)", '\u03C4{:.5f}'.format(total_stake), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]RANK", '{:.5f}'.format(total_rank), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]TRUST", '{:.5f}'.format(total_trust), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]CONSENSUS", '{:.5f}'.format(total_consensus), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]INCENTIVE", '{:.5f}'.format(total_incentive), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]DIVIDENDS", '{:.5f}'.format(total_dividends), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]EMISSION(\u03C1)", '\u03C1{}'.format(int(total_emission)), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]VTRUST", '{:.5f}'.format(total_validator_trust), footer_style = "overline white", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]VAL", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]UPDATED", justify='right', no_wrap=True)
        table.add_column("[overline white]ACTIVE", justify='right', style='green', no_wrap=True)
        table.add_column("[overline white]AXON", justify='left', style='dim blue', no_wrap=True) 
        table.add_column("[overline white]HOTKEY", style='dim blue', no_wrap=False)
        table.add_column("[overline white]COLDKEY", style='dim purple', no_wrap=False)
        table.show_footer = True

        for row in TABLE_DATA:
            table.add_row(*row)
        table.box = None
        table.pad_edge = False
        table.width = None
        console.print(table)

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        check_netuid_set( config, subtensor = bittensor.subtensor( config = config ) )

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        metagraph_parser = parser.add_parser(
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
