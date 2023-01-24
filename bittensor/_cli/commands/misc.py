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

import os
import sys
import argparse
import bittensor
from typing import List
from rich.prompt import Prompt
from rich.table import Table
console = bittensor.__console__

class HelpCommand:
    @staticmethod
    def run (cli):
        cli.config.to_defaults()
        sys.argv = [sys.argv[0], '--help']
        # Run miner.
        if cli.config.model == 'core_server':
            bittensor.neurons.core_server.neuron().run()
        elif cli.config.model == 'core_validator':
            bittensor.neurons.core_validator.neuron().run()
        elif cli.config.model == 'multitron_server':
            bittensor.neurons.multitron_server.neuron().run()

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        if config.model == 'None':
            model = Prompt.ask('Enter miner name', choices = list(bittensor.neurons.__text_neurons__.keys()), default = 'core_server')
            config.model = model

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        help_parser = parser.add_parser(
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

class UpdateCommand:
    @staticmethod
    def run (cli):
        if cli.config.no_prompt or cli.config.answer == 'Y':
            os.system(' (cd ~/.bittensor/bittensor/ ; git checkout master ; git pull --ff-only )')
            os.system('pip install -e ~/.bittensor/bittensor/')

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        if not config.no_prompt:
            answer = Prompt.ask('This will update the local bittensor package', choices = ['Y','N'], default = 'Y')
            config.answer = answer

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        update_parser = parser.add_parser(
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

class ListSubnetsCommand:
    @staticmethod
    def run (cli):
        r"""List all subnet netuids in the network. """
        subtensor = bittensor.subtensor( config = cli.config )
        subnets: List[bittensor.SubnetInfo] = subtensor.get_all_subnets_info()

        rows = []
        total_neurons = 0
        
        for subnet in subnets:
            total_neurons += subnet.max_n
            rows.append((
                str(subnet.netuid),
                str(subnet.subnetwork_n),
                str(subnet.max_n),
                str(bittensor.utils.registration.millify(subnet.difficulty)) + "M",
                str(subnet.immunity_period),
                str(subnet.validator_batch_size),
                str(subnet.validator_sequence_length),
                str(subnet.tempo),
                str(subnet.modality),
                str(list(subnet.connection_requirements.keys())),
                str(subnet.emission_value),
            ))

        table = Table(show_footer=True, width=cli.config.get('width', None), pad_edge=False, box=None)
        table.title = (
            "[white]Subnets - {}".format(subtensor.network)
        )
        table.add_column("[overline white]NETUID",  str(len(subnets)), footer_style = "overline white", style='bold white')
        table.add_column("[overline white]N", str(total_neurons), footer_style = "overline white", style='white')
        table.add_column("[overline white]MAX_N", style='white')
        table.add_column("[overline white]DIFFICULTY", style='white')
        table.add_column("[overline white]IMMUNITY", style='white')
        table.add_column("[overline white]BATCH SIZE", style='white')
        table.add_column("[overline white]SEQ_LEN", style='white')
        table.add_column("[overline white]TEMPO", style='white')
        table.add_column("[overline white]MODALITY", style='white')
        table.add_column("[overline white]CON_REQ", style='white')
        table.add_column("[overline white]EMISSION", "1.0", style='white', footer_style="overline white") # sums to 1.0
        
        for row in rows:
            table.add_row(*row)

        bittensor.__console__.print(table)

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        if config.subtensor.get('network') == bittensor.defaults.subtensor.network and not config.no_prompt:
            config.subtensor.network = Prompt.ask("Enter subtensor network", choices=bittensor.__networks__, default = bittensor.defaults.subtensor.network)

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        list_subnets_parser = parser.add_parser(
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
        