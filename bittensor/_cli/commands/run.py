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
from rich.prompt import Prompt, Confirm
from .utils import check_netuid_set, check_for_cuda_reg_config
console = bittensor.__console__

class RunCommand:
    def run ( cli ):
        cli.config.to_defaults()
        subtensor = bittensor.subtensor( config = cli.config )

        # Verify subnet exists
        if not subtensor.subnet_exists( netuid = cli.config.netuid ):
            bittensor.__console__.print(f"[red]Subnet {cli.config.netuid} does not exist[/red]")
            sys.exit(1)

        # Check coldkey.
        wallet = bittensor.wallet( config = cli.config )
        if not wallet.coldkeypub_file.exists_on_device():
            if Confirm.ask("Coldkey: [bold]'{}'[/bold] does not exist, do you want to create it".format(cli.config.wallet.get('name', bittensor.defaults.wallet.name))):
                wallet.create_new_coldkey()
            else:
                sys.exit()

        # Check hotkey.
        if not wallet.hotkey_file.exists_on_device():
            if Confirm.ask("Hotkey: [bold]'{}'[/bold] does not exist, do you want to create it".format(cli.config.wallet.hotkey)):
                wallet.create_new_hotkey()
            else:
                sys.exit()

        if wallet.hotkey_file.is_encrypted():
            bittensor.__console__.print("Decrypting hotkey ... ")
        wallet.hotkey

        if wallet.coldkeypub_file.is_encrypted():
            bittensor.__console__.print("Decrypting coldkeypub ... ")
        wallet.coldkeypub

        # Check registration
        ## Will exit if --wallet.reregister is False
        if cli.config.wallet.get('reregister', bittensor.defaults.wallet.reregister) and not cli.config.no_prompt and not wallet.is_registered(netuid = cli.config.netuid):
            console.print("Wallet not registered.")
            check_for_cuda_reg_config(wallet.config)
            print(wallet.config)
            
        wallet.reregister( subtensor=subtensor, netuid = cli.config.netuid )

        # Run miner.
        if cli.config.model == 'core_server':
            if cli.config.synapse == 'TextLastHiddenState':
                bittensor.neurons.core_server.neuron(lasthidden=True, causallm=False, seq2seq = False, netuid = cli.config.netuid).run()
            elif cli.config.synapse == 'TextCausalLM':
                bittensor.neurons.core_server.neuron(lasthidden=False, causallm=True, seq2seq = False, netuid = cli.config.netuid).run()
            elif cli.config.synapse == 'TextSeq2Seq':
                bittensor.neurons.core_server.neuron(lasthidden=False, causallm=False, seq2seq = True, netuid = cli.config.netuid).run()
            else:
                bittensor.neurons.core_server.neuron(netuid = cli.config.netuid).run()

        elif cli.config.model == 'core_validator':
            bittensor.neurons.core_validator.neuron(netuid = cli.config.netuid).run()
        elif cli.config.model == 'multitron_server':
            bittensor.neurons.multitron_server.neuron(netuid = cli.config.netuid).run()
   
    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        # Check network.        
        check_netuid_set( config, subtensor = bittensor.subtensor( config = config ) )

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


    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        run_parser = parser.add_parser(
            'run', 
            add_help=True,
            help='''Run the miner.'''
        )
        run_parser.add_argument( '--no_version_checking', action='store_true', help='''Set false to stop cli version checking''', default = False )
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
        bittensor.subtensor.add_args( run_parser )
        bittensor.wallet.add_args( run_parser )
