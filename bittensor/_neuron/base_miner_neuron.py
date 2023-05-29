# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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
import time
import torch
import threading
import argparse
import bittensor

from rich import print
from typing import Union, Tuple
from datetime import datetime

class BaseMinerNeuron:

    def priority( self, forward_call: "bittensor.SynapseCall" ) -> float:
        return self.prioritizer.priority( forward_call, metagraph = self.metagraph )

    def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
        return self.blacklister.blacklist( forward_call, metagraph = self.metagraph )
    
    @classmethod
    def config( cls ) -> "bittensor.Config":
        parser = argparse.ArgumentParser()
        cls.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def help( cls ):
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        print( cls.__new__.__doc__ )
        parser.print_help()

    @classmethod
    def check_config( cls, config: "bittensor.Config" ):
        bittensor.axon.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.logging.check_config( config )
        bittensor.subtensor.check_config( config )
        full_path = os.path.expanduser(
            '{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.get('name', bittensor.defaults.wallet.name),
                                  config.wallet.get('hotkey', bittensor.defaults.wallet.hotkey), config.neuron.name ) )
        config.neuron.full_path = os.path.expanduser( full_path )
        if not os.path.exists( config.neuron.full_path ):
            os.makedirs( config.neuron.full_path )

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser, prefix: str = None ):
        prefix_str = "" if prefix is None else prefix + "."
        parser.add_argument(
            '--' + prefix_str + 'netuid', 
            type = int, 
            help = 'Subnet netuid', 
            default = 1
        )
        parser.add_argument(
            '--' + prefix_str + 'neuron.name', 
            type = str,
            help = 'Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ',
            default = 'openai_prompting_miner'
        )
        parser.add_argument(
            '--' + prefix_str + 'neuron.blocks_per_epoch', 
            type = str, 
            help = 'Blocks until the miner sets weights on chain',
            default = 100
        )
        parser.add_argument(
            '--' + prefix_str + 'neuron.no_set_weights', 
            action = 'store_true', 
            help = 'If True, the model does not set weights.',
            default = False
        )
        bittensor.wallet.add_args( parser, prefix = prefix )
        bittensor.axon.add_args( parser, prefix = prefix )
        bittensor.subtensor.add_args( parser, prefix = prefix )
        bittensor.logging.add_args( parser, prefix = prefix )
        bittensor.blacklist.add_args( parser, prefix = prefix_str + 'neuron' )
        bittensor.priority.add_args( parser, prefix = prefix_str + 'neuron' )

    def __init__(self, netuid: int = None, config: "bittensor.Config" = None ):
        # Build config.
        self.config = config if config != None else BaseMinerNeuron.config()
        self.config.netuid = netuid or self.config.netuid
        BaseMinerNeuron.check_config( self.config )

        # Build objects.
        bittensor.logging( config = self.config, logging_dir = self.config.neuron.full_path )
        self.subtensor = bittensor.subtensor( self.config )
        self.wallet = bittensor.wallet( self.config )
        self.metagraph = self.subtensor.metagraph( self.config.netuid )
        self.axon = bittensor.axon( wallet = self.wallet, config = self.config )
        self.blacklister = bittensor.blacklist( config = self.config.neuron )
        self.prioritizer = bittensor.priority( config = self.config.neuron )

        # Used for backgounr process.
        self.is_running = False
        self.should_exit = False 
        self.background_thread = None

    def attach( self, synapse: "bittensor.Synapse" ):
        # pass through attach function.
        self.axon.attach( synapse )

    def __enter__(self):
        bittensor.logging.trace( 'BaseMinerNeuron.__enter__()' )
        self.start_in_background()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        bittensor.logging.trace( 'BaseMinerNeuron.__exit__()' )
        self.stop()

    def start_in_background(self):
        if self.is_running:
            bittensor.logging.warning( 'The base miner neuron is already running.')
        else:
            self.should_exit = False
            self.background_thread = threading.Thread( target = self.run, daemon = True )
            self.background_thread.start()
            self.is_running = True
            bittensor.logging.trace( 'Starting the base miner neuron in the background.')

    def stop(self):
        if self.is_running:
            self.should_exit = True
        else:
            bittensor.logging.warning( 'The base miner neuron is not running.')

    def run( self ):
        bittensor.logging.debug( 'BaseMinerNeuron.run()' )

        # --- Start the miner.
        self.is_running = True
        self.wallet.reregister( netuid = self.config.netuid, subtensor = self.subtensor )
        self.axon.start()
        self.subtensor.serve_axon( netuid = self.config.netuid, axon = self.axon, wait_for_finalization = False, wait_for_inclusion = False ) #TODO: fix finalization & inclusion

        # --- Run Forever.
        last_update = self.subtensor.get_current_block()
        retries = 0
        while not self.should_exit:

            # --- Wait until next epoch.
            current_block = self.subtensor.get_current_block()
            while (current_block - last_update) < self.config.neuron.blocks_per_epoch:
                if self.should_exit: continue
                time.sleep( 0.1 ) #bittensor.__blocktime__
                current_block = self.subtensor.get_current_block()
            last_update = self.subtensor.get_current_block()

            # --- Update the metagraph with the latest network state.
            try:
                self.metagraph.sync( lite = True )
                uid = self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )
            except:
                # --- If we fail to sync the metagraph, wait and try again.
                if(retries > 8):
                    bittensor.logging.error( f'Failed to sync metagraph, exiting.')
                    self.stop()
                    break 
                seconds_to_sleep = 5 * 1.5**(retries)
                bittensor.logging.error( f'Failed to sync metagraph, retrying in {seconds_to_sleep} seconds.')
                time.sleep( seconds_to_sleep )
                retries += 1
                continue

            if(retries > 0):
                retries = 0

            # --- Log performance.
            print(
                f"[white not bold]{datetime.now():%Y-%m-%d %H:%M:%S}[/white not bold]{' ' * 4} | "
                f"{f'UID [bright_cyan]{uid}[/bright_cyan]'.center(16 + len('[bright_cyan][/bright_cyan]'))} | "
                f'[dim white not bold] [green]{str(self.metagraph.S[uid].item()):.4}[/green] Stake [/dim white not bold]'
                f'[dim white not bold]| [yellow]{str(self.metagraph.trust[uid].item()) :.3}[/yellow] Trust [/dim white not bold]'
                f'[dim white not bold]| [green]{str(self.metagraph.incentive[uid].item()):.3}[/green] Incentive [/dim white not bold]')

            # --- Set weights.
            if not self.config.neuron.no_set_weights:
                try:
                    # --- query the chain for the most current number of peers on the network
                    chain_weights = torch.zeros( self.subtensor.subnetwork_n( netuid = self.config.netuid ))
                    chain_weights[uid] = 1
                    did_set = self.subtensor.set_weights(
                        uids = torch.arange(0, len(chain_weights)),
                        netuid = self.config.netuid,
                        weights = chain_weights,
                        wait_for_inclusion = False,
                        walle = self.wallet,
                        version_key = 1
                    )
                except:
                    pass

        self.axon.stop()