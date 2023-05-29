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
from typing import List, Dict, Union, Tuple, Optional
from datetime import datetime

class BaseValidator:

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
        bittensor.subtensor.add_args( parser, prefix = prefix )
        bittensor.logging.add_args( parser, prefix = prefix )

    def __init__(self, netuid: int = None, config: "bittensor.Config" = None ):
        # Build config.
        self.config = config if config != None else BaseValidator.config()
        self.config.netuid = netuid or self.config.netuid
        BaseValidator.check_config( self.config )

        # Build objects.
        bittensor.logging( config = self.config, logging_dir = self.config.neuron.full_path )
        self.subtensor = bittensor.subtensor( self.config )
        self.wallet = bittensor.wallet( self.config )
        self.metagraph = self.subtensor.metagraph( self.config.netuid )

        # Used for backgounr process.
        self.is_running = False
        self.should_exit = False 
        self.background_thread = None

    def __enter__(self):
        bittensor.logging.trace( 'BaseValidator.__enter__()' )
        self.start_in_background()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        bittensor.logging.trace( 'BaseValidator.__exit__()' )
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
        bittensor.logging.debug( 'BaseMinBaseValidatorerNeuron.run()' )

        # --- Start the miner.
        self.is_running = True
        self.wallet.reregister( netuid = self.config.netuid, subtensor = self.subtensor )

        # --- Run Forever.
        last_update = self.subtensor.get_current_block()
        while not self.should_exit:

            # --- Wait until next epoch.
            current_block = self.subtensor.get_current_block()
            while (current_block - last_update) < self.config.neuron.blocks_per_epoch:
                if self.should_exit: continue
                time.sleep( 12 )
                current_block = self.subtensor.get_current_block()
            last_update = self.subtensor.get_current_block()

            # --- Update the metagraph with the latest network state.
            self.metagraph.sync( lite = True )
            uid = self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )

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