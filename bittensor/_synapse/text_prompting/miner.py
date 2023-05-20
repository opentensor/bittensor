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
import time
import copy
import torch
import argparse
import bittensor

from rich import print
from typing import List, Dict, Union, Tuple, Optional
from datetime import datetime
from abc import ABC, abstractmethod

class BasePromptingMiner( ABC ):

    @classmethod
    @abstractmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        ...

    def priority( self, forward_call: "bittensor.TextPromptingForwardCall" ) -> float:
        if self.metagraph is not None:
            uid = self.metagraph.hotkeys.index(forward_call.src_hotkey)
            return self.metagraph.S[uid].item()
        else:
            return self.config.neuron.default_priority

    def blacklist( self, forward_call: "bittensor.TextPromptingForwardCall" ) -> Union[ Tuple[bool, str], bool ]:
        def is_registered() -> bool:
            """
            Return true if the hotkey is registered or if the miner doesn't require it.
            """
            if self.config.neuron.blacklist.allow_non_registered:
                return True
            
            hotkey_registered = forward_call.src_hotkey in self.metagraph.hotkeys
            return hotkey_registered

        def has_vpermit() -> bool:
            """
            Return true if the neuron querying this miner has a vpermit or if the miner doesn't require one.
            """
            if self.config.neuron.blacklist.vpermit_required:
                hotkey_registered = forward_call.src_hotkey in self.metagraph.hotkeys
                if hotkey_registered:
                    uid = self.metagraph.hotkeys.index(forward_call.src_hotkey)
                    return self.metagraph.neurons[uid].validator_permit
                return False
            return True


        # Blacklist based on stake.
        def enough_stake() -> bool:
            """
            Returns true if required stake is <= 0 or <= the neuron's stake, otherwise false.
            """
            required_stake = self.config.neuron.blacklist.default_stake
            if required_stake <= 0.0:
                return True
            
            hotkey_registered = forward_call.src_hotkey in self.metagraph.hotkeys
            if hotkey_registered:
                uid = self.metagraph.hotkeys.index(forward_call.src_hotkey)
                if required_stake <= self.metagraph.S[uid].item():
                    return True
            return False

        # Optionally blacklist based on checks.
        try:
            checks = [
                (is_registered(), "Key is not registered"),
                (enough_stake(), "Key doesn't have enough stake"),
                (has_vpermit(), "Key doesn't have a vpermit"),
            ]
            for passed, error_message in checks:
                if not passed:
                    return True, error_message

            return False, 'passed blacklist'
        except Exception as e:
            bittensor.logging.warning( "Blacklisted. Error in `registration_check` or `stake_check()" )
            return True, 'Error in `registration_check` or `stake_check()'

    @abstractmethod
    def forward( self, messages: List[Dict[str, str]] ) -> str:
        ...

    @classmethod
    @abstractmethod
    def check_config( cls, config: 'bittensor.Config' ):
        ...

    @classmethod
    def config( cls ) -> "bittensor.Config":
        parser = argparse.ArgumentParser()
        cls.add_super_args( parser )
        return bittensor.config( parser )

    @classmethod
    def help( cls ):
        parser = argparse.ArgumentParser()
        cls.add_super_args( parser )
        cls.add_args(parser)
        print( cls.__new__.__doc__ )
        parser.print_help()

    @classmethod
    def super_check_config( cls, config: "bittensor.Config" ):
        cls.check_config( config )
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
    def add_super_args( cls, parser: argparse.ArgumentParser ):
        cls.add_args(parser)
        parser.add_argument(
            '--netuid',
            type = int,
            help = 'Subnet netuid',
            default = 41
        )
        parser.add_argument(
            '--neuron.name',
            type = str,
            help = 'Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ',
            default = 'openai_prompting_miner'
        )
        parser.add_argument(
            '--neuron.blocks_per_epoch',
            type = str,
            help = 'Blocks until the miner sets weights on chain',
            default = 100
        )
        parser.add_argument(
            '--neuron.no_set_weights',
            action = 'store_true',
            help = 'If True, the model does not set weights.',
            default = False
        )
        parser.add_argument(
            '--neuron.max_batch_size',
            type = int,
            help = 'The maximum batch size for forward requests.',
            default = -1
        )
        parser.add_argument(
            '--neuron.max_sequence_len',
            type = int,
            help = 'The maximum sequence length for forward requests.',
            default = -1
        )
        parser.add_argument(
            '--neuron.blacklist.hotkeys',
            type = str,
            required = False,
            nargs = '*',
            action = 'store',
            help = 'To blacklist certain hotkeys', default=[]
        )
        parser.add_argument(
            '--neuron.blacklist.allow_non_registered',
            action = 'store_true',
            help = 'If True, this miner will allow non-registered hotkeys to query it.',
            default = True
        )
        parser.add_argument(
            '--neuron.blacklist.default_stake',
            type = float,
            help = 'Set default stake for miners.',
            default = 0.0
        )
        parser.add_argument(
            '--neuron.blacklist.vpermit_required',
            action="store_true",
            help = 'Require vpermit to query this miner.',
            default = False
        )
        parser.add_argument(
            '--neuron.default_priority',
            type = float,
            help = 'Set default priority for miners.',
            default = 0.0
        )
        bittensor.wallet.add_args( parser )
        bittensor.axon.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.logging.add_args( parser )

    def __init__(
        self,
        config: "bittensor.Config" = None
    ):
        config = config if config != None else self.config()
        self.config = copy.deepcopy( config )
        self.super_check_config( self.config )
        self.config.to_defaults()
        bittensor.logging( config = self.config, logging_dir = self.config.neuron.full_path )
        self.subtensor = bittensor.subtensor( self.config )
        self.wallet = bittensor.wallet( self.config )
        self.metagraph = self.subtensor.metagraph( self.config.netuid )
        self.axon = bittensor.axon(
            wallet = self.wallet,
            metagraph = self.metagraph,
            config = self.config,
        )
        class Synapse( bittensor.TextPromptingSynapse ):
            def priority( _, forward_call: "bittensor.TextPromptingForwardCall" ) -> float:
                return self.priority( forward_call )
            def blacklist( _, forward_call: "bittensor.TextPromptingForwardCall" ) -> Union[ Tuple[bool, str], bool ]:
                return self.blacklist( forward_call )
            def backward( self, messages: List[Dict[str, str]], response: str, rewards: torch.FloatTensor ) -> str: pass
            def forward( _, messages: List[Dict[str, str]] ) -> str:
                return self.forward( messages )
        self.synapse = Synapse( axon = self.axon )

    def run( self ):

        # --- Start the miner.
        self.wallet.reregister( netuid = self.config.netuid, subtensor = self.subtensor )
        self.axon.start()
        self.axon.netuid = self.config.netuid
        self.axon.protocol = 4
        self.subtensor.serve_axon( netuid = self.config.netuid, axon = self.axon )

        # --- Run Forever.
        last_update = self.subtensor.get_current_block()
        while True:

            # --- Wait until next epoch.
            current_block = self.subtensor.get_current_block()
            while (current_block - last_update) < self.config.neuron.blocks_per_epoch:
                time.sleep( 0.1 ) #bittensor.__blocktime__
                current_block = self.subtensor.get_current_block()
            last_update = self.subtensor.get_current_block()

            # --- Update the metagraph with the latest network state.
            self.metagraph.sync( lite = True )
            uid = self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )

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
                        uids=torch.arange(0, len(chain_weights)),
                        netuid=self.config.netuid,
                        weights=chain_weights,
                        wait_for_inclusion=False,
                        wallet=self.wallet,
                        version_key=1
                    )
                except:
                    pass