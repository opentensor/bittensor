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

""" Main Validator script.

Example:
    $ python3 bittensor/_neurons/text/core_validator/main.py ...
"""

import argparse
import time
import datetime
import bittensor
import torch
import os
import wandb
import math
import random
import sys
import pandas
import traceback
from rich import print
from rich.console import Console
from rich.traceback import install
from typing import List, Tuple, Callable, Dict, Any, Union, Set

from bittensor._neuron.text.neuron_utilities import ThreadQueue, PositionalEncoding, calc_loss_fct
from bittensor._neuron.text.log_utilities import ValidatorLogger
from model import nucleus 
from bittensor.utils.tokenizer_utils import phrase_cross_entropy, topk_tokens_to_vocab_size, prune_tokens

from torch.nn.functional import kl_div
from torch.nn.utils import clip_grad_norm_
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from loguru import logger
from threading import Lock
from prometheus_client import Counter, Gauge, Histogram, Summary, Info

logger = logger.opt( colors=True )
console = Console()
install(show_locals=True)

class neuron:
    def __init__( 
        self, 
        config: 'bittensor.Config' = None,
        wallet: 'bittensor.Wallet' = None,
        subtensor: 'bittensor.Subtensor' = None,
        metagraph: 'bittensor.Metagraph' = None,
        dendrite: 'bittensor.Dendrite' = None,
        dataset: 'bittensor.dataset' = None,
        axon: 'bittensor.axon' = None,
        netuid: int = None
    ):
        # === Set up Config ===
        self.config = neuron.config() if config == None else config
        neuron.check_config( self.config )
        self.config.to_defaults()
         
        # === Create Bittensor objects ===
        bittensor.logging( config = self.config, logging_dir = self.config.neuron.full_path )
        self.vlogger = ValidatorLogger( config = self.config )
        self.subtensor = bittensor.subtensor ( config = self.config ) if subtensor == None else subtensor
        self.wallet = bittensor.wallet ( config = self.config ) if wallet == None else wallet
        self.metagraph = bittensor.metagraph ( config = self.config ) if metagraph == None else metagraph
        # self.dendrite = bittensor.dendrite ( config = self.config, wallet = self.wallet, max_active_receptors = 0 ) if dendrite == None else dendrite # Dendrite should not store receptor in validator.
        # self.axon = bittensor.axon ( netuid=self.config.netuid, config = self.config, wallet = self.wallet ) if axon == None else axon
        self.device = torch.device ( device = self.config.neuron.device )    
        self.nucleus = nucleus ( config = self.config, device = self.device, subtensor = self.subtensor, vlogger = self.vlogger ).to( self.device )
        self.dataset = bittensor.dataset(config=self.config, 
                                          batch_size=self.subtensor.validator_batch_size(self.config.netuid),
                                          block_size=self.subtensor.validator_sequence_length(self.config.netuid) + self.config.neuron.validation_len + self.subtensor.validator_prune_len(netuid=self.config.netuid)
                        )if dataset is None else dataset
        self.optimizer = torch.optim.SGD(self.nucleus.parameters(), lr=self.config.neuron.learning_rate, momentum=self.config.neuron.momentum)
        
        self.loss = None
        self.loss_agg_mutex = Lock()

        # === Neuron statistics variables ===
        self.neuron_stats = {}  # neuron statistics dict of dicts: [uid] -> {'stat1': val1, 'stat2': val2, ...}
        self.neuron_hotkeys = []  # keep neuron hotkeys to compare and check for changes after metagraph.sync()
        self.alpha = 0.1  # EMA coefficient in [0, 1], higher alpha discounts older observations faster
        self.weight_key = 'shapley_values_nxt'  # stat key + ! to calculate neuron weights with
        self.synapse_keys = ['shapley_values_nxt']

        
    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
        nucleus.check_config( config )
        bittensor.logging.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        bittensor.dataset.check_config( config )
        # bittensor.dendrite.check_config( config )
        bittensor.wandb.check_config( config )
        bittensor.axon.check_config( config )
        bittensor.prometheus.check_config( config )
        
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        config.using_wandb = config.wandb.api_key != 'default'

        subtensor = bittensor.subtensor(config)
        config.netuid = subtensor.get_subnets()[0] if config.netuid == None else config.netuid
        if not subtensor.subnet_exists( netuid = config.netuid ):
            bittensor.__console__.print(f"[red]Subnet {config.netuid} does not exist[/red]")
            sys.exit(1)
    
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)

    @classmethod
    def add_args( cls, parser ):
        parser.add_argument('--neuron.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='core_validator')
        parser.add_argument('--neuron.learning_rate', type=float, help='Training initial learning rate.', default=0.1 )
        parser.add_argument('--neuron.momentum', type=float, help='optimizer momentum.', default=0.8 )
        parser.add_argument('--neuron.blocks_per_epoch', type=int, help='Blocks per epoch, -1 value means we use the chain value.', default = -1 )
        parser.add_argument('--neuron.epochs_until_reset', type=int, help='Number of epochs before weights are reset.', default = -1 )
        parser.add_argument('--neuron.validation_len', type=int, help='Number of tokens to holdout for phrase validation beyond sequence context.', default=8)
        parser.add_argument('--neuron.prune_len', type=int, help='Number of tokens to prune from each validation input sequence.  (default value: -1, pulling from subtensor directly)', default=-1)
        parser.add_argument('--neuron.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--neuron.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0 )
        parser.add_argument('--neuron.track_hotkey_changes', action='store_true', help='If True, track hotkey changes.', default=False)
        parser.add_argument('--neuron.restart', action='store_true', help='If True, reset neuron_stats and validate anew.', default=False)
        parser.add_argument('--neuron.restart_on_failure',  action='store_true', help='''Restart neuron on unknown error.''', default=True )
        parser.add_argument('--neuron._mock', action='store_true', help='To turn on neuron mocking for testing purposes.', default=False )
        parser.add_argument('--neuron.wait_for_finalization', action='store_true', help='''when setting weights the miner waits for trnasaction finalization.''', default=False)
        parser.add_argument('--neuron.forward_num', type=int, help='''How much forward request before a backward call.''', default=3)
        parser.add_argument('--neuron.validation_synapse', type=str, help='''Synapse used for validation.''', default='TextCausalLMNext', choices = ['TextCausalLMNext', 'TextCausalLM'])
        parser.add_argument('--neuron.exclude_quantile', type=float, help='Exclude the lowest quantile from weight setting. (default value: -1, pulling from subtensor directly)', default=-1)
        parser.add_argument('--neuron.topk', type=int, help='the number of peers queried during each remote forward call', default = 20 )

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        nucleus.add_args( parser )    
        
        # Netuid Arg
        parser.add_argument('--netuid', type=int , help='Subnet netuid', default=1)

        bittensor.wallet.add_args( parser )
        # bittensor.dendrite.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.metagraph.add_args( parser )
        bittensor.logging.add_args( parser )
        bittensor.dataset.add_args( parser )
        bittensor.wandb.add_args(parser)
        bittensor.axon.add_args( parser )
        bittensor.prometheus.add_args( parser )
        return bittensor.config( parser )
    
    def __str__(self) -> str:
        return (f'[bold]UID {self.uid}[/bold] \[{self.dendrite.receptor_pool.external_ip}] '
                f'({self.wallet.name}:[bold]{self.wallet.coldkeypub.ss58_address[:7]}[/bold]/'
                f'{self.config.wallet.hotkey}:[bold]{self.wallet.hotkey.ss58_address[:7]}[/bold])')

    def __del__(self):
        if getattr(self, 'dataset', None) is not None:
            self.dataset.close()
        
        if getattr(self, 'dendrite', None) is not None:
            self.dendrite.__del__()

    def __enter__(self):
        r""" Sanity checks and begin validator.
        """
        # === Wallet ===
        # Connects wallet to network. 
        self.wallet.create()
        self.wallet.reregister( subtensor = self.subtensor, netuid=self.config.netuid )

        # === UID ===
        # Get our uid from the chain. 
        # At this point we should have a uid because we are already registered.
        self.uid = self.wallet.get_uid( subtensor = self.subtensor, netuid=self.config.netuid )    

    def create_dendrites(self):
        self.dendrites = [bittensor.text_last_hidden_state( endpoint = ep, wallet = self.wallet ) for ep in self.metagraph.endpoint_objs]
        random.shuffle(self.dendrites)

    def metagraph_sync(self):
        r""" Syncing metagraph together with other metagraph-size related objects
        """
        old_hotkeys = self.neuron_hotkeys + [] if self.neuron_hotkeys else self.metagraph.hotkeys
        self.metagraph.sync( subtensor=self.subtensor, netuid=self.config.netuid)
        self.neuron_hotkeys = self.metagraph.hotkeys

        changed_hotkeys = []
        # === Reset neuron stats if uid got replaced
        for uid, old_hotkey in enumerate(old_hotkeys):
            if old_hotkey != self.neuron_hotkeys[uid]:
                if self.config.neuron.track_hotkey_changes:
                    block = self.subtensor.block
                    self.neuron_changes.setdefault(uid, {})  # [uid] -> dict() of blocks
                    self.neuron_changes[uid][block] = {'new_hotkey': self.neuron_hotkeys[uid], 'old_hotkey': old_hotkey}
                    if uid in self.neuron_stats:
                        self.neuron_changes[uid][block]['old_stats'] = self.neuron_stats[uid]

                if uid in self.neuron_stats:
                    del self.neuron_stats[uid]
                    changed_hotkeys += [uid]


    def run(self):
        self.metagraph_sync()
        self.create_dendrites()
        count = 0
        flag = 0
        
        while True:
            scores = self.nucleus(
                inputs = next(self.dataset),
                dendrites = self.dendrites[flag:flag + self.config.neuron.topk]
            )

            if flag + self.config.neuron.topk > self.metagraph.n:
                flag = 0
                random.shuffle(self.dendrites)
            else:
                flag += self.config.neuron.topk

            count += 1
            print(count, scores)
             