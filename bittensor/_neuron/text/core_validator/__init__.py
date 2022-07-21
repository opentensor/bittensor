#!/bin/python3
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
""" The bittensor base validator

Example:
    $ python3 miners/text/core_validator.py --logging.debug

"""
import sys
import argparse
import time
from types import SimpleNamespace
import bittensor
import torch
import os
import wandb
import math
import pandas
import random
import traceback
from rich import print
from rich.console import Console
from rich.style import Style
from rich.table import Table
from rich.traceback import install
from typing import List, Tuple, Callable, Dict, Any, Union

from ..neuron_utilities import ThreadQueue, PositionalEncoding, calc_loss_fct
from bittensor.utils.tokenizer_utils import unravel_topk_token_phrases, phrase_cross_entropy

import torch.nn as nn
import random
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from loguru import logger
from threading import Lock

logger = logger.opt( colors=True )
console = Console()
install(show_locals=True)

# Neuron stats recorded by validator neuron/nucleus
#   [Column_name, key_name, format_string, rich_style]  # description
neuron_stats_columns = [
    ['UID', 'uid', '{:.0f}', 'cyan'],  # neuron UID
    ['Upd!', 'updates!', '{}', 'bright_yellow'],  # number of exponential moving average updates with zeroing on
    ['mUpd', 'updates_shapley_values_min', '{}', 'bright_yellow'],  # number of exponential moving average updates to mShap
    ['nUpd', 'updates_shapley_values_nxt', '{}', 'bright_yellow'],  # number of exponential moving average updates to nShap
    ['sTime', 'response_time', '{:.2f}', 'yellow'],  # response time to TextCausalLM forward requests
    ['nTime', 'response_time_nxt', '{:.2f}', 'yellow'],  # response time to TextCausalLMNext forward requests
    ['Route', 'routing_score', '{:.3f}', 'grey30'],  # validator routing score (higher preferred)
    ['Weight', 'weight', '{:.5f}', 'green'],  # weight set on substrate (each epoch)
    ['mShap!', 'shapley_values_min!', '{:.0f}', 'bright_magenta'],  # min(Shap, vShap) of sequence and validation Shapley (zeroing)
    ['mShap', 'shapley_values_min', '{:.0f}', 'bright_magenta'],  # min(Shap, vShap) of sequence and validation Shapley
    ['sLoss', 'loss', '{:.2f}', 'bright_cyan'],  # next token prediction loss average over sequence
    ['vLoss', 'loss_val', '{:.2f}', 'bright_cyan'],  # next token prediction loss for validation task
    ['nLoss', 'loss_nxt', '{:.2f}', 'bright_cyan'],  # next token phrase prediction loss for phrase validation task
    ['RLoss', 'routing_loss', '{:.3f}', 'grey30'],  # MSE between routing_score and conditioned loss
    ['nRLoss', 'routing_loss_nxt', '{:.3f}', 'grey30'],  # MSE between routing_score_nxt and conditioned loss
    ['sShap', 'shapley_values', '{:.0f}', 'magenta'],  # Shapley value (=Base+Syn) over sequence
    ['vShap', 'shapley_values_val', '{:.0f}', 'magenta'],  # Shapley value (=vBase+vSyn) for validation
    ['nShap!', 'shapley_values_nxt!', '{:.0f}', 'magenta'],  # Shapley value (=vBase+vSyn) for phrase validation (zeroing)
    ['nShap', 'shapley_values_nxt', '{:.0f}', 'magenta'],  # Shapley value (=vBase+vSyn) for phrase validation
    ['sBase', 'base_params', '{:.0f}', ''],  # parameter count estimate via adjusted scaling law
    ['vBase', 'base_params_val', '{:.0f}', ''],  # parameter count estimate for validation task
    ['nBase', 'base_params_nxt', '{:.0f}', ''],  # parameter count estimate for phrase validation task
    ['sSyn', 'synergy', '{:.0f}', 'white'],  # Shapley pairwise synergy over sequence loss (parameter count estimate)
    ['vSyn', 'synergy_val', '{:.0f}', 'white'],  # Shapley pairwise synergy over validation loss (count estimate)
    ['nSyn', 'synergy_nxt', '{:.0f}', 'white'],  # Shapley pairwise synergy over phrase validation loss (count estimate)
    ['sSynD', 'synergy_loss_diff', '{:.2f}', 'bright_blue'],  # Shapley pairwise synergy over sequence loss (loss difference)
    ['vSynD', 'synergy_loss_diff_val', '{:.2f}', 'bright_blue'],  # Shapley pairwise synergy over validation loss (loss difference)
    ['nSynD', 'synergy_loss_diff_nxt', '{:.2f}', 'bright_blue'],  # Shapley pairwise synergy over phrase validation loss (loss difference)
]


class neuron:
    r"""
    Creates a bittensor neuron that specializes validating other peers. The core validator
    finetunes on the bittensor network with a mixture of experts model and shapely scoring.
    The validator's main jobs are to identify important/useful peers in the network and correctly
    weight them. To achieve this, the validator will send requests to different peers on the network
    and evalute their responses.

    Args: 
            config (:obj:`bittensor.Config`, `optional`): 
                bittensor.server.config()
            subtensor (:obj:bittensor.subtensor , `optional`):
                bittensor subtensor connection
            dataset (:obj:bittensor.dataset , `optional`):
                bittensor dataset 
            wallet (:obj:bittensor.wallet, `optional`):
                bittensor wallet object
            metagraph (:obj:bittensor.metagraph, `optional`):
                bittensor metagraph object
            dendrite (:obj:bittensor.dendrite, `optional`):
                bittensor dendrite object
            dataset (:obj:bittensor.dendrite, `optional`):
                bittensor dendrite object
    Examples:: 
            >>> subtensor = bittensor.subtensor(network='nakamoto')
            >>> validator = bittensor.neuron.text.core_validator.neuron(subtensor=subtensor)
            >>> validator.run()
    """
    def __init__( 
        self, 
        config: 'bittensor.Config' = None,
        wallet: 'bittensor.Wallet' = None,
        subtensor: 'bittensor.Subtensor' = None,
        metagraph: 'bittensor.Metagraph' = None,
        dendrite: 'bittensor.Dendrite' = None,
        dataset: 'bittensor.dataset' = None
    ):

        # === Set up Config ===
        if config == None: config = neuron.config()
        self.config = config
        neuron.check_config( self.config )
        self.config.to_defaults()
        if self.config.neuron._mock == True:
            self.config.subtensor._mock = True
            self.config.wallet._mock = True
            self.config.dataset._mock = True
            self.config.dendrite._mock = True
            self.config.metagraph._mock = True
            self.config.subtensor._mock = True
        print ( self.config )

        # === Create Bittensor objects ===
        bittensor.logging( config = self.config, logging_dir = self.config.neuron.full_path )
        self.wallet = bittensor.wallet ( config = self.config ) if wallet == None else wallet
        self.subtensor = bittensor.subtensor ( config = self.config ) if subtensor == None else subtensor
        self.metagraph = bittensor.metagraph ( config = self.config, subtensor = self.subtensor ) if metagraph == None else metagraph
        self.dendrite = bittensor.dendrite ( config = self.config, wallet = self.wallet ) if dendrite == None else dendrite
        self.device = torch.device ( device = self.config.neuron.device )    
        self.nucleus = nucleus ( config = self.config, device = self.device, subtensor = self.subtensor ).to( self.device )
        self.dataset = (bittensor.dataset(config=self.config, batch_size=self.subtensor.validator_batch_size,
                                          block_size=self.subtensor.validator_sequence_length + self.config.neuron.validation_len)
                        if dataset is None else dataset)
        self.optimizer = torch.optim.SGD(
            self.nucleus.parameters(), lr=self.config.neuron.learning_rate, momentum=self.config.neuron.momentum
        )

        # === Create thread queue ===
        self.forward_thread_queue = ThreadQueue(num_jobs = self.config.neuron.forward_num, target = self.forward)
        self.loss = None
        self.loss_agg_mutex = Lock()

        # === Neuron statistics variables ===
        self.alpha = 0.05  # EMA coefficient in [0, 1], higher alpha discounts older observations faster
        self.weight_key = 'shapley_values_min'  # stat key + ! to calculate neuron weights with
        # stat keys to duplicate (['key']->['key!']) and push zero to its EMA if neuron non-responsive
        self.synapse_keys = ['shapley_values_min', 'shapley_values_nxt']
        self.neuron_stats = {}

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
        bittensor.dendrite.check_config( config )
        bittensor.wandb.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        config.using_wandb = config.wandb.api_key != 'default'
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
        parser.add_argument('--neuron.device', type=str, help='miner default training device cpu/cuda', default=("cuda" if torch.cuda.is_available() else "cpu"))
        parser.add_argument('--neuron.clip_gradients', type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.', default=1.0 )
        parser.add_argument('--neuron.restart_on_failure',  action='store_true', help='''Restart neuron on unknown error.''', default=True )
        parser.add_argument('--neuron._mock', action='store_true', help='To turn on neuron mocking for testing purposes.', default=False )
        parser.add_argument('--neuron.wait_for_finalization', action='store_true', help='''when setting weights the miner waits for trnasaction finalization.''', default=False)
        parser.add_argument('--neuron.forward_num', type=int, help='''How much forward request before a backward call.''', default=3)

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        nucleus.add_args( parser )        
        bittensor.wallet.add_args( parser )
        bittensor.dendrite.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.metagraph.add_args( parser )
        bittensor.logging.add_args( parser )
        bittensor.dataset.add_args( parser )
        bittensor.wandb.add_args(parser)
        return bittensor.config( parser )

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (f'[bold]UID {self.uid}[/bold] \[{self.dendrite.receptor_pool.external_ip}] '
                f'({self.wallet.name}:[bold]{self.wallet.coldkeypub.ss58_address[:7]}[/bold]/'
                f'{self.config.wallet.hotkey}:[bold]{self.wallet.hotkey.ss58_address[:7]}[/bold])')

    def __del__(self):
        self.__exit__()

    def __exit__ ( self, exc_type, exc_value, exc_traceback ):
        r""" Close down neuron.
        """
        print(exc_type, exc_value, exc_traceback)
        self.dataset.close()
        self.dendrite.__del__()
        self.forward_thread_queue.stop()
        self.forward_thread_queue.join()

    def __enter__(self):
        r""" Sanity checks and begin validator.
        """
        # === Wallet ===
        # Connects wallet to network. 
        self.wallet.create()
        # NOTE: This registration step should likely be solved offline first.
        self.wallet.reregister( subtensor = self.subtensor )


        # === UID ===
        # Get our uid from the chain. 
        # At this point we should have a uid because we are already registered.
        self.uid = self.wallet.get_uid( subtensor = self.subtensor )    

        # === Monitoring ===
        # Optionally set up wandb logging.
        if self.config.using_wandb:
            bittensor.wandb(
                config = self.config,
                cold_pubkey = self.wallet.coldkeypub.ss58_address,
                hot_pubkey = self.wallet.hotkey.ss58_address,
                root_dir = self.config.neuron.full_path
            )

    def forward(self):
        r""" Run the nucleus forward request
        This function is supposed to be ran multi-threaded.
        """
        loss, stats = self.nucleus( next(self.dataset) , self.metagraph, self.dendrite )

        # === Backward ===
        # Backwards gradients through model to train gating and remote endpoints.
        if hasattr(loss, 'grad_fn') and loss.grad_fn is not None:
            print(f'Backward \t| Loss: {loss:.3f} ... backpropagation ... ', end='')
            start_time = time.time()
            (loss / self.config.neuron.forward_num).backward()
            print(f'complete [{time.time() - start_time:.3g}s]')

        return loss, stats

    def run ( self ):
        r""" Run the validator and terminate on Keyboard interrupt.
        """
        # === Setup ===
        # Checks wallet and starts monitoring with wandb.
        with self:

            # === Start forward requests ===
            self.metagraph_sync()
            self.forward_thread_queue.start()
            
            # === Run ===
            # Iterates through epochs.
            self.epoch = 0
            self.global_step = 0
            while True:
                try:

                    # === Epoch ===
                    # Each epoch runs for blocks_per_epoch and resets
                    # the model every epochs_until_reset.
                    self.run_epoch()

                # === Stops on interrupt otherwise restarts ===
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    console.print_exception(show_locals=False)
                    print( traceback.format_exc() )
                    print( 'Unknown exception: {}', e )
                    if not self.config.neuron.restart_on_failure:
                        break

    def run_epoch( self ):
        r""" Runs a validator epoch. We apply batches until the epoch length is exhausted.
            Occasionally the validator nucleus is completely reset to ensure we dont converge to far.
            At the end of the epoch we set weights on the chain and optionally log to wandb.
        """
        # === Get params for epoch ===
        # Pulling the latest chain parameters.
        current_block = self.subtensor.block
        batch_size = self.subtensor.validator_batch_size 
        sequence_length = self.subtensor.validator_sequence_length
        validation_len = self.config.neuron.validation_len  # Number of tokens to holdout for phrase validation beyond sequence context
        n_topk_peer_weights = self.subtensor.min_allowed_weights
        max_allowed_ratio = self.subtensor.max_allowed_min_max_ratio
        blocks_per_epoch = self.subtensor.validator_epoch_length if self.config.neuron.blocks_per_epoch == -1 else self.config.neuron.blocks_per_epoch
        epochs_until_reset = self.subtensor.validator_epochs_per_reset if self.config.neuron.epochs_until_reset == -1 else self.config.neuron.epochs_until_reset

        # === Update dataset size ===
        if (batch_size != self.dataset.batch_size) or (sequence_length + validation_len != self.dataset.block_size):
            self.dataset.set_data_size(batch_size, sequence_length + validation_len)

        # === Logs ===
        if self.config.using_wandb:
            wandb.log({'era/batch_size': batch_size, 'era/sequence_length': sequence_length,
                       'era/validation_len': validation_len,
                       'era/n_topk_peer_weights': n_topk_peer_weights, 'era/max_allowed_ratio': max_allowed_ratio,
                       'era/blocks_per_epoch': blocks_per_epoch, 'era/epochs_until_reset': epochs_until_reset},
                      step=current_block)

        # === Run Epoch ===
        # Each block length lasts blocks_per_epoch blocks.
        # This gives us a consistent network wide timer.
        # Here we run until blocks_per_epochs have progressed.
        self.metagraph_sync() # Reset metagraph.
        epoch_steps = 0

        start_block = self.subtensor.block
        while self.subtensor.block < start_block + blocks_per_epoch:
            start_time = time.time()

            # === Forward ===
            # Forwards inputs through the network and returns the loss
            # and endpoint scores using shapely approximation of salience.
            loss, stats = self.forward_thread_queue.get()

            # === Stats update ===
            # Updates moving averages and history.
            self.neuron_stats_update(stats)

            # === State update ===
            # Prints step logs to screen.
            epoch_steps += 1
            self.global_step += 1
            current_block = self.subtensor.block
            step_time = time.time() - start_time

            print(f'UID {self.uid}   \t| '
                  f'Updated {current_block - self.metagraph.last_update[self.uid]} [white]blocks ago[/white] | '
                  f'Dividends {self.metagraph.dividends[self.uid]:.5f} | '
                  f'Stake \u03C4{self.metagraph.stake[self.uid]:.5f} '
                  f'[dim](retrieved {current_block - start_block} blocks ago from {self.subtensor.network})[/dim]')

            # === Print stats update (table) ===
            # Prints exponential moving average statistics of valid neurons from latest validator forward
            stats_table({uid: self.neuron_stats[uid]
                         for uid, stat in stats.items() if len(set(stat.keys()) & set(self.synapse_keys))},
                        self.weight_key, self.config.get('width', None),
                        f'[white] Stats update [/white] | ' + str(self),  # title
                        f'#{current_block}: '
                        f'[bold]{current_block - start_block}[/bold]/{blocks_per_epoch} (blocks/epoch) | '
                        f'Epoch {self.epoch} | '
                        f'[white] Step {epoch_steps} ({self.global_step} global) \[{step_time:.3g}s] [/white]')  # caption

            # === Calculate neuron weights ===
            topk_uids, topk_weights = self.calculate_weights()
            self.weights_table(topk_uids, topk_weights)  # print weights table

            # === Logs ===
            if self.config.using_wandb:
                wandb.log({'epoch/epoch': self.epoch, 'epoch/epoch_steps': epoch_steps,
                           'epoch/global_steps': self.global_step, 'epoch/loss': loss.item(),
                           'epoch/time': step_time}, step=current_block)
                for uid, vals in self.neuron_stats.items():
                    for key in vals:  # detailed neuron evaluation fields, e.g. loss, shapley_values, synergy
                        wandb.log({f'stats/{key}_{uid}': vals[key]}, step=current_block)

            # Do the backward request after the a queue of forward requests got finished.  
            if self.forward_thread_queue.paused() and self.forward_thread_queue.is_empty():
                start_time = time.time()
                print('Model update \t| Optimizer step ... ', end='')

                # === Apply gradients ===
                # Applies local gradients to parameters.
                clip_grad_norm_(self.nucleus.parameters(), self.config.neuron.clip_gradients)
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(f'complete \[{time.time() - start_time:.3g}s]')
                
                # === Get another round of forward requests ===
                self.forward_thread_queue.resume()

        # Iterate epochs.
        self.epoch += 1

        # === Calculate neuron weights ===
        topk_uids, topk_weights = self.calculate_weights()
        self.weights_table(topk_uids, topk_weights)  # print weights table

        self.subtensor.set_weights(
            uids = topk_uids.detach().to('cpu'),
            weights = topk_weights.detach().to('cpu'),
            wallet = self.wallet,
            wait_for_finalization = self.config.neuron.wait_for_finalization,
        )

        # === Wandb Logs ===
        # Optionally send validator logs to wandb.
        if self.config.using_wandb:
            # Logging history to wandb.
            df = pandas.concat( [
                bittensor.utils.indexed_values_to_dataframe( prefix = 'weights', index = topk_uids, values = torch.zeros( self.metagraph.n ).scatter( dim = 0, src = topk_weights, index = topk_uids ) ),
                self.dendrite.to_dataframe( metagraph = self.metagraph )
            ], axis = 1); df['uid'] = df.index
            wandb_data_dend = self.dendrite.to_wandb()
            wandb_data = { 'stake': self.metagraph.S[ self.uid ].item(), 'dividends': self.metagraph.D[ self.uid ].item() } 
            wandb.log( { 'stats': wandb.Table( dataframe = df ) }, step = current_block )
            wandb.log( { **wandb_data, **wandb_data_dend }, step = current_block )

    def metagraph_sync(self):
        r""" Syncing metagraph together with other metagraph-size related objects
        """
        old_hotkeys = self.metagraph.hotkeys 
        self.metagraph.sync()

        # === Reset neuron stats if uid got replaced
        for uid, old_hotkey in enumerate(old_hotkeys):
            if old_hotkey != self.metagraph.hotkeys[uid]:
                if uid in self.neuron_stats:
                    del self.neuron_stats[uid]

    def neuron_stats_update(self, neuron_stats: Dict[int, Dict[str, Any]]):
        r""" Updates self.neuron_stats with new individual dictionaries per uid.
        """
        for _uid, _stats in neuron_stats.items():
            stats = self.neuron_stats.setdefault(_uid, {})

            # === EMA zeroing update ===
            # Push zero into EMA for synapse_keys to exponentially decay weighting keys if neuron non-responsive
            if 'updates!' in stats:
                stats['updates!'] += 1  # increment number of EMA zeroing updates
            else:
                stats.setdefault('updates!', 1)  # number of EMA zeroing updates init to zero

            for key in self.synapse_keys:
                zkey = key + '!'
                if zkey in stats:
                    if key in _stats:
                        stats[zkey] = (1 - self.alpha) * stats[zkey] + self.alpha * _stats[key]
                    else:
                        stats[zkey] = (1 - self.alpha) * stats[zkey]  # + self.alpha * 0
                else:
                    if key in _stats:
                        stats[zkey] = _stats[key]
                    else:
                        stats.setdefault(zkey, 0.)

            # === EMA normal update ===
            # If synapse responsive push available values into EMA for normal update.
            # Normal EMA values provide a view on neuron performance if fully responsive.
            for key in self.synapse_keys:
                if key in _stats:
                    updates = 'updates_' + key
                    if updates in stats:
                        stats[updates] += 1  # increment number of normal EMA updates made
                    else:
                        stats.setdefault(updates, 1)  # add updates fields for new uid entries

            for key in _stats:  # detailed neuron evaluation fields, e.g. loss, shapley_values, synergy
                if key in stats:
                    stats[key] = (1 - self.alpha) * stats[key] + self.alpha * _stats[key]  # update EMA
                else:
                    stats.setdefault(key, _stats[key])

    def calculate_weights(self):
        r""" Calculates neuron set-weights from weight_key mapped values. Defines weight_key as the neuron stats key
        used to obtain the mapped stat value (typically a Shapley value) that the final set-weights are calculated from.
        """

        weight_key = self.weight_key + '!'  # use zeroing key to penalize non-responsive neurons
        n_topk_peer_weights = self.subtensor.min_allowed_weights
        max_allowed_ratio = self.subtensor.max_allowed_min_max_ratio

        # === Calculate neuron weights ===
        neuron_weights = torch.zeros_like(self.metagraph.S)  # allow unevaluated UIDs to be selected to meet min topk

        for uid in self.neuron_stats:
            if weight_key in self.neuron_stats[uid]:
                neuron_weights[uid] = torch.tensor([self.neuron_stats[uid][weight_key]])

        # Find the n_topk_peer_weights peers to set weights to.
        topk_weights, topk_uids = bittensor.unbiased_topk(neuron_weights, k=n_topk_peer_weights)
        topk_weights = bittensor.utils.weight_utils.normalize_max_multiple(x=topk_weights,
                                                                           multiple=max_allowed_ratio)
        return topk_uids, topk_weights

    def weights_table(self, topk_uids, topk_weights):
        r""" Prints weights table given topk_uids and topk_weights.
        """
        n_topk_peer_weights = self.subtensor.min_allowed_weights
        max_allowed_ratio = self.subtensor.max_allowed_min_max_ratio

        # === Weight table ===
        # Prints exponential moving average statistics of valid neurons and latest weights
        _neuron_stats = {}
        unvalidated = []
        for uid, weight in zip(topk_uids.tolist(), topk_weights.tolist()):
            if uid in self.neuron_stats:
                _neuron_stats[uid] = {k: v for k, v in self.neuron_stats[uid].items()}
                _neuron_stats[uid]['weight'] = weight
            else:
                unvalidated += [uid]

        print()
        stats_table(_neuron_stats, 'weight', self.config.get('width', None),
                    f'[white] Neuron weights [/white] | ' + str(self),  # title
                    f'Validated [bold]{(n_topk_peer_weights - len(unvalidated))}[/bold]'
                    f'/{n_topk_peer_weights}/{self.metagraph.n} (valid/min/total) | '
                    f'sum:{topk_weights.sum().item():.2g} '
                    f'[white] max:[bold]{topk_weights.max().item():.4g}[/bold] / '
                    f'min:[bold]{topk_weights.min().item():.4g}[/bold] [/white] '
                    f'\[{topk_weights.max().item() / topk_weights.min().item():.1f}:1] '
                    f'({max_allowed_ratio} allowed)')  # caption


class nucleus( torch.nn.Module ):
    """ Nucleus class which holds the validator model.
    """
    def __init__( self, config, device, subtensor ):
        super(nucleus, self).__init__()
        self.config = config
        self.device = device
        self.max_n = subtensor.max_n 
        tokenizer = bittensor.tokenizer()
        self.pad_token = tokenizer(tokenizer.pad_token)['input_ids'][0]

        # Token embeddings project int64 tokens onto representations.
        self.token_embedding = torch.nn.Embedding( bittensor.__vocab_size__,  bittensor.__network_dim__ )
        
        # Routing encoder, projects token embeddings onto context for routing inputs.
        self.routing_encoder_layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout, batch_first=True)
        self.routing_encoder = TransformerEncoder( self.routing_encoder_layers, 1 )

        # Encoder projects response representations onto hidden units.
        self.encoder_layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout, batch_first=True)
        self.encoder = TransformerEncoder( self.encoder_layers, config.nucleus.nlayers )

        # Decoder which projects hidden unit representations on to the token dimension.
        self.decoder = torch.nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__ , bias=False)

        # Positional Encoding
        self.local_pos_encoder = PositionalEncoding( bittensor.__network_dim__, self.config.nucleus.dropout )

        # Crosss entropy loss for NTP.    
        self.loss_fct = torch.nn.CrossEntropyLoss()
    
        # SGMOE Gates: Instantiating the gates per expert.
        self.gates = torch.nn.Linear( bittensor.__network_dim__, self.max_n, bias=True ).to( self.device )

        self.sigmoid = torch.nn.Sigmoid()

        self.reset_weights()

    @classmethod
    def add_args( cls, parser ):
        parser.add_argument('--nucleus.topk', type=int, help='the number of peers queried during each remote forward call', default = 20 )
        parser.add_argument('--nucleus.nhid', type=int, help='the dimension of the feedforward network model in nn.TransformerEncoder', default=200 )
        parser.add_argument('--nucleus.nhead', type=int, help='the number of heads in the multiheadattention models', default = 2 )
        parser.add_argument('--nucleus.nlayers', type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=2 )
        parser.add_argument('--nucleus.dropout', type=float, help='the dropout value', default=0.2)
        parser.add_argument('--nucleus.importance', type=float, help='hyperparameter for the importance loss', default=3)
        parser.add_argument('--nucleus.noise_multiplier', type=float, help='Standard deviation multipler on weights', default=2 )
        parser.add_argument('--nucleus.dendrite_backward', action='store_true', help='Pass backward request to the server side or not', default=False )

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    def reset_weights ( self ):
        r""" Resets the validator weights.
        """
        # === Resets all the weights using xavier initialization. ===
        torch.nn.init.xavier_uniform_ ( self.token_embedding.weight )
        torch.nn.init.xavier_uniform_ ( self.decoder.weight )
        torch.nn.init.xavier_uniform_( self.gates.weight )
        def init_xavier( component ):
            try:
                torch.nn.init.xavier_uniform_( component.weight )
            except: pass
        self.routing_encoder.apply( init_xavier )
        self.encoder.apply( init_xavier )
        torch.nn.init.xavier_uniform_( self.gates.weight )

    def forward(
            self,
            inputs: torch.FloatTensor,
            metagraph: 'bittensor.Metagraph',
            dendrite: 'bittensor.Dendrite',
    ):
        r"""
        Forward validator pass. Selects endpoints to query and validate, calculates routing_score and Shapley values
        for validated synapses.
            Args:
                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, *-1*)`, `required`): 
                    Tensor inputs to distribute to neurons using query context.
                metagraph (bittensor.Metagraph):
                    Metagraph object used to query network information.
                dendrite (bittensor.Dendrite):
                    Dendrite RPC client used to make network queries.
            Returns:
                loss (:obj:`torch.FloatTensor`):
                    Loss for training validator nucleus and dendrite backward to endpoints.
                neuron_stats (:obj:`Dict`, `required`):
                    Statistics per endpoint for this batch.
        """
        start_time = time.time()
        print(f'Forward \t| Model forward ... ', end='')

        val_len = self.config.neuron.validation_len  # Number of tokens to holdout for phrase validation beyond sequence context
        inputs = inputs.to(self.device)
        inputs_seq = inputs[..., :-val_len]  # input sequence without last validation tokens [batch_size, sequence_len]

        # === Create the local context used to select endpoints ===
        # The context tensor returns a hidden unit representation for the text inputs
        # this context can be used as input to the gates in the next step.
        # embedding: retrieve learned representation vectors for input vocabulary tokens.
        # inputs.shape = [batch_size, sequence_len]
        # embedding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        embedding = self.token_embedding(inputs_seq) * math.sqrt(bittensor.__network_dim__)

        # === Create an attention mask ===
        # The attention mask will mask out parts of the context
        # This prevents cheating and forward-looking when predicting each token in the sequence.
        # src_mask: (torch.FloatTensor) attention mask adds -inf to positions not allowed to attend
        # src_mask.shape = [sequence_len, sequence_len]
        src_mask = torch.triu(torch.ones(embedding.size(1), embedding.size(1)) * float('-inf'), diagonal=1)
        src_mask = src_mask.to(self.device)

        # === Apply the positional encoding to help select endpoints ===
        # The positional encoder provides information based on the relative postion of each token
        # embedding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        # pos_embedding: (torch.FloatTensor) positional encoded embedding.
        # pos_embedding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        pos_embedding = self.local_pos_encoder(embedding)

        # routing_context: (torch.FloatTensor): context tensor which is used to select endpoints.
        # routing_context.shape = [ batch size, __network_dim__ ]
        routing_context = self.routing_encoder(pos_embedding, mask=src_mask)

        # === Get gate values for UIDs. ===
        # We iterate over each of the network UIDs and compute a querying score for each
        # using the gating function. This returns a score per endpoint per example.
        # routing_score: (torch.FloatTensor): score per example, per endpoint.
        # routing_score.shape = [metagraph.n]
        # The gates act over the last embedding of the routing_context.
        routing_score = torch.mean(self.sigmoid(self.gates(routing_context[:, -1, :])), dim=0)

        # Ensure number of queried neurons does not exceed metagraph.n
        num_endpoints = min([self.config.nucleus.topk, metagraph.n])

        print(f'complete \[{time.time() - start_time:.3g}s]')
        print(f'Dendrite \t| Request {num_endpoints} x {list(inputs_seq.shape)} ... ', end='')
        request_start_time = time.time()

        # === Randomly select num_endpoints UIDs ===
        random_uids = torch.randperm(metagraph.n)[:num_endpoints]

        # === Get endpoint information for the selected UIDs ===
        # We index into the metagraph's endpoints and return a list of the filtered set of endpoints we wish to query.
        # random_endpoints: List[bittensor.endpoints]: endpoint information for filtered uids.
        # len(neurons) == self.config.nucleus.topk
        random_endpoints = [metagraph.endpoints[uid] for uid in random_uids]

        # === Define which synapse we want to use ===
        # The synapse defines the task we are sending to the neurons
        # synapses: List[bittensor.synapse]: synapse information
        # TODO: WORK IN PROGRESS, prototype
        synapses = [(bittensor.synapse.TextCausalLM(), textcausallm),
                    (bittensor.synapse.TextCausalLMNext(), textcausallmnext)]

        # === Query the endpoints ===
        # Makes the dendrite call into the network returning the representations
        # for each of the endpoints. The return ops can be used to filter weights and outputs.
        # query_responses: (List[torch.float64]): responses from each endpoint.
        # query_responses.shape = self.config.nucleus.topk * num_synapses * [batch_size, sequence_len, synapse_dim]
        # return_ops: (torch.int64): Return ops.
        # return_ops.shape = self.config.nucleus.topk * [num_synapses]
        # TODO: WORK IN PROGRESS, prototype
        query_responses, return_ops, times = dendrite.text(
            endpoints=random_endpoints,
            inputs=inputs_seq,
            synapses=[syn for syn, _ in synapses],
            timeout=bittensor.__blocktime__
        )

        if not self.config.nucleus.dendrite_backward:
            query_responses = [[syn.detach().to(self.device) for syn in res] for res in query_responses]
            return_ops = [ops.detach().to(self.device) for ops in return_ops]
            times = [t.detach().to(self.device) for t in times]

        # Send responses to device. This is required to ensure we move the responses
        # Onto the correct device.
        for responses in query_responses:
            for response in responses:
                response.to(self.device)

        print(f'complete \[{time.time() - request_start_time:.3g}s]')

        # === Prepare validation parameter set ===
        console_width = self.config.get('width', None)  # console width for rich table displays of synapse measures
        validation_params = (random_uids, query_responses, return_ops, times, routing_score,
                             inputs, val_len, self.loss_fct, console_width)

        loss = torch.tensor(0.).to(self.device)  # to accumulate neuron_loss and routing_loss over synapses
        neuron_stats = {}  # to gather neuron synapse validation measures and statistics

        # === Validate synapse responses ===
        # Iterate over all queried synapses and validate responses
        for i, (synapse, validate_func) in enumerate(synapses):
            _loss, stats = validate_func(*validation_params, synapse=synapse, index_s=i)  # validate individual synapse
            loss += _loss  # add neuron_loss and routing_loss

            for _uid, _stats in stats.items():
                neuron_stats.setdefault(_uid, {})
                neuron_stats[_uid].update(_stats)  # gather neuron synapse validation measures and statistics

        return loss, neuron_stats


def scaling_law_loss_to_params(loss):
    r""" (OpenAI scaling laws) Kaplan, Jared, et al. "Scaling laws for neural language models." arXiv:2001.08361 (2020)
    """
    num_params = torch.exp(torch.log(torch.tensor(8.8e13).to(loss.device)) - torch.log(torch.clamp(loss, 1.69)) / 0.076)
    pow_num_params = torch.pow(num_params, 0.5)  # powered down number of params, dynamic range 3 → 6 nats
    return pow_num_params  # modified scaling law, powered down to improve dynamic range (subject to change)


def textcausallm(uids: torch.Tensor, query_responses: List[List[torch.FloatTensor]], return_ops: List[torch.LongTensor],
                 times: List[torch.FloatTensor], routing_score: torch.FloatTensor,
                 inputs: torch.FloatTensor, validation_len: int, loss_fct: Callable,
                 console_width: int, synapse: 'bittensor.TextCausalLM' = None, index_s: int = 0
                 ) -> Tuple[torch.FloatTensor, Dict]:
    r"""
    Calculate Shapley values and neuron response validation measure statistics, given TextCausalLM synapse responses.
        Args:
            uids (:obj:`torch.Tensor`, `required`): [num_neurons]
                Neuron UIDs.
            query_responses (:obj:`List[List[torch.FloatTensor]]`, `required`):
                List of outputs from synapses, each a list of size num_endpoints of tensors with relevant size. Non-responses are zeroes of relevant
                synapse shape. Shape num_synapses * ( num_endpoints * ( -1, -1, -1 ) )
            return_ops (:obj:`List[torch.LongTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Return code per call per synapse.
            times (:obj:`List [torch.FloatTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Times per call per synapse.
            routing_score (:obj:`torch.FloatTensor`, `required`):
                [metagraph.n] Predictive routing score per endpoint in the metagraph, mean over the batch.
            inputs (:obj:`torch.FloatTensor`, `required`):
                [batch_size, sequence_len + validation_len] Token batch of original inputs with validation tokens.
            validation_len (:obj:`int`, `required`):
                Number of held-out phrase token batch for extended validation, not sent to neurons.
            loss_fct (:obj:`Callable`, `required`):
                CrossEntropy loss function to use.
            console_width (:obj:`int`, `required`):
                Config console width for table print.
            synapse (:obj:`bittensor.TextCausalLM`, `optional`):
                TextCausalLM synapse object.
            index_s (:obj:`int`, `optional`):
                Index of synapse to extract responses.

        Returns:
            loss (:obj:`torch.FloatTensor`):
                Loss for training validator nucleus and dendrite backward to endpoints.
            stats (:obj:`Dict`, `required`):
                Statistics per endpoint for this batch.
    """

    inputs_seq = inputs[..., :-validation_len]  # input sequence without last token [batch_size, sequence_len]
    inputs_val = inputs[..., -validation_len]  # input validation with next token [batch_size]

    def _base_params(_stats, query_response):
        _stats.update({'logits': query_response[:, :-1, :],
                       'logits_val': query_response[:, -1:, :]})

        for target, _ext in [(inputs_seq[:, 1:], ''), (inputs_val, '_val')]:
            _loss = calc_loss_fct(loss_fct, _stats['logits' + _ext], target)  # CausalLM loss
            _num_params = scaling_law_loss_to_params(_loss)  # estimate the effective number of model parameters

            _stats.update({'loss' + _ext: _loss, 'base_params' + _ext: _num_params,
                           'synergy' + _ext: 0, 'synergy_loss_diff' + _ext: 0})

    def _synergy(first, second, target, _ext):
        # Combined logits: log of average probabilities per token between responses
        combined_logits = torch.log((torch.softmax(first['logits' + _ext], dim=-1) +
                                     torch.softmax(second['logits' + _ext], dim=-1)) / 2 + 1e-40)
        measured_loss = calc_loss_fct(loss_fct, combined_logits, target)  # actual measured loss

        return measured_loss

    print(f'\[{str(synapse)}] Shapley values \t| Calculating base ... ', end='')
    shapley_start_time = time.time()

    loss, stats, unsuccessful = shapley_base(uids, query_responses, return_ops, times, routing_score,
                                             _base_params, index_s, ext='')

    print(f'\[{time.time() - shapley_start_time:.3g}s] | synergy ... ', end='')

    syn_loss_diff = shapley_synergy(stats, _synergy, ext='', target=inputs_seq[:, 1:])
    syn_loss_diff_val = shapley_synergy(stats, _synergy, ext='_val', target=inputs_val)

    # === Shapley value combination ===
    # Combine base values with synergy approximation to get final Shapley values.
    for s in stats.values():
        for ext in ['', '_val']:
            if 'base_params' + ext in s and 'synergy' + ext in s:
                s['shapley_values' + ext] = (s['base_params' + ext] + s['synergy' + ext])

            if 'logits' + ext in s:
                del s['logits' + ext]  # remove logits - not needed for stats anymore

        if 'shapley_values' in s and 'shapley_values_val' in s:
            s['shapley_values_min'] = torch.min(s['shapley_values'], s['shapley_values_val'])

        for key in s:
            if hasattr(s[key], 'item'):
                s[key] = s[key].item()

    print(f'complete \[{time.time() - shapley_start_time:.3g}s]')

    # === Synergy table ===
    # Prints the synergy loss diff matrix with pairwise loss reduction due to synergy (original loss on diagonal)
    synergy_table(stats, syn_loss_diff, 'shapley_values_min', console_width=console_width)

    # === Neuron responses (table) ===
    # Prints the evaluation of the neuron responses to the validator request
    synapse_table(str(synapse), stats, 'shapley_values_min', console_width, shapley_start_time)

    # === Unsuccessful responses ===
    # Prints the return codes and response times of unsuccessful responses
    unsuccess(str(synapse), unsuccessful)

    return loss, stats


def textcausallmnext(uids: torch.Tensor, query_responses: List[List[torch.FloatTensor]], return_ops: List[torch.LongTensor],
                     times: List[torch.FloatTensor], routing_score: torch.FloatTensor,
                     inputs: torch.FloatTensor, validation_len: int, loss_fct: Callable,
                     console_width: int, synapse: 'bittensor.TextCausalLMNext' = None, index_s: int = 0
                     ) -> Tuple[torch.FloatTensor, Dict]:
    r"""
    Calculate Shapley values and neuron response validation measure statistics, given TextCausalLMNext synapse responses.
        Args:
            uids (:obj:`torch.Tensor`, `required`): [num_neurons]
                Neuron UIDs.
            query_responses (:obj:`List[List[torch.FloatTensor]]`, `required`):
                List of outputs from synapses, each a list of size num_endpoints of tensors with relevant size. Non-responses are zeroes of relevant
                synapse shape. Shape num_synapses * ( num_endpoints * ( -1, -1, -1 ) )
            return_ops (:obj:`List[torch.LongTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Return code per call per synapse.
            times (:obj:`List [torch.FloatTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Times per call per synapse.
            routing_score (:obj:`torch.FloatTensor`, `required`):
                [metagraph.n] Predictive routing score per endpoint in the metagraph, mean over the batch.
            inputs (:obj:`torch.FloatTensor`, `required`):
                [batch_size, sequence_len + validation_len] Token batch of original inputs with validation tokens.
            validation_len (:obj:`int`, `required`):
                Number of held-out phrase token batch for extended validation, not sent to neurons.
            loss_fct (:obj:`Callable`, `required`):
                CrossEntropy loss function to use.
            console_width (:obj:`int`, `required`):
                Config console width for table print.
            synapse (:obj:`bittensor.TextCausalLMNext`, `optional`):
                TextCausalLMNext Synapse object.
            index_s (:obj:`int`, `optional`):
                Index of synapse to extract responses.

        Returns:
            loss (:obj:`torch.FloatTensor`):
                Loss for training validator nucleus and dendrite backward to endpoints.
            stats (:obj:`Dict`, `required`):
                Statistics per endpoint for this batch.
    """

    inputs_nxt = inputs[..., -validation_len:]  # input validation with next token target phrase [batch_size, val_len]

    def _base_params(_stats, query_response):
        result = unravel_topk_token_phrases(query_response, topk=synapse.topk)
        topk_tokens, topk_probs, floor_probs = result
        # topk_tokens: [batch_size, topk, max_len] Phrase tokens with ignore_index token for padding.
        # topk_probs: [batch_size, topk] Probabilities for each phrase in topk.
        # floor_probs: [batch_size] Floor probabilities as mean probability for non-topk tokens.
        # inputs_nxt: [batch_size] Target phrases in standard token sequence list.

        _losses = phrase_cross_entropy(inputs_nxt, topk_tokens, topk_probs, floor_probs, reduce=False)  # [batch_size]
        _loss = _losses.mean()
        _num_params = scaling_law_loss_to_params(_loss)  # estimate the effective number of model parameters

        _stats.update({'losses_nxt': _losses, 'loss_nxt': _loss,
                       'base_params_nxt': _num_params, 'synergy_nxt': 0, 'synergy_loss_diff_nxt': 0})

    def _synergy(first, second, target, ext):
        # average first + second probabilities per batch item, convert to loss
        measured_loss = -torch.log((torch.exp(-first['losses_nxt']) +
                                    torch.exp(-second['losses_nxt'])) / 2).mean()

        return measured_loss

    print(f'\[{str(synapse)}] Shapley values \t| Calculating base ... ', end='')
    shapley_start_time = time.time()

    loss, stats, unsuccessful = shapley_base(uids, query_responses, return_ops, times, routing_score,
                                             _base_params, index_s, ext='_nxt')

    print(f'\[{time.time() - shapley_start_time:.3g}s] | synergy ... ', end='')

    syn_loss_diff = shapley_synergy(stats, _synergy, '_nxt')

    # === Shapley value combination ===
    # Combine base values with synergy approximation to get final Shapley values.
    for s in stats.values():
        if 'base_params_nxt' in s and 'synergy_nxt' in s:
            s['shapley_values_nxt'] = s['base_params_nxt'] + s['synergy_nxt']

        if 'losses_nxt' in s:
            del s['losses_nxt']  # remove batch losses - not needed for stats anymore

        for key in s:
            if hasattr(s[key], 'item'):
                s[key] = s[key].item()

    print(f'complete \[{time.time() - shapley_start_time:.3g}s]')

    # === Synergy table ===
    # Prints the synergy loss diff matrix with pairwise loss reduction due to synergy (original loss on diagonal)
    synergy_table(stats, syn_loss_diff, 'shapley_values_nxt', console_width)

    # === Neuron responses (table) ===
    # Prints the evaluation of the neuron responses to the validator request
    synapse_table(str(synapse), stats, 'shapley_values_nxt', console_width, shapley_start_time)

    # === Unsuccessful responses ===
    # Prints the return codes and response times of unsuccessful responses
    unsuccess(str(synapse), unsuccessful)

    return loss, stats


def shapley_base(uids: torch.Tensor, query_responses: List[List[torch.FloatTensor]], return_ops: List[torch.LongTensor],
                 times: List[torch.FloatTensor], routing_score: torch.FloatTensor,
                 base_params: Callable, index_s: int = 0, ext: str = None) -> Tuple[Union[float, torch.FloatTensor],
                                                                                    Dict,
                                                                                    List]:
    r"""
    Calculate Shapley base values and neuron response validation measure statistics, given responses from a synapse.
        Args:
            uids (:obj:`torch.Tensor`, `required`): [num_neurons]
                Neuron UIDs.
            query_responses (:obj:`List[List[torch.FloatTensor]]`, `required`):
                List of outputs from synapses, each a list of size num_endpoints of tensors with relevant size. Non-responses are zeroes of relevant
                synapse shape. Shape num_synapses * ( num_endpoints * ( -1, -1, -1 ) )
            return_ops (:obj:`List[torch.LongTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Return code per call per synapse.
            times (:obj:`List [torch.FloatTensor]` of shape :obj:`[num_endpoints]`, `required`):
                Times per call per synapse.
            routing_score (:obj:`torch.FloatTensor`, `required`):
                [metagraph.n] Predictive routing score per endpoint in the metagraph, mean over the batch.
            base_params (:obj:`Callable`, `required`):
                CrossEntropy loss function to use.
            index_s (:obj:`int`, `optional`):
                Index of synapse to extract responses.
            ext (:obj:`str`, `optional`):
                Extension to parameter string for stats key.

        Returns:
            loss (:obj:`torch.FloatTensor`):
                Loss for training validator nucleus and dendrite backward to endpoints.
            stats (:obj:`Dict`, `required`):
                Statistics per endpoint for this batch.
            unsuccessful (:obj:`List`, `required`):
                Unsuccessful endpoints [(uid, return_op, time)].
    """
    stats = {}
    unsuccessful = []
    neuron_loss = 0.  # neuron losses to accumulate to then backward() via dendrite
    routing_loss = 0.  # validator routing loss for local model update

    # === Base parameter estimation ===
    # Shapley values - base level - coalition size 1
    # Collect successful neuron responses, calculate base Shapley values.
    # Measured in effective number of model parameters, according to OpenAI scaling laws.
    for index, _uid in enumerate(uids.tolist()):
        if return_ops[index][index_s] == bittensor.proto.ReturnCode.Success:
            _stats = {'uid': _uid,
                      'response_time' + ext: times[index][index_s],
                      'routing_score': routing_score[_uid]}

            base_params(_stats, query_responses[index][index_s])

            neuron_loss += _stats['loss' + ext]  # add sequence loss to be backward() to neuron

            # === Add routing loss ===
            # MSE loss between predicted routing score and ideal target routing score.
            # The Bayes risk approx. 1.69, i.e. the minimal loss achievable for next-token
            # prediction on the full distribution 𝑃, a.k.a the "entropy of natural text"
            # Hoffmann, Jordan, et al. "Training Compute-Optimal Large Language Models." arXiv:2203.15556 (2022).
            routing_score_target = torch.exp(-torch.clamp(_stats['loss' + ext].detach() - 1.69, 0))
            _routing_loss = (routing_score[_uid] - routing_score_target) ** 2  # MSE loss
            routing_loss += _routing_loss
            _stats.update({'routing_score_target' + ext: routing_score_target, 'routing_loss' + ext: _routing_loss})

            stats[_uid] = _stats
        else:
            stats[_uid] = {'uid': _uid,
                           'response_time' + ext: times[index][index_s],
                           'routing_score': routing_score[_uid]}
            unsuccessful += [(_uid, return_ops[index][index_s], times[index][index_s])]

    return neuron_loss + routing_loss, stats, unsuccessful


def shapley_synergy(stats: Dict, synergy: Callable, ext: str, target: torch.Tensor = None):
    r"""
    Calculates Shapley synergy for coalition size 2, measured performance above expected performance.
    Measured in effective number of model parameters, just like base Shapley values.
        Args:
            stats (:obj:`Dict`, `required`):
                Statistics per endpoint for this batch.
            synergy (:obj:`Callable`, `required`)
                Function to calculate measured loss.
            ext (:obj:`str`, `optional`):
                Extension to parameter string for stats key.
            target (:obj:`torch.Tensor`, `optional`):
                Target to measure loss against.

        Returns:
            syn_loss_diff (:obj:`Dict`, `required`):
                Dictionary table of pairwise synergies as loss reductions, with direct loss on diagonal.
    """
    # === Shapley synergy approximation ===
    # Shapley values - second level - coalition size 2
    # Synergy = measured performance above expected performance
    # Measured in effective number of model parameters, just like base Shapley values.
    syn_loss_diff = {}  # expected_loss - measured_loss (where > 0)
    for _first, first in stats.items():
        if 'loss' + ext not in first:
            continue
        first_diff = syn_loss_diff.setdefault(_first, {})
        first_diff[_first] = first['loss' + ext]  # diagonal keeps direct loss

        for _second, second in stats.items():
            if 'loss' + ext not in second or _second <= _first:
                continue
            second_diff = syn_loss_diff.setdefault(_second, {})

            with torch.no_grad():
                expected_loss = torch.min(first['loss' + ext], second['loss' + ext])  # expecting min loss

                measured_loss = synergy(first, second, target, ext)

                loss_diff_share = torch.clamp(expected_loss - measured_loss, 0) / 2  # record direct loss diff
                first['synergy_loss_diff' + ext] += loss_diff_share
                second['synergy_loss_diff' + ext] += loss_diff_share

                # pairwise loss reduction of expected to measured loss due to synergy between first and second
                first_diff[_second] = loss_diff_share
                second_diff[_first] = loss_diff_share

                synergy_share = torch.clamp(scaling_law_loss_to_params(measured_loss) -
                                            scaling_law_loss_to_params(expected_loss), 0) / 2
                first['synergy' + ext] += synergy_share  # share synergy amongst coalition members
                second['synergy' + ext] += synergy_share

    return syn_loss_diff


def synergy_table(stats, syn_loss_diff, sort_col, console_width):
    r""" Prints the synergy loss diff matrix with pairwise loss reduction due to synergy (original loss on diagonal)
    """
    sort = sorted([(uid, s[sort_col]) for uid, s in stats.items() if sort_col in s], reverse=True, key=lambda _row: _row[1])
    uid_col = neuron_stats_columns[0]  # [Column_name, key_name, format_string, rich_style]
    columns = [uid_col] + [[f'{s[0]}', '', '{:.2f}', ''] for s in sort]
    rows = [[uid_col[2].format(s[0])] +
            [('[bright_cyan]{:.2f}[/bright_cyan]' if t == s else
              '[magenta]{:.2f}[/magenta]' if syn_loss_diff[s[0]][t[0]] > 0 else
              '[dim]{:.0f}[/dim]').format(syn_loss_diff[s[0]][t[0]]) for t in sort] for s in sort]

    # === Synergy table ===
    table = Table(width=console_width, box=None)
    table.title = f'[white] Synergy [/white]'
    table.caption = f'loss decrease'

    for col, _, _, stl in columns:  # [Column_name, key_name, format_string, rich_style]
        table.add_column(col, style=stl, justify='right')
    for row in rows:
        table.add_row(*row)

    if len(rows):
        print(table)
        print()


def stats_table(stats, sort_col, console_width, title, caption):
    r""" Gathers data and constructs neuron statistics table and prints it
    """
    # === Gather columns and rows ===
    stats_keys = [set(k for k in stat)
                  for stat in stats.values() if sort_col in stat]  # all available stats keys with sort_col

    if len(stats_keys) == 0:
        return  # nothing to print

    stats_keys = set.union(*stats_keys)
    columns = [c[:] for c in neuron_stats_columns if c[1] in stats_keys]  # available columns intersecting with stats_keys
    rows = [['' if key not in stat else txt.format(stat[key]) for _, key, txt, _ in columns]
            for stat in stats.values() if sort_col in stat]  # only keep rows with at least one non-empty cell

    if len(columns) == 0 or len(rows) == 0:
        return  # nothing to print

    # === Sort rows ===
    col_keys = [c[1] for c in columns]
    if sort_col in col_keys:
        sort_idx = col_keys.index(sort_col)  # sort column with key of sort_col
        columns[sort_idx][0] += '\u2193'  # ↓ downwards arrow (sort)
        rows = sorted(rows, reverse=True, key=lambda _row: _row[sort_idx])  # sort according to _sortcol

    # === Instantiate stats table ===
    table = Table(width=console_width, box=None, row_styles=[Style(bgcolor='grey15'), ""])
    table.title = title
    table.caption = caption

    for col, _, _, stl in columns:  # [Column_name, key_name, format_string, rich_style]
        table.add_column(col, style=stl, justify='right')
    for row in rows:
        table.add_row(*row)

    # === Print table ===
    print(table)


def synapse_table(name, stats, sort_col, console_width, start_time):
    r""" Prints the evaluation of the neuron responses to the validator request
    """

    stats_table(stats, sort_col, console_width,
                f'[white] \[{name}] responses [/white] | Validator forward',  # title
                f'[bold]{len([s for s in stats.values() if len(s)])}[/bold]/{len(stats)} (respond/topk) | '
                f'[bold]Synapse[/bold] | [white]\[{time.time() - start_time:.3g}s][/white]'  # caption
                )


def unsuccess(_name, _unsuccessful):
    r""" Prints the return codes and response times of unsuccessful responses
    """
    # === Unsuccessful responses ===
    unsuccess_txt = f'\[{_name}] Unsuccessful \t| [cyan]UID[/cyan]\[[red]return_op[/red] [yellow]time[/yellow]]: '
    for _uid, _return_op, _time in _unsuccessful:
        unsuccess_txt += f'{_uid}[[red]{_return_op}[/red] [yellow not bold]{_time:.2f}[/yellow not bold]] '
    print(unsuccess_txt)
