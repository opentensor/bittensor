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
from ..neuron_utilities import joining_context, partial_contexts, ThreadQueue
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
    ['Upd', 'updates', '{}', 'bright_yellow'],  # number of exponential moving average updates
    ['Time', 'response_time', '{:.2f}', 'yellow'],  # response time to forward requests
    ['Route', 'routing_score', '{:.3f}', 'grey30'],  # validator routing score (higher preferred)
    ['Weight', 'weight', '{:.5f}', 'green'],  # weight set on substrate (each epoch)
    ['mShap', 'shapley_values_min', '{:.0f}', 'bright_magenta'],  # min(Shap, vShap) of sequence and validation Shapley
    ['Loss', 'loss', '{:.2f}', 'bright_cyan'],  # next token prediction loss average over sequence
    ['vLoss', 'loss_val', '{:.2f}', 'bright_cyan'],  # next token prediction loss for validation task
    ['RLoss', 'routing_loss', '{:.3f}', 'grey30'],  # MSE between routing_score and conditioned loss
    ['Shap', 'shapley_values', '{:.0f}', 'magenta'],  # Shapley value (=Base+Syn) over sequence
    ['vShap', 'shapley_values_val', '{:.0f}', 'magenta'],  # Shapley value (=vBase+vSyn) for validation
    ['Base', 'base_params', '{:.0f}', ''],  # parameter count estimate via adjusted scaling law
    ['vBase', 'base_params_val', '{:.0f}', ''],  # parameter count estimate for validation task
    ['Syn', 'synergy', '{:.0f}', 'white'],  # Shapley pairwise synergy over sequence loss (parameter count estimate)
    ['vSyn', 'synergy_val', '{:.0f}', 'white'],  # Shapley pairwise synergy over validation loss (count estimate)
    ['SynD', 'synergy_loss_diff', '{:.2f}', 'bright_blue'],  # Shapley pairwise synergy over sequence loss (loss difference)
    ['vSynD', 'synergy_loss_diff_val', '{:.2f}', 'bright_blue']]  # Shapley pairwise synergy over validation loss (loss difference)


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
        self.dataset = bittensor.dataset ( config = self.config, batch_size = self.subtensor.validator_batch_size, block_size = self.subtensor.validator_sequence_length + 1 ) if dataset == None else dataset
        self.optimizer = torch.optim.SGD(
            self.nucleus.parameters(), lr=self.config.neuron.learning_rate, momentum=self.config.neuron.momentum
        )

        # === Create thread queue ===
        self.forward_thread_queue = ThreadQueue(num_jobs = self.config.neuron.forward_num, target = self.forward)
        self.loss = None
        self.loss_agg_mutex = Lock()
        self.server_stats = {}

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
        sequence_length = self.subtensor.validator_sequence_length + 1  # add validation token
        n_topk_peer_weights = self.subtensor.min_allowed_weights
        max_allowed_ratio = self.subtensor.max_allowed_min_max_ratio
        blocks_per_epoch = self.subtensor.validator_epoch_length if self.config.neuron.blocks_per_epoch == -1 else self.config.neuron.blocks_per_epoch
        epochs_until_reset = self.subtensor.validator_epochs_per_reset if self.config.neuron.epochs_until_reset == -1 else self.config.neuron.epochs_until_reset

        # === Update dataset size ===
        if (batch_size != self.dataset.batch_size) or (sequence_length != self.dataset.block_size):
            self.dataset.set_data_size(batch_size, sequence_length)

        # === Logs ===
        if self.config.using_wandb:
            wandb.log({'era/batch_size': batch_size, 'era/sequence_length': sequence_length,
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

            # === Scoring ===
            # Updates moving averages and history.
            for s in stats:
                history = self.server_stats.setdefault(s['uid'], s)
                history.setdefault('updates', 0)  # add updates fields for new uid entries
                history['updates'] += 1  # increase number of updates made
                sum_ratio = 1. / min(20, history['updates'])  # moving average window size of 20 max
                for key in s:  # detailed server evaluation fields, e.g. loss, shapley_values, synergy
                    if key not in ['updates']:
                        history[key] = (1 - sum_ratio) * history[key] + sum_ratio * s[key]  # update EMA

            # === State update ===
            # Prints step logs to screen.
            epoch_steps += 1
            self.global_step += 1
            current_block = self.subtensor.block
            step_time = time.time() - start_time

            # === Stats update (table) ===
            # Prints exponential moving average statistics of valid neurons from latest validator forward
            columns = [_[:] for _ in neuron_stats_columns if _[0] not in ['Weight']]
            sort_col = [_[0] for _ in columns].index('mShap')  # sort column with key of shapley_values_min
            columns[sort_col][0] += '\u2193'  # ↓ downwards arrow (sort)
            rows = [[txt.format(self.server_stats[s['uid']][key]) for _, key, txt, _ in columns] for s in stats]
            rows = sorted(rows, reverse=True, key=lambda _row: int(_row[sort_col]))  # sort according to mShap column

            table = Table(width=self.config.get('width', None), box=None, row_styles=[Style(bgcolor='grey15'), ''])
            table.title = f'[white] Stats update [/white] | [bold]UID {self.uid}[/bold] ' \
                          f'\[{self.dendrite.receptor_pool.external_ip}] ' \
                          f'({self.wallet.name}:[bold]{self.wallet.coldkeypub.ss58_address[:7]}[/bold]/' \
                          f'{self.config.wallet.hotkey}:[bold]{self.wallet.hotkey.ss58_address[:7]}[/bold])'
            table.caption = f'#{current_block}: ' \
                            f'[bold]{current_block - start_block}[/bold]/{blocks_per_epoch} (blocks/epoch) | ' \
                            f'Epoch {self.epoch} | ' \
                            f'[white] Step {epoch_steps} ({self.global_step} global) \[{step_time:.3g}s] [/white]'

            for col, _, _, stl in columns:
                table.add_column(col, style=stl, justify='right')
            for row in rows:
                table.add_row(*row)

            print(f'UID {self.uid}   \t| '
                  f'Updated {current_block - self.metagraph.last_update[self.uid]} [white]blocks ago[/white] | '
                  f'Dividends {self.metagraph.dividends[self.uid]:.5f} | '
                  f'Stake \u03C4{self.metagraph.stake[self.uid]:.5f} '
                  f'[dim](retrieved {current_block - start_block} blocks ago from {self.subtensor.network})[/dim]')
            print(table)
            print()

            # === Set weights ===
            score_key = 'shapley_values_min'  # server score based on Shapley value approximation
            moving_avg_scores = torch.zeros_like(
                self.metagraph.S)  # allow unevaluated UIDs to be selected to meet min topk

            for key in self.server_stats:
                moving_avg_scores[key] = torch.tensor([self.server_stats[key][score_key]])
            # Find the n_topk_peer_weights peers to set weights to.
            topk_scores, topk_uids = bittensor.unbiased_topk(moving_avg_scores, k=n_topk_peer_weights)
            topk_scores = bittensor.utils.weight_utils.normalize_max_multiple(x=topk_scores, multiple=max_allowed_ratio)

            # === Set weights (table) ===
            # Prints exponential moving average statistics of valid neurons and latest weights
            columns = [_[:] for _ in neuron_stats_columns]  # clone neuron_stats_columns
            rows = []
            unvalidated = []  # unsuccessful neurons that still receive a weighting
            for i in range(len(topk_uids)):
                _weight = topk_scores[i].item()
                _uid = topk_uids[i].item()
                if _uid in self.server_stats:
                    _stats = {k: v for k, v in self.server_stats[_uid].items()}
                    _stats['weight'] = _weight  # add weight entry into _stats dictionary
                    rows += [[txt.format(_stats[key]) for _, key, txt, _ in columns]]
                else:
                    unvalidated += [_uid]

            sort_col = [_[0] for _ in columns].index('Weight')  # sort column with key of weight
            columns[sort_col][0] += '\u2193'  # ↓ downwards arrow (sort)
            rows = sorted(rows, reverse=True, key=lambda _row: float(_row[sort_col]))  # sort according to weights

            table = Table(width=self.config.get('width', None), box=None, row_styles=[Style(bgcolor='grey15'), ""])
            table.title = f'[white] Set weights [/white] | [bold]UID {self.uid}[/bold] ' \
                          f'\[{self.dendrite.receptor_pool.external_ip}] ' \
                          f'({self.wallet.name}:[bold]{self.wallet.coldkeypub.ss58_address[:7]}[/bold]/' \
                          f'{self.config.wallet.hotkey}:[bold]{self.wallet.hotkey.ss58_address[:7]}[/bold])'
            table.caption = f'Validated [bold]{(n_topk_peer_weights - len(unvalidated))}[/bold]' \
                            f'/{n_topk_peer_weights}/{self.metagraph.n} (valid/min/total) | ' \
                            f'sum:{topk_scores.sum().item():.2g} ' \
                            f'[white] max:[bold]{topk_scores.max().item():.4g}[/bold] / ' \
                            f'min:[bold]{topk_scores.min().item():.4g}[/bold] [/white] ' \
                            f'\[{topk_scores.max().item() / topk_scores.min().item():.1f}:1] ' \
                            f'({max_allowed_ratio} allowed)'

            for col, _, _, stl in columns:
                table.add_column(col, style=stl, justify='right')
            for row in rows:
                table.add_row(*row)

            print(table)
            print(f'Unvalidated \t| [dim]\[weight={topk_scores.min().item():.4g}][/dim]: {unvalidated}')
            print()

            # === Logs ===
            if self.config.using_wandb:
                wandb.log({'epoch/epoch': self.epoch, 'epoch/epoch_steps': epoch_steps,
                           'epoch/global_steps': self.global_step, 'epoch/loss': loss.item(),
                           'epoch/time': step_time}, step=current_block)
                for uid, vals in self.server_stats.items():
                    for key in vals:  # detailed server evaluation fields, e.g. loss, shapley_values, synergy
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
                print(f'complete [{time.time() - start_time:.3g}s]')
                
                # === Get another round of forward requests ===
                self.forward_thread_queue.resume()

        # Iterate epochs.
        self.epoch += 1

        # === Set weights ===
        score_key = 'shapley_values_min'  # server score based on Shapley value approximation
        moving_avg_scores = torch.zeros_like(self.metagraph.S)  # allow unevaluated UIDs to be selected to meet min topk

        for key in self.server_stats:
            moving_avg_scores[key] = torch.tensor([self.server_stats[key][score_key]])
        # Find the n_topk_peer_weights peers to set weights to.
        topk_scores, topk_uids = bittensor.unbiased_topk(moving_avg_scores, k=n_topk_peer_weights)
        topk_scores = bittensor.utils.weight_utils.normalize_max_multiple(x=topk_scores, multiple=max_allowed_ratio)

        # === Set weights (table) ===
        # Prints exponential moving average statistics of valid neurons and latest weights
        columns = [_[:] for _ in neuron_stats_columns]  # clone neuron_stats_columns
        rows = []
        unvalidated = []  # unsuccessful neurons that still receive a weighting
        for i in range(len(topk_uids)):
            _weight = topk_scores[i].item()
            _uid = topk_uids[i].item()
            if _uid in self.server_stats:
                _stats = {k: v for k, v in self.server_stats[_uid].items()}
                _stats['weight'] = _weight  # add weight entry into _stats dictionary
                rows += [[txt.format(_stats[key]) for _, key, txt, _ in columns]]
            else:
                unvalidated += [_uid]

        sort_col = [_[0] for _ in columns].index('Weight')  # sort column with key of weight
        columns[sort_col][0] += '\u2193'  # ↓ downwards arrow (sort)
        rows = sorted(rows, reverse=True, key=lambda _row: float(_row[sort_col]))  # sort according to weights

        table = Table(width=self.config.get('width', None), box=None, row_styles=[Style(bgcolor='grey15'), ""])
        table.title = f'[white] Set weights [/white] | [bold]UID {self.uid}[/bold] ' \
                      f'\[{self.dendrite.receptor_pool.external_ip}] ' \
                      f'({self.wallet.name}:[bold]{self.wallet.coldkeypub.ss58_address[:7]}[/bold]/' \
                      f'{self.config.wallet.hotkey}:[bold]{self.wallet.hotkey.ss58_address[:7]}[/bold])'
        table.caption = f'Validated [bold]{(n_topk_peer_weights - len(unvalidated))}[/bold]' \
                        f'/{n_topk_peer_weights}/{self.metagraph.n} (valid/min/total) | ' \
                        f'sum:{topk_scores.sum().item():.2g} ' \
                        f'[white] max:[bold]{topk_scores.max().item():.4g}[/bold] / ' \
                        f'min:[bold]{topk_scores.min().item():.4g}[/bold] [/white] ' \
                        f'\[{topk_scores.max().item() / topk_scores.min().item():.1f}:1] ' \
                        f'({max_allowed_ratio} allowed)'

        for col, _, _, stl in columns:
            table.add_column(col, style=stl, justify='right')
        for row in rows:
            table.add_row(*row)

        print(table)
        print(f'Unvalidated \t| [dim]\[weight={topk_scores.min().item():.4g}][/dim]: {unvalidated}')
        print()

        self.subtensor.set_weights(
            uids = topk_uids.detach().to('cpu'),
            weights = topk_scores.detach().to('cpu'),
            wallet = self.wallet,
            wait_for_finalization = self.config.neuron.wait_for_finalization,
        )

        # === Wandb Logs ===
        # Optionally send validator logs to wandb.
        if self.config.using_wandb:
            # Logging history to wandb.
            df = pandas.concat( [
                bittensor.utils.indexed_values_to_dataframe( prefix = 'weights', index = topk_uids, values = torch.zeros( self.metagraph.n ).scatter( dim = 0, src = topk_scores, index = topk_uids ) ),
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

        # === Reset server stats if uid got replaced
        for uid, old_hotkey in enumerate(old_hotkeys):
            if old_hotkey != self.metagraph.hotkeys[uid]:
                if uid in self.server_stats:
                    del self.server_stats[uid]


class PositionalEncoding(nn.Module):
    r""" Positional Encoder which adds information based on the relative position of each token
    
    """
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # === Create position matrix ===
        # Creates a positional matrix with alternating frequencies 
        # pe: (torch.FloatTensor) positional encoding matrix
        # pe.shape: [1, max_len, network_dim]
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, : , 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # === Positional Encoding ===
        # Inject some information of the relative position of the token in the sequence.
        #  Finally, Dropout is applied to tokens
        # x: (torch.FloatTensor) input sequence tokens with position information injected
        # x.shape: [batch_size, seq_len, network_dim]
        x = x + self.pe[0, :x.size(1)]
        return self.dropout(x)


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

    # === Compute loss given joined responses ===
    # This function computes target loss for next token prediction given 
    # the joined responses as a hidden unit input.
    # target_loss: (torch.float64): loss after decoding responses to targets.
    # target_loss.shape = [ 1 ]
    def get_target_loss_casuallm ( self, logits, targets, eval_type = '' ):
        # hidden: (torch.float64): [ batch_size, sequence_len, __network_dim__ ]
        #   Hidden units which are encoded and decoded onto targets for loss computation.
        # targets: (torch.float64): [n]
        #   Token targets,
        #src_mask = torch.triu(torch.ones(hidden.size(1), hidden.size(1)) * float('-inf'), diagonal=1)
        #src_mask = src_mask.to(self.config.neuron.device)
        #encoded_hidden = self.encoder( hidden, mask = src_mask )
        #decoded_targets = self.decoder( encoded_hidden )
        #shift_logits = decoded_targets[..., :-1, :].contiguous()
        if eval_type == '':
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
        elif eval_type == '_val':
            shift_logits = logits.contiguous()
            shift_labels = targets.contiguous()
        return self.loss_fct( shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1) )

    def forward ( 
        self, 
        inputs: torch.FloatTensor,
        metagraph: 'bittensor.Metagraph',
        dendrite: 'bittensor.Dendrite',
    ):
        r""" Forward validator pass. Selects peer to query, joins results and computes scoring.
            Args:
                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, *-1*)`, `required`): 
                    Tensor inputs to distribute to neurons using query context.
                metagraph (bittensor.Metagraph):
                    Metagraph object used to query network information.
                dendrite (bittensor.Dendrite):
                    Dendrite RPC client used to make network queries.
            Returns:
                global_loss (torch.FloatTensor, [1] ):
                    Loss for training validator nucleus.
                scores (torch.FloatTensor, [ metagraph.n ]):
                    Scores per endpoint for this batch.
        """
        start_time = time.time()
        batch_size, sequence_len = inputs.shape
        print(f'Forward \t| Model forward ... ', end='')

        inputs = inputs.to(self.device)
        inputs_seq = inputs[..., :-1]  # input sequence without last token [batch_size, sequence_len-1]
        inputs_val = inputs[..., -1]  # input validation with last token [batch_size]

        # === Create the local context used to select endpoints ===
        # The context tensor returns a hidden unit representation for the text inputs
        # this context can be used as input to the gates in the next step.
        # embedding: retrieve learned representation vectors for input vocabulary tokens.
        # inputs.shape = [batch_size, sequence_len]
        # embedding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        
        embedding =  self.token_embedding( inputs_seq ) * math.sqrt( bittensor.__network_dim__ )
        
        # === Create an attention mask ===
        # The attention mask will mask out parts of the context
        # This prevents cheating and forward-looking when predicting each token in the sequence.
        # src_mask: (torch.FloatTensor) attention mask adds -inf to positions not allowed to attend
        # src_mask.shape = [sequence_len, sequence_len]
        src_mask = torch.triu(torch.ones(embedding.size(1), embedding.size(1)) * float('-inf'), diagonal=1)
        src_mask = src_mask.to(self.config.neuron.device)

        # === Apply the positional encoding to help select endpoints ===
        # The positional encoder provides information based on the relative postion of each token 
        # embedding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        # pos_embedding: (torch.FloatTensor) positional encoded embedding.
        # pos_embedding.shape = [batch_size, sequence_len, bittensor.__network_dim__]
        pos_embedding = self.local_pos_encoder(embedding)

        # routing_context: (torch.FloatTensor): context tensor which is used to select endpoints.
        # routing_context.shape = [ batch size, __network_dim__ ]
        routing_context = self.routing_encoder( pos_embedding, mask = src_mask )

        # === Get gate values for UIDs. ===
        # We iterate over each of the network UIDs and compute a querying score for each
        # using the gating function. This returns a score per endpoint per example.
        # routing_score: (torch.FloatTensor): score per example, per endpoint.
        # routing_score.shape = [ batch size, __network_n__ ]
        # The gates act over the last embedding of the routing_context.
        routing_score = torch.mean(self.sigmoid(self.gates(routing_context[:, -1, :])), axis=0)

        # Ensure number of queried servers does not exceed metagraph.n
        num_servers = min([self.config.nucleus.topk, metagraph.n])

        print(f'complete \[{time.time() - start_time:.3g}s]')
        print(f'Dendrite \t| Request {num_servers} x \[{batch_size}, {sequence_len - 1}, {bittensor.__network_dim__}] ... ', end='')
        request_start_time = time.time()

        # === Randomly select num_servers UIDs ===
        random_uids = torch.randperm(metagraph.n)[:num_servers]

        # === Get endpoint information for the selected UIDs ===
        # We index into the metagraph's endpoints and return a list of the filtered set of endpoints we wish to query.
        # random_endpoints: List[bittensor.endpoints]: endpoint information for filtered uids.
        # len(neurons) == self.config.nucleus.topk
        random_endpoints = [metagraph.endpoints[uid] for uid in random_uids]

        # === Define which synapse we want to use ===
        # The synapse defines the task we are sending to the servers
        # synapses: List[bittensor.synapse]: synapse information 
        # TODO: WORK IN PROGRESS, prototype
        synapses = [bittensor.synapse.TextCausalLM()]

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
            synapses=synapses,
            timeout=bittensor.__blocktime__
        )

        if not self.config.nucleus.dendrite_backward:
            query_responses = [(res[0].detach(),) for res in query_responses]
            return_ops = [ops.detach() for ops in return_ops]
            times = [time.detach() for time in times]

        print(f'complete \[{time.time() - request_start_time:.3g}s]')
        print(f'Shapley values \t| Calculating ... ', end='')
        shapley_start_time = time.time()

        # Send responses to device. This is required to ensure we move the responses
        # Onto the correct device.
        query_responses = [[response.to(self.device) for response in responses] for responses in query_responses]

        # Send return ops to device.
        return_ops = [ops.to(self.device) for ops in return_ops]

        stats = []
        routing_loss = torch.tensor(0.).to( self.device )
        unsuccessful = []

        def get_num_params(_loss):
            # (OpenAI scaling laws) Kaplan, Jared, et al. "Scaling laws for neural language models." arXiv:2001.08361 (2020)
            num_params = torch.exp(torch.log(torch.tensor(8.8e13)) - torch.log(torch.clamp(_loss, 1.69)) / 0.076)
            scaled_num_params = torch.pow(num_params, 0.5)  # powered down number of params, dynamic range 3 → 6 nats
            return scaled_num_params  # modified scaling law, powered down to improve dynamic range

        # === Base parameter estimation ===
        # Shapley values - base level - coalition size 1
        # Collect successful server responses, calculate base Shapley values.
        # Measured in effective number of model parameters, according to OpenAI scaling laws.
        index_s = 0  # synapse = bittensor.synapse.TextCausalLM()
        for index in range(num_servers):
            _uid = random_uids[index]
            if return_ops[index][index_s] == bittensor.proto.ReturnCode.Success:
                _stats = {'uid': _uid.item(),
                          'response_time': times[index][index_s],
                          'routing_score': routing_score[_uid]}
                
                _stats.update({'logits': query_responses[index][index_s],
                               'logits_val': query_responses[index][index_s][:, -1:, :]})

                for target, ext in [(inputs_seq, ''), (inputs_val, '_val')]:
                    _loss = self.get_target_loss_casuallm(_stats['logits' + ext], target, eval_type = ext)  # CausalLM loss
                    _num_params = get_num_params(_loss)  # estimate the effective number of model parameters

                    _stats.update({'loss' + ext: _loss, 'base_params' + ext: _num_params,
                                   'synergy' + ext: 0, 'synergy_loss_diff' + ext: 0})

                # === Add routing loss ===
                # MSE loss between predicted routing score and ideal target routing score.
                # The Bayes risk approx. 1.69, i.e. the minimal loss achievable for next-token
                # prediction on the full distribution 𝑃, a.k.a the "entropy of natural text"
                # Hoffmann, Jordan, et al. "Training Compute-Optimal Large Language Models." arXiv:2203.15556 (2022).
                routing_score_target = torch.exp(-torch.clamp(_stats['loss'] - 1.69, 0))
                _routing_loss = (routing_score[_uid] - routing_score_target) ** 2  # MSE loss
                routing_loss += _routing_loss
                _stats.update({'routing_score_target': routing_score_target, 'routing_loss': _routing_loss})
                stats += [_stats]
            else:
                unsuccessful += [(_uid, return_ops[index][index_s], times[index][index_s])]

        # === Shapley synergy approximation ===
        # Shapley values - second level - coalition size 2
        # Synergy = measured performance above expected performance
        # Measured in effective number of model parameters, just like base Shapley values.
        syn_loss_diff = {}  # expected_loss - measured_loss (where > 0)
        for _first in range(len(stats)):
            first = stats[_first]
            first_diff = syn_loss_diff.setdefault(first['uid'], {})
            first_diff[first['uid']] = torch.min(first['loss'], first['loss_val'])  # diagonal keeps best direct loss

            for _second in range(_first + 1, len(stats)):
                second = stats[_second]
                second_diff = syn_loss_diff.setdefault(second['uid'], {})
                second_diff.setdefault(first['uid'], torch.tensor(0.))
                first_diff.setdefault(second['uid'], torch.tensor(0.))

                with torch.no_grad():
                    for target, ext in [(inputs_seq, ''), (inputs_val, '_val')]:
                        expected_loss = torch.min(first['loss' + ext], second['loss' + ext])  # expecting min loss
                        combined_logits = (first['logits' + ext] + second['logits' + ext]) / 2  # combined logits
                        measured_loss = self.get_target_loss_casuallm(combined_logits, target, eval_type=ext)  # actual loss

                        loss_diff_share = torch.clamp(expected_loss - measured_loss, 0) / 2  # record direct loss diff
                        first['synergy_loss_diff' + ext] += loss_diff_share
                        second['synergy_loss_diff' + ext] += loss_diff_share

                        # pairwise loss reduction of expected and measured loss due to synergy between first and second
                        first_diff[second['uid']] = torch.max(loss_diff_share, first_diff[second['uid']])
                        second_diff[first['uid']] = torch.max(loss_diff_share, second_diff[first['uid']])

                        synergy_share = torch.clamp(get_num_params(measured_loss) -
                                                    get_num_params(expected_loss), 0) / 2
                        first['synergy' + ext] += synergy_share  # share synergy amongst coalition members
                        second['synergy' + ext] += synergy_share

        # === Shapley value combination ===
        # Combine base values with synergy approximation to get final Shapley values.
        for s in stats:
            for ext in ['', '_val']:
                s['shapley_values' + ext] = (s['base_params' + ext] + s['synergy' + ext])
                del s['logits' + ext]  # remove logits - not needed for stats anymore

            # use minimum of sequence/validation Shapley values, for added adversarial resilience
            s['shapley_values_min'] = torch.min(s['shapley_values'], s['shapley_values_val'])

            for key in s:
                if hasattr(s[key], 'item'):
                    s[key] = s[key].item()

        print(f'complete \[{time.time() - shapley_start_time:.3g}s]')

        # === Synergy table ===
        # Prints the synergy loss diff matrix with pairwise loss reduction due to synergy (original loss on diagonal)
        sort = sorted([(s['uid'], s['shapley_values_min']) for s in stats], reverse=True, key=lambda _row: _row[1])
        uid_col = neuron_stats_columns[0]  # [Column_name, key_name, format_string, rich_style]
        columns = [uid_col] + [[f'{s[0]}', '', '{:.2f}', ''] for s in sort]
        rows = [[uid_col[2].format(s[0])] +
                [('[bright_cyan]{:.2f}[/bright_cyan]' if t == s else
                  '[magenta]{:.2f}[/magenta]' if syn_loss_diff[s[0]][t[0]] > 0 else
                  '[dim]{:.0f}[/dim]').format(syn_loss_diff[s[0]][t[0]]) for t in sort] for s in sort]

        table = Table(width=self.config.get('width', None), box=None)
        table.title = f'[white] Synergy [/white]'
        table.caption = f'loss decrease'
        for col, _, _, stl in columns:
            table.add_column(col, style=stl, justify='right')
        for row in rows:
            table.add_row(*row)

        if len(rows):
            print(table)
            print()

        # === Neuron responses (table) ===
        # Prints the evaluation of the neuron responses to the validator request
        columns = [_[:] for _ in neuron_stats_columns if _[0] not in ['Upd', 'Weight']]
        sort_col = [_[0] for _ in columns].index('mShap')  # sort column with key of shapley_values_min
        columns[sort_col][0] += '\u2193'  # ↓ downwards arrow (sort)
        rows = [[txt.format(s[key]) for _, key, txt, _ in columns] for s in stats]
        rows = sorted(rows, reverse=True, key=lambda _row: int(_row[sort_col]))  # sort according to mShap column

        table = Table(width=self.config.get('width', None), box=None, row_styles=[Style(bgcolor='grey15'), ""])
        table.title = f'[white] Neuron responses [/white] | Validator forward'
        table.caption = f'[bold]{num_servers}[/bold]/{metagraph.n} (topk/total) | [bold]TextCausalLM[/bold] | ' \
                        f'[white] {len(stats)} x \[{batch_size}, {sequence_len - 1}, {bittensor.__network_dim__}] ' \
                        f'\[{time.time() - start_time:.3g}s] [/white]'

        for col, _, _, stl in columns:
            table.add_column(col, style=stl, justify='right')
        for row in rows:
            table.add_row(*row)

        print(table)

        unsuccess_txt = f'Unsuccessful \t| [cyan]UID[/cyan]\[[red]return_op[/red] [yellow]time[/yellow]]: '
        for _uid, _return_op, _time in unsuccessful:
            unsuccess_txt += f'{_uid}[[red]{_return_op}[/red] [yellow not bold]{_time:.2f}[/yellow not bold]] '
        print(unsuccess_txt)

        return routing_loss, stats
