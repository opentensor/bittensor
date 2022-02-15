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
""" Base neuron version 1.

Example:
    $ import neurons
    $ neurons.text.base_neuron_v1().run()

"""

from re import I
import pandas
from pandas.core.frame import DataFrame
import bittensor
import math
import torch
import traceback
import sys
import wandb
from termcolor import colored
from qqdm import qqdm, format_str
from loguru import logger

from bittensor._metagraph import metagraph
logger = logger.opt(colors=True)

from types import SimpleNamespace
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
from functools import partial

import torch.nn.functional as F

class BaseStatsKeeper():
    def __init__(self):
        self.stats = SimpleNamespace(
            global_step = 0,
            last_sync_block = 0,
            epoch_data_size = 0,
            epoch_sync_count = 0,
            local_target_epoch_loss = math.inf,
            distillation_epoch_loss = math.inf,
            remote_target_epoch_loss = math.inf,
            local_epoch_acc = 0,
            best_epoch_loss = math.inf,
            scores = torch.nn.Parameter(torch.zeros(0), requires_grad = False).to(self.device),
            ema_scores = torch.nn.Parameter(torch.zeros(0), requires_grad = False).to(self.device)
        )

        self.restart_total_epoch_loss()

        assert self.config != None, 'self.config was not initialized.'
        assert self.epoch != None, 'self.epoch was not initialized.'
        assert self.wallet != None, 'self.wallet was not initialized.'
        assert self.subtensor != None, 'self.subtensor was not initialized.'
        assert self.metagraph != None, 'self.metagraph was not initialized.'

    def start_wandb(self):
        bittensor.wandb(
            config = self.config,
            cold_pubkey = self.wallet.coldkeypub.ss58_address,
            hot_pubkey = self.wallet.hotkey.ss58_address,
            root_dir = self.config.neuron.full_path
        )

    def restart_total_epoch_loss(self):
        self.total_epoch_loss = SimpleNamespace(
            local_target = 0,
            distillation = 0,
            remote_target = 0,
            local_acc = 0,
            batches_count = 0
        ) 
    def restart_epoch_stats(self):
        self.stats.epoch_data_size = 0
        self.stats.epoch_sync_count = 0
        self.restart_total_epoch_loss()
        
    def agg_total_epoch_loss(self, output):
        self.total_epoch_loss.local_target += output.local_target_loss.item()
        self.total_epoch_loss.distillation += output.distillation_loss.item()
        self.total_epoch_loss.remote_target += output.remote_target_loss.item()
        self.total_epoch_loss.local_acc += output.local_accuracy
        self.total_epoch_loss.batches_count += 1

    def update_avg_epoch_loss(self):
        batches_count = self.total_epoch_loss.batches_count
        self.stats.local_target_epoch_loss = self.total_epoch_loss.local_target / batches_count
        self.stats.distillation_epoch_loss = self.total_epoch_loss.distillation / batches_count
        self.stats.remote_target_epoch_loss = self.total_epoch_loss.remote_target / batches_count
        self.stats.local_epoch_acc = self.total_epoch_loss.local_acc / batches_count
        self.restart_total_epoch_loss()

    def init_scores(self):
        self.stats.ema_scores = torch.nn.Parameter(torch.ones(self.metagraph.n.item()).to(self.device) * (1 / self.metagraph.n.item()), requires_grad = False)

    def update_scores(self, scores):
        # ---- Expand ema_scores tensor if the chain grew and aggrigate the score
        chain_growth = max(scores.shape[0] - self.stats.ema_scores.shape[0], 0)
        if chain_growth > 0:
            self.stats.ema_scores = torch.nn.Parameter(torch.cat( [self.stats.ema_scores, torch.zeros([chain_growth], dtype=torch.float32, device = self.device)]), requires_grad=False)
        self.stats.ema_scores = self.fisher_ema_decay * self.stats.ema_scores + (1 - self.fisher_ema_decay) * scores
        self.stats.scores = scores

    def log_progress_bar(self, progress_bar, output, my_neuron, block):
        # ---- Progress bar log
        info = {
            'Step': colored('{}'.format(self.stats.global_step), 'red'),
            'Epoch': colored('{}'.format(self.epoch+1), 'yellow'),
            'Best-loss': colored('{:.4f}'.format(self.stats.best_epoch_loss), 'green'),          
            'L-loss': colored('{:.4f}'.format(output.local_target_loss.item()), 'blue'),
            'R-loss': colored('{:.4f}'.format(output.remote_target_loss.item()), 'red'),
            'D-loss': colored('{:.4f}'.format(output.distillation_loss.item()), 'yellow'),
            'L-acc': colored('{:.4f}'.format(output.local_accuracy), 'green'),
            'nPeers': colored(self.metagraph.n.item(), 'blue'),
            'Stake(\u03C4)': colored('{:.3f}'.format(my_neuron.stake), 'red'),
            'Rank(\u03C4)': colored('{:.3f}'.format(my_neuron.rank), 'yellow'),
            'Incentive(\u03C4/block)': colored('{:.6f}'.format(my_neuron.incentive), 'green'),
            'Current Block': colored('{}'.format(block), 'blue'),
            'Synced Block': colored('{}'.format(self.stats.last_sync_block), 'yellow'),
        }
        progress_bar.set_infos( info )

    def wandb_log(self, my_neuron, block):
        my_uid = my_neuron.uid
        self.update_average_epoch_loss()

        # ---- Miner summary for wandb
        wandb_info = {
            'neuron/stake': my_neuron.stake,
            'neuron/rank': my_neuron.rank,
            'neuron/incentive': my_neuron.incentive,
            'neuron/num_peers':self.metagraph.n.item(),
            'nucleus/remote_target_epoch_loss': self.stats.remote_target_epoch_loss,
            'nucleus/distillation_epoch_loss': self.stats.distillation_epoch_loss,
            'nucleus/local_target_epoch_loss': self.stats.local_target_epoch_loss,
            'nucleus/local_epoch_acc': self.stats.local_epoch_acc,
            'neuron/num_sync_metagraph': self.stats.epoch_sync_count,
            'neuron/data_size': self.stats.epoch_data_size,
        }

        # ---- Build stats dataframe.
        normalized_peer_weights = F.softmax (self.nucleus.peer_weights.detach(), dim=0)
        k = min( self.config.neuron.n_topk_peer_weights, self.metagraph.n.item() )
        topk_scores, topk_uids = bittensor.unbiased_topk( self.stats.ema_scores, k, dim=0 )

        df = pandas.concat( [
            bittensor.utils.indexed_values_to_dataframe( prefix = 'fisher_ema_score', index = topk_uids, values = self.stats.ema_scores, filter_zeros = True),
            bittensor.utils.indexed_values_to_dataframe( prefix = 'raw_peer_weight', index = topk_uids, values = self.nucleus.peer_weights, filter_zeros = True),
            bittensor.utils.indexed_values_to_dataframe( prefix = 'normalized_peer_weight', index = topk_uids, values = normalized_peer_weights, filter_zeros = True),
            bittensor.utils.indexed_values_to_dataframe( prefix = 'w_{}_i'.format(my_uid), index = topk_uids, values = self.metagraph.W[ my_uid, : ], filter_zeros = True),
            bittensor.utils.indexed_values_to_dataframe( prefix = 'w_i_{}'.format(my_uid), index = topk_uids, values = self.metagraph.W[ :, my_uid ], filter_zeros = True),
            self.axon.to_dataframe( metagraph = self.metagraph ),
            self.dendrite.to_dataframe( metagraph = self.metagraph )
        ], axis = 1)
        df['uid'] = df.index
        stats_data_table = wandb.Table( dataframe = df)

        wandb_info_axon = self.axon.to_wandb()
        wandb_info_dend = self.dendrite.to_wandb()
        wandb.log( { **wandb_info, **wandb_info_axon, **wandb_info_dend }, step = block)
        wandb.log( { 'stats': stats_data_table}, step = block)
        wandb.log( { 'axon_query_times': wandb.plot.scatter( stats_data_table, "uid", "axon_query_time", title="Axon Query time vs UID") } )
        wandb.log( { 'dendrite_query_times': wandb.plot.scatter( stats_data_table, "uid", "dendrite_query_time", title="Dendrite Query time vs UID") } )
        
    # ---- Training logs ----
    def logs( self, progress_bar, iteration:int, output: SimpleNamespace ):
        r""" Called after every training step. Displays miner state to screen.
        """
        my_neuron = self.subtensor.neuron_for_pubkey( self.wallet.hotkey.ss58_address )
        current_block = self.subtensor.get_current_block()

        self.log_progress_bar(progress_bar, output, my_neuron, current_block)
        
        if self.config.neuron.use_wandb and ((iteration + 1) % (self.config.neuron.epoch_length ) == 0):
            self.log_wandb(my_neuron, current_block)