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

class BaseNeuron():
    def __init__( self, config: 'bittensor.config', nucleus: 'Nucleus'):

        self.config = config
        self.wallet = bittensor.wallet ( config = self.config )
        self.subtensor = bittensor.subtensor ( config = self.config )
        self.metagraph = bittensor.metagraph ( config = self.config, subtensor = self.subtensor )
        self.dendrite = bittensor.dendrite ( config = self.config, wallet = self.wallet )
        self.dataset = bittensor.dataset ( config = self.config )
        self.device = torch.device( device = self.config.neuron.device )

        self.nucleus.metagraph = lambda : self.metagraph
        self.nucleus.dendrite = self.dendrite
        
    def init_load(self):
        # ---- reloads previous run if not restart ----
        if self.config.neuron.no_restart:
            self.save()

        try:
            self.reload()
            self.axon.check()
        except Exception as e:
            logger.error("Error when trying to reload model: {}".format(e))
            self.save()
            self.reload()
            self.axon.check()

    def save( self ):
        r""" Saves the training state to disk.
        """
        try:
            state_dict = {
                'epoch': self.epoch,
                'epoch_loss': self.stats.local_target_epoch_loss,
                'global_step': self.stats.global_step,
                'nucleus_state': self.nucleus.state_dict(), # Save nucleus state.
                'optimizer_state': self.optimizer.state_dict(), # Save optimizer.
                'network': self.subtensor.network # Save Network
            }
            torch.save( state_dict, "{}/model.torch".format( self.config.neuron.full_path ) )
            bittensor.logging.success(prefix='Saved model', sufix='<blue>{}/model.torch</blue>'.format( self.config.neuron.full_path ) )
        except Exception as e:
            logger.exception('Failed to save model with error:{}', e)

    def get_saved_state( self ):
        r""" Returns a saved state dict or none.
        """
        try:
            return torch.load("{}/model.torch".format( self.config.neuron.full_path ))
        except Exception as e:
            logger.warning('No saved model found with error: {}', e)
            return None

    def reload_stat(self, state_dict):
        self.epoch = state_dict['epoch']
        self.stats.local_target_epoch_loss = state_dict['epoch_loss']
        self.stats.best_epoch_loss = state_dict['epoch_loss']
        self.stats.global_step = state_dict['global_step']

    def reload_nucleus(self, state_dict):

        # ---- Model parameters
        try:
            self.nucleus.load_state_dict( state_dict['nucleus_state'], strict=False )
        except Exception as e:
            logger.exception('Failed to load nucleus state with error, updating the current state')
            state_dict['nucleus_state'] = self.nucleus.state_dict()
            torch.save( state_dict, "{}/model.torch".format( self.config.neuron.full_path ) )

        # ---- Optimizer
        self.optimizer = torch.optim.SGD(
            [{"params": self.nucleus.parameters()}],
            lr = state_dict['optimizer_state']['param_groups'][0]['lr'],
            momentum = state_dict['optimizer_state']['param_groups'][0]['momentum'],
        )

        # ---- Peer weight
        self.nucleus.peer_weights = nn.Parameter(
            torch.ones(
                list(state_dict['nucleus_state']['peer_weights'].shape),
                requires_grad=True
            ).to(self.device)
        )
        chain_growth = max(0, self.metagraph.n.item() - state_dict['nucleus_state']['peer_weights'].shape[0])
        self.nucleus.peer_weights = nn.Parameter(torch.cat([self.nucleus.peer_weights, torch.ones([chain_growth],dtype=torch.float32,requires_grad=True, device = self.device)]))

        # ---- device, dendrite, metagraph
        self.nucleus.to( self.device ) # Load nucleus
        self.nucleus.dendrite = self.dendrite # Set local dendrite.
        self.nucleus.metagraph = self.metagraph_callback # Set local metagraph.
        bittensor.logging.success( prefix = 'Reloaded model', sufix = '<blue>{}/model.torch</blue>'.format( self.config.neuron.full_path ))        

    def reload( self ):
        r""" Reloads/updates the training state from the disk.
        """        
        # --- Loads and syncs metagraph.
        self.meta_sync()

        # ---- Load training state.
        state_dict = self.get_saved_state()
        self.reload_stat(state_dict)

    def checkpoint( self ):
        r""" Optionally Saves, updates and then reloads the miner training state.
        """
        last_saved = self.get_saved_state()
        if last_saved == None or last_saved['epoch_loss'] >= self.stats.local_target_epoch_loss:
            self.stats.best_epoch_loss = self.stats.local_target_epoch_loss
            self.save()

        # Checks if epochs managed to diverage
        if not math.isfinite(self.stats.local_target_epoch_loss):
            logger.error('Incorrect epoch loss detected, reloading to previous saved state')
            self.reload()

    def meta_sync (self, current_block = None ):
        """ Miner sync with metagraph and update chain weight
        """
        # ---- Sync with metagraph ----
        self.metagraph.sync()
        self.stats.last_sync_block= self.subtensor.get_current_block()
        chain_growth = max(self.metagraph.n.item()- self.nucleus.peer_weights.shape[0], 0)
        self.nucleus.peer_weights = nn.Parameter(torch.cat([self.nucleus.peer_weights, torch.ones([chain_growth],dtype=torch.float32,requires_grad=True).to(self.device)]))
        self.stats.scores = torch.nn.Parameter(torch.cat( [self.stats.scores, torch.zeros([chain_growth], dtype=torch.float32, requires_grad=False).to(self.device)]))
        self.stats.ema_scores = torch.nn.Parameter(torch.cat( [self.stats.ema_scores, torch.zeros([chain_growth], dtype=torch.float32, requires_grad=False).to(self.device)]))
        bittensor.logging.success( 'Synced metagraph:', 'Block: {}'.format(current_block))

    def set_peer_weights( self ):
        r""" Sets the fisher ema score to peers.
        """

        try:
            k = min( self.config.neuron.n_topk_peer_weights, self.metagraph.n.item() )
            inactive_uids = torch.where(self.metagraph.active == 0)[0]
            self.stats.ema_scores[inactive_uids] = 0
            topk_scores, topk_uids = bittensor.unbiased_topk( self.stats.ema_scores , k = k )
            topk_uids = topk_uids.detach().to('cpu')
            topk_scores = topk_scores.detach().to('cpu')
            self.subtensor.set_weights(
                uids = topk_uids,
                weights = topk_scores,
                wait_for_inclusion = False,
                wallet = self.wallet,
            )

        except Exception as e:
            logger.error('Failure setting weights on chain with error: {}', e)

    def check_sync(self):
        # ---- Sync with metagraph if the current block >= last synced block + sync block time 
        current_block = self.subtensor.get_current_block()
        block_diff = current_block - self.stats.last_sync_block
        if block_diff >= self.config.neuron.sync_block_time:
            self.set_peer_weights()
            self.meta_sync(current_block)                                                                                                                
            self.stats.last_sync_block = current_block
            self.stats.epoch_sync_count += 1