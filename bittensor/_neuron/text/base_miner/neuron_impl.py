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
from ..base.neuron import BaseNeuron  
from ..base.neuron import BaseServer  
from ..base.neuron import BaseNeuronLog  


class Neuron(BaseNeuron, BaseServer, BaseNeuronLog):

    def __init__( self, config: 'bittensor.config', nucleus: 'Nucleus'):
        r""" Initializes the neuron with the passed config.
        """
        self.nucleus = nucleus
        super().__init__(config, nucleus) # init for BaseNeuron              
        super(bittensor.neurons.text.base.BaseNeuron, self).__init__() # init for BaseServer
        self.nucleus = nucleus.to(self.device)
        self.optimizer = torch.optim.SGD(
            [ {'params': self.nucleus.peer_weights, 'lr': self.config.neuron.learning_rate_chain} ],
            lr = self.config.neuron.learning_rate,
            momentum = self.config.neuron.momentum,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
            step_size = 1.0,
            gamma = 0.95
        )
        # ---- Decay factor for fisher ema score 
        self.fisher_ema_decay = 0.995

    def __enter__(self):
        self.wallet.create()
        self.subtensor.register( self.wallet )
        self.axon_start()

    def __exit__ ( self, exc_type, exc_value, exc_traceback ):
        self.axon_stop()
        print(exc_type, exc_value, exc_traceback)
    
    def run( self ):
        r""" Miner main loop.
        """
        # ---- Build Bittensor neuron ----
        with self:
            if self.config.neuron.use_wandb:
                bittensor.wandb(
                    config = self.config,
                    cold_pubkey = self.wallet.coldkeypub.ss58_address,
                    hot_pubkey = self.wallet.hotkey.ss58_address,
                    root_dir = self.config.neuron.full_path
                )

            # ---- Init run state ----
            self.epoch = 0   

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
            
            self.stats.ema_scores = torch.nn.Parameter(torch.ones(self.metagraph.n.item()).to(self.device) * (1 / self.metagraph.n.item()), requires_grad = False)

            # --- Run until n_epochs ----
            while self.epoch < self.config.neuron.n_epochs:
                try:
                    # --- Init epoch stat----
                    self.stats.epoch_data_size = 0
                    self.stats.epoch_sync_count = 0
                    total_local_target_epoch_loss = 0
                    total_distillation_epoch_loss = 0
                    total_remote_target_epoch_loss = 0
                    total_local_epoch_acc = 0
                    batches_count = 0

                    # ---- Run epoch ----
                    start_block = self.subtensor.get_current_block() + 1
                    end_block = start_block + self.config.neuron.epoch_length
                    block_steps = [ block_delta for block_delta in range(start_block, end_block)]
                    progress_bar = qqdm( block_steps, total=len(block_steps), desc=format_str('blue', f'Epoch:'))
                    progress_bar.set_bar = partial(progress_bar.set_bar,  element='#')
                    for block in progress_bar:

                        # --- Iterate over batches until the end of the block.
                        current_block = self.subtensor.get_current_block()
                        while block >= current_block:
                            # ---- Forward pass ----
                            inputs = next( self.dataset )
                            output = self.nucleus.remote_forward (
                                inputs = inputs.to( self.device ),
                                training = True,
                            )
                            
                            # ---- Backward pass ----
                            output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
                            scores = torch.nn.functional.normalize ( torch.relu( self.nucleus.compute_scores(output.remote_target_loss) ), p=1, dim = 0 )
                            scores[output.query_uids] += 1e-6

                            output.loss.backward() # Accumulates gradients on the nucleus.
                            clip_grad_norm_(self.nucleus.parameters(), self.config.neuron.clip_gradients)
                            
                            # ---- Apply and zero accumulated gradients.
                            self.optimizer.step() 
                            self.optimizer.zero_grad()
                            current_block = self.subtensor.get_current_block()
                            
                            # ---- Aggrigate outputs and losses 
                            total_local_target_epoch_loss += output.local_target_loss.item()
                            total_distillation_epoch_loss += output.distillation_loss.item()
                            total_remote_target_epoch_loss += output.remote_target_loss.item()
                            total_local_epoch_acc += output.local_accuracy
                            self.stats.epoch_data_size += inputs.nelement()
                            batches_count += 1
                            
                            # ---- Expand ema_scores tensor if the chain grew and aggrigate the score
                            chain_growth = max(scores.shape[0] - self.stats.ema_scores.shape[0], 0)
                            if chain_growth > 0:
                                self.stats.ema_scores = torch.nn.Parameter(torch.cat( [self.stats.ema_scores, torch.zeros([chain_growth], dtype=torch.float32, device = self.device)]), requires_grad=False)
                            self.stats.ema_scores = self.fisher_ema_decay * self.stats.ema_scores + (1 - self.fisher_ema_decay) * scores
                            self.stats.scores = scores


                        # ---- Sync with metagraph if the current block >= last synced block + sync block time 
                        current_block = self.subtensor.get_current_block()
                        block_diff = current_block - self.stats.last_sync_block
                        if block_diff >= self.config.neuron.sync_block_time:
                            self.set_peer_weights()
                            self.meta_sync(current_block)                                                                                                                
                            self.stats.last_sync_block = current_block
                            self.stats.epoch_sync_count += 1
                            
                        # ---- Update the epoch loss if it is the last iteration within epoch
                        if block+1 == end_block :
                            self.stats.local_target_epoch_loss = total_local_target_epoch_loss / batches_count
                            self.stats.distillation_epoch_loss = total_distillation_epoch_loss / batches_count
                            self.stats.remote_target_epoch_loss = total_remote_target_epoch_loss / batches_count
                            self.stats.local_epoch_acc = total_local_epoch_acc / batches_count

                        # ---- Block logs.
                        self.logs (
                            progress_bar,
                            iteration = block-start_block,
                            output = output,
                        )
                        self.stats.global_step += 1

                    # ---- Update params ----
                    self.epoch += 1

                    # ---- Checkpoint state ----
                    self.checkpoint()

                except KeyboardInterrupt:
                    # --- User ended session ----
                    break

                except Exception as e:
                    # --- Unknown error ----
                    logger.exception('Unknown exception: {} with traceback {}', e, traceback.format_exc())
                    if self.config.neuron.restart_on_failure == True:
                        logger.info('Restarting from last saved state.')
                        self.reload()
                    else:
                        break



