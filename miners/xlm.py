#!/bin/python3.7

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

"""XLM Language Modelling miner

This file demonstrates training the XLM miner with language modelling.

Example:
    $ python miners/text/xlm.py

To run with a config file and debug
    $ python miners/text/xlm.py --debug --config <path to config file>

"""

import argparse
import copy
import os
import math
import random
import torch
import sys
import torch.nn.functional as F
import bittensor

from qqdm import qqdm, format_str
from tqdm import tqdm
from munch import Munch
from termcolor import colored
from types import SimpleNamespace
from synapses.xlm import XLMSynapse
from typing import Tuple, List, Optional
from bittensor.dataloaders.text_dataloader import GenesisTextDataloader
from pytorch_transformers import WarmupCosineWithHardRestartsSchedule

from loguru import logger
logger = logger.opt(colors=True)

class Miner( bittensor.miner.BaseMiner ):

    def __init__( 
            self, 
            config: Munch = None,
            **kwargs
        ):
        # ---- Load Config ----
        if config == None:
            config = Miner.default_config();   
        config = copy.deepcopy(config); bittensor.config.Config.update_with_kwargs(config, kwargs )
        Miner.check_config( config )
        logger.info( bittensor.config.Config.toString( config ) )
        self.config = config

        # ---- Row Weights ----
        self.row_weights = torch.ones([1])

        # ---- Synapse ----
        self.synapse = XLMSynapse( self.config )

        # ---- Optimizer ----
        self.optimizer = torch.optim.SGD(self.synapse.parameters(), lr = self.config.miner.learning_rate, momentum=self.config.miner.momentum)
        self.scheduler = WarmupCosineWithHardRestartsSchedule(self.optimizer, 50, 300)

        # ---- Dataset ----
        self.dataset = GenesisTextDataloader( self.config.miner.batch_size_train, 20 )
        super(Miner, self).__init__( self.config, **kwargs )
    
    @staticmethod
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        Miner.add_args( parser ) 
        config = bittensor.config.Config.to_config( parser ); 
        return config
    
    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        parser.add_argument('--miner.learning_rate', default=0.01, type=float, help='Training initial learning rate.')
        parser.add_argument('--miner.momentum', default=0.98, type=float, help='Training initial momentum for SGD.')
        parser.add_argument('--miner.epoch_length', default=500, type=int, help='Iterations of training per epoch')
        parser.add_argument('--miner.n_epochs', default=-1, type=int, help='Number of training epochs, if < 0 runs for ever.')
        parser.add_argument('--miner.batch_size_train', default=1, type=int, help='Training batch size.')
        parser.add_argument('--miner.name', default='xlm', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ')
        XLMSynapse.add_args( parser )
        bittensor.miner.BaseMiner.add_args( parser )
        GenesisTextDataloader.add_args( parser )

    @staticmethod
    def check_config(config: Munch):
        assert config.miner.momentum > 0 and config.miner.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.miner.batch_size_train > 0, "batch_size_train must be a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
        XLMSynapse.check_config( config )
        bittensor.miner.BaseMiner.check_config( config )
        GenesisTextDataloader.check_config( config )

    # ---- Axon Forward call ----
    def forward_call( self, pubkey:str, inputs: torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        r""" Called by miner.forward_loop which can be overridden by the child class.
            The arguments reflect an RPC request from another miner in the network, the response tensor
            should be the hidden units of the local synapse of shape [batch_size, sequence_len, __network_dim__].
            
            Args:
                pubkey ( str, `required`): 
                    The public key of the caller.
                inputs ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                modality ( bittensor.proto.Modality, `required`):
                    modality of inputs e.g. bittensor.proto.Modality.TEXT.
            
            Returns:
                outputs (:obj:`torch.FloatTensor`): 
                    The synapse's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
        """
        output = self.synapse.local_forward (
            inputs = inputs        
        )
        return output.local_hidden

    # ---- Axon Backward call ----
    def backward_call( self, pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        r""" Called by miner.backward_loop which can be overridden in the child class.
            Arguments reflect an RPC backward request from another miner in the network, the response tensor
            should be the gradients of the miner's synapse w.r.t to the inputs and the passed output grads.
            
            Args:
                pubkey ( str, `required`): 
                    The public key of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs from previous forward call.
                grads_dy ( :obj:`torch.Tensor`, `required`):
                    torch grads of forward output.
                modality ( bittensor.proto.Modality, `required`):
                    modality of inputs e.g. bittensor.proto.Modality.TEXT.
            
            Returns:
                outputs (:obj:`torch.FloatTensor`): 
                    The gradients w.r.t to the inputs [batch_size, sequence_len, __network_dim__]
        """
        # Not processing backward requests
        return None

    def should_run( self, epoch: int ) -> bool:
        r""" Called by miner.run() every epoch, if the response is false, training stops.
        """
        if self.config.miner.n_epochs < 0:
            return True
        elif epoch < self.config.miner.n_epochs:
            return True
        else:
            return False

    def should_save( self ) -> bool:
        r""" Called by miner.run() after every epoch.
            If this function returns True, the model is saved to disk and can be reloaded later.
            Returns:
                should_save (bool):
                    True by default. Saves model after each epoch.
        """
        if self.epoch_loss < self.last_saved_loss:
            return True
        else:
            return False

    def should_reload(self) -> bool:
        r""" Called by miner.run() after every epoch.
            If the function returns True the model state dict is saved to miner.full_path.
            Returns:
                should_reload (bool):
                    False by default. Does not reload the model after each epoch.
        """
        if torch.any(torch.isnan(torch.cat([param.view(-1) for param in self.synapse.parameters()]))):
            return True

    def get_state_dict( self ) -> dict:
        r""" Called by miner.save_model().
            Returns a state dict which can be passed to miner.reload_from_state_dict on reload.
            Returns:
                state_dict (:obj:`dict`): 
                    Dictionary containing run state information such as the model parameters.
        """
        return {
            'synapse_state': self.synapse.state_dict(), 
            'optimizer_state': self.optimizer.state_dict(),
        }

    def reload_from_state_dict( self, state_dict: dict):
        r""" Called by miner.reload_model().
            Reloads the training state from the passed state_dict. 
            Args:
                state_dict (:obj:`dict`): 
                    Dictionary containing run state information such as the model parameters. Output 
                    of get_state_dict.
        """
        self.synapse.load_state_dict( state_dict['synapse_state'] )
        self.optimizer.load_state_dict( state_dict['optimizer_state'] )

    # ---- Get Row Weights ----
    def get_row_weights( self ) -> torch.FloatTensor:
        r""" Called after each training epoch. Returns row_weights to be set on chain.
            Returns:
                row_weights ( torch.FloatTensor, shape=(self.metagraph.n) ): 
                    torch row_weights matching the metagraph size.
                    weight values should be normalized and be in range [0,1].
        """
        self.row_weights = torch.nn.functional.pad( self.row_weights, pad = [0, self.metagraph.n - self.row_weights.numel()] )
        self.row_weights = F.normalize( self.row_weights, p = 1, dim = 0) # Ensure normalization.
        return self.row_weights

    # ---- Epcoch ending logs ---
    def epoch_logs(self):
        r""" Called by miner.run() after each epoch.
            Sends miner state to tensorboard.
        """
        self.axon.__to_tensorboard__( self.tensorboard, self.global_step )
        self.dendrite.__to_tensorboard__( self.tensorboard, self.global_step )
        self.metagraph.__to_tensorboard__( self.tensorboard, self.global_step )

    # ---- Training logs ----
    def training_logs( self, progress_bar, iteration:int, output: SimpleNamespace ):
        r""" Called by miner.run_training_epoch() after each training step.
            The function populates and displays the passed progress bar.
        """
        index = self.metagraph.state.index_for_uid[self.metagraph.uid]
        progress_bar.set_infos({
            'GS': colored('{}'.format(self.global_step), 'red'),
            'LS': colored('{}'.format(iteration), 'blue'),
            'Epoch': colored('{}'.format(self.epoch+1), 'green'),
            'Epoch-loss': colored('{:.4f}'.format(self.epoch_loss), 'yellow'),
            'Saved-loss': colored('{:.4f}'.format(self.last_saved_loss), 'red'),
            'L-loss': colored('{:.4f}'.format(output.local_target_loss.item()), 'blue'),
            'R-loss': colored('{:.4f}'.format(output.remote_target_loss.item()), 'green'),
            'D-loss': colored('{:.4f}'.format(output.distillation_loss.item()), 'yellow'),
            'nPeers': colored(self.metagraph.n, 'red'),
            'Stake(\u03C4)': colored('{:.3f}'.format(self.metagraph.S[index]), 'green'),
            'Rank(\u03C4)': colored('{:.3f}'.format(self.metagraph.R[index]), 'blue'),
            'Incentive(\u03C4/block)': colored('{:.6f}'.format(self.metagraph.I[index]), 'yellow'),
            'Axon': self.axon.__str__(),
            'Dendrite': self.dendrite.__str__(),
        })
        self.tensorboard.add_scalar('R-loss', output.remote_target_loss.item(), self.global_step)
        self.tensorboard.add_scalar('L-loss', output.local_target_loss.item(), self.global_step)
        self.tensorboard.add_scalar('D-loss', output.distillation_loss.item(), self.global_step)

    # ---- Training call ----
    def training_call( self, batch: dict ) -> SimpleNamespace:
        r""" Runs a single training batch through the synapse and applies a gradient update.
            Args:
                batch ( dict, `required`): 
                    training batch dictionary as returned from get_epoch_batches            
            Returns:
                outputs ( SimpleNamespace ): 
                    SimpleNamespace output as returned by a synapse forward call.
                    Must include fields local_loss, remote_loss, distillation_loss
        """
        # ---- Forward pass ----
        inputs = batch['inputs'].to(self.synapse.device)
        output = self.synapse.remote_forward(
            neuron = self,
            inputs = inputs,
            training = True,
        )

        # ---- Backward pass ----
        output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
        output.loss.backward() # Accumulates gradients on the synapse.
        self.optimizer.step() # Applies accumulated gradients.
        self.optimizer.zero_grad() # Zeros out gradients for next accummulation

        # ---- Train row weights ----
        batch_weights = torch.mean(output.router.weights, axis = 0).to( self.synapse.device ) # Average over batch.
        self.row_weights = (1 - 0.03) * self.row_weights + 0.03 * batch_weights # Moving avg update.
        self.row_weights = F.normalize( self.row_weights, p = 1, dim = 0) # Ensure normalization.

        # ---- Update global loss ----
        return output

    # --- Run Epoch ----
    def run_epoch( self ):
        r""" Called by miner.run(), calls training_call for passed batches.
        """
        # --- Init Epoch ----
        total_epoch_loss = 0.0
        training_batches = self.dataset.dataloader( self.config.miner.epoch_length )
        progress_bar = qqdm(enumerate(training_batches), total=len(training_batches), desc=format_str('blue', f'Epoch Progress'))
        for iteration, (inputs) in progress_bar:

            # ---- Forward / Backward ----
            output = self.training_call( batch = { 'inputs': inputs }  )

            # ---- Update training state ----
            total_epoch_loss += output.local_target_loss.item()
            self.epoch_loss = total_epoch_loss / (iteration + 1) 
            self.global_step += 1
            self.training_logs( progress_bar, iteration = iteration, output = output )

        self.epoch += 1

    # --- Run Miner ----
    def run( self ):
        r""" Miner main loop.
        """
        # ---- Setup ----
        with self:

            # --- Run state ----
            self.epoch = -1
            self.epoch_loss = math.inf
            self.global_step = 0        
            self.save_state()

            # --- Run until ----
            while self.should_run( self.epoch ):
                try:
                    # ---- Train ----
                    self.run_epoch()

                    # ---- Save or Reload state ----
                    if self.should_save():
                        self.save_state()
                    elif self.should_reload():
                        self.reload_state()

                    # ---- Set weights ----
                    self.metagraph.set_weights(
                        weights = self.get_row_weights(), 
                        wait_for_inclusion = True
                    )

                    # ---- Metagraph ----
                    self.sync_metagraph()

                    # ---- Update Tensorboard ----
                    self.epoch_logs() 
                
                except KeyboardInterrupt:
                    # User ended.
                    break

                except Exception as e:
                    # Unintended.
                    logger.exception('Uncaught Error in run loop: {}', e )
                    logger.info('Reload and continue.')
                    self.reload_state()
                    continue

if __name__ == "__main__":
    # ---- Build and Run ----
    miner = Miner()
    miner.run()

