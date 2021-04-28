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

        # ---- Nucleus ----
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

    # ---- Get epoch batches ----
    def get_epoch_batches( self, epoch:int ) -> List[ dict ]:
        r""" Returns training batches for each epoch.
            Returns:
                batches ( List[dict], shape=(self.config.miner.epoch_length) ): 
                    List of batches as dictionary containing tokenized sentences
                    'inputs' = torch.LongTensor.
        """
        batches = []
        epoch_data = self.dataset.dataloader( self.config.miner.epoch_length )
        for iteration, inputs in tqdm( enumerate( epoch_data ) ):
            batch = { 'inputs': inputs }
            batches.append( batch )
            if iteration == self.config.miner.epoch_length:
                break
        return batches

    # ---- Training call ----
    def training_call( self, batch: dict ) -> SimpleNamespace:
        r""" Runs a single training batch through the nucleus and applies a gradient update.
            Args:
                batch ( dict, `required`): 
                    training batch dictionary as returned from get_epoch_batches            
            Returns:
                outputs ( SimpleNamespace ): 
                    SimpleNamespace output as returned by a nucleus forward call.
                    Must include fields local_loss, remote_loss, distillation_loss
        """
        # ---- Forward pass ----
        inputs = batch['inputs'].to( self.synapse.device )
        output = self.synapse.remote_forward(
            neuron = self,
            inputs = inputs,
            training = True,
        )

        # ---- Backward pass ----
        output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
        output.loss.backward() # Accumulates gradients on the nucleus.
        self.optimizer.step() # Applies accumulated gradients.
        self.optimizer.zero_grad() # Zeros out gradients for next accummulation
        

        # ---- Train row weights ----
        batch_weights = torch.mean(output.router.weights, axis = 0).to( self.synapse.device ) # Average over batch.
        self.row_weights = (1 - 0.03) * self.row_weights + 0.03 * batch_weights # Moving avg update.
        self.row_weights = F.normalize( self.row_weights, p = 1, dim = 0) # Ensure normalization.

        # ---- Update global loss ----
        return output

if __name__ == "__main__":
    # ---- Build and Run ----
    miner = Miner()
    miner.run()

