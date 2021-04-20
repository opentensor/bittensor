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

This file demonstrates training the XLM neuron with language modelling.

Example:
    $ python miners/text/xlm.py

To run with a config file:
    $ python miners/text/xlm.py --config <path to config file>

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
from loguru import logger
logger = logger.opt(colors=True)
from termcolor import colored
from types import SimpleNamespace
from datasets import load_dataset
from nuclei.xlm import XLMNucleus
from typing import Tuple, List, Optional
from torch.utils.data.dataloader import DataLoader
from pytorch_transformers import WarmupCosineWithHardRestartsSchedule

class Miner( bittensor.neuron.BasicNeuron ):

    def __init__( 
            self, 
            config: Munch = None,
            **kwargs
        ):
        # ---- Load Config ----
        if config == None:
            config = Miner.default_config();   
        config = copy.deepcopy(config); bittensor.config.Config.update_with_kwargs( copy.deepcopy(config), kwargs )
        logger.info( bittensor.config.Config.toString( config ) )
        Miner.check_config( config )
        self.config = config

        # ---- Row Weights ----
        # Neuron specific mechanism weights.
        self.row_weights = torch.ones([1])

        # ---- Model ----
        self.model = XLMNucleus( self.config )

        # ---- Optimizer ----
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.config.miner.learning_rate, momentum=self.config.miner.momentum)
        self.scheduler = WarmupCosineWithHardRestartsSchedule(self.optimizer, 50, 300)

        # ---- Dataset ----
        self.dataset = bittensor.datasets.TextCorpus ( 
            block_size = 20,
            dataset = load_dataset('glue', 'cola')['train'],
            tokenizer = bittensor.__tokenizer__()
        )
        self.data_loader = DataLoader(
            self.dataset, 
            shuffle=True,
            batch_size=self.config.miner.batch_size_train,
            num_workers=self.config.miner.num_data_loader_workers
        )        
        super(Miner, self).__init__( self.config )
    
    @staticmethod
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        Miner.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config
    
    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
        parser.add_argument('--miner.learning_rate', default=0.01, type=float, help='Training initial learning rate.')
        parser.add_argument('--miner.momentum', default=0.98, type=float, help='Training initial momentum for SGD.')
        parser.add_argument('--miner.n_epochs', default=-1, type=int, help='Miner runs for this many epochs, or forever if < 0')
        parser.add_argument('--miner.epoch_length', default=500, type=int, help='Iterations of training per epoch')
        parser.add_argument('--miner.batch_size_train', default=2, type=int, help='Training batch size.')
        parser.add_argument('--miner.num_data_loader_workers', default=1, type=int, help='Number of workers for data loader.')
        parser.add_argument (
            '--config',
            default=None,
            type=str,
            help='Path to optional config file.'
        )
        XLMNucleus.add_args(parser)
        bittensor.neuron.BasicNeuron.add_args(parser)

    @staticmethod
    def check_config(config: Munch):
        assert config.miner.momentum > 0 and config.miner.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.miner.batch_size_train > 0, "batch_size_train must be a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
        XLMNucleus.check_config( config )
        bittensor.neuron.Neuron.check_config( config )

    def should_reload(self) -> bool:
        r""" This function is called by neuron.run() after every epoch.
            If the function returns True the model state is saved to neuron.full_path.
            Returns:
                should_reload (bool):
                    False by default. Does not reload the model after each epoch.
        """
        if torch.any(torch.isnan(torch.cat([param.view(-1) for param in self.model.parameters()]))):
            return True

    def should_save( self ) -> bool:
        r""" This function is called by neuron.run() after every epoch.
            If this function returns True, the model is saved to disk and can be reloaded late.
            Returns:
                should_save (bool):
                    True by default. Saves model after each epoch.
        """
        if self.epoch_loss < self.best_epoch_loss:
            self.best_epoch_loss = self.epoch_loss
            return True
        else:
            return False

    def get_state_dict( self ) -> dict:
        r""" This function is called by neuron.save_model() on save.
            The function must return a state dict which will be passed to neuron.reload_from_state_dict.       
            Returns:
                state_dict (:obj:`dict`): 
                    Dictionary containing run state information such as the model parameters.
        """
        return {
            'model_state': self.model.state_dict(), 
            'optimizer_state': self.optimizer.state_dict(),
        }

    def reload_from_state_dict( self, state_dict: dict):
        r""" This function is called by neuron.reload_model() on reload.
            The function must reload the training state from the passed state_dict. 
            Args:
                state_dict (:obj:`dict`): 
                    Dictionary containing run state information such as the model parameters. Output 
                    of get_state_dict.
        """
        self.model.load_state_dict( state_dict['model_state'] )
        self.optimizer.load_state_dict( state_dict['optimizer_state'] )

    # ---- Axon Forward call ----
    def forward_call( self, pubkey:str, inputs: torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        r""" This function is called by neuron.forward_loop which can be overridden by the base class.
            The arguments reflect an RPC request from another neuron in the network, the response tensor
            from this function is processed and returned to the caller.
            
            Args:
                pubkey ( str, `required`): 
                    The public key of the caller.
                inputs ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                modality ( bittensor.proto.Modality, `required`):
                    modality of inputs e.g. bittensor.proto.Modality.TEXT.
            
            Returns:
                outputs (:obj:`torch.FloatTensor`): 
                    The model's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
        """
        output = self.model.local_forward (
            inputs = inputs        
        )
        return output.local_hidden

    # ---- Axon Backward call ----
    def backward_call( self, pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        r""" This function is called by neuron.backward_loop which can be overridden by the base class.
            The arguments reflect an RPC backward request from another neuron in the network, the response tensor
            should be the gradients of the miner's model w.r.t to the inputs and the passed output grads.
            
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
        # This function is null because the inputs are int64s without gradients. 
        return None

    # ---- Get Row Weights ----
    def get_row_weights( self ) -> torch.FloatTensor:
        r""" This function is called after each training epoch to recieve. Returns row_weights
            to submit to chain.
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
        r""" Returns training batches for an epoch.
            Returns:
                batches ( List[dict], shape=(self.config.miner.epoch_length) ): 
                    List of batches as dictionary containing tokenized sentences
                    'inputs' = torch.LongTensor.
        """
        batches = []
        for iteration, inputs in tqdm( enumerate( self.data_loader ) ):
            batch = { 'inputs': inputs }
            batches.append( batch )
            if iteration == self.config.miner.epoch_length:
                break
        return batches

    # ---- Training call ----
    def training_call( self, batch: dict ) -> SimpleNamespace:
        r""" Run a single training batch through the model and apply a gradient update.
            Args:
                batch ( dict, `required`): 
                    training batch dictionary as returned from get_epoch_batches            
            Returns:
                outputs ( SimpleNamespace ): 
                    SimpleNamespace output as returned by a nucleus forward call.
                    Must include fields local_loss, remote_loss, distillation_loss
        """
        # ---- Forward pass ----
        inputs = batch['inputs'].to( self.model.device )
        output = self.model.remote_forward(
            neuron = self,
            inputs = inputs,
            training = True,
        )

        # ---- Backward pass ----
        output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
        output.loss.backward() # Accumulates gradients on the model.
        self.optimizer.step() # Applies accumulated gradients.
        self.optimizer.zero_grad() # Zeros out gradients for next accummulation

        # ---- Train row weights ----
        batch_weights = torch.mean(output.router.weights, axis = 0).to( self.model.device ) # Average over batch.
        self.row_weights = (1 - 0.03) * self.row_weights + 0.03 * batch_weights # Moving avg update.
        self.row_weights = F.normalize( self.row_weights, p = 1, dim = 0) # Ensure normalization.

        # ---- Update global loss ----
        return output

    # ---- Training logs ----
    def training_logs( self, progress_bar, iteration:int, output: SimpleNamespace ):
        r""" This function is called by neuron.run_training_epoch() after each training step.
            The function must populate the passed progress bar with training step state.
        """
        index = self.metagraph.state.index_for_uid[self.metagraph.uid]
        progress_bar.set_infos({
            'GS': colored('{}'.format(self.global_step), 'red'),
            'LS': colored('{}'.format(iteration), 'blue'),
            'Epoch': colored('{}'.format(self.epoch+1), 'green'),
            'Loss': colored('{:.5f}'.format(self.epoch_loss), 'yellow'),
            'L-loss': colored('{:.5f}'.format(output.local_target_loss.item()), 'red'),
            'R-loss': colored('{:.5f}'.format(output.remote_target_loss.item()), 'green'),
            'D-loss': colored('{:.5f}'.format(output.distillation_loss.item()), 'blue'),
            'nPeers': colored(self.metagraph.n, 'yellow'),
            'Stake(\u03C4)': colored('{:.3f}'.format(self.metagraph.S[index]), 'red'),
            'Rank(\u03C4)': colored('{:.3f}'.format(self.metagraph.R[index]), 'green'),
            'Incentive(\u03C4/block)': colored('{:.6f}'.format(self.metagraph.I[index]), 'yellow'),
            'Axon': self.axon.__str__(),
            'Dendrite': self.dendrite.__str__(),
        })
        self.tensorboard.add_scalar('R-loss', output.remote_target_loss.item(), self.global_step)
        self.tensorboard.add_scalar('L-loss', output.local_target_loss.item(), self.global_step)
        self.tensorboard.add_scalar('D-loss', output.distillation_loss.item(), self.global_step)

    # --- To Tensorboard ----
    def epoch_to_tensorboard(self):
        r""" This function is called by neuron.run() after each epoch.
            The subclass may override this function to send custom data to tensorboard after every epoch.
        """
        self.axon.__to_tensorboard__( self.tensorboard, self.global_step )
        self.dendrite.__to_tensorboard__( self.tensorboard, self.global_step )
        self.metagraph.__to_tensorboard__( self.tensorboard, self.global_step )
        self.tensorboard.add_scalar('neuron/epoch_loss', self.epoch_loss, self.global_step )
        self.tensorboard.add_scalar('neuron/best_epoch_loss', self.best_epoch_loss, self.global_step )

    # --- Run Epoch ----
    def run_next_training_epoch( self ):
        r""" Called by neuron.run(), runs a training epoch of length self.config.miner.epoch_length
        """
        total_epoch_loss = 0.0
        training_batches = self.get_epoch_batches( epoch = self.epoch )
        progress_bar = qqdm(enumerate(training_batches), total=len(training_batches), desc=format_str('blue', f'Epoch Progress'))
        for iteration, (training_batch) in progress_bar:
            output = self.training_call( batch = training_batch )
            total_training_loss += output.loss.item()
            self.epoch_loss = total_training_loss / (iteration + 1) 
            self.global_step += 1
            self.training_logs( progress_bar, iteration = iteration, output = output )
        self.scheduler.step()
    
    # --- Run Neuron ----
    def run( self ):
        # --- Run startup ----
        with self:

            # --- Run state ----
            self.epoch = 0
            self.global_step = 0
            self.best_epoch_loss = math.inf

            # ---- Save model ----
            self.save_model()

            # --- Run forever, or n_epochs ----
            for self.epoch in range( self.epoch, sys.maxsize if self.config.miner.n_epochs < 0 else self.config.miner.n_epochs ):

                # ---- Train ----
                self.run_next_training_epoch()

                # ---- Optionally Reload ----
                if self.should_reload():
                    self.reload_model()

                # ---- Optionally Save ----
                elif self.should_save():
                    self.save_model()

                # ---- Metagraph ----
                self.metagraph.sync() # Pulls the latest metagraph state.

                # ---- Set weights ----
                self.metagraph.set_weights(
                    weights = self.get_row_weights(), 
                    wait_for_inclusion = True
                )

                # ---- Update Tensorboard ----
                self.epoch_to_tensorboard()


if __name__ == "__main__":
    # ---- Build and Run ----
    miner = Miner()
    miner.run()
