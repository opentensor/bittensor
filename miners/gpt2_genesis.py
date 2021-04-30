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
"""GPT2 Language Modelling miner

The genesis miner.

Example:
    $ python miners/gpt2_genesis.py

To run with a config file:
    $ python miners/gpt2_genesis.py --config <path to config file>

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
from transformers import AdamW
from types import SimpleNamespace
from synapses.gpt2 import GPT2Synapse
from typing import Tuple, List, Optional
from torch.nn.utils import clip_grad_norm_
from bittensor.dataloaders.text_dataloader import GenesisTextDataloader
from pytorch_transformers import WarmupCosineWithHardRestartsSchedule

from loguru import logger
logger = logger.opt(colors=True)

class Miner( bittensor.miner.BaseMiner ):

    def __init__(self, config: Munch = None, **kwargs):
        # ---- Load Config ----
        if config == None:
            config = Miner.default_config();   
        config = copy.deepcopy(config); bittensor.config.Config.update_with_kwargs(config, kwargs )
        Miner.check_config( config )
        logger.info( bittensor.config.Config.toString( config ) )
        self.config = config

        # ---- Row Weights ----
        self.row_weights = torch.ones([1])

        # ---- Model ----
        self.synapse = GPT2Synapse( self.config )

        # ---- Optimizer ----
        self.optimizer = self.configure_optimizers()
        self.lr = self.config.miner.learning_rate

        # ---- Dataset ----
        # The Genesis Dataset:
        # The dataset used to train Adam and his first 100 children.
        self.dataset = GenesisTextDataloader(self.config.miner.batch_size_train, self.synapse.get_block_size())
        self.tokens = 0
        super( Miner, self ).__init__( self.config, **kwargs )
               
    @staticmethod
    def default_config() -> Munch:
        parser = argparse.ArgumentParser()
        Miner.add_args(parser)
        config = bittensor.config.Config.to_config(parser)
        return config

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            '--miner.learning_rate', 
            default=3e-2, 
            type=float, 
            help='Training initial learning rate.'
        )
        parser.add_argument(
            '--miner.weight_decay', 
            default=0.25, 
            type=float, 
            help='Model parameter weight decay.'
        )
        parser.add_argument(
            '--miner.lr_decay',
            default=True,
            type=bool,
            help='learning rate decay params: linear warmup followed by cosine decay to 10%% of original.'
        )
        parser.add_argument(
            '--miner.warmup_tokens',
            default=375e6,
            type=float,
            help='A linear LR warmup over the first miner.warmup_tokens tokens (default is 365 million)'
        )
        parser.add_argument(
            '--miner.final_tokens',
            default=260e9,
            type=float,
            help='At what point we reach 10%% of original LR'
        )
        parser.add_argument(
            '--miner.clip_gradients',
            default=1.0,
            type=float,
            help='Implement gradient clipping to avoid exploding loss on smaller architectures.'
        )
        parser.add_argument(
            '--miner.n_epochs', 
            default=int(sys.maxsize), 
            type=int, 
            help='Number of training epochs.'
        )
        parser.add_argument(
            '--miner.epoch_length', 
            default=500, 
            type=int, 
            help='Iterations of training per epoch'
        )
        parser.add_argument(
            '--miner.batch_size_train', 
            default=2, 
            type=int, 
            help='Training batch size.'
        )
        parser.add_argument('--miner.name', default='gpt2_genesis', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ')
        GPT2Synapse.add_args( parser )
        bittensor.miner.BaseMiner.add_args( parser )
        GenesisTextDataloader.add_args( parser )

    @staticmethod
    def check_config(config: Munch):
        assert config.miner.batch_size_train > 0, "batch_size_train must a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
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
        clip_grad_norm_(self.synapse.parameters(), self.config.miner.clip_gradients)
        self.optimizer.step() # Applies accumulated gradients.
        self.optimizer.zero_grad() # Zeros out gradients for next accummulation
        self.decay_learning_rate( inputs )

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
            output = self.training_call( batch = { 'inputs': inputs } )

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
                    # Unintended exception.
                    logger.exception('Uncaught Error in run loop: {}', e )
                    logger.info('Reload and continue.')
                    self.reload_state()
                    continue

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.

        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.Tanh)
        for mn, m in self.synapse.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.synapse.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.config.miner.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.miner.learning_rate, betas=(0.9, 0.95))
        return optimizer

    def decay_learning_rate(self, batch):
        """Decay the learning rate based on the progress thus far.
        Adjusts the self.config.miner.learning_rate according to the
        tokens processed so far, returns number of tokens.

        Args:
            tokens (int): Number of tokens processed so far.
        """

        if self.config.miner.lr_decay:
            # number of tokens processed this step
            self.tokens += (batch >= 0).sum()
            if self.tokens < self.config.miner.warmup_tokens:
                # linear warmup
                lr_mult = float(self.tokens) / float(max(1, self.config.miner.warmup_tokens))
            else:
                # cosine learning rate decay
                progress = float(self.tokens - self.config.miner.warmup_tokens) / float(max(1, self.config.miner.final_tokens - self.config.miner.warmup_tokens))
                lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

            self.lr = self.config.miner.learning_rate * lr_mult

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        else:
            self.lr = self.config.miner.learning_rate

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


if __name__ == "__main__":
    # ---- Build and Run ----
    miner = Miner()
    miner.run()
