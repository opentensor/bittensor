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
    $ python miners/text/gpt2_genesis.py

To run with a config file:
    $ python miners/text/gpt2_genesis.py --config <path to config file>

"""
import argparse
import copy
import math
import os
import sys
import time
import torch
import time
import bittensor
import torch.nn.functional as F

from qqdm import qqdm, format_str
from tqdm import tqdm
from munch import Munch
from loguru import logger
logger = logger.opt(colors=True)
from termcolor import colored
from typing import Tuple, List, Optional

from nuclei.gpt2 import GPT2Nucleus
from torch.nn.utils import clip_grad_norm_
from transformers import AdamW
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset
from types import SimpleNamespace

class Miner( bittensor.neuron.BasicNeuron ):

    def __init__(self, config: Munch = None, **kwargs ):
        if config == None:
            config = Miner.default_config()
        config = copy.deepcopy(config); bittensor.config.Config.update_with_kwargs(config, kwargs )
        logger.info( bittensor.config.Config.toString( config ) )
        Miner.check_config( config )
        self.config = config

        # ---- Row Weights ----
        # Neuron specific mechanism weights.
        self.row_weights = torch.ones([1])

        # ---- Model ----
        self.model = GPT2Nucleus( self.config )

        # ---- Optimizer ----
        self.optimizer = self.configure_optimizers()
        self.lr = self.config.miner.learning_rate

        # ---- Dataset ----
        self.dataset = bittensor.datasets.TextCorpus( 
            dataset = load_dataset('glue', 'cola')['train'],
            block_size = self.model.get_block_size(),
            tokenizer = bittensor.__tokenizer__()
        )
        self.data_loader = DataLoader(
            self.dataset, 
            shuffle=True,
            batch_size=self.config.miner.batch_size_train,
            num_workers=self.config.miner.num_dataloader_workers
        )        
        self.tokens = 0
        super(Miner, self).__init__( self.config )
                
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
            help='learning rate decay params: linear warmup followed by cosine decay to 10% of original.'
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
            help='At what point we reach 10% of original LR'
        )
        parser.add_argument(
            '--miner.num_dataloader_workers',
            default=1,
            type=int,
            help='Number of workers for data loader.'
        )
        parser.add_argument(
            '--miner.clip_gradients',
            default=1.0,
            type=float,
            help='Implement gradient clipping to avoid exploding loss on smaller architectures.'
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
        parser.add_argument (
            '--miner.custom_dataset',
            default="~/.bittensor/bittensor/miners/TEXT/gpt2_genesis/genesis_dataset/",
            type=str,
            help='Custom datasets to train on.'
        )
        parser.add_argument (
            '--miner.config_file',
            type=str,
            help='config file to run this neuron, if not using cmd line arguments.'
        )
        parser.add_argument (
            '--config',
            default=None,
            type=str,
            help='Path to optional config file'
        )
        parser.add_argument('--miner.n_epochs', default=-1, type=int, help='Miner runs for this many epochs, or forever if < 0')
        GPT2Nucleus.add_args(parser)
        bittensor.neuron.BasicNeuron.add_args(parser)

    @staticmethod
    def check_config(config: Munch):
        assert config.miner.batch_size_train > 0, "batch_size_train must a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
        GPT2Nucleus.check_config( config )
        bittensor.neuron.BasicNeuron.check_config( config )

    def should_run( self, epoch: int ) -> bool:
        r""" Called by neuron.run() every epoch, if the response is false, training stops.
        """
        if self.config.miner.n_epochs < 0:
            return True
        elif epoch < self.config.miner.n_epochs:
            return True
        else:
            return False
    
    def should_save( self ) -> bool:
        r""" This function is called by neuron.run() after every epoch.
            If this function returns True, the model is saved to disk and can be reloaded late.
            Returns:
                should_save (bool):
                    True by default. Saves model after each epoch.
        """
        if self.epoch_loss < self.last_saved_loss:
            return True
        else:
            return False

    def should_reload(self) -> bool:
        r""" This function is called by neuron.run() after every epoch.
            If the function returns True the model state is saved to neuron.full_path.
            Returns:
                should_reload (bool):
                    False by default. Does not reload the model after each epoch.
        """
        if torch.any(torch.isnan(torch.cat([param.view(-1) for param in self.model.parameters()]))):
            return True

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
        for iteration, inputs in  tqdm( enumerate( self.data_loader )) :
            batch = { 'inputs': inputs }
            batches.append( batch )
            if iteration == self.config.miner.epoch_length:
                break
        return batches

    # ---- Training call ----
    def training_call( self, batch: dict ) -> SimpleNamespace:
        """ Run a single training batch through the model and apply a gradient update.
            Args:
                batch ( dict, `required`): 
                    training batch dictionary as returned from get_epoch_batches            
            Returns:
                outputs ( SimpleNamespace ): 
                    SimpleNamespace output as returned by a nucleus forward call.
                    Must include fields local_loss, remote_loss, distillation_loss
        """
        # ---- Init for forward pass ----
        self.model.train(True)
        self.row_weights = torch.nn.functional.pad(self.row_weights, pad = [0, self.metagraph.n - self.row_weights.numel() ])

        # ---- Forward pass ----
        inputs = batch[ 'inputs' ].to( self.model.device )
        output = self.model.remote_forward( 
            neuron = self, 
            inputs = inputs, 
            training = True
        )
        # ---- Backward pass ----
        output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
        output.loss.backward()

        # ---- Gradient Step ----
        clip_grad_norm_( self.model.parameters(), self.config.miner.clip_gradients )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.decay_learning_rate( inputs )

        # ---- Train row weights ----
        batch_weights = torch.mean(output.router.weights, axis = 0).to(self.model.device) # Average over batch.
        self.row_weights = (1 - 0.03) * self.row_weights + 0.03 * batch_weights # Moving avg update.
        self.row_weights = F.normalize(self.row_weights, p = 1, dim = 0) # Ensure normalization.
        return output

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

    def reset_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

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
        for mn, m in self.model.named_modules():
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
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
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

if __name__ == "__main__":
    # ---- Build and Run ----
    miner = Miner()
    miner.run()
