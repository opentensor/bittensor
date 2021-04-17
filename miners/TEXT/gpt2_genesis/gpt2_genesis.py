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

This file demonstrates training the GPT2 neuron with language modelling.

Example:
        $ python miners/TEXT/gpt2_genesis/gpt2_genesis.py

Look at the yaml config file to tweak the parameters of the model. To run with those
default configurations, run:
        $ cd miners/TEXT
        $ python gpt2_genesis/gpt2_genesis.py --session.config_file gpt2_genesis/gpt2_genesis_config.yaml


"""
import argparse
import math
import os
import torch
import bittensor
import torch.nn.functional as F

from tqdm import tqdm
from munch import Munch
from loguru import logger
logger = logger.opt(ansi=True)
from typing import Tuple, List, Optional

from transformers import AdamW
from datasets import load_dataset
from types import SimpleNamespace
from synapses.gpt2 import GPT2Synapse
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.dataloader import DataLoader

class Miner( bittensor.neuron.Neuron ):

    def __init__( self, config: Munch = None ):
        if config == None:
            config = Miner.default_config()
        Miner.check_config(config)
        self.config = config

        # ---- Row Weights ----
        # Neuron specific mechanism weights.
        self.row_weights = torch.ones([1])

        # ---- Model ----
        # Unique ML model, served and used to train row_weights
        self.model = GPT2Synapse( self.config )

        # ---- Optimizer ----
        self.optimizer = self.configure_optimizers()
        self.lr = self.config.miner.learning_rate

        # ---- Dataset ----
        self.dataset = bittensor.datasets.TextCorpus ( 
            dataset = load_dataset('glue', 'cola')['train'],
            block_size = self.model.get_block_size(),
            tokenizer = bittensor.__tokenizer__()
        )
        self.data_loader = DataLoader(
            self.dataset, 
            shuffle=True,
            batch_size=self.config.miner.batch_size_train,
            num_workers=self.config.miner.num_workers
        )
        self.tokens = 0
        super(Miner, self).__init__( config )
               
    @staticmethod
    def default_config() -> Munch:
        parser = argparse.ArgumentParser()
        Miner.add_args(parser)
        config = bittensor.config.Config.to_config(parser)
        return config

    @staticmethod
    def add_args( parser: argparse.ArgumentParser ):
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
            '--miner.num_workers',
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
        GPT2Synapse.add_args(parser)
        bittensor.neuron.Neuron.add_args(parser)

    @staticmethod
    def check_config( config: Munch ):
        assert config.miner.batch_size_train > 0, "batch_size_train must a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
        config.miner.custom_dataset = os.path.expanduser(config.miner.custom_dataset)
        GPT2Synapse.check_config( config )
        bittensor.neuron.Neuron.check_config( config )

    def get_row_weights( self ) -> torch.FloatTensor:
        self.row_weights = torch.nn.functional.pad(self.row_weights, pad = [0, self.metagraph.n - self.row_weights.numel() ])
        return self.row_weights

    def next_training_batches( self, epoch:int ) -> List[ dict ]:
        batches = []
        for iteration, inputs in  tqdm( enumerate( self.data_loader ) ):
            batch = { 'inputs': inputs }
            batches.append( batch )
            if iteration == self.config.miner.epoch_length:
                break
        return batches

    def training_forward( self, batch: dict ) -> SimpleNamespace:

        # ---- Init for forward pass ----
        self.model.train(True)
        self.row_weights = torch.nn.functional.pad(self.row_weights, pad = [0, self.metagraph.n - self.row_weights.numel() ])

        # ---- Forward pass ----
        inputs = batch['inputs'].to(self.model.device)
        output = self.model.remote_forward(
            neuron = self, 
            inputs = inputs, 
            training = True
        )

        # ---- Backward pass ----
        output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
        output.loss.backward()

        # ---- Gradient Step ----
        clip_grad_norm_(self.model.parameters(), self.config.miner.clip_gradients)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.decay_learning_rate( inputs )

        # ---- Train row weights ----
        batch_weights = torch.mean(output.router.weights, axis = 0).to(self.model.device) # Average over batch.
        self.row_weights = (1 - 0.03) * self.row_weights + 0.03 * batch_weights # Moving avg update.
        self.row_weights = F.normalize(self.row_weights, p = 1, dim = 0) # Ensure normalization.
        return output

    def get_lr( self ):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def configure_optimizers( self ):
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

    def decay_learning_rate( self, batch ):
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

if __name__ == "__main__":
    # ---- Build and Run ----
    miner = Miner()
    logger.info(bittensor.config.Config.toString(miner.config))
    miner.run()
