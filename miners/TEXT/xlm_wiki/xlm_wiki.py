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
        $ python miners/TEXT/xlm_wiki.py

"""
import argparse
import os
import random
import torch
import torch.nn.functional as F
import bittensor

from munch import Munch
from loguru import logger
from termcolor import colored
from datasets import load_dataset
from synapses.xlm import XLMSynapse
from typing import Tuple, List, Optional
from pytorch_transformers import WarmupCosineWithHardRestartsSchedule

class Miner( bittensor.neuron.Neuron ):

    def __init__(self, config: Munch = None, **kwargs):
        if config == None:
            config = Miner.default_config();       
        bittensor.config.Config.update_with_kwargs(config.miner, kwargs) 
        Miner.check_config(config)
        self.config = config

        # ---- Row Weights ----
        # Neuron specific mechanism weights.
        self.row_weights = torch.ones([1])

        # ---- Model ----
        self.model = XLMSynapse( self.config )

        # ---- Optimizer ----
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.config.miner.learning_rate, momentum=self.config.miner.momentum)
        self.scheduler = WarmupCosineWithHardRestartsSchedule(self.optimizer, 50, 300)

        # ---- Dataset ----
        self.dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
        self.tokenizer = bittensor.__tokenizer__()
        super(Miner, self).__init__( self.config )
    
    @staticmethod
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        Miner.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config
    
    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument('--miner.learning_rate', default=0.01, type=float, help='Training initial learning rate.')
        parser.add_argument('--miner.momentum', default=0.98, type=float, help='Training initial momentum for SGD.')
        parser.add_argument('--miner.epoch_length', default=500, type=int, help='Iterations of training per epoch')
        parser.add_argument('--miner.batch_size_train', default=2, type=int, help='Training batch size.')
        XLMSynapse.add_args(parser)
        bittensor.neuron.Neuron.add_args(parser)

    @staticmethod
    def check_config(config: Munch):
        assert config.miner.momentum > 0 and config.miner.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.miner.batch_size_train > 0, "batch_size_train must be a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
        XLMSynapse.check_config( config )
        bittensor.neuron.Neuron.check_config( config )

    def get_row_weights( self ) -> torch.FloatTensor:
        self.row_weights = torch.nn.functional.pad( self.row_weights, pad = [0, self.metagraph.n - self.row_weights.numel()] )
        return self.row_weights

    def next_training_batches(self, epoch:int ) -> List[dict]:
        logger.info('Preparing {} batches for epoch ...', self.config.miner.epoch_length)
        batches = []
        for _ in range( self.config.miner.epoch_length ):
            batch_text = [] 
            for _ in range( self.config.miner.batch_size_train ):
                rnd_index = random.randint(0, len(self.dataset['train']))
                batch_text.append( self.dataset['train'][rnd_index]['text'] )
            print ( batch_text )
            tokenized_batch = self.tokenizer( batch_text, return_tensors='pt', padding=True, truncation=True )['input_ids']
            batch = { 'inputs': tokenized_batch }
            batches.append( batch )
        return batches
    
    def training_forward( self, batch: dict ):
        # ---- Forward pass ----
        inputs = batch['inputs'].to( self.model.device )
        output = self.model.remote_forward(
            self.neuron,
            inputs = inputs,
            training = True,
        )

        # ---- Backward pass ----
        loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
        loss.backward() # Accumulates gradients on the model.
        self.optimizer.step() # Applies accumulated gradients.
        self.optimizer.zero_grad() # Zeros out gradients for next accummulation

        # ---- Train row weights ----
        batch_weights = torch.mean(output.router.weights, axis = 0).to( self.model.device ) # Average over batch.
        self.row_weights = (1 - 0.03) * self.row_weights + 0.03 * batch_weights # Moving avg update.
        self.row_weights = F.normalize( self.row_weights, p = 1, dim = 0) # Ensure normalization.


if __name__ == "__main__":
    # ---- Build and Run ----
    miner = Miner()
    logger.info(bittensor.config.Config.toString(miner.config))
    miner.run()
