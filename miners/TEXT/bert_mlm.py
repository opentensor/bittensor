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
"""BERT Masked Language Modelling.

This file demonstrates training the BERT neuron with masked language modelling.

Example:
    $ python miners/text/bert_mlm.py

To run with a config file:
    $ python miners/text/bert_mlm.py --config <path to config file>

"""
import argparse
import os
import random
import torch
import torch.nn.functional as F
import bittensor

from tqdm import tqdm
from munch import Munch
from termcolor import colored
from datasets import load_dataset
from types import SimpleNamespace
from typing import Tuple, List, Optional
from torch.nn.utils import clip_grad_norm_
from transformers import DataCollatorForLanguageModeling
from pytorch_transformers import WarmupCosineWithHardRestartsSchedule

from nuclei.bert import BertMLMNucleus
from loguru import logger
logger = logger.opt(ansi=True)

class Miner( bittensor.neuron.Neuron ):

    def __init__(self, config: Munch = None ):
        if config == None:
            config = Miner.default_config();       
        Miner.check_config(config)
        self.config = config

        # ---- Row Weights ----
        self.row_weights = torch.ones([1])

        # ---- Model ----
        self.model = BertMLMNucleus( self.config )

        # ---- Optimizer ----
        self.optimizer = torch.optim.SGD( self.model.parameters(), lr = self.config.miner.learning_rate, momentum=self.config.miner.momentum )
        self.scheduler = WarmupCosineWithHardRestartsSchedule( self.optimizer, 50, 300 )

        # ---- Dataset ----
        self.corpus = bittensor.datasets.MLMCorpus (
            dataset = load_dataset('glue', 'cola')['train'],
            tokenizer = bittensor.__tokenizer__(),
            collator = DataCollatorForLanguageModeling (
                tokenizer=bittensor.__tokenizer__(), 
                mlm=True, 
                mlm_probability=0.15
            )   
        )
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
        parser.add_argument('--miner.clip_gradients', default=0.8, type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.')
        parser.add_argument('--miner.epoch_length', default=500, type=int, help='Iterations of training per epoch')
        parser.add_argument('--miner.batch_size_train', default=1, type=int, help='Training batch size.')
        BertMLMNucleus.add_args( parser )
        bittensor.neuron.Neuron.add_args( parser )

    @staticmethod
    def check_config(config: Munch):
        assert config.miner.momentum > 0 and config.miner.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.miner.batch_size_train > 0, "batch_size_train must a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
        BertMLMNucleus.check_config( config )
        bittensor.neuron.Neuron.check_config( config )
           
    # ---- Get Row Weights ----
    # Returns mechanism weights (to be submit to chain)
    def get_row_weights( self ) -> torch.FloatTensor:
        self.row_weights = torch.nn.functional.pad( self.row_weights, pad = [0, self.metagraph.n - self.row_weights.numel()] )
        return self.row_weights

    # ---- Get Batches ----
    # Returns a list of batches for the next training epoch.
    def get_epoch_batches( self, epoch:int ) -> List[ dict ]:
        logger.info('Preparing {} batches for epoch ...', self.config.miner.epoch_length)
        batches = []
        for _ in tqdm(range( self.config.miner.epoch_length )):
            batches.append( self.corpus.next_batch( self.config.miner.batch_size_train ) )
        return batches
    
    # ---- Training call ----
    # Applies a training forward + backward pass for a given input batch.
    def training_call( self, batch: dict ) -> SimpleNamespace:
        # ---- Forward pass ----
        inputs = batch['inputs'].to( self.model.device )
        targets = batch['labels'].to( self.model.device )
        output = self.model.remote_forward(
            neuron = self,
            inputs = inputs, 
            targets = targets,
        )

        # ---- Backward pass ----
        output.loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
        output.loss.backward() # Accumulates gradients on the model.
        clip_grad_norm_( self.model.parameters(), self.config.miner.clip_gradients ) # clip model gradients
        self.optimizer.step() # Applies accumulated gradients.
        self.optimizer.zero_grad() # Zeros out gradients for next accummulation

        # ---- Train row weights ----
        batch_weights = torch.mean(output.router.weights, axis = 0).to( self.model.device ) # Average over batch.
        self.row_weights = (1 - 0.03) * self.row_weights + 0.03 * batch_weights # Moving avg update.
        self.row_weights = F.normalize( self.row_weights, p = 1, dim = 0) # Ensure normalization.

        return output

    # ---- Forward call ----
    # Returns the nucleus hidden representation w.r.t the passed inputs.
    def forward_call( self, pubkey:str, inputs: torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        output = self.model.local_forward(
            inputs = inputs, 
        )
        return output.local_hidden

    # ---- Backward call ----
    # Returns the input gradients w.r.t the passed inputs and grads.
    def backward_call( self, pubkey:str, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, modality:int ) -> torch.FloatTensor:
        outputs_y = self.model.local_forward(
            inputs_x = inputs 
        )
        grads_dx = torch.autograd.grad(
            outputs = outputs_y, 
            inputs = inputs_x,
            grad_outputs = grads_dy, 
            only_inputs = True,
            create_graph = False, 
            retain_graph = False
        )
        return grads_dx

    

if __name__ == "__main__":
    # ---- Build and Run ----
    miner = Miner()
    logger.info(bittensor.config.Config.toString(miner.config))
    miner.run()

