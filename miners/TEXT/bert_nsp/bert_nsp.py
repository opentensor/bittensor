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
"""BERT Next Sentence Prediction Neuron.

This file demonstrates training the BERT neuron with next sentence prediction.

Example:
        $ python miners/TEXT/bert_nsp/bert_nsp.py

Look at the yaml config file to tweak the parameters of the model. To run with those 
default configurations, run:
        $ cd miners/TEXT
        $ python bert_nsp/bert_nsp.py --session.config_file bert_nsp/bert_nsp_config.yaml

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
from synapses.bert import BertNSPSynapse
from torch.nn.utils import clip_grad_norm_
from pytorch_transformers import WarmupCosineWithHardRestartsSchedule

class Miner( bittensor.neuron.Neuron ):

    def __init__( self, config: Munch = None ):

        if config == None:
            config = Miner.default_config();       
        Miner.check_config( config )
        self.config = config

        # ---- Row Weights ----
        # Neuron specific mechanism weights.
        self.row_weights = torch.ones([1])

        # ---- Model ----
        self.model = BertNSPSynapse( self.config )

        # ---- Optimizer ----
        self.optimizer = torch.optim.SGD( self.model.parameters(), lr = self.config.miner.learning_rate, momentum = self.config.miner.momentum )
        self.scheduler = WarmupCosineWithHardRestartsSchedule( self.optimizer, 50, 300 )

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
        parser.add_argument('--miner.clip_gradients', default=0.8, type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.')
        parser.add_argument('--miner.epoch_length', default=500, type=int, help='Iterations of training per epoch')
        parser.add_argument('--miner.batch_size_train', default=1, type=int, help='Training batch size.')
        BertNSPSynapse.add_args( parser )
        bittensor.neuron.Neuron.add_args( parser )

    @staticmethod
    def check_config(config: Munch):
        assert config.miner.momentum > 0 and config.miner.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.miner.batch_size_train > 0, "batch_size_train must a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
        BertNSPSynapse.check_config( config )
        bittensor.neuron.Neuron.check_config( config )

    def nsp_batch(data, batch_size, tokenizer):
        """ Returns a random batch from text dataset with 50 percent NSP.            
            Returns:
                inputs List[str]: List of sentences.
                targets torch.Tensor(batch_size): 1 if random next sentence, otherwise 0.
        """
        batch_inputs = []
        batch_next = []
        batch_labels = []
        for _ in range(batch_size):
            if random.random() > 0.5:
                pos = random.randint(0, len(data))
                batch_inputs.append(data[pos]['text'])
                batch_next.append(data[pos + 1]['text'])
                batch_labels.append(0)
            else:
                while True:
                    pos_1 = random.randint(0, len(data))
                    pos_2 = random.randint(0, len(data))
                    batch_inputs.append(data[pos_1]['text'])
                    batch_next.append(data[pos_2]['text'])
                    batch_labels.append(1)
                    if (pos_1 != pos_2) and (pos_1 != pos_2 - 1):
                        break
        tokenized_inputs = self.tokenizer(batch_inputs, text_pair = batch_next, return_tensors='pt', padding=True)
        return {'inputs': tokenized_inputs['input_ids'], 'attention_mask': tokenized_inputs['attention_mask'], 'targets': torch.tensor(batch_labels, dtype=torch.long)}

    def next_training_batches(self, epoch:int ) -> List[dict]:
        logger.info('Preparing {} batches for epoch ...', self.config.miner.epoch_length)
        batches = []
        for _ in range( self.config.miner.epoch_length ):
            batches.append( nsp_batch() )
        return batches
    
    def training_forward( self, batch: dict ):
        # ---- Forward pass ----
        inputs = batch['inputs'].to( self.model.device )
        attention_mask = batch['attention_mask'].to( self.model.device )
        targets = batch['targets'].to( self.model.device )
        output = self.model.remote_forward(
            neuron = self,
            inputs = inputs, 
            attention_mask = attention_mask,
            targets = labels,
        )

        # ---- Backward pass ----
        loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
        loss.backward() # Accumulates gradients on the model.
        clip_grad_norm_( self.model.parameters(), self.config.miner.clip_gradients ) # clip model gradients
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
