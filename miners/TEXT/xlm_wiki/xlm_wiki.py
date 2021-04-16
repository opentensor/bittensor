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
import math
import os
import sys
import time
import torch
import torch.nn.functional as F
import traceback
import time
import bittensor

from termcolor import colored
from synapses.xlm import XLMSynapse, nextbatch
from bittensor.utils.model_utils import ModelToolbox
from munch import Munch
from loguru import logger
from pytorch_transformers import WarmupCosineWithHardRestartsSchedule
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter


class Miner():
    """
    Initializes, trains, and tests models created inside of 'bittensor/synapses'. 
    During instantiation, this class takes a config as a [Munch](https://github.com/Infinidat/munch) object. 
    """

    def __init__(self, config: Munch = None, **kwargs):
        if config == None:
            config = Miner.default_config();       
        bittensor.config.Config.update_with_kwargs(config.miner, kwargs) 
        Miner.check_config(config)
        self.config = config

        # ---- Neuron ----
        self.neuron = bittensor.neuron.Neuron(self.config)

        # ---- Model ----
        self.model = XLMSynapse( self.config )

        # ---- Optimizer ----
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.config.miner.learning_rate, momentum=self.config.miner.momentum)
        self.scheduler = WarmupCosineWithHardRestartsSchedule(self.optimizer, 50, 300)

        # ---- Model Load/Save tools ----
        self.model_toolbox = ModelToolbox(XLMSynapse, torch.optim.SGD)

        # ---- Dataset ----
        # Dataset: 74 million sentences pulled from books.
        self.dataset = load_dataset('amazon_reviews_multi', 'en')['train']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.config.synapse.device:
            self.device = torch.device(self.config.synapse.device)

        # ---- Logging ----
        self.tensorboard = SummaryWriter(log_dir = self.config.miner.full_path)
        if self.config.miner.record_log == True:
            filepath = f"{self.config.miner.full_path}/{self.config.miner.name}_ {self.config.miner.trial_uid}.log"
            logger.add (
                filepath,
                format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
                rotation="250 MB",
                retention="10 days"
            )
    
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
        parser.add_argument('--miner.n_epochs', default=int(sys.maxsize), type=int, help='Number of training epochs.')
        parser.add_argument('--miner.epoch_length', default=500, type=int, help='Iterations of training per epoch')
        parser.add_argument('--miner.batch_size_train', default=1, type=int, help='Training batch size.')
        parser.add_argument('--miner.sync_interval', default=100, type=int, help='Batches before we sync with chain and emit new weights.')
        parser.add_argument('--miner.log_interval', default=10, type=int, help='Batches before we log miner info.')
        parser.add_argument('--miner.accumulation_interval', default=1, type=int, help='Batches before we apply acummulated gradients.')
        parser.add_argument('--miner.apply_remote_gradients', default=False, type=bool, help='If true, neuron applies gradients which accumulate from remotes calls.')
        parser.add_argument('--miner.root_dir', default='~/.bittensor/miners/', type=str,  help='Root path to load and save data associated with each miner')
        parser.add_argument('--miner.name', default='xlm_wiki', type=str, help='Trials for this miner go in miner.root / miner.name')
        parser.add_argument('--miner.trial_uid', default=str(time.time()).split('.')[0], type=str, help='Saved models go in miner.root_dir / miner.name / miner.uid')
        parser.add_argument('--miner.record_log', default=False, help='Record all logs when running this miner')
        parser.add_argument('--miner.config_file', type=str, help='config file to run this neuron, if not using cmd line arguments.')
        parser.add_argument('--debug', dest='debug', action='store_true', help='''Turn on bittensor debugging information''')
        parser.set_defaults( debug=False )
        XLMSynapse.add_args(parser)
        bittensor.neuron.Neuron.add_args(parser)

    @staticmethod
    def check_config(config: Munch):
        if config.debug:  bittensor.__log_level__ = 'TRACE'; logger.debug('DEBUG is ON')
        else: logger.info('DEBUG is OFF') 
        assert config.miner.momentum > 0 and config.miner.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.miner.batch_size_train > 0, "batch_size_train must be a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
        full_path = '{}/{}/{}'.format(config.miner.root_dir, config.miner.name, config.miner.trial_uid)
        config.miner.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.miner.full_path):
            os.makedirs(config.miner.full_path)
    
    def training_forward( self, batch: dict ):

    # ---- Forward pass ----
    inputs = nextbatch(self.dataset, self.config.miner.batch_size_train, bittensor.__tokenizer__())
    output = self.model.remote_forward(
        self.neuron,
        inputs.to(self.model.device),
        training = True,
    )

    # ---- Backward pass ----
    loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
    loss.backward() # Accumulates gradients on the model.
    self.optimizer.step() # Applies accumulated gradients.
    self.optimizer.zero_grad() # Zeros out gradients for next accummulation

    # ---- Train row weights ----
    batch_weights = torch.mean(output.router.weights, axis = 0).to(self.model.device) # Average over batch.
    self.row = (1 - 0.03) * self.row + 0.03 * batch_weights # Moving avg update.
    self.row = F.normalize(self.row, p = 1, dim = 0) # Ensure normalization.


if __name__ == "__main__":
    # ---- Build and Run ----
    miner = Miner()
    logger.info(bittensor.config.Config.toString(miner.config))
    miner.run()
