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
        $ python miners/TEXT/bert_mlm/bert_mlm.py

Look at the yaml config file to tweak the parameters of the model. To run with those 
default configurations, run:
        $ cd miners/TEXT
        $ python bert_mlm/bert_mlm.py --session.config_file bert_mlm/bert_mlm_config.yaml


"""
import argparse
import math
import os
import sys
import random
import time
import torch
import torch.nn.functional as F
import traceback
import time
import bittensor

from termcolor import colored
from munch import Munch
from datasets import load_dataset
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from transformers import DataCollatorForLanguageModeling
from pytorch_transformers import WarmupCosineWithHardRestartsSchedule
from bittensor.utils.model_utils import ModelToolbox
from synapses.bert import BertMLMSynapse
from torch.nn.utils import clip_grad_norm_

def mlm_batch(data, batch_size, tokenizer, collator):
    """ Returns a random batch from text dataset with 50 percent NSP.
        Args:
            data: (List[dict{'text': str}]): Dataset of text inputs.
            batch_size: size of batch to create.
        
        Returns:
            tensor_batch torch.Tensor (batch_size, sequence_length): List of tokenized sentences.
            labels torch.Tensor (batch_size, sequence_length)
    """
    batch_text = []
    for _ in range(batch_size):
        batch_text.append(data[random.randint(0, len(data))]['text'])

    # Tokenizer returns a dict { 'input_ids': list[], 'attention': list[] }
    # but we need to convert to List [ dict ['input_ids': ..., 'attention': ... ]]
    # annoying hack...
    tokenized = tokenizer(batch_text)
    tokenized = [dict(zip(tokenized,t)) for t in zip(*tokenized.values())]

    # Produces the masked language model inputs aw dictionary dict {'inputs': tensor_batch, 'labels': tensor_batch}
    # which can be used with the Bert Language model. 
    collated_batch =  collator(tokenized)
    return collated_batch['input_ids'], collated_batch['labels']


class Miner( bittensor.miner.Miner ):

    def __init__(self, config: Munch = None, **kwargs):
        if config == None:
            config = Miner.default_config();       
        bittensor.config.Config.update_with_kwargs(config.miner, kwargs) 
        Miner.check_config(config)
        self.config = config

        # ---- Model ----
        self.model = BertMLMSynapse( self.config )

        # ---- Optimizer ----
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.config.miner.learning_rate, momentum=self.config.miner.momentum)
        self.scheduler = WarmupCosineWithHardRestartsSchedule(self.optimizer, 50, 300)

        # ---- Model Load/Save tools ----
        self.model_toolbox = ModelToolbox(BertMLMSynapse, torch.optim.SGD)

        # ---- Dataset ----
        # Dataset: 74 million sentences pulled from books.
        self.dataset = load_dataset('ag_news')['train']
        # The collator accepts a list [ dict{'input_ids, ...; } ] where the internal dict 
        # is produced by the tokenizer.
        self.data_collator = DataCollatorForLanguageModeling (
            tokenizer=bittensor.__tokenizer__(), mlm=True, mlm_probability=0.15
        )
        super( Miner, self ).__init__( self.config, **kwargs )

    @staticmethod
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        Miner.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod
    def check_config(config: Munch):
        assert config.miner.momentum > 0 and config.miner.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.miner.batch_size_train > 0, "batch_size_train must a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument('--miner.learning_rate', default=0.01, type=float, help='Training initial learning rate.')
        parser.add_argument('--miner.momentum', default=0.98, type=float, help='Training initial momentum for SGD.')
        parser.add_argument('--miner.clip_gradients', default=0.8, type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.')
        parser.add_argument('--miner.n_epochs', default=int(sys.maxsize), type=int, help='Number of training epochs.')
        parser.add_argument('--miner.epoch_length', default=500, type=int, help='Iterations of training per epoch')
        parser.add_argument('--miner.batch_size_train', default=1, type=int, help='Training batch size.')
        parser.add_argument('--miner.name', default='bert_mlm', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ')
        BertMLMSynapse.add_args(parser)
        bittensor.neuron.Neuron.add_args(parser)

    # --- Main loop ----
    def run (self):

        # ---- Subscribe ----
        with self:

            # ---- Weights ----
            self.row = self.metagraph.row

            # --- Run state ---
            self.global_step = 0
            self.best_train_loss = math.inf

            # --- Loop for epochs ---
            for self.epoch in range(self.config.miner.n_epochs):
                try:
                    # ---- Serve ----
                    self.axon.serve( self.model )

                    # ---- Train Model ----
                    self.train()
                    self.scheduler.step()

                    # If model has borked for some reason, we need to make sure it doesn't emit weights
                    # Instead, reload into previous version of model
                    if torch.any(torch.isnan(torch.cat([param.view(-1) for param in self.model.parameters()]))):
                        self.model, self.optimizer = self.model_toolbox.load_model(self.config)    
                        continue

                    # ---- Emitting weights ----
                    self.metagraph.set_weights(self.row, wait_for_inclusion = True) # Sets my row-weights on the chain.

                    # ---- Sync metagraph ----
                    self.metagraph.sync() # Pulls the latest metagraph state (with my update.)
                    self.row = self.metagraph.row
                    logger.info(self.metagraph)

                    # ---- Update Tensorboard ----
                    self.dendrite.__to_tensorboard__(self.tensorboard, self.global_step)
                    self.metagraph.__to_tensorboard__(self.tensorboard, self.global_step)
                    self.axon.__to_tensorboard__(self.tensorboard, self.global_step)
                
                    # ---- Save best loss and model ----
                    if self.training_loss and self.epoch % 10 == 0:
                        if self.training_loss < self.best_train_loss:
                            self.best_train_loss = self.training_loss # update best train loss
                            self.model_toolbox.save_model(
                                self.config.miner.full_path,
                                {
                                    'epoch': self.epoch, 
                                    'model_state_dict': self.model.state_dict(), 
                                    'loss': self.best_train_loss,
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                }
                            )
                            self.tensorboard.add_scalar('Neuron/Train_loss', self.training_loss, self.global_step)
                    
                # --- Catch Errors ----
                except Exception as e:
                    logger.error('Exception in training script with error: {}', e)
                    logger.info(traceback.print_exc())
                    logger.info('Continuing to train.')
                    time.sleep(1)
    
    # ---- Train Epoch ----
    def train(self):
        self.training_loss = 0.0
        for local_step in range(self.config.miner.epoch_length):
            # ---- Forward pass ----
            inputs, targets = mlm_batch(self.dataset, self.config.miner.batch_size_train, bittensor.__tokenizer__(), self.data_collator)
            output = self.model.remote_forward (
                    self,
                    inputs = inputs.to(self.model.device), 
                    targets = targets.to(self.model.device)
            )

            # ---- Backward pass ----
            loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
            loss.backward() # Accumulates gradients on the model.
            clip_grad_norm_(self.model.parameters(), self.config.miner.clip_gradients) # clip model gradients
            self.optimizer.step() # Applies accumulated gradients.
            self.optimizer.zero_grad() # Zeros out gradients for next accummulation

            # ---- Train row weights ----
            batch_weights = torch.mean(output.router.weights, axis = 0) # Average over batch.
            self.row = (1 - 0.03) * self.row + 0.03 * batch_weights # Moving avg update.
            self.row = F.normalize(self.row, p = 1, dim = 0) # Ensure normalization.

            # ---- Step logs ----
            logger.info('GS: {} LS: {} Epoch: {}\tLocal Target Loss: {}\tRemote Target Loss: {}\tDistillation Loss: {}\tAxon: {}\tDendrite: {}',
                    colored('{}'.format(self.global_step), 'red'),
                    colored('{}'.format(local_step), 'blue'),
                    colored('{}'.format(self.epoch), 'green'),
                    colored('{:.4f}'.format(output.local_target_loss.item()), 'green'),
                    colored('{:.4f}'.format(output.remote_target_loss.item()), 'blue'),
                    colored('{:.4f}'.format(output.distillation_loss.item()), 'red'),
                    self.axon,
                    self.dendrite)
            logger.info('Codes: {}', output.router.return_codes.tolist())
            
            self.tensorboard.add_scalar('Neuron/Rloss', output.remote_target_loss.item(), self.global_step)
            self.tensorboard.add_scalar('Neuron/Lloss', output.local_target_loss.item(), self.global_step)
            self.tensorboard.add_scalar('Neuron/Dloss', output.distillation_loss.item(), self.global_step)

            # ---- Step increments ----
            self.global_step += 1
            self.training_loss += output.local_target_loss.item()

            # --- Memory clean up ----
            torch.cuda.empty_cache()
            del output

if __name__ == "__main__":
    # ---- Build and Run ----
    miner = Miner()
    logger.info(bittensor.config.Config.toString(miner.config))
    miner.run()

