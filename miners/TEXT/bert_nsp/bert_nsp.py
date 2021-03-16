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
from bittensor.utils.model_utils import ModelToolbox
from nucleuss.bert import BertNSPNucleus
from pytorch_transformers import WarmupCosineWithHardRestartsSchedule
from torch.nn.utils import clip_grad_norm_


def nsp_batch(data, batch_size, tokenizer):
    """ Returns a random batch from text dataset with 50 percent NSP.

        Args:
            data: (List[dict{'text': str}]): Dataset of text inputs.
            batch_size: size of batch to create.
        
        Returns:
            input_ids List[str]: List of sentences.
            batch_labels torch.Tensor(batch_size): 1 if random next sentence, otherwise 0.
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

    tokenized = tokenizer(batch_inputs, text_pair = batch_next, return_tensors='pt', padding=True)
    return tokenized, torch.tensor(batch_labels, dtype=torch.long)


class Miner():

    def __init__(self, config: Munch = None, **kwargs):
        if config == None:
            config = Miner.default_config();       
        bittensor.Config.update_with_kwargs(config.miner, kwargs) 
        Miner.check_config(config)
        self.config = config

        # ---- Neuron ----
        bittensor.neuron = bittensor.neuron.Neuron(self.config)

        # ---- Model ----
        self.model = BertNSPNucleus( self.config )

        # ---- Optimizer ----
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.config.miner.learning_rate, momentum=self.config.miner.momentum)
        self.scheduler = WarmupCosineWithHardRestartsSchedule(self.optimizer, 50, 300)

        # ---- Model Load/Save tools ----
        self.model_toolbox = ModelToolbox(BertNSPNucleus, torch.optim.SGD)

        # ---- Dataset ----
        # Dataset: News headlines
        self.dataset = load_dataset('ag_news')['train']


        # ---- Logging ----
        self.tensorboard = SummaryWriter(log_dir = self.config.miner.full_path)
        if self.config.miner.record_log:
            logger.add(self.config.miner.full_path + "/{}_{}.log".format(self.config.miner.name, self.config.miner.trial_uid),format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")

    @staticmethod
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        Miner.add_args(parser) 
        config = bittensor.Config.to_config(parser); 
        return config

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument('--miner.learning_rate', default=0.01, type=float, help='Training initial learning rate.')
        parser.add_argument('--miner.momentum', default=0.98, type=float, help='Training initial momentum for SGD.')
        parser.add_argument('--miner.clip_gradients', default=0.8, type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.')
        parser.add_argument('--miner.n_epochs', default=int(sys.maxsize), type=int, help='Number of training epochs.')
        parser.add_argument('--miner.epoch_length', default=500, type=int, help='Iterations of training per epoch')
        parser.add_argument('--miner.batch_size_train', default=1, type=int, help='Training batch size.')
        parser.add_argument('--miner.sync_interval', default=100, type=int, help='Batches before we sync with chain and emit new weights.')
        parser.add_argument('--miner.log_interval', default=10, type=int, help='Batches before we log miner info.')
        parser.add_argument('--miner.accumulation_interval', default=1, type=int, help='Batches before we apply acummulated gradients.')
        parser.add_argument('--miner.apply_remote_gradients', default=False, type=bool, help='If true, neuron applies gradients which accumulate from remotes calls.')
        parser.add_argument('--miner.root_dir', default='~/.bittensor/miners/', type=str,  help='Root path to load and save data associated with each miner')
        parser.add_argument('--miner.name', default='bert-nsp', type=str, help='Trials for this miner go in miner.root / miner.name')
        parser.add_argument('--miner.trial_uid', default=str(time.time()).split('.')[0], type=str, help='Saved models go in miner.root_dir / miner.name / miner.uid')
        parser.add_argument('--miner.record_log', default=False, help='Record all logs when running this miner')
        parser.add_argument('--miner.config_file', type=str, help='config file to run this neuron, if not using cmd line arguments.')
        BertNSPNucleus.add_args(parser)
        bittensor.neuron.Neuron.add_args(parser)

    @staticmethod
    def check_config(config: Munch):
        assert config.miner.momentum > 0 and config.miner.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.miner.batch_size_train > 0, "batch_size_train must a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
        full_path = '{}/{}/{}'.format(config.miner.root_dir, config.miner.name, config.miner.trial_uid)
        config.miner.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.miner.full_path):
            os.makedirs(config.miner.full_path)

    # --- Main loop ----
    def run (self):

        # ---- Subscribe ----
        with bittensor.neuron:

            # ---- Weights ----
            self.row = bittensor.metagraph.row()

            # --- Run state ---
            self.global_step = 0
            self.best_train_loss = math.inf

            # --- Loop forever ---
            for self.epoch in range(self.config.miner.n_epochs):
                try:
                    # ---- Serve ----
                    bittensor.axon.serve( self.model )

                    # ---- Train Model ----
                    self.train()
                    self.scheduler.step()

                    # If model has borked for some reason, we need to make sure it doesn't emit weights
                    # Instead, reload into previous version of model
                    if torch.any(torch.isnan(torch.cat([param.view(-1) for param in self.model.parameters()]))):
                        self.model, self.optimizer = self.model_toolbox.load_model(self.config)     
                        continue               

                    # ---- Emit row-weights ----
                    bittensor.metagraph.set_weights(self.row, wait_for_inclusion = True) # Sets my row-weights on the chain.

                    # ---- Sync metagraph ----
                    bittensor.metagraph.sync() # Pulls the latest metagraph state (with my update.)
                    self.row = bittensor.metagraph.row

                    # --- Epoch logs ----
                    print(bittensor.axon.fullToString())
                    print(bittensor.dendrite.fullToString())
                    print(bittensor.metagraph.toString())

                    # ---- Update Tensorboard ----
                    bittensor.dendrite.toTensorboard(self.tensorboard, self.global_step)
                    bittensor.metagraph.toTensorboard(self.tensorboard, self.global_step)
                    bittensor.axon.toTensorboard(self.tensorboard, self.global_step)
                
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
            inputs, targets = nsp_batch(self.dataset, self.config.miner.batch_size_train, bittensor.__tokenizer__())
            output = self.model.remote_forward (
                    bittensor.neuron,
                    inputs = inputs['input_ids'].to(self.model.device), 
                    attention_mask = inputs['attention_mask'].to(self.model.device),
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
                    bittensor.axon,
                    bittensor.dendrite)
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
    logger.info(bittensor.Config.toString(miner.config))
    miner.run()
