#!/bin/python3
"""GPT2 Language Modelling miner

This file demonstrates training the GPT2 neuron with language modelling.

Example:
        $ python examples/gpt2-wiki.py

"""
import argparse
import math
import os
import time
import torch
import torch.nn.functional as F
import traceback
import time

from termcolor import colored
from munch import Munch
from datasets import load_dataset
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

import bittensor
from bittensor.neuron import Neuron
from bittensor.config import Config
from bittensor.synapses.gpt2 import GPT2LMSynapse, nextbatch
from pytorch_transformers import WarmupCosineWithHardRestartsSchedule


class Session():

    def __init__(self, config: Munch):
        self.config = config

        # ---- Neuron ----
        self.neuron = Neuron(self.config)

        # ---- Model ----
        self.model = GPT2LMSynapse( self.config )

        # ---- Optimizer ----
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.config.session.learning_rate, momentum=self.config.session.momentum)
        self.scheduler = WarmupCosineWithHardRestartsSchedule(self.optimizer, 50, 300)

        # ---- Dataset ----
        # 74 million sentences pulled from books.
        self.dataset = load_dataset('ag_news')['train']

        # ---- Logging ----
        self.tensorboard = SummaryWriter(log_dir = self.config.session.full_path)
        if self.config.session.record_log:
            logger.add(self.config.session.full_path + "/{}_{}.log".format(self.config.session.name, self.config.session.trial_uid),format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):
        parser.add_argument('--session.learning_rate', default=0.01, type=float, help='Training initial learning rate.')
        parser.add_argument('--session.momentum', default=0.98, type=float, help='Training initial momentum for SGD.')
        parser.add_argument('--session.epoch_length', default=10, type=int, help='Iterations of training per epoch')
        parser.add_argument('--session.batch_size_train', default=1, type=int, help='Training batch size.')
        parser.add_argument('--session.sync_interval', default=100, type=int, help='Batches before we sync with chain and emit new weights.')
        parser.add_argument('--session.log_interval', default=10, type=int, help='Batches before we log session info.')
        parser.add_argument('--session.accumulation_interval', default=1, type=int, help='Batches before we apply acummulated gradients.')
        parser.add_argument('--session.apply_remote_gradients', default=False, type=bool, help='If true, neuron applies gradients which accumulate from remotes calls.')
        parser.add_argument('--session.root_dir', default='data/', type=str,  help='Root path to load and save data associated with each session')
        parser.add_argument('--session.name', default='gpt-wiki', type=str, help='Trials for this session go in session.root / session.name')
        parser.add_argument('--session.trial_uid', default=str(time.time()).split('.')[0], type=str, help='Saved models go in session.root_dir / session.name / session.uid')
        parser.add_argument('--session.record_log', default=True, help='Record all logs when running this session')
        GPT2LMSynapse.add_args(parser)
        Neuron.add_args(parser)

    @staticmethod
    def check_config(config: Munch):
        assert config.session.momentum > 0 and config.session.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.session.batch_size_train > 0, "batch_size_train must a positive value"
        assert config.session.learning_rate > 0, "learning_rate must be a positive value."
        full_path = '{}/{}/{}'.format(config.session.root_dir, config.session.name, config.session.trial_uid)
        config.session.full_path = full_path
        if not os.path.exists(config.session.full_path):
            os.makedirs(config.session.full_path)
        GPT2LMSynapse.check_config(config)
        Neuron.check_config(config)

    # --- Main loop ----
    def run (self):

        # ---- Subscribe ----
        with self.neuron:

            # ---- Weights ----
            self.row = self.neuron.metagraph.row

            # --- Run state ---
            self.epoch = -1
            self.global_step = 0
            self.best_train_loss = math.inf

            # --- Loop forever ---
            while True:
                try:
                    self.epoch += 1

                    # ---- Serve ----
                    self.neuron.axon.serve( self.model )

                    # ---- Train Model ----
                    self.train()
                    self.scheduler.step()

                    # ---- Emitting weights ----
                    self.neuron.metagraph.emit( self.weights, wait_for_inclusion = True ) # Sets my row-weights on the chain.

                    # ---- Sync metagraph ----
                    self.neuron.metagraph.sync() # Pulls the latest metagraph state (with my update.)
                    self.row = self.neuron.metagraph.row

                    # --- Epoch logs ----
                    print(self.neuron.axon.__full_str__())
                    print(self.neuron.dendrite.__full_str__())
                    print(self.neuron.metagraph.__full_str__())
                
                    # ---- Save best loss and model ----
                    if self.training_loss and self.epoch % 10 == 0:
                        if self.training_loss < self.best_train_loss:
                            self.best_train_loss = self.training_loss # update best train loss
                            logger.info( 'Saving/Serving model: epoch: {}, loss: {}, path: {}/model.torch'.format(self.epoch, self.best_train_loss, self.config.session.full_path))
                            torch.save( {'epoch': self.epoch, 'model': self.model.state_dict(), 'loss': self.best_train_loss},"{}/model.torch".format(self.config.session.full_path))
                            self.tensorboard.add_scalar('Train loss', self.training_loss, self.global_step)
                    
                # --- Catch Errors ----
                except Exception as e:
                    logger.error('Exception in training script with error: {}', e)
                    logger.info(traceback.print_exc())
                    logger.info('Continuing to train.')
    
    # ---- Train Epoch ----
    def train(self):
        self.training_loss = 0.0
        for local_step in range(self.config.session.epoch_length):
            # ---- Forward pass ----
            inputs = nextbatch(self.dataset, self.config.session.batch_size_train, bittensor.__tokenizer__())
            output = self.model.remote_forward(
                self.neuron,
                inputs,
                training = True,
            )

            # ---- Backward pass ----
            loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
            loss.backward() # Accumulates gradients on the model.
            self.optimizer.step() # Applies accumulated gradients.
            self.optimizer.zero_grad() # Zeros out gradients for next accummulation

            # ---- Train row weights ----
            batch_weights = torch.mean(output.dendrite.weights, axis = 0) # Average over batch.
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
                    self.neuron.axon,
                    self.neuron.dendrite,
                    self.neuron.metagraph)
            logger.info('Codes: {}', output.dendrite.return_codes.tolist())
            self.tensorboard.add_scalar('Rloss', output.remote_target_loss.item(), self.global_step)
            self.tensorboard.add_scalar('Lloss', output.local_target_loss.item(), self.global_step)
            self.tensorboard.add_scalar('Dloss', output.distillation_loss.item(), self.global_step)

            # ---- Step increments ----
            self.global_step += 1
            self.training_loss += output.local_target_loss.item()

            # --- Memory clean up ----
            torch.cuda.empty_cache()
            del output

if __name__ == "__main__":
    # ---- Config ----
    parser = argparse.ArgumentParser(); Session.add_args(parser) 
    config = Config.to_config(parser); Session.check_config(config)
    logger.info(Config.toString(config))
   
    # ---- Build + Run ----
    session = Session(config)
    session.run()
