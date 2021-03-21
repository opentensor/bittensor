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
import torch.multiprocessing as mp 
import traceback
import time
import threading
import bittensor

from termcolor import colored
from nuclei.xlm import XLMNucleus, nextbatch
from bittensor.utils.model_utils import ModelToolbox
from munch import Munch
from loguru import logger
from pytorch_transformers import WarmupCosineWithHardRestartsSchedule
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter


class Miner():
    """
    Initializes, trains, and tests models created inside of 'bittensor/nucleus'. 
    During instantiation, this class takes a config as a [Munch](https://github.com/Infinidat/munch) object. 
    """

    def __init__(self, config: Munch = None, **kwargs):
        if config == None:
            config = Miner.default_config();       
        bittensor.Config.update_with_kwargs(config.miner, kwargs) 
        Miner.check_config(config)
        logger.info(bittensor.Config.toString( config ))
        self.config = config

        # ---- Wallet ----
        self.wallet = bittensor.Wallet( self.config )
        if not self.wallet.has_coldkeypub:
            self.wallet.create_new_coldkey(n_words = 12, use_password = True )
        if not self.wallet.has_hotkey:
            self.wallet.create_new_hotkey(n_words = 12)
        
        # ---- Bittensor components ----
        self.axon = bittensor.Axon( config = config, wallet = wallet )
        self.dendrite = bittensor.Dendrite( config = config, wallet = wallet )
        self.subtensor = bittensor.Subtensor( config = config, wallet = wallet )
        self.metagraph = bittensor.Metagraph( subtensor = subtensor )

        # ---- Model ----
        self.model = XLMNucleus ( self.config, self.dendrite, self.metagraph )

        # ---- Serving thread ----
        self.quit_serving = None
        self.serving_thread = None # Thread for running queries on our model.

        # ---- Optimizer ----
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.config.miner.learning_rate, momentum=self.config.miner.momentum)
        self.scheduler = WarmupCosineWithHardRestartsSchedule(self.optimizer, 50, 300)

        # ---- Model Load/Save tools ----
        self.model_toolbox = ModelToolbox(XLMNucleus, torch.optim.SGD)

        # ---- Dataset ----
        # Dataset: 74 million sentences pulled from books.
        self.dataset = load_dataset('amazon_reviews_multi', 'en')['train']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Logging ----
        self.tensorboard = SummaryWriter(log_dir = self.config.miner.full_path)
        if self.config.miner.record_log:
            filepath = self.config.miner.full_path + "/{}_{}.log".format(self.config.miner.name, self.config.miner.trial_uid),
            logger.add (
                filepath,
                format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
                rotation="500 MB",
                retention="10 days"
            )
    
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
        XLMNucleus.add_args(parser)
        bittensor.Axon.add_args(parser)
        bittensor.Dendrite.add_args(parser)
        bittensor.Subtensor.add_args(parser)

    @staticmethod
    def check_config(config: Munch):
        assert config.miner.momentum > 0 and config.miner.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.miner.batch_size_train > 0, "batch_size_train must be a positive value"
        assert config.miner.learning_rate > 0, "learning_rate must be a positive value."
        full_path = '{}/{}/{}'.format(config.miner.root_dir, config.miner.name, config.miner.trial_uid)
        config.miner.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.miner.full_path):
            os.makedirs(config.miner.full_path)       
    
    # --- Main loop ----
    def run (self):

        # --- Connect to the chain. ---
        if not self.subtensor.connect():
            raise RuntimeError('Failed to connect miner to network: {}'.format(self.subtensor.config.network))

        # --- Subscribe our endpoint ----
        if not self.subtensor.subscribe(
                self.config.axon.external_ip, 
                self.config.axon.external_port,
                self.config.neuron.modality,
                self.wallet.coldkeypub,
                wait_for_finalization = True,
                timeout = 4 * bittensor.__blocktime__
            )
            raise RuntimeError('Failed to subscribe miner on network: {}'.format(self.subtensor.config.network))

        # --- Start the serving endpoint ----
        self.axon.start()

        # --- Start the serving thread ----
        self.quit_serving = mp.Event()
        self.serving_thread = threading.Thread( target = self.serving_loop,  name = 'serving', daemon=True)
        self.serving_thread.start()

        # --- Sync metagraph  ----
        self.metagraph.sync()
        self.weights = torch.rand([self.metagraph.n()])

        # --- Init running state ---
        self.global_step = 0
        self.best_train_loss = math.inf

        # --- Loop for epochs ---
        for self.epoch in range(self.config.miner.n_epochs):
            try:
                # ---- Train Model ----
                self.train()
                self.scheduler.step()
                
                # ---- Catch NaNs ----
                # If model has borked for some reason, we need to make sure it doesn't emit weights
                # Instead, reload into previous version of model
                if torch.any(torch.isnan(torch.cat([param.view(-1) for param in self.model.parameters()]))):
                    self.model, self.optimizer = self.model_toolbox.load_model( self.config )
                    continue

                # ---- Set weights on chain ----
                self.subtensor.set_weights (
                    uids = self.metagraph.uids(),
                    weights = self.weights,
                    wait_for_inclusion = False
                )

                # ---- Sync metagraph ----
                self.metagraph.sync() # Pulls latest chain info.
                self.weights = torch.nn.functional.pad(self.weights, pad = [0, self.metagraph.n() - self.weights.numel() ]) # Pads weights to the correct size.

                # --- Epoch logs ----
                print(self.axon)
                print(self.dendrite)
                print(self.metagraph)

                # ---- Update Tensorboard ----
                self.axon.__to_tensorboard__(self.tensorboard, self.global_step)
                self.dendrite.__to_tensorboard__(self.tensorboard, self.global_step)
                self.metagraph.__to_tensorboard__(self.tensorboard, self.global_step)
            
                # ---- Save best loss and model ----
                if self.training_loss and self.epoch % 10 == 0 and self.training_loss < self.best_train_loss:
                    self.best_train_loss = self.training_loss / 10 # update best train loss
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
                logger.error('Exception in training script with error: {}, {}', e, traceback.format_exc())
                logger.info('Continuing to train.')
    
    # ---- Train Epoch ----
    def train(self):
        self.training_loss = 0.0
        for local_step in range(self.config.miner.epoch_length):
            # ---- Forward pass ----
            inputs = nextbatch(self.dataset, self.config.miner.batch_size_train, bittensor.__tokenizer__())
            output = self.model.remote_forward(
                inputs.to( self.model.device ),
                training = True,
            )

            # ---- Backward pass ----
            loss = output.local_target_loss + output.distillation_loss + output.remote_target_loss
            loss.backward() # Accumulates gradients on the model.
            self.optimizer.step() # Applies accumulated gradients.
            self.optimizer.zero_grad() # Zeros out gradients for next accummulation

            # ---- Train row weights ----
            batch_weights = torch.mean(output.router.weights, axis = 0).to(self.model.device) # Average over batch.
            self.weights = (1 - 0.03) * self.weights + 0.03 * batch_weights # Moving avg update.
            self.weights = F.normalize( self.weights, p = 1, dim = 0 ) # Ensure normalization.

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

    # ---- Serving loop -----
    def serving_loop ( self ): 
        # ---- Loop until event is set -----
        logger.info('Serving thread started: ')
        while not self.quit_serving.is_set():

            # ---- Pull request ----
            logger.info('Axon:{}, waiting for query ... ', self.axon.toString())
            pong, pubkey, inputs, modality = self.axon.next_forward_item( timeout = 10.0 )

            # ---- Process request ----
            if None not in [ pong, pubkey, inputs, modality]:
                logger.info('Recieved Query: from:{}, inputs.shape:{}', pubkey, inputs.shape)
                try:          
                    outputs = self.model.local_forward( inputs, training = False ).local_hidden
                    pong.send( outputs.detach() )
                    logger.info('Sent response: to:{}, outputs.shape:{}', pubkey, outputs.shape)

                except Exception as e:
                    logger.exception('Error in forward process with error {}', e)
                    continue

        # ---- Tensorboard ----
        #bittensor.neuron.axon.toTensorboard(serving_tensorboard, serving_step)

    def __del__(self):
        if self.serving_thread != None:
            self.quit_serving.set()
            self.serving_thread.join( timeout = 12.0 )
            if self.serving_thread.is_alive():
                logger.error('Failed to join serving thread.')

if __name__ == "__main__":
    # ---- Build and Run ----
    miner = Miner()
    miner.run()