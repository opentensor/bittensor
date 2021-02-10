"""FFNN Grunt node

This fil demonstrates how to train a FeedForward Neural network on the network
without a training ste.

Example:
        $ python examples/ffnn_grunt.py

"""
import argparse
import math
import os
import sys
import pathlib
import time
import torch
import torch.nn.functional as F

from munch import Munch
from loguru import logger
from termcolor import colored
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from bittensor.utils.model_utils import ModelToolbox

import bittensor
from bittensor.config import Config
from bittensor.synapses.ffnn import FFNNSynapse

class Session():

    def __init__(self, config: Munch = None):
        if config == None:
            config = Session.build_config(); logger.info(bittensor.config.Config.toString(config))
        self.config = config

        # ---- Build Neuron ----
        self.neuron = bittensor.neuron.Neuron(config)

        # ---- Build FFNN Model ----
        self.model = FFNNSynapse( self.config )
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.neuron.axon.serve( self.model )

        # ---- Optimizer ----
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.config.session.learning_rate, momentum=self.config.session.momentum)

        # ---- Model Load/Save tools ----
        self.model_toolbox = ModelToolbox(FFNNSynapse, torch.optim.SGD)

        # ---- Logging ----
        self.tensorboard = SummaryWriter(log_dir = self.config.session.full_path)
        if self.config.session.record_log:
            logger.add(self.config.session.full_path + "/{}_{}.log".format(self.config.session.name, self.config.session.trial_uid),format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
     
    @staticmethod
    def build_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        Session.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        Session.check_config(config)
        return config

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):    
        parser.add_argument('--session.learning_rate', default=0.01, type=float, help='Training initial learning rate.')
        parser.add_argument('--session.momentum', default=0.9, type=float, help='Training initial momentum for SGD.')
        parser.add_argument('--session.n_epochs', default=int(sys.maxsize), type=int, help='Number of training epochs.')
        parser.add_argument('--session.sync_interval', default=150, type=int, help='Batches before we we sync with chain and emit new weights.')
        parser.add_argument('--session.root_dir', default='~/.bittensor/sessions/', type=str,  help='Root path to load and save data associated with each session')
        parser.add_argument('--session.name', default='ffnn-grunt', type=str, help='Trials for this session go in session.root / session.name')
        parser.add_argument('--session.trial_uid', default=str(time.time()).split('.')[0], type=str, help='Saved models go in session.root_dir / session.name / session.uid')
        parser.add_argument('--session.record_log', default=True, help='Record all logs when running this session')
        parser.add_argument('--session.config_file', type=str, help='config file to run this neuron, if not using cmd line arguments.')
        bittensor.neuron.Neuron.add_args(parser)
        FFNNSynapse.add_args(parser)

    @staticmethod
    def check_config(config: Munch):
        assert config.session.momentum > 0 and config.session.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.session.learning_rate > 0, "learning rate must be be a positive value."
        full_path = '{}/{}/{}/'.format(config.session.root_dir, config.session.name, config.session.trial_uid)
        config.session.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.session.full_path):
            os.makedirs(config.session.full_path)
        FFNNSynapse.check_config(config)
        bittensor.neuron.Neuron.check_config(config)

    # ---- Main loop ----
    def run(self):

        # --- Subscribe / Update neuron ---
        with self.neuron:

            # ---- Loop for epochs ----
            self.model.train()
            for self.epoch in range(self.config.session.n_epochs):

                # ---- Poll until gradients ----
                public_key, inputs_x, grads_dy, modality_x = self.neuron.axon.gradients.get(block = True)

                # ---- Backward Gradients ----
                # TODO (const): batch normalization over the gradients for consistency.
                grads_dy = torch.where(torch.isnan(grads_dy), torch.zeros_like(grads_dy), grads_dy)
                self.model.backward(inputs_x, grads_dy, modality_x)

                # ---- Apply Gradients ----
                self.optimizer.step() # Apply accumulated gradients.
                self.optimizer.zero_grad() # Clear any lingering gradients

                # If model has borked for some reason, we need to make sure it doesn't emit weights
                # Instead, reload into previous version of the model
                if torch.any(torch.isnan(torch.cat([param.view(-1) for param in self.model.parameters()]))):
                    self.model, self.optimizer = self.model_toolbox.load_model(self.config)

                # ---- Serve latest model ----
                self.neuron.axon.serve( self.model ) # Serve the newest model.
                logger.info('Step: {} \t Key: {} \t sum(W[:,0])', self.epoch, public_key, torch.sum(self.neuron.metagraph.col).item())
            
                # ---- Sync State ----
                if (self.epoch + 1) % self.config.session.sync_interval == 0:

                    # --- Display Epoch ----
                    print(self.neuron.axon.__full_str__())
                    print(self.neuron.dendrite.__full_str__())
                    print(self.neuron.metagraph)
                    
                    # ---- Sync metagrapn from chain ----
                    self.neuron.metagraph.sync() # Sync with the chain.
                    
                    # --- Save Model ----
                    self.model_toolbox.save_model(
                        self.config.session.full_path,
                        {
                            'epoch': self.epoch, 
                            'model_state_dict': self.model.state_dict(), 
                            'optimizer_state_dict': self.optimizer.state_dict(),
                        }
                    )                

   
if __name__ == "__main__":
    # ---- Build and Run ----
    session = Session()
    session.run()