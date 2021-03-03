"""FFNN Grunt node

This fil demonstrates how to train a FeedForward Neural network on the network
without a training ste.

Example:
        $ python miners/IMAGE/ffnn_grunt/ffnn_grunt.py

Look at the yaml config file to tweak the parameters of the model. To run with those 
default configurations, run:
        $ cd miners/IMAGE
        $ python ffnn_grunt/ffnn_grunt.py --session.config_file ffnn_grunt/ffnn_grunt_config.yaml
"""
import argparse
import os
import sys
import time
import torch
import bittensor

from munch import Munch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from bittensor.utils.model_utils import ModelToolbox
from synapses.ffnn import FFNNSynapse

class Miner():

    def __init__(self, config: Munch = None, **kwargs):
        if config == None:
            config = Miner.default_config();       
        bittensor.config.Config.update_with_kwargs(config.miner, kwargs) 
        Miner.check_config(config)
        self.config = config

        # ---- Build Neuron ----
        self.neuron = bittensor.neuron.Neuron(config)

        # ---- Build FFNN Model ----
        self.model = FFNNSynapse( self.config )
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.neuron.axon.serve( self.model )

        # ---- Optimizer ----
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr = self.config.miner.learning_rate, momentum=self.config.miner.momentum)

        # ---- Model Load/Save tools ----
        self.model_toolbox = ModelToolbox(FFNNSynapse, torch.optim.SGD)

        # ---- Logging ----
        self.tensorboard = SummaryWriter(log_dir = self.config.miner.full_path)
        if self.config.miner.record_log:
            logger.add(self.config.miner.full_path + "/{}_{}.log".format(self.config.miner.name, self.config.miner.trial_uid),format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")
     
    @staticmethod
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        Miner.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):    
        parser.add_argument('--miner.learning_rate', default=0.01, type=float, help='Training initial learning rate.')
        parser.add_argument('--miner.momentum', default=0.9, type=float, help='Training initial momentum for SGD.')
        parser.add_argument('--miner.n_epochs', default=int(sys.maxsize), type=int, help='Number of training epochs.')
        parser.add_argument('--miner.sync_interval', default=150, type=int, help='Batches before we we sync with chain and emit new weights.')
        parser.add_argument('--miner.root_dir', default='~/.bittensor/miners/', type=str,  help='Root path to load and save data associated with each miner')
        parser.add_argument('--miner.name', default='ffnn-grunt', type=str, help='Trials for this miner go in miner.root / miner.name')
        parser.add_argument('--miner.trial_uid', default=str(time.time()).split('.')[0], type=str, help='Saved models go in miner.root_dir / miner.name / miner.uid')
        parser.add_argument('--miner.record_log', default=False, help='Record all logs when running this miner')
        parser.add_argument('--miner.config_file', type=str, help='config file to run this neuron, if not using cmd line arguments.')
        bittensor.neuron.Neuron.add_args(parser)
        FFNNSynapse.add_args(parser)

    @staticmethod
    def check_config(config: Munch):
        assert config.miner.momentum > 0 and config.miner.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.miner.learning_rate > 0, "learning rate must be be a positive value."
        full_path = '{}/{}/{}/'.format(config.miner.root_dir, config.miner.name, config.miner.trial_uid)
        config.miner.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.miner.full_path):
            os.makedirs(config.miner.full_path)

    # ---- Main loop ----
    def run(self):

        # --- Subscribe / Update neuron ---
        with self.neuron:

            # ---- Loop for epochs ----
            self.model.train()
            for self.epoch in range(self.config.miner.n_epochs):

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
                if (self.epoch + 1) % self.config.miner.sync_interval == 0:

                    # --- Display Epoch ----
                    print(self.neuron.axon.__full_str__())
                    print(self.neuron.dendrite.__full_str__())
                    print(self.neuron.metagraph)
                    
                    # ---- Sync metagrapn from chain ----
                    self.neuron.metagraph.sync() # Sync with the chain.
                    
                    # --- Save Model ----
                    self.model_toolbox.save_model(
                        self.config.miner.full_path,
                        {
                            'epoch': self.epoch, 
                            'model_state_dict': self.model.state_dict(), 
                            'optimizer_state_dict': self.optimizer.state_dict(),
                        }
                    )                

   
if __name__ == "__main__":
    # ---- Build and Run ----
    miner = Miner()
    logger.info(bittensor.config.Config.toString(miner.config))
    miner.run()