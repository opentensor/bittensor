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

"""Training a MNIST Neuron.

This file demonstrates a training pipeline for an MNIST Neuron.

Example:
        $ python miners/IMAGE/mnist/mnist.py

Look at the yaml config file to tweak the parameters of the model. To run with those 
default configurations, run:
        $ cd miners/IMAGE
        $ python mnist/mnist.py --session.config_file mnist/mnist_config.yaml
"""
import argparse
import math
import os
import sys
import time
import torch
import bittensor
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from termcolor import colored
from torch.utils.tensorboard import SummaryWriter
from bittensor.utils.model_utils import ModelToolbox
from munch import Munch
from loguru import logger
from bittensor.config import Config
from nucleuss.ffnn import FFNNNucleus
from torch.nn.utils import clip_grad_norm_



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
        self.model = FFNNNucleus( config ) # Feedforward neural network with PKMRouter.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to( self.device ) # Set model to device
        
        # ---- Optimizer ---- 
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.miner.learning_rate, momentum=self.config.miner.momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10.0, gamma=0.1)

        # ---- Model Load/Save tools ----
        self.model_toolbox = ModelToolbox(FFNNNucleus, optim.SGD)

        # ---- Dataset ----
        self.train_data = torchvision.datasets.MNIST(root = self.config.miner.root_dir + "datasets/", train=True, download=True, transform=transforms.ToTensor())
        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size = self.config.miner.batch_size_train, shuffle=True, num_workers=2)
        self.test_data = torchvision.datasets.MNIST(root = self.config.miner.root_dir + "datasets/", train=False, download=True, transform=transforms.ToTensor())
        self.testloader = torch.utils.data.DataLoader(self.test_data, batch_size = self.config.miner.batch_size_test, shuffle=False, num_workers=2)

        # ---- Tensorboard ----
        self.global_step = 0
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
        parser.add_argument('--miner.momentum', default=0.9, type=float, help='Training initial momentum for SGD.')
        parser.add_argument('--miner.clip_gradients', default=0.8, type=float, help='Implement gradient clipping to avoid exploding loss on smaller architectures.')
        parser.add_argument('--miner.n_epochs', default=int(sys.maxsize), type=int, help='Number of training epochs.')
        parser.add_argument('--miner.epoch_length', default=int(sys.maxsize), type=int, help='Iterations of training per epoch (or dataset EOF)')
        parser.add_argument('--miner.batch_size_train', default=64, type=int, help='Training batch size.')
        parser.add_argument('--miner.batch_size_test', default=64, type=int, help='Testing batch size.')
        parser.add_argument('--miner.log_interval', default=150, type=int, help='Batches until miner prints log statements.')
        parser.add_argument('--miner.sync_interval', default=10, type=int, help='Batches before we we sync with chain and emit new weights.')
        parser.add_argument('--miner.root_dir', default='~/.bittensor/miners/', type=str,  help='Root path to load and save data associated with each miner')
        parser.add_argument('--miner.name', default='mnist', type=str, help='Trials for this miner go in miner.root / miner.name')
        parser.add_argument('--miner.trial_uid', default=str(time.time()).split('.')[0], type=str, help='Saved models go in miner.root_dir / miner.name / miner.uid')
        parser.add_argument('--miner.record_log', default=False, help='Record all logs when running this miner')
        parser.add_argument('--miner.config_file', type=str, help='config file to run this neuron, if not using cmd line arguments.')
        bittensor.neuron.Neuron.add_args(parser)
        FFNNNucleus.add_args(parser)

    @staticmethod
    def check_config(config: Munch):
        assert config.miner.log_interval > 0, "log_interval dimension must be positive"
        assert config.miner.momentum > 0 and config.miner.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.miner.batch_size_train > 0, "batch_size_train must be a positive value"
        assert config.miner.batch_size_test > 0, "batch_size_test must be a positive value"
        assert config.miner.learning_rate > 0, "learning rate must be be a positive value."
        full_path = '{}/{}/{}/'.format(config.miner.root_dir, config.miner.name, config.miner.trial_uid)
        config.miner.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.miner.full_path):
            os.makedirs(config.miner.full_path)

    # --- Main loop ----
    def run(self):

        # ---- Subscribe neuron ---- 
        with bittensor.neuron:

            # ---- Weights ----
            self.row = bittensor.neuron.metagraph.row().to(self.model.device)

            # --- Loop for epochs ---
            self.best_test_loss = math.inf; self.global_step = 0
            for self.epoch in range(self.config.miner.n_epochs):
                # ---- Serve ----
                bittensor.neuron.axon.serve( self.model )

                # ---- Train ----
                self.train()
                self.scheduler.step()
                                    
                # If model has borked for some reason, we need to make sure it doesn't emit weights
                # Instead, reload into previous version of model
                if torch.any(torch.isnan(torch.cat([param.view(-1) for param in self.model.parameters()]))):
                    self.model, self.optimizer = self.model_toolbox.load_model(self.config)
                    continue

                # ---- Test ----
                test_loss, test_accuracy = self.test()

                # ---- Emit ----
                bittensor.neuron.metagraph.set_weights(self.row, wait_for_inclusion = True) # Sets my row-weights on the chain.
                        
                # ---- Sync ----  
                bittensor.neuron.metagraph.sync() # Pulls the latest metagraph state (with my update.)
                self.row = bittensor.neuron.metagraph.row().to(self.device)

                # --- Display Epoch ----
                print(bittensor.neuron.axon.fullToString())
                print(bittensor.neuron.dendrite.fullToString())
                print(bittensor.neuron.metagraph.toString())

                # ---- Update Tensorboard ----
                bittensor.neuron.dendrite.toTensorboard(self.tensorboard, self.global_step)
                bittensor.neuron.metagraph.toTensorboard(self.tensorboard, self.global_step)
                bittensor.neuron.axon.toTensorboard(self.tensorboard, self.global_step)

                # ---- Save ----
                if test_loss < self.best_test_loss:
                    self.best_test_loss = test_loss # Update best loss.
                    self.model_toolbox.save_model(
                        self.config.miner.full_path,
                        {
                            'epoch': self.epoch, 
                            'model_state_dict': self.model.state_dict(), 
                            'loss': self.best_test_loss,
                            'optimizer_state_dict': self.optimizer.state_dict(),
                        }
                    )
                    self.tensorboard.add_scalar('Test loss', test_loss, self.global_step)

    # ---- Train epoch ----
    def train(self):
        # ---- Init training state ----
        self.model.train() # Turn on dropout etc.
        for batch_idx, (images, targets) in enumerate(self.trainloader): 
            if batch_idx >= self.config.miner.epoch_length:
                break   
            self.global_step += 1 

            # ---- Remote Forward pass ----
            output = self.model.remote_forward(  
                neuron = bittensor.neuron,
                images = images.to(self.device), 
                targets = torch.LongTensor(targets).to(self.device), 
            ) 
            
            # ---- Remote Backward pass ----
            loss = output.remote_target_loss + output.local_target_loss + output.distillation_loss
            loss.backward() # Accumulates gradients on the model.
            clip_grad_norm_(self.model.parameters(), self.config.miner.clip_gradients) # clip model gradients
            self.optimizer.step() # Applies accumulated gradients.
            self.optimizer.zero_grad() # Zeros out gradients for next accummulation 

            # ---- Train weights ----
            batch_weights = torch.mean(output.router.weights, axis = 0).to(self.model.device) # Average over batch.
            self.row = (1 - 0.03) * self.row + 0.03 * batch_weights # Moving avg update.
            self.row = F.normalize(self.row, p = 1, dim = 0) # Ensure normalization.

            # ---- Step Logs + Tensorboard ----
            processed = ((batch_idx + 1) * self.config.miner.batch_size_train)
            progress = (100. * processed) / len(self.train_data)
            logger.info('GS: {}\t Epoch: {} [{}/{} ({})]\tLoss: {}\tAcc: {}\tAxon: {}\tDendrite: {}', 
                    colored('{}'.format(self.global_step), 'blue'), 
                    colored('{}'.format(self.epoch), 'blue'), 
                    colored('{}'.format(processed), 'green'), 
                    colored('{}'.format(len(self.train_data)), 'red'),
                    colored('{:.2f}%'.format(progress), 'green'),
                    colored('{:.4f}'.format(output.local_target_loss.item()), 'green'),
                    colored('{:.4f}'.format(output.local_accuracy.item()), 'green'),
                    bittensor.neuron.axon,
                    bittensor.neuron.dendrite)
            self.tensorboard.add_scalar('Rloss', output.remote_target_loss.item(), self.global_step)
            self.tensorboard.add_scalar('Lloss', output.local_target_loss.item(), self.global_step)
            self.tensorboard.add_scalar('Dloss', output.distillation_loss.item(), self.global_step)


    # --- Test epoch ----
    def test (self):
        with torch.no_grad(): # Turns off gradient computation for inference speed up.
            self.model.eval() # Turns off Dropoutlayers, BatchNorm etc.
            loss = 0.0; accuracy = 0.0
            for _, (images, labels) in enumerate(self.testloader):

                # ---- Local Forward pass ----
                outputs = self.model.local_forward(
                    images = images.to(self.device), 
                    targets = torch.LongTensor(labels).to(self.device), 
                )
                loss += outputs.local_target_loss.item()
                accuracy += outputs.local_accuracy.item()
                
            return loss / len(self.testloader), accuracy / len(self.testloader) 

        
if __name__ == "__main__":
    # ---- Build and Run ----
    miner = Miner()
    logger.info(bittensor.Config.toString(miner.config))
    miner.run()



