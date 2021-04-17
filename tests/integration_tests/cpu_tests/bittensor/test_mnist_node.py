"""Training a MNIST Neuron.
This file demonstrates a training pipeline for an MNIST Neuron.
Example:
        $ python miners/IMAGE/mnist.py
"""
import argparse
import math
import os
import time
import torch
from termcolor import colored
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from munch import Munch
from loguru import logger
logger = logger.opt(ansi=True)

import bittensor
from bittensor.neuron import Neuron
from bittensor.config import Config
from synapses.ffnn import FFNNSynapse

class Session():

    def __init__(self, config: Munch):
        if config == None:
            config = Session.default_config()
        self.config = config

        # ---- Neuron ----
        self.neuron = Neuron(self.config)
    
        # ---- Model ----
        self.model = FFNNSynapse( config ) # Feedforward neural network with PKMRouter.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to( self.device ) # Set model to device
        
        # ---- Optimizer ---- 
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.miner.learning_rate, momentum=self.config.miner.momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10.0, gamma=0.1)

        # ---- Dataset ----
        self.train_data = torchvision.datasets.MNIST(root = self.config.miner.root_dir + "datasets/", train=True, download=True, transform=transforms.ToTensor())
        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size = self.config.miner.batch_size_train, shuffle=True, num_workers=2)
        self.test_data = torchvision.datasets.MNIST(root = self.config.miner.root_dir + "datasets/", train=False, download=True, transform=transforms.ToTensor())
        self.testloader = torch.utils.data.DataLoader(self.test_data, batch_size = self.config.miner.batch_size_test, shuffle=False, num_workers=2)

        # ---- Tensorboard ----
        self.global_step = 0
        self.tensorboard = SummaryWriter(log_dir = self.config.miner.full_path)

    @staticmethod
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        Session.add_args(parser) 
        config = Config.to_config(parser); 
        Session.check_config(config)
        return config

    @staticmethod
    def add_args(parser: argparse.ArgumentParser):    
        parser.add_argument('--session.learning_rate', default=0.01, type=float, help='Training initial learning rate.')
        parser.add_argument('--session.momentum', default=0.9, type=float, help='Training initial momentum for SGD.')
        parser.add_argument('--session.batch_size_train', default=64, type=int, help='Training batch size.')
        parser.add_argument('--session.batch_size_test', default=64, type=int, help='Testing batch size.')
        parser.add_argument('--session.log_interval', default=150, type=int, help='Batches until session prints log statements.')
        parser.add_argument('--session.sync_interval', default=150, type=int, help='Batches before we we sync with chain and emit new weights.')
        parser.add_argument('--session.root_dir', default='data/', type=str,  help='Root path to load and save data associated with each session')
        parser.add_argument('--session.name', default='mnist', type=str, help='Trials for this session go in session.root / session.name')
        parser.add_argument('--session.uid', default=str(time.time()).split('.')[0], type=str, help='Saved models go in session.root_dir / session.name / session.uid')
        Neuron.add_args(parser)
        FFNNSynapse.add_args(parser)

    @staticmethod
    def check_config(config: Munch):
        assert config.miner.log_interval > 0, "log_interval dimension must be positive"
        assert config.miner.momentum > 0 and config.miner.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.miner.batch_size_train > 0, "batch_size_train must be a positive value"
        assert config.miner.batch_size_test > 0, "batch_size_test must be a positive value"
        assert config.miner.learning_rate > 0, "learning rate must be be a positive value."
        full_path = '{}/{}/{}/'.format(config.miner.root_dir, config.miner.name, config.miner.uid)
        config.miner.full_path = full_path
        if not os.path.exists(config.miner.full_path):
            os.makedirs(config.miner.full_path)
        FFNNSynapse.check_config(config)
        Neuron.check_config(config)

    # --- Main loop ----
    def run(self):

        # ---- Subscribe neuron ---- 
        with self.neuron:

            # ---- Loop forever ----
            start_time = time.time()
            self.epoch = -1; self.best_test_loss = math.inf; self.global_step = 0
            self.weights = self.neuron.metagraph.row # Trained weights.
            while True:
                self.epoch += 1

                # ---- Emit ----
                self.neuron.metagraph.set_weights(self.weights, wait_for_inclusion = True) # Sets my row-weights on the chain.
                        
                # ---- Sync ----  
                self.neuron.metagraph.sync() # Pulls the latest metagraph state (with my update.)
                self.weights = self.neuron.metagraph.row.to(self.device)
                        
                # ---- Train ----
                self.train()
                self.scheduler.step()
                
                # ---- Test ----
                test_loss, test_accuracy = self.test()

                # ---- Test checks ----
                time_elapsed = time.time() - start_time
                assert test_accuracy > 0.8
                assert test_loss < 0.2
                assert len(self.neuron.metagraph.state.neurons) > 0
                assert time_elapsed < 300 # 1 epoch of MNIST should take less than 5 mins.

                # ---- End test ----
                break

    # ---- Train epoch ----
    def train(self):
        # ---- Init training state ----
        self.model.train() # Turn on dropout etc.
        for batch_idx, (images, targets) in enumerate(self.trainloader):    
            self.global_step += 1 

            # ---- Remote Forward pass ----
            output = self.model.remote_forward(  
                neuron = self.neuron,
                images = images.to(self.device), 
                targets = torch.LongTensor(targets).to(self.device), 
            ) 
            
            # ---- Remote Backward pass ----
            loss = output.remote_target_loss + output.local_target_loss + output.distillation_loss
            loss.backward() # Accumulates gradients on the model.
            self.optimizer.step() # Applies accumulated gradients.
            self.optimizer.zero_grad() # Zeros out gradients for next accummulation 

            # ---- Train weights ----
            batch_weights = torch.mean(output.router.weights, axis = 0) # Average over batch.
            self.weights = (1 - 0.03) * self.weights + 0.03 * batch_weights # Moving avg update.
            self.weights = F.normalize(self.weights, p = 1, dim = 0) # Ensure normalization.

            # ---- Step Logs + Tensorboard ----
            processed = ((batch_idx + 1) * self.config.miner.batch_size_train)
            progress = (100. * processed) / len(self.train_data)
            logger.info('GS: {}\t Epoch: {} [{}/{} ({})]\t Loss: {}\t Acc: {}\t Axon: {}\t Dendrite: {}', 
                    colored('{}'.format(self.global_step), 'blue'), 
                    colored('{}'.format(self.epoch), 'blue'), 
                    colored('{}'.format(processed), 'green'), 
                    colored('{}'.format(len(self.train_data)), 'red'),
                    colored('{:.2f}%'.format(progress), 'green'),
                    colored('{:.4f}'.format(output.local_target_loss.item()), 'green'),
                    colored('{:.4f}'.format(output.local_accuracy.item()), 'green'),
                    self.neuron.axon,
                    self.neuron.dendrite)
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
    config = Session.config(); logger.info(Config.toString(config))
    session = Session(config)
    session.run()

