"""Training a MNIST Neuron.
This file demonstrates a training pipeline for an MNIST Neuron.
Example:
        $ python examples/mnist.py
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

import bittensor
from bittensor.neuron import Neuron
from bittensor.config import Config
from bittensor.synapses.ffnn import FFNNSynapse

class Session():

    def __init__(self, config: Munch = None):
        if config == None:
            config = Session.build_config(); logger.info(bittensor.config.Config.toString(config))
        self.config = config

        # ---- Neuron ----
        self.neuron = bittensor.neuron.Neuron(self.config)
    
        # ---- Model ----
        self.model = FFNNSynapse( config ) # Feedforward neural network with PKMRouter.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to( self.device ) # Set model to device
        
        # ---- Optimizer ---- 
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.session.learning_rate, momentum=self.config.session.momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10.0, gamma=0.1)

        # ---- Dataset ----
        self.train_data = torchvision.datasets.MNIST(root = self.config.session.root_dir + "datasets/", train=True, download=True, transform=transforms.ToTensor())
        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size = self.config.session.batch_size_train, shuffle=True, num_workers=2)
        self.test_data = torchvision.datasets.MNIST(root = self.config.session.root_dir + "datasets/", train=False, download=True, transform=transforms.ToTensor())
        self.testloader = torch.utils.data.DataLoader(self.test_data, batch_size = self.config.session.batch_size_test, shuffle=False, num_workers=2)

        # ---- Tensorboard ----
        self.global_step = 0
        self.tensorboard = SummaryWriter(log_dir = self.config.session.full_path)
        if self.config.session.record_log:
            logger.add(self.config.session.full_path + "/{}_{}.log".format(self.config.session.name, self.config.session.trial_uid),format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}")

    @staticmethod
    def build_config() -> Munch:
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
        parser.add_argument('--session.sync_interval', default=10, type=int, help='Batches before we we sync with chain and emit new weights.')
        parser.add_argument('--session.root_dir', default='~/.bittensor/sessions/', type=str,  help='Root path to load and save data associated with each session')
        parser.add_argument('--session.name', default='mnist', type=str, help='Trials for this session go in session.root / session.name')
        parser.add_argument('--session.trial_uid', default=str(time.time()).split('.')[0], type=str, help='Saved models go in session.root_dir / session.name / session.uid')
        parser.add_argument('--session.record_log', default=True, help='Record all logs when running this session')
        parser.add_argument('--session.config_file', type=str, help='config file to run this neuron, if not using cmd line arguments.')
        bittensor.neuron.Neuron.add_args(parser)
        FFNNSynapse.add_args(parser)

    @staticmethod
    def check_config(config: Munch):
        assert config.session.log_interval > 0, "log_interval dimension must be positive"
        assert config.session.momentum > 0 and config.session.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.session.batch_size_train > 0, "batch_size_train must be a positive value"
        assert config.session.batch_size_test > 0, "batch_size_test must be a positive value"
        assert config.session.learning_rate > 0, "learning rate must be be a positive value."
        full_path = '{}/{}/{}/'.format(config.session.root_dir, config.session.name, config.session.trial_uid)
        config.session.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.session.full_path):
            os.makedirs(config.session.full_path)
        FFNNSynapse.check_config(config)
        bittensor.neuron.Neuron.check_config(config)

    # --- Main loop ----
    def run(self):

        # ---- Subscribe neuron ---- 
        with self.neuron:

            # ---- Weights ----
            self.row = self.neuron.metagraph.row.to(self.model.device)

            # ---- Loop forever ----
            self.epoch = -1; self.best_test_loss = math.inf; self.global_step = 0
            while True:
                self.epoch += 1

                # ---- Serve ----
                self.neuron.axon.serve( self.model )

                # ---- Train ----
                self.train()
                self.scheduler.step()

                # ---- Test ----
                test_loss, test_accuracy = self.test()

                # ---- Emit ----
                self.neuron.metagraph.set_weights(self.row, wait_for_inclusion = True) # Sets my row-weights on the chain.
                        
                # ---- Sync ----  
                self.neuron.metagraph.sync() # Pulls the latest metagraph state (with my update.)
                self.row = self.neuron.metagraph.row.to(self.device)

                # --- Display Epoch ----
                print(self.neuron.axon.__full_str__())
                print(self.neuron.dendrite.__full_str__())
                print(self.neuron.metagraph)

                # ---- Update Tensorboard ----
                self.neuron.dendrite.__to_tensorboard__(self.tensorboard, self.global_step)
                self.neuron.metagraph.__to_tensorboard__(self.tensorboard, self.global_step)
                self.neuron.axon.__to_tensorboard__(self.tensorboard, self.global_step)

                # ---- Save ----
                if test_loss < self.best_test_loss:
                    self.best_test_loss = test_loss # Update best loss.
                    logger.info( 'Saving model: epoch: {}, accuracy: {}, loss: {}, path: {}/model.torch'.format(self.epoch, test_accuracy, self.best_test_loss, self.config.session.full_path))
                    torch.save( {'epoch': self.epoch, 'model': self.model.state_dict(), 'loss': self.best_test_loss},"{}/model.torch".format(self.config.session.full_path))
                    self.tensorboard.add_scalar('Test loss', test_loss, self.global_step)

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
            batch_weights = torch.mean(output.router.weights, axis = 0).to(self.model.device) # Average over batch.
            self.row = (1 - 0.03) * self.row + 0.03 * batch_weights # Moving avg update.
            self.row = F.normalize(self.row, p = 1, dim = 0) # Ensure normalization.

            # ---- Step Logs + Tensorboard ----
            processed = ((batch_idx + 1) * self.config.session.batch_size_train)
            progress = (100. * processed) / len(self.train_data)
            logger.info('GS: {}\t Epoch: {} [{}/{} ({})]\tLoss: {}\tAcc: {}\tAxon: {}\tDendrite: {}', 
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
    config = Session.build_config(); logger.info(Config.toString(config))
    session = Session(config)
    session.run()



