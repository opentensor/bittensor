"""Training a MNIST Neuron.

This file demonstrates a training pipeline for an MNIST Neuron.

Example:
        $ python examples/mnist/main.py
"""

import bittensor
from bittensor.synapse import Synapse
from bittensor.config import Config
from bittensor.synapses.ffnn import FFNNSynapse, FFNNConfig
from substrateinterface import Keypair

import random
from loguru import logger
import math
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import traceback
import unittest

default_config = """
    session_settings:
        axon_port: 8081
        chain_endpoint: http://206.189.254.5:12345
        logdir: /tmp/

        metagraph:
            polls_every_sec: 15
            re_poll_neuron_every_blocks: 5
            stale_emit_limit: 30

    training:
        datapath: /tmp/
        batch_size: 10
        log_interval: 10
        total_epochs: 1
        batch_size_train: 64
        batch_size_test: 64
        learning_rate: 0.05
        momentum: 0.9
"""

class MnistNode(unittest.TestCase):

    def setUp(self):
        # Load config, keys, and build session.
        self.config = Config.load_from_yaml_string(yaml_str = default_config)
        self.config = Config.obtain_ip_address(self.config)
        mnemonic = Keypair.generate_mnemonic()
        self.keypair = Keypair.create_from_mnemonic(mnemonic)
        self.session = bittensor.init(self.config, self.keypair)
    
        # Build and server the synapse.
        model_config = FFNNConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FFNNSynapse(model_config, self.session)
        self.model.to( self.device ) # Set model to device.
        self.session.serve( self.model.deepcopy() )

        # Build the optimizer.
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.training.learning_rate, momentum=self.config.training.momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10.0, gamma=0.1)

        # Load (Train, Test) datasets into memory.
        self.train_data = torchvision.datasets.MNIST(root = self.config.training.datapath + "datasets/", train=True, download=True, transform=transforms.ToTensor())
        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size = self.config.training.batch_size_train, shuffle=True, num_workers=2)
        self.test_data = torchvision.datasets.MNIST(root = self.config.training.datapath + "datasets/", train=False, download=True, transform=transforms.ToTensor())
        self.testloader = torch.utils.data.DataLoader(self.test_data, batch_size = self.config.training.batch_size_test, shuffle=False, num_workers=2)
    
    # Test training.
    def test(self):
        with self.session:
            best_loss = math.inf
            best_accuracy = 0
            epoch = 0
            total_epochs = 1
            while epoch < total_epochs:

                self.train(self.model, epoch)
                test_loss, accuracy = self.validate(self.model)
                self.scheduler.step()
                epoch += 1

                best_accuracy = accuracy if accuracy >= best_accuracy else best_accuracy
                if test_loss < best_loss:
                    best_loss = test_loss
                    logger.info("Best test loss: {}".format(best_loss))
                    self.session.serve( self.model.deepcopy() )

            assert best_loss <= 0.1
            assert best_accuracy >= 75 
            # Neurons should be 3, but we are setting to >= 2 due to the
            # unpredictability of circleci booting up neurons, some neurons might
            # start and finish before others even boot up.
            # TODO: bring back this assert once chain issues are resolved
            #assert len(self.session.metagraph.neurons()) >= 2

    # Train loop: Single threaded training of MNIST.
    def train(self, model, epoch):
        # Turn on Dropoutlayers BatchNorm etc.
        model.train()
        for batch_idx, (images, targets) in enumerate(self.trainloader):
            # Clear gradients.
            self.optimizer.zero_grad()

            # Forward pass.
            images = images.to(self.device)
            targets = torch.LongTensor(targets).to(self.device)
            output = model(images, targets, remote = True)
            
            # Backprop.
            loss = output.remote_target_loss + output.distillation_loss
            loss.backward()
            self.optimizer.step()

            # Logs:
            if (batch_idx + 1) % self.config.training.log_interval == 0: 
                n = len(self.train_data)
                max_logit = output.remote_target.data.max(1, keepdim=True)[1]
                correct = max_logit.eq( targets.data.view_as(max_logit) ).sum()
                loss_item  = output.remote_target_loss.item()
                processed = ((batch_idx + 1) * self.config.training.batch_size_train)
                
                progress = (100. * processed) / n
                accuracy = (100.0 * correct) / self.config.training.batch_size_train
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLocal Loss: {:.6f}\t Accuracy: {:.6f}\t nP: {}', 
                    epoch, processed, n, progress, loss_item, accuracy, len(self.session.metagraph.neurons()))

    # Test loop.
    # Evaluates the local model on the hold-out set.
    # Returns the test_accuracy and test_loss.
    def validate(self, model: Synapse):
        
        # Turns off Dropoutlayers, BatchNorm etc.
        model.eval()
        
        # Turns off gradient computation for inference speed up.
        with torch.no_grad():
        
            loss = 0.0
            correct = 0.0
            for _, (images, labels) in enumerate(self.testloader):                
                
                images = images.to(self.device)
                # Labels to Tensor
                labels = torch.LongTensor(labels).to(self.device)

                # Compute full pass and get loss.
                outputs = model(images, labels, remote = False)
                loss = loss + outputs.loss
                
                # Count accurate predictions.
                max_logit = outputs.local_target.data.max(1, keepdim=True)[1]
                correct = correct + max_logit.eq( labels.data.view_as(max_logit) ).sum()
        
        # # Log results.
        n = len(self.test_data)
        loss /= n
        accuracy = (100. * correct) / n
        logger.info('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, n, accuracy))  
        return loss, accuracy




    
if __name__ == "__main__":
    node = MnistNode()
    node.setUp()
    node.test()
    

