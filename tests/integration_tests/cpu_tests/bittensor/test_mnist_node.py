"""Training a MNIST Neuron.

This file demonstrates a training pipeline for an MNIST Neuron.

Example:
        $ python examples/mnist/main.py
"""
import bittensor
from bittensor.synapse import Synapse
from bittensor.config import Config
from bittensor.neuron import NeuronBase
from bittensor.synapses.ffnn import FFNNSynapse
from bittensor.subtensor import Keypair
from bittensor.utils.asyncio import Asyncio

import random
from loguru import logger
import math
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import traceback
import unittest
import pytest


class Neuron(NeuronBase):

    def __init__(self, config):
        self.config = config
       
    def start(self, session):

        # Build and server the synapse.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FFNNSynapse(self.config, session)
        model.to( device ) # Set model to device.
        session.serve( model.deepcopy() )

        # Build the optimizer.
        optimizer = optim.SGD(model.parameters(), lr=self.config.neuron.learning_rate, momentum=self.config.neuron.momentum)

        # Load (Train, Test) datasets into memory.
        train_data = torchvision.datasets.MNIST(root = self.config.neuron.datapath + "datasets/", train=True, download=True, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(train_data, batch_size = self.config.neuron.batch_size_train, shuffle=True, num_workers=2)
        
        # Train 1 epoch.
        model.train()
        best_loss = math.inf
        best_accuracy = 0
        for batch_idx, (images, targets) in enumerate(trainloader):
            # Clear gradients.
            optimizer.zero_grad()

            # Forward pass.
            images = images.to(device)
            targets = torch.LongTensor(targets).to(device)
            output = model(images, targets, remote = True)

            # Backprop.
            loss = output.remote_target_loss + output.distillation_loss
            loss.backward()
            optimizer.step()

            # Metrics.
            max_logit = output.remote_target.data.max(1, keepdim=True)[1]
            correct = max_logit.eq( targets.data.view_as(max_logit) ).sum()
            target_loss  = output.remote_target_loss.item()
            accuracy = (100.0 * correct) / self.config.neuron.batch_size_train

            # Update best vals.
            best_accuracy = accuracy if accuracy >= best_accuracy else best_accuracy
            if target_loss < best_loss:
                best_loss = target_loss
                session.serve( model.deepcopy() )

            # Logs:
            if (batch_idx + 1) % self.config.neuron.log_interval == 0: 
                n = len(train_data)
                max_logit = output.remote_target.data.max(1, keepdim=True)[1]
                correct = max_logit.eq( targets.data.view_as(max_logit) ).sum()
                loss_item  = output.remote_target_loss.item()
                processed = ((batch_idx + 1) * self.config.neuron.batch_size_train)
                progress = (100. * processed) / n
                accuracy = (100.0 * correct) / self.config.neuron.batch_size_train
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLocal Loss: {:.6f}\t Accuracy: {:.6f}\t nP: {}', 
                    1, processed, n, progress, loss_item, accuracy, len(session.metagraph.neurons()))

        assert best_loss <= 0.1
        assert best_accuracy > 0.80
        
def main():
    # 1. Load Config.
    logger.info('Load Config ...')
    config = Config.load(neuron_path='bittensor/neurons/mnist')
    logger.info(Config.toString(config))

    # 2. Load Keypair.
    logger.info('Load Keyfile ...')
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
   
    # 3. Load Neuron.
    logger.info('Load Neuron ... ')
    neuron = Neuron( config )

    # 4. Load Session.
    logger.info('Build Session ... ')
    session = bittensor.init(config, keypair)

    # 5. Start Neuron.
    logger.info('Start ... ')
    with session:
        Asyncio.init()
        Asyncio.start_in_thread(neuron.start, session)
        Asyncio.run_forever()

    
if __name__ == "__main__":
    main()
    

