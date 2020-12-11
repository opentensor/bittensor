"""Training a MNIST Neuron.

This file demonstrates a training pipeline for an MNIST Neuron.

Example:
        $ python examples/mnist/main.py
"""
import bittensor
from bittensor.config import Config
from bittensor.synapses.ffnn import FFNNSynapse
from bittensor.subtensor import Keypair
from bittensor.utils.logging import (log_outputs, log_batch_weights, log_chain_weights, log_request_sizes)

import argparse
import numpy as np
from loguru import logger
from munch import Munch
import math
from termcolor import colored
import torch
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import time
import sys

def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:    
    parser.add_argument('--neuron.datapath', default='data/', type=str, 
                        help='Path to load and save data.')
    parser.add_argument('--neuron.learning_rate', default=0.01, type=float, 
                        help='Training initial learning rate.')
    parser.add_argument('--neuron.momentum', default=0.9, type=float, 
                        help='Training initial momentum for SGD.')
    parser.add_argument('--neuron.batch_size_train', default=64, type=int, 
                        help='Training batch size.')
    parser.add_argument('--neuron.batch_size_test', default=64, type=int, 
                        help='Testing batch size.')
    parser.add_argument('--neuron.log_interval', default=10, type=int, 
                        help='Batches until neuron prints log statements.')
    parser.add_argument('--neuron.sync_interval', default=100, type=int,
                        help='How often to sync with chain')
    # Load args from FFNNSynapse.
    parser = FFNNSynapse.add_args(parser)
    return parser

def check_config(config: Munch) -> Munch:
    assert config.neuron.log_interval > 0, "log_interval dimension must positive"
    assert config.neuron.momentum > 0 and config.neuron.momentum < 1, "momentum must be a value between 0 and 1"
    assert config.neuron.batch_size_train > 0, "batch_size_train must a positive value"
    assert config.neuron.batch_size_test > 0, "batch_size_test must a positive value"
    assert config.neuron.learning_rate > 0, "learning rate must be a positive value."
    Config.validate_path_create('neuron.datapath', config.neuron.datapath)
    config = FFNNSynapse.check_config(config)
    return config

def start(config, session):
    # Build and server the synapse.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFNNSynapse(config, session)
    model.to( device ) # Set model to device.
    session.serve( model.deepcopy() )

    # Build the optimizer.
    optimizer = optim.SGD(model.parameters(), lr=config.neuron.learning_rate, momentum=config.neuron.momentum)

    # Load (Train, Test) datasets into memory.
    train_data = torchvision.datasets.MNIST(root = config.neuron.datapath + "datasets/", train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = config.neuron.batch_size_train, shuffle=True, num_workers=2)
    
    # Train 1 epoch.
    model.train()
    best_loss = math.inf
    best_accuracy = 0
    start_time = time.time()
    weights = None
    epoch = 0
    history = []
    for batch_idx, (images, targets) in enumerate(trainloader):
        # Clear gradients.
        optimizer.zero_grad()

        # Syncs chain state and emits learned weights to the chain.
        if batch_idx % config.neuron.sync_interval == 0:
            weights = session.metagraph.sync(weights)
        
        # Forward pass.
        images = images.to(device)
        targets = torch.LongTensor(targets).to(device)
        output = model(images, targets, remote = True)
        history.append(output)
        
        # Backprop.
        loss = output.remote_target_loss + output.distillation_loss
        loss.backward()
        optimizer.step()

        # Update weights.
        batch_weights = F.softmax(torch.mean(output.weights, axis=0), dim=0)
        weights = (1 - 0.05) * weights + 0.05 * batch_weights
        weights = weights / torch.sum(weights)

        # Metrics.
        max_logit = output.remote_target.data.max(1, keepdim=True)[1]
        correct = max_logit.eq( targets.data.view_as(max_logit) ).sum()
        target_loss  = output.remote_target_loss.item()
        accuracy = (100.0 * correct) / config.neuron.batch_size_train

        # Update best vars.
        best_accuracy = accuracy if accuracy >= best_accuracy else best_accuracy
        if target_loss < best_loss:
            best_loss = target_loss
            session.serve( model.deepcopy() )

        # Logs: 
        if (batch_idx + 1) % config.neuron.log_interval == 0: 
            total_examples = len(trainloader) * config.neuron.batch_size_train
            processed = ((batch_idx + 1) * config.neuron.batch_size_train)
            progress = (100. * processed) / total_examples
            logger.info('Epoch: {} [{}/{} ({})]', 
                        colored('{}'.format(epoch), 'blue'), 
                        colored('{}'.format(processed), 'green'), 
                        colored('{}'.format(total_examples), 'red'),
                        colored('{:.2f}%'.format(progress), 'green'))
            log_outputs(history)
            log_batch_weights(session, history)
            log_chain_weights(session)
            log_request_sizes(session, history)
            history = []
        
        epoch += 1

    # Test checks.
    time_elapsed = time.time() - start_time
    logger.info("Total time elapsed: {}".format(time_elapsed))
    assert best_loss <= 0.1
    assert best_accuracy > 0.80
    assert len(session.metagraph.state.neurons()) > 0
    assert time_elapsed < 300 # 1 epoch of MNIST should take less than 5 mins.
        
def main():

    # 1. Load bittensor config.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    config = Config.load(parser)
    config = check_config(config)    
    logger.info(Config.toString(config))

    # 2. Load Keypair.
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
   
    # 3. Load Session.
    session = bittensor.init(config, keypair)

    # 4. Start Neuron.
    with session:
        start(config, session)
    
if __name__ == "__main__":
    main()
    

