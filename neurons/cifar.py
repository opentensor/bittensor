"""Training a CIFAR Neuron.

This file demonstrates a training pipeline for an CIFAR Neuron.

Example:
        $ python examples/cifar.py
"""

import argparse
import math
import time
import torch
from termcolor import colored
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os

from munch import Munch
from loguru import logger

import bittensor
from bittensor import Session
from bittensor.utils.logging import (log_outputs, log_batch_weights, log_chain_weights, log_request_sizes)
from bittensor.subtensor.interface import Keypair
from bittensor.config import Config
from bittensor.synapse import Synapse
from bittensor.synapses.dpn import DPNSynapse

def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:    
    parser.add_argument('--neuron.datapath', default='data/', type=str, 
                        help='Path to load and save data.')
    parser.add_argument('--neuron.learning_rate', default=0.01, type=float, 
                        help='Training initial learning rate.')
    parser.add_argument('--neuron.momentum', default=0.9, type=float, 
                        help='Training initial momentum for SGD.')
    parser.add_argument('--neuron.batch_size_train', default=32, type=int, 
                        help='Training batch size.')
    parser.add_argument('--neuron.batch_size_test', default=16, type=int, 
                        help='Testing batch size.')
    parser.add_argument('--neuron.log_interval', default=10, type=int, 
                        help='Batches until neuron prints log statements.')
    parser.add_argument('--neuron.sync_interval', default=100, type=int, 
                        help='Batches before we we sync with chain and emit new weights.')
    parser.add_argument('--neuron.name', default='cifar', type=str, help='Trials for this neuron go in neuron.datapath / neuron.name')
    parser.add_argument('--neuron.trial_id', default=str(time.time()).split('.')[0], type=str, help='Saved models go in neuron.datapath / neuron.name / neuron.trial_id')
    parser = DPNSynapse.add_args(parser)
    return parser

def check_config(config: Munch) -> Munch:
    assert config.neuron.log_interval > 0, "log_interval dimension must positive"
    assert config.neuron.momentum > 0 and config.neuron.momentum < 1, "momentum must be a value between 0 and 1"
    assert config.neuron.batch_size_train > 0, "batch_size_train must a positive value"
    assert config.neuron.batch_size_test > 0, "batch_size_test must a positive value"
    assert config.neuron.learning_rate > 0, "learning rate must be a positive value."
    if config.neuron.name == None:
        config.neuron.name = str(time.time())
    config = DPNSynapse.check_config(config)
    return config

def train(
    epoch: int,
    model: Synapse,
    session: Session,
    config: Munch,
    optimizer: optim.Optimizer,
    trainloader: torch.utils.data.DataLoader):

    model.train() # Turn on Dropoutlayers BatchNorm etc.
    weights = None
    history = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch_idx, (images, targets) in enumerate(trainloader):
        optimizer.zero_grad() # Clear gradients.

        # Sync with chain.
        if batch_idx % config.neuron.sync_interval == 0:
            weights = session.metagraph.sync(weights)
            weights = weights.to(device)
                        
        # Forward pass.
        output = model(images.to(model.device), torch.LongTensor(targets).to(model.device), remote = True)
        history.append(output)

        # Backprop.
        loss = output.remote_target_loss + output.distillation_loss
        loss.backward()
        optimizer.step()

        # Update weights.
        batch_weights = F.softmax(torch.mean(output.weights, axis=0), dim=0) # Softmax weights.
        weights = (1 - 0.05) * weights + 0.05 * batch_weights # Moving Avg
        weights = weights / torch.sum(weights) # Normalize.

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

        batch_idx += 1

def test ( 
    model: Synapse,
    session: Session,
    testloader: torch.utils.data.DataLoader,
    num_tests: int):

    model.eval() # Turns off Dropoutlayers, BatchNorm etc.
    with torch.no_grad(): # Turns off gradient computation for inference speed up.
        loss = 0.0
        correct = 0.0
        for _, (images, labels) in enumerate(testloader):                
            # Compute full pass and get loss.
            outputs = model.forward(images.to(model.device), torch.LongTensor(labels).to(model.device), remote = False)
            loss = loss + outputs.loss
            
            # Count accurate predictions.
            max_logit = outputs.local_target.data.max(1, keepdim=True)[1]
            correct = correct + max_logit.eq( labels.data.view_as(max_logit) ).sum()
    
    # Log results.
    n = num_tests * config.neuron.batch_size_test
    loss /= n
    accuracy = (100. * correct) / n
    logger.info('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, n, accuracy))  
    session.tbwriter.write_loss('test loss', loss)
    session.tbwriter.write_accuracy('test accuracy', accuracy)
    return loss, accuracy

def main(config: Munch, session: Session): 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build local synapse to serve on the network.
    model = DPNSynapse( config, session )
    model.to( device ) # Set model to device.
    session.serve( model.deepcopy() )

    # Build the optimizer.
    optimizer = optim.SGD(model.parameters(), lr=config.neuron.learning_rate, momentum=config.neuron.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10.0, gamma=0.1)

    # Load (Train, Test) datasets into memory.
    train_data = torchvision.datasets.CIFAR10(
        root=config.neuron.datapath, train=True, download=True, transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]))
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = config.neuron.batch_size_train, shuffle=True, num_workers=2)
    test_data = torchvision.datasets.CIFAR10(root=config.neuron.datapath, train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(test_data, batch_size = config.neuron.batch_size_test, shuffle=False, num_workers=2)
    test_data = torchvision.datasets.MNIST(root = config.neuron.datapath + "datasets/", train=False, download=True, transform=transforms.ToTensor())

    # Create directory if it does not exist.
    data_directory = '{}/{}/{}'.format(config.neuron.datapath, config.neuron.name, config.neuron.trial_id)
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)

    epoch = 0
    best_test_loss = math.inf
    while True:
        # Train model
        train( 
            epoch = epoch,
            model = model,
            session = session,
            config = config,
            optimizer = optimizer,
            trainloader = trainloader
        )
        scheduler.step()

        # Test model.
        test_loss, test_accuracy = test( 
            model = model,
            session = session,
            testloader = testloader,
            num_tests = len(test_data)
        )

        # Save best model. 
        if test_loss < best_test_loss:
            best_test_loss = test_loss # Update best loss.
            # Save and serve the new best local model.
            logger.info( 'Saving/Serving model: epoch: {}, accuracy: {}, loss: {}, path: {}/{}/{}/model.torch', epoch, test_accuracy, best_test_loss, config.neuron.datapath, config.neuron.name, config.neuron.trial_id)
            torch.save( {'epoch': epoch, 'model': model.state_dict(), 'loss': best_test_loss},"{}/{}/{}/model.torch".format(config.neuron.datapath , config.neuron.name, config.neuron.trial_id))
            # Save experiment metrics
            session.serve( model.deepcopy() )

        epoch += 1

if __name__ == "__main__":
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
        main(config, session)


