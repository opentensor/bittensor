"""Training a MNIST Neuron.

This file demonstrates a training pipeline for an MNIST Neuron.

Example:
        $ python examples/mnist/main.py
"""

import bittensor
from bittensor.synapses.mnist.model import MnistSynapse
from bittensor.utils.model_utils import ModelToolbox

import argparse
import copy
from loguru import logger
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import traceback
from typing import List, Tuple, Dict, Optional

def main(hparams):
     
    # Additional training params.
    batch_size_train = 64
    batch_size_test = 64
    learning_rate = 0.05
    momentum = 0.9
    log_interval = 10
    epoch = 0
    global_step = 0
    trial_uid = "mnist-{}".format(str(time.time()).split('.')[0])
    best_test_loss = math.inf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # Setup Bittensor.
    # Create background objects.
    # Connect the metagraph.
    # Start the axon server.
    config = bittensor.Config.from_hparams( hparams )
    logger.info(config)
    bittensor.init( config )
    bittensor.start()
    
    # Build local synapse to serve on the network.
    model = MnistSynapse()
    model.to( device ) # Set model to device.
    bittensor.serve( model.deepcopy() )

    # Build the optimizer.
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        
    # Load (Train, Test) datasets into memory.
    train_data = torchvision.datasets.MNIST(root = config.datapath + "datasets/", train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size_train, shuffle=True, num_workers=2)
    test_data = torchvision.datasets.MNIST(root = config.datapath + "datasets/", train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(test_data, batch_size = batch_size_test, shuffle=False, num_workers=2)
    
    # Build summary writer for tensorboard.
    writer = SummaryWriter(log_dir = config.datapath + trial_uid + "/logs/")

    # Train loop: Single threaded training of MNIST.
    def train(model, epoch, global_step):
        # Turn on Dropoutlayers BatchNorm etc.
        model.train()
        for batch_idx, (images, targets) in enumerate(trainloader):
            # Clear gradients.
            optimizer.zero_grad()

            # Forward pass.
            images = images.to(device)
            targets = torch.LongTensor(targets).to(device)
            output = model(images, targets, query = True)

            # Backprop.
            loss = output['network_target_loss'] + output['distillation_loss']
            loss.backward()
            optimizer.step()
            global_step += 1
                            
            # Logs:
            if batch_idx % log_interval == 0:            
                n = len(train_data)
                max_logit = output['network_target'].data.max(1, keepdim=True)[1]
                correct = max_logit.eq( targets.data.view_as(max_logit) ).sum()
                loss_item  = output['network_target_loss'].item()
                processed = ((batch_idx + 1) * batch_size_train)
                progress = (100. * processed) / n
                accuracy = (100.0 * correct) / batch_size_train
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLocal Loss: {:.6f}\t Accuracy: {:.6f}', 
                    epoch, processed, n, progress, loss_item, accuracy)

    # Test loop.
    # Evaluates the local model on the hold-out set.
    # Returns the test_accuracy and test_loss.
    def test( model: bittensor.Synapse ):
        
        # Turns off Dropoutlayers, BatchNorm etc.
        model.eval()
        
        # Turns off gradient computation for inference speed up.
        with torch.no_grad():
        
            loss = 0.0
            correct = 0.0
            for _, (images, labels) in enumerate(testloader):                
               
                # Labels to Tensor
                labels = torch.LongTensor(labels).to(device)

                # Compute full pass and get loss.
                outputs = model.forward(images, labels, query=False)
                            
                # Count accurate predictions.
                max_logit = outputs['student_target'].data.max(1, keepdim=True)[1]
                correct += max_logit.eq( labels.data.view_as(max_logit) ).sum()
        
        # # Log results.
        n = len(test_data)
        loss /= n
        accuracy = (100. * correct) / n
        logger.info('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, n, accuracy))        
        return loss, accuracy
    
    while True:
        try:
            # Train model
            train( model, epoch, global_step )
            
            # Test model.
            test_loss, _ = test( model )
        
            # Save best model. 
            if test_loss < best_test_loss:
                # Update best loss.
                best_test_loss = test_loss
                
                # Save and serve the new best local model.
                logger.info('Saving/Serving model: epoch: {}, loss: {}, path: {}', epoch, test_loss, config.datapath + trial_uid + '/model.torch')
                torch.save({ 'epoch': epoch, 'model': model.state_dict(), 'test_loss': test_loss}, config.datapath + trial_uid + '/model.torch')
                bittensor.serve( model.deepcopy()  )

            epoch += 1

        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            bittensor.stop()
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    hparams = bittensor.Config.add_args(parser)
    hparams = parser.parse_args()
    main(hparams)
