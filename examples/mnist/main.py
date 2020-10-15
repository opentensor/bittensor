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
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    epoch = 0
    global_step = 0
    trial_uid = 'mnist-' + str(time.time()).split('.')[0]
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
    bittensor.serve( model )

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
        correct = 0.0
        total_epoch = 0
        for batch_idx, (images, labels) in enumerate(trainloader):
            
            # Clear gradients on model parameters.
            optimizer.zero_grad()

            # Targets and images to correct device.
            labels = torch.LongTensor(labels).to(device)
            images = images.to(device)
            
            # Computes model outputs and loss.
            output = model(images, labels, query = False)

            # Loss and step.
            max_logit = output['local_target'].data.max(1, keepdim=True)[1]
            correct += max_logit.eq( labels.data.view_as(max_logit) ).sum()
            loss = output['loss']
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            global_step += 1
            total_epoch += batch_size_train
                            
            # Logs:
            if batch_idx % log_interval == 0:
                n_peers = len(bittensor.metagraph.peers())
                n_synapses = len(bittensor.metagraph.synapses())
                writer.add_scalar('n_peers', n_peers, global_step)
                writer.add_scalar('n_synapses', n_synapses, global_step)
                writer.add_scalar('train_loss', float(loss.item()), global_step)
            
                n = len(train_data)
                accuracy = (100.0 * correct) / total_epoch
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLocal Loss: {:.6f}\t Accuracy: {:.6f}\tnP|nS: {}|{}'.format(
                    epoch, (batch_idx * batch_size_train), n, (100. * batch_idx * batch_size_train)/n, output['local_target_loss'].item(), accuracy, len(bittensor.metagraph.peers()), 
                            len(bittensor.metagraph.synapses())))
                # logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLocal Loss: {:.6f}\nNetwork Loss: {:.6f}\tDistillation Loss: {:.6f}\tnP|nS: {}|{}'.format(
                #     epoch, (batch_idx * batch_size_train), n, (100. * batch_idx * batch_size_train)/n, output['local_target_loss'].item(), output['network_target_loss'].item(), output['distillation_loss'].item(), len(bittensor.metagraph.peers()), 
                #             len(bittensor.metagraph.synapses())))

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
                max_logit = outputs['local_target'].data.max(1, keepdim=True)[1]
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
                
                # Save the best local model.
                logger.info('Serving / Saving model: epoch: {}, loss: {}, path: {}', epoch, test_loss, config.datapath + trial_uid + '/model.torch')
                torch.save({ 'epoch': epoch, 'model': model.state_dict(), 'test_loss': test_loss}, config.datapath + trial_uid + '/model.torch')
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
