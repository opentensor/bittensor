"""Training a MNIST Neuron.
This file demonstrates a training pipeline for an MNIST Neuron.
Example:
        $ python neurons/mnist.py
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
from bittensor.utils.logging import log_all
from bittensor.config import Config
from bittensor.synapses.ffnn import FFNNSynapse

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

def check_config(config: Munch):
    assert config.session.log_interval > 0, "log_interval dimension must be positive"
    assert config.session.momentum > 0 and config.session.momentum < 1, "momentum must be a value between 0 and 1"
    assert config.session.batch_size_train > 0, "batch_size_train must be a positive value"
    assert config.session.batch_size_test > 0, "batch_size_test must be a positive value"
    assert config.session.learning_rate > 0, "learning rate must be be a positive value."
    full_path = '{}/{}/{}/'.format(config.session.root_dir, config.session.name, config.session.uid)
    config.session.full_path = full_path
    if not os.path.exists(config.session.full_path):
        os.makedirs(config.session.full_path)
    FFNNSynapse.check_config(config)
    Neuron.check_config(config)

# --- Train epoch ----
def main(config: Munch, neuron: Neuron):
    
    # ---- Model ----
    model = FFNNSynapse(config, neuron) # Feedforward neural network with PKMDendrite.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to( device ) # Set model to device
    
    # ---- Optimizer ---- 
    optimizer = optim.SGD(model.parameters(), lr=config.session.learning_rate, momentum=config.session.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10.0, gamma=0.1)

    # ---- Dataset ----
    train_data = torchvision.datasets.MNIST(root = config.session.root_dir + "datasets/", train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = config.session.batch_size_train, shuffle=True, num_workers=2)
    test_data = torchvision.datasets.MNIST(root = config.session.root_dir + "datasets/", train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(test_data, batch_size = config.session.batch_size_test, shuffle=False, num_workers=2)

    # ---- Tensorboard ----
    global_step = 0
    tensorboard = SummaryWriter(log_dir = config.session.full_path)

    # --- Test epoch ----
    def test ():

        model.eval() # Turns off Dropoutlayers, BatchNorm etc.
        loss = 0.0; accuracy = 0.0
        for _, (images, labels) in enumerate(testloader):

            # ---- Forward pass ----
            outputs = model.forward(
                images = images.to(model.device), 
                targets = torch.LongTensor(labels).to(model.device), 
                remote = False # *without* rpc-queries being made.
            )
            loss = loss + outputs.loss / len(test_data)
            accuracy = outputs.metadata['local_accuracy'] / len(test_data)
            
        return loss, accuracy

    # ---- Train epoch ----
    def train(epoch: int, global_step: int):

        # ---- Init training state ----
        model.train() # Turn on dropout etc.
        row_weights = neuron.metagraph.row_weights.to(device) # My weights on the chain-state (zeros initially).

        history = []
        for batch_idx, (images, targets) in enumerate(trainloader):    
            global_step += 1 

            # ---- Forward pass ----
            output = model(  
                images = images.to(model.device), 
                targets = torch.LongTensor(targets).to(model.device), 
                remote = True # *with* rpc-queries made to the network.
            ) 
            
            # ---- Backward pass ----
            output.loss.backward() # Accumulates gradients on the model.
            optimizer.step() # Applies accumulated gradients.
            optimizer.zero_grad() # Zeros out gradients for next accummulation 

            # ---- Serve Model ----
            neuron.serve( model.deepcopy() ) # Serve the newest model.

            # ---- Step Logs + Tensorboard ----
            history.append(output) # Save for later analysis/logs.
            processed = ((batch_idx + 1) * config.session.batch_size_train)
            progress = (100. * processed) / len(train_data)
            logger.info('GS: {}\t Epoch: {} [{}/{} ({})]\t Loss: {}\t Acc: {}\t', 
                    colored('{}'.format(global_step), 'blue'), 
                    colored('{}'.format(epoch), 'blue'), 
                    colored('{}'.format(processed), 'green'), 
                    colored('{}'.format(len(train_data)), 'red'),
                    colored('{:.2f}%'.format(progress), 'green'),
                    colored('{:.4f}'.format(output.local_target_loss.item()), 'green'),
                    colored('{:.4f}'.format(output.metadata['local_accuracy'].item()), 'green'))
            logger.info('Axon: {}', neuron.axon.__full_str__())
            logger.info('Dendrite: {}', neuron.dendrite.__full_str__())
            tensorboard.add_scalar('Rloss', output.remote_target_loss.item(), global_step)
            tensorboard.add_scalar('Lloss', output.local_target_loss.item(), global_step)
            tensorboard.add_scalar('Dloss', output.distillation_loss.item(), global_step)

            # ---- Update State ----
            batch_weights = torch.mean(output.weights, axis = 0) # Average over batch.
            row_weights = (1 - 0.03) * row_weights.to(device) + 0.03 * batch_weights # Moving avg update.
            row_weights = F.normalize(row_weights, p = 1, dim = 0) # Ensure normalization.

            if (batch_idx+1) % config.session.sync_interval == 0:
                # ---- Sync Metagraph State ----
                logger.info('Emitting with weights {}', row_weights.tolist())
                neuron.metagraph.emit( row_weights, wait_for_inclusion = True ) # Sets my row-weights on the chain.
                neuron.metagraph.sync() # Pulls the latest metagraph state (with my update.)
                row_weights = neuron.metagraph.row_weights
                
                # ---- Update Axon Priority ----
                col_weights = neuron.metagraph.col_weights # weights to me.
                neuron.axon.set_priority( neuron.metagraph.neurons, col_weights ) # Sets the nucleus-backend request priority.


    # ---- Loop forever ----
    epoch = -1; best_test_loss = math.inf
    while True:
        epoch += 1

        # ---- Train model ----
        train(epoch, global_step)
        scheduler.step()
        
        # ---- Test model ----
        with torch.no_grad(): # Turns off gradient computation for inference speed up.
            test_loss, test_accuracy = test()

         # ---- Save Best ----
        if test_loss < best_test_loss:
            best_test_loss = test_loss # Update best loss.
            logger.info( 'Saving/Serving model: epoch: {}, accuracy: {}, loss: {}, path: {}/model.torch'.format(epoch, test_accuracy, best_test_loss, config.session.full_path))
            torch.save( {'epoch': epoch, 'model': model.state_dict(), 'loss': best_test_loss},"{}/model.torch".format(config.session.full_path))
            tensorboard.add_scalar('Test loss', test_loss, global_step)

        
if __name__ == "__main__":
    # ---- Load command line args ----
    parser = argparse.ArgumentParser(); add_args(parser) 
    config = Config.to_config(parser); check_config(config)
    logger.info(Config.toString(config))
   
    # ---- Build Neuron ----
    neuron = Neuron(config)

    # ---- Start Neuron ----
    with neuron:
        main(config, neuron)

