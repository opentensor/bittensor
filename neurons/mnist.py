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

from munch import Munch
from loguru import logger

import bittensor
from bittensor import Session
from bittensor.utils.logging import log_all
from bittensor.subtensor import Keypair
from bittensor.config import Config
from bittensor.synapse import Synapse
from bittensor.synapses.ffnn import FFNNSynapse

def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:    
    parser.add_argument('--neuron.datapath', default='data/', type=str,  help='Path to load and save data.')
    parser.add_argument('--neuron.learning_rate', default=0.01, type=float, help='Training initial learning rate.')
    parser.add_argument('--neuron.momentum', default=0.9, type=float, help='Training initial momentum for SGD.')
    parser.add_argument('--neuron.batch_size_train', default=64, type=int, help='Training batch size.')
    parser.add_argument('--neuron.batch_size_test', default=64, type=int, help='Testing batch size.')
    parser.add_argument('--neuron.log_interval', default=150, type=int, help='Batches until neuron prints log statements.')
    parser.add_argument('--neuron.sync_interval', default=150, type=int, help='Batches before we we sync with chain and emit new weights.')
    parser.add_argument('--neuron.accumulation_interval', default=1, type=int, help='Batches before we apply acummulated gradients.')
    parser.add_argument('--neuron.apply_remote_gradients', default=False, type=bool, help='If true, neuron applies gradients which accumulate from remotes calls.')
    parser.add_argument('--neuron.name', default='mnist', type=str, help='Trials for this neuron go in neuron.datapath / neuron.name')
    parser.add_argument('--neuron.trial_id', default=str(time.time()).split('.')[0], type=str, help='Saved models go in neuron.datapath / neuron.name / neuron.trial_id')
    parser = FFNNSynapse.add_args(parser)
    return parser

def check_config(config: Munch) -> Munch:
    assert config.neuron.log_interval > 0, "log_interval dimension must positive"
    assert config.neuron.momentum > 0 and config.neuron.momentum < 1, "momentum must be a value between 0 and 1"
    assert config.neuron.batch_size_train > 0, "batch_size_train must a positive value"
    assert config.neuron.batch_size_test > 0, "batch_size_test must a positive value"
    assert config.neuron.learning_rate > 0, "learning rate must be a positive value."
    data_directory = '{}/{}/{}'.format(config.neuron.datapath, config.neuron.name, config.neuron.trial_id)
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    config = FFNNSynapse.check_config(config)
    return config


# --- Train epoch ----
def train(
    epoch: int,
    model: Synapse,
    session: Session,
    config: Munch,
    optimizer: optim.Optimizer,
    trainloader: torch.utils.data.DataLoader):

    # ---- Init training state ----
    session.metagraph.sync() # Sync with the chain.
    row_weights = session.metagraph.W[0, :] # Weight to others
    col_weights = session.metagraph.W[:, 0] # Other to me.
    priority_map = dict(zip(session.metagraph.public_keys, col_weights.tolist()))
    session.axon.set_priority( priority_map )

    # ---- Do epoch ----
    model.train()
    history = []
    for batch_idx, (images, targets) in enumerate(trainloader):     
        # ---- Forward pass ----
        output = model(images.to(model.device), torch.LongTensor(targets).to(model.device), remote = True)
        history.append(output)

        # ---- Update Weights ----
        batch_weights = torch.mean(output.weights, axis = 0)
        row_weights = (1 - 0.05) * row_weights + 0.05 * batch_weights # Moving Avg weights.
        row_weights = F.normalize(row_weights, p = 1, dim = 0)

        # ---- Accumulate Local Gradients ----
        loss = output.loss / config.neuron.accumulation_interval # Need to average accross accumulation steps.
        loss.backward() # Accumulates gradients on model via sum.

        # ---- Accumulate Remote Gradients  ----
        if config.neuron.apply_remote_gradients:
            # TODO (const): batch normalization over the gradients for consistency.
            n_grads = session.axon.gradients.qsize
            for _, (_, input_x, grads_dy, modality) in list(session.axon.gradients.queue):
                grads_dy = grads_dy / (config.neuron.accumulation_interval * n_grads)
                model.backward(input_x, grads_dy, modality)

        # ---- Apply Gradient Batch ----
        if (batch_idx+1) % config.neuron.accumulation_interval == 0:
            optimizer.step() # Apply accumulated gradients.
            optimizer.zero_grad() # Zero grads for next accummulation 
            session.serve( model ) # Serve the newest model.

            # ---- Logs for gradient batch ----
            total_examples = len(trainloader) * config.neuron.batch_size_train
            processed = ((batch_idx + 1) * config.neuron.batch_size_train)
            progress = (100. * processed) / total_examples
            logger.info('Epoch: {} [{}/{} ({})]\t Loss: {}\t Acc: {}', 
                    colored('{}'.format(epoch), 'blue'), 
                    colored('{}'.format(processed), 'green'), 
                    colored('{}'.format(total_examples), 'red'),
                    colored('{:.2f}%'.format(progress), 'green'),
                    colored('{:.4f}'.format(output.local_target_loss.item()), 'green'),
                    colored('{:.4f}'.format(output.metadata['local_accuracy'].item()), 'green'))

        # ---- Sync State ----
        if (batch_idx+1) % config.neuron.sync_interval == 0:
            pass
            # ---- Emit weights and sync from chain ----
            session.metagraph.emit( row_weights ) # Set weights on chain.
            session.metagraph.sync() # Sync with the chain.
            
            # ---- Get row and col Weights ----
            row_weights = session.metagraph.W[0, :] # Weight to others.
            col_weights = session.metagraph.W[:, 0] # Other to me.

            # ---- Update Axon Priority ----
            priority_map = dict(zip(session.metagraph.public_keys, col_weights.tolist()))
            session.axon.set_priority( priority_map )

        # ---- Session Logs ----
        if (batch_idx+1) % config.neuron.log_interval == 0:
            log_all(session, history); history = [] # Log batch history.
    

# --- Test epoch ----
def test ( 
    model: Synapse,
    session: Session,
    testloader: torch.utils.data.DataLoader,
    num_tests: int):
    with torch.no_grad(): # Turns off gradient computation for inference speed up.

        model.eval() # Turns off Dropoutlayers, BatchNorm etc.
        loss = 0.0; correct = 0.0
        for _, (images, labels) in enumerate(testloader):                
            # ---- Forward pass ----
            outputs = model.forward(images.to(model.device), torch.LongTensor(labels).to(model.device), remote = False)
            loss = loss + outputs.loss
            
            # ---- Metric ----
            max_logit = outputs.local_target.data.max(1, keepdim=True)[1]
            correct = correct + max_logit.eq( labels.data.view_as(max_logit) ).sum()
    
    # --- Log results ----
    n = num_tests * config.neuron.batch_size_test
    loss /= n
    accuracy = (100. * correct) / n
    logger.info('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, num_tests, accuracy))  
    session.tbwriter.write_loss('test loss', loss)
    session.tbwriter.write_accuracy('test accuracy', accuracy)
    return loss, accuracy

def main(config: Munch, session: Session):

    # ---- Model ----
    model = FFNNSynapse(config, session)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to( device ) # Set model to device.
    session.serve( model.deepcopy() )

    # ---- Optimizer ---- 
    optimizer = optim.SGD(model.parameters(), lr=config.neuron.learning_rate, momentum=config.neuron.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10.0, gamma=0.1)

    # ---- Dataset ----
    train_data = torchvision.datasets.MNIST(root = config.neuron.datapath + "datasets/", train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = config.neuron.batch_size_train, shuffle=True, num_workers=2)
    test_data = torchvision.datasets.MNIST(root = config.neuron.datapath + "datasets/", train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(test_data, batch_size = config.neuron.batch_size_test, shuffle=False, num_workers=2)

    # ---- Train and Test ----
    epoch = -1
    best_test_loss = math.inf
    while True:
        epoch += 1

        # ---- Train model ----
        train( 
            epoch = epoch,
            model = model,
            session = session,
            config = config,
            optimizer = optimizer,
            trainloader = trainloader
        )
        scheduler.step()

        # ---- Test model ----
        test_loss, _ = test( 
            epoch = epoch,
            model = model,
            session = session,
            testloader = testloader,
            num_tests = len(test_data),
        )

        # ---- Save Best ----
        if test_loss < best_test_loss:
            best_test_loss = test_loss # Update best loss.
            logger.info( 'Saving/Serving model: epoch: {}, accuracy: {}, loss: {}, path: {}/{}/{}/model.torch'.format(epoch, test_accuracy, best_test_loss, config.neuron.datapath, config.neuron.name, config.neuron.trial_id))
            torch.save( {'epoch': epoch, 'model': model.state_dict(), 'loss': best_test_loss},"{}/{}/{}/model.torch".format(config.neuron.datapath , config.neuron.name, config.neuron.trial_id))
           

if __name__ == "__main__":
    # ---- Load config ----
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    config = Config.load(parser)
    config = check_config(config)
    logger.info(Config.toString(config))

    # ---- Load Keypair ----
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
   
    # ---- Build Session ----
    session = bittensor.init(config, keypair)

    # ---- Start Neuron ----
    with session:
        main(config, session)

