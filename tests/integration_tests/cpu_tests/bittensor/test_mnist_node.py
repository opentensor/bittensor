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

import numpy as np
from loguru import logger
import math
from termcolor import colored
import torch
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import time


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
        start_time = time.time()
        for batch_idx, (images, targets) in enumerate(trainloader):
            # Clear gradients.
            optimizer.zero_grad()

            # Emit and sync.
            if (session.metagraph.block() - session.metagraph.state.block) > 5:
                session.metagraph.emit()
                session.metagraph.sync()

            # Forward pass.
            images = images.to(device)
            targets = torch.LongTensor(targets).to(device)
            output = model(images, targets, remote = True)

            # Backprop.
            loss = output.remote_target_loss + output.distillation_loss
            loss.backward()
            optimizer.step()

            # Update weights.
            state_weights = session.metagraph.state.weights
            learned_weights = F.softmax(torch.mean(output.weights, axis=0))
            state_weights = (1 - 0.05) * state_weights + 0.05 * learned_weights
            norm_state_weights = F.softmax(state_weights)
            session.metagraph.state.set_weights( norm_state_weights )

            # Metrics.
            max_logit = output.remote_target.data.max(1, keepdim=True)[1]
            correct = max_logit.eq( targets.data.view_as(max_logit) ).sum()
            target_loss  = output.remote_target_loss.item()
            accuracy = (100.0 * correct) / self.config.neuron.batch_size_train

            # Update best vars.
            best_accuracy = accuracy if accuracy >= best_accuracy else best_accuracy
            if target_loss < best_loss:
                best_loss = target_loss
                session.serve( model.deepcopy() )

            # Logs:
            if (batch_idx + 1) % self.config.neuron.log_interval == 0:
                n = len(train_data)
                max_logit = output.remote_target.data.max(1, keepdim=True)[1]
                correct = max_logit.eq( targets.data.view_as(max_logit) ).sum()
                n_str = colored('{}'.format(n), 'red')

                loss_item = output.remote_target_loss.item()
                loss_item_str = colored('{:.3f}'.format(loss_item), 'green')

                processed = ((batch_idx + 1) * self.config.neuron.batch_size_train)
                processed_str = colored('{}'.format(processed), 'green')

                progress = (100. * processed) / n
                progress_str = colored('{:.2f}%'.format(progress), 'green')

                accuracy = (100.0 * correct) / self.config.neuron.batch_size_train
                accuracy_str = colored('{:.3f}'.format(accuracy), 'green')

                nN = session.metagraph.state.n
                nN_str = colored('{}'.format(nN), 'red')

                # nA = len(torch.unique(torch.flatten(output.keys))) 
                # nA_str = colored('{}'.format(nA), 'green')

                logger.info('Epoch: {} [{}/{} ({})] | Loss: {} | Acc: {} | Act/Tot: {}/{}', 
                    1, processed_str, n_str, progress_str, loss_item_str, accuracy_str, 1, nN_str)

                np.set_printoptions(precision=2, suppress=True, linewidth=500, sign=' ')
                numpy_uids = np.array(session.metagraph.state.uids.tolist())
                numpy_weights = np.array(session.metagraph.state.weights.tolist())
                numpy_stack = np.stack((numpy_uids, numpy_weights), axis=0)
                stack_str = colored(numpy_stack, 'green')
                logger.info('Weights: \n {}', stack_str)

        # Test checks.
        time_elapsed = time.time() - start_time
        logger.info("Total time elapsed: {}".format(time_elapsed))
        assert best_loss <= 0.1
        assert best_accuracy > 0.80
        assert len(session.metagraph.state.neurons()) > 0
        assert time_elapsed < 300 # 1 epoch of MNIST should take less than 5 mins.
        
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
        neuron.start(session)

    
if __name__ == "__main__":
    main()
    

