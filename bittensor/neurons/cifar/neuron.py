"""Training a CIFAR Neuron.

This file demonstrates a training pipeline for an CIFAR Neuron.

Example:
        $ python examples/cifar/main.py
"""

import bittensor
from bittensor import BTSession
from bittensor.config import Config
from bittensor.synapse import Synapse
from bittensor.synapses.dpn import DPNSynapse
from bittensor.neuron import NeuronBase

import argparse
from loguru import logger
import math
from munch import Munch
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class Neuron (NeuronBase):
    def __init__(self, config):
        self.config = config

    @staticmethod
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
        parser = DPNSynapse.add_args(parser)
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        assert config.neuron.log_interval > 0, "log_interval dimension must positive"
        assert config.neuron.momentum > 0 and config.neuron.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.neuron.batch_size_train > 0, "batch_size_train must a positive value"
        assert config.neuron.batch_size_test > 0, "batch_size_test must a positive value"
        assert config.neuron.learning_rate > 0, "learning rate must be a positive value."
        Config.validate_path_create('neuron.datapath', config.neuron.datapath)
        config = DPNSynapse.check_config(config)
        return config

    def start(self, session: BTSession): 
        
        # Build local synapse to serve on the network.
        model = DPNSynapse( self.config, session )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to( device ) # Set model to device.
        session.serve( model.deepcopy() )

        # Build the optimizer.
        optimizer = optim.SGD(model.parameters(), lr=self.config.neuron.learning_rate, momentum=self.config.neuron.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10.0, gamma=0.1)

        # Load (Train, Test) datasets into memory.
        train_data = torchvision.datasets.CIFAR10(
            root=self.config.neuron.datapath, train=True, download=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]))
        trainloader = torch.utils.data.DataLoader(train_data, batch_size = self.config.neuron.batch_size_train, shuffle=True, num_workers=2)
        test_data = torchvision.datasets.CIFAR10(root=self.config.neuron.datapath, train=False, download=True, transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(test_data, batch_size = self.config.neuron.batch_size_test, shuffle=False, num_workers=2)

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
                output = model(images, targets, remote = True)

                # Backprop.
                output.loss.backward()
                optimizer.step()
                global_step += 1
                                
                # Logs:
                if batch_idx % self.config.neuron.log_interval == 0:            
                    n = len(train_data)
                    max_logit = output.remote_target.data.max(1, keepdim=True)[1]
                    correct = max_logit.eq(targets.data.view_as(max_logit) ).sum()
                    loss_item  = output.remote_target_loss.item()
                    processed = ((batch_idx + 1) * self.config.neuron.batch_size_train)
                    
                    progress = (100. * processed) / n
                    accuracy = (100.0 * correct) / self.config.neuron.batch_size_train
                    logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLocal Loss: {:.6f}\t Accuracy: {:.6f}\tnP:{}', 
                        epoch, processed, n, progress, loss_item, accuracy, len(session.metagraph.neurons()))

        # Test loop.
        # Evaluates the local model on the hold-out set.
        # Returns the test_accuracy and test_loss.
        def test( model: Synapse ):
            
            # Turns off Dropoutlayers, BatchNorm etc.
            model.eval()
            
            # Turns off gradient computation for inference speed up.
            with torch.no_grad():
            
                loss = 0.0
                correct = 0.0
                for _, (images, labels) in enumerate(testloader):                
                
                    images = images.to(device)

                    # Labels to Tensor
                    labels = torch.LongTensor(labels).to(device)
                    
                    # Compute full pass and get loss.
                    outputs = model (images, labels, remote = False)
                    loss = loss + outputs.loss
                    
                    # Count accurate predictions.
                    max_logit = outputs.local_target.data.max(1, keepdim=True)[1]
                    correct += max_logit.eq( labels.data.view_as(max_logit) ).sum()
            
            # # Log results.
            n = len(test_data)
            loss /= n
            accuracy = (100. * correct) / n
            logger.info('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\tnP:{}'.format(loss, correct, n, accuracy, len(bittensor.metagraph.neurons())))        
            return loss, accuracy
        
        
        epoch = 0
        global_step = 0
        best_test_loss = math.inf
        while True:
            # Train model
            train( model, epoch, global_step )
            scheduler.step()
            # Test model.
            test_loss, _ = test( model )
        
            # Save best model. 
            if test_loss < best_test_loss:
                # Update best loss.
                best_test_loss = test_loss
                
                # Save the best local model.
                logger.info('Serving / Saving model: epoch: {}, loss: {}, path: {}', epoch, test_loss, self.config.logger.logdir + '/model.torch')
                torch.save( {'epoch': epoch, 'model': model.state_dict(), 'test_loss': test_loss}, self.config.logger.logdir + '/model.torch' )
                session.serve( model.deepcopy() )

            epoch += 1
