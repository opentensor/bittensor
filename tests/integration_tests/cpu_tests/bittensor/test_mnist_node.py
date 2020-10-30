"""Training a MNIST Neuron.

This file demonstrates a training pipeline for an MNIST Neuron.

Example:
        $ python examples/mnist/main.py
"""

import bittensor
from bittensor.synapses.ffnn import FFNNSynapse, FFNNConfig

import random
from loguru import logger
import math
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import traceback
import unittest

class MnistNode(unittest.TestCase):
    log_interval = 10
    epoch = 0
    total_epochs = 5
    global_step = 0
    best_test_loss = math.inf
    best_accuracy = 0
    global_step = 0
    scheduler = None
    trainloader = None
    train_data = None
    testloader = None
    test_data = None
    optimizer = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    # Additional training params.
    batch_size_train = 64
    batch_size_test = 64
    learning_rate = 0.05
    momentum = 0.9
    
    # Evaluation parameters
    accuracy = 0
    loss = 0

    def setUp(self):
        # Setup Bittensor.
        # Create background objects.
        # Connect the metagraph.
        # Start the axon server.
        config = bittensor.Config()
        config.bootstrap = '134.122.37.200:8121'
        config.metagraph_port = str(random.randint(8000, 9000))
        config.axon_port = str(random.randint(8000, 9000))
        logger.info(config)
        bittensor.init( config )
        bittensor.start()
        # Build local synapse to serve on the network.
        model_config = FFNNConfig()
        self.model = FFNNSynapse(model_config)
        self.model.to( self.device ) # Set model to device.
        bittensor.serve( self.model.deepcopy() )

        # Build the optimizer.
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10.0, gamma=0.1)

        # Load (Train, Test) datasets into memory.
        self.train_data = torchvision.datasets.MNIST(root = config.datapath + "datasets/", train=True, download=True, transform=transforms.ToTensor())
        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size = self.batch_size_train, shuffle=True, num_workers=2)
        self.test_data = torchvision.datasets.MNIST(root = config.datapath + "datasets/", train=False, download=True, transform=transforms.ToTensor())
        self.testloader = torch.utils.data.DataLoader(self.test_data, batch_size = self.batch_size_test, shuffle=False, num_workers=2)
    
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
            self.global_step += 1

            # Logs:
            if (batch_idx + 1) % self.log_interval == 0: 
                n = len(self.train_data)
                max_logit = output.remote_target.data.max(1, keepdim=True)[1]
                correct = max_logit.eq( targets.data.view_as(max_logit) ).sum()
                loss_item  = output.remote_target_loss.item()
                processed = ((batch_idx + 1) * self.batch_size_train)
                
                progress = (100. * processed) / n
                accuracy = (100.0 * correct) / self.batch_size_train
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLocal Loss: {:.6f}\t Accuracy: {:.6f}\t nP: {}', 
                    epoch, processed, n, progress, loss_item, accuracy, len(bittensor.metagraph.peers()))

    # Test loop.
    # Evaluates the local model on the hold-out set.
    # Returns the test_accuracy and test_loss.
    def validate(self, model: bittensor.Synapse):
        
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
                outputs = model.forward(images, labels, remote = False)
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
    
    def testModelTraining(self):
        while self.epoch < self.total_epochs:
            try:
                # Train model
                logger.info("Training epoch {}...".format(self.epoch))
                self.train( self.model, self.epoch )
                self.scheduler.step()
                # Test model.
                test_loss, accuracy = self.validate( self.model )
            
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                # Save best model. 
                if test_loss < self.best_test_loss:
                    # Update best loss.
                    self.best_test_loss = test_loss
                    logger.info("Best test loss: {}".format(self.best_test_loss))
                    # Save and serve the new best local model.
                    #logger.info( 'Saving/Serving model: epoch: {}, loss: {}, path: {}', epoch, test_loss, config.logdir + '/model.torch' )
                    #torch.save( {'epoch': self.epoch, 'model': model.state_dict(), 'test_loss': test_loss}, config.logdir + '/model.torch' )
                    bittensor.serve( self.model.deepcopy() )
                 
                self.epoch += 1
                
            except Exception as e:
                traceback.print_exc()
                logger.error(e)
                bittensor.stop()
                break
        
        assert self.best_test_loss <= 0.1
        assert self.best_accuracy >= 90 
        
