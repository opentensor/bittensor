"""Training a MNIST Neuron.
This file demonstrates a training pipeline for an MNIST Neuron.
Example:
        $ python examples/mnist.py
"""

import math
import bittensor
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse

from munch import Munch
from loguru import logger
from termcolor import colored
from bittensor.synapse import Synapse
from bittensor.miner import Miner
from bittensor.synapses.ffnn import FFNNSynapse

class Session(Miner):

    def __init__(self, model_type: Synapse, config: Munch = None):

        super(Session, self).__init__(model_type = model_type, config=config)

        # ---- Optimizer ---- 
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.miner.learning_rate, momentum=self.config.miner.momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10.0, gamma=0.1)

        # ---- Dataset ----
        self.train_data = torchvision.datasets.MNIST(root = self.config.miner.root_dir + "datasets/", train=True, download=True, transform=transforms.ToTensor())
        self.trainloader = torch.utils.data.DataLoader(self.train_data, batch_size = self.config.miner.batch_size_train, shuffle=True, num_workers=2)
        self.test_data = torchvision.datasets.MNIST(root = self.config.miner.root_dir + "datasets/", train=False, download=True, transform=transforms.ToTensor())
        self.testloader = torch.utils.data.DataLoader(self.test_data, batch_size = self.config.miner.batch_size_test, shuffle=False, num_workers=2)

        # ---- Tensorboard ----
        self.global_step = 0
       
    # --- Main loop ----
    def run(self):

        # ---- Subscribe neuron ---- 
        with self.neuron:

            # ---- Weights ----
            self.update_row_weights()

            # ---- Loop forever ----
            self.epoch = -1; 
            self.best_test_loss = math.inf; 
            self.global_step = 0
            
            for self.epoch in range(self.config.miner.n_epochs):
                self.epoch += 1

                # ---- Serve ----
                self.neuron.axon.serve( self.model )

                # ---- Train ----
                self.train()
                self.scheduler.step()
                                    
                # If model has borked for some reason, we need to make sure it doesn't emit weights
                # Instead, reload into previous version of model
                if torch.any(torch.isnan(torch.cat([param.view(-1) for param in self.model.parameters()]))):
                    self.model, self.optimizer = self.model_toolbox.load_model(self.config)
                    continue

                # ---- Test ----
                test_loss, test_accuracy = self.test()

                # ---- Emit and sync metagraph ----
                self.set_metagraph_weights_and_sync()

                # --- Display Epoch ----
                self.display_epoch()

                # ---- Update Tensorboard ----
                self.update_tensorboard()

                # ---- Save ----
                if test_loss < self.best_test_loss:
                    self.best_test_loss = test_loss # Update best loss.
                    self.save_model(
                        {
                            'epoch': self.epoch, 
                            'model_state_dict': self.model.state_dict(), 
                            'loss': self.best_test_loss,
                            'optimizer_state_dict': self.optimizer.state_dict(),
                        }
                    )
                    self.tensorboard.add_scalar('Test loss', test_loss, self.global_step)

    # ---- Train epoch ----
    def train(self):
        # ---- Init training state ----
        self.model.train() # Turn on dropout etc.
        
        for batch_idx, (images, targets) in enumerate(self.trainloader):    
            self.global_step += 1 

            # ---- Remote Forward pass ----
            output = self.model.remote_forward(  
                neuron = self.neuron,
                images = images.to(self.device), 
                targets = torch.LongTensor(targets).to(self.device), 
            ) 
            
            # ---- Remote Backward pass ----
            loss = output.remote_target_loss + output.local_target_loss + output.distillation_loss
            loss.backward() # Accumulates gradients on the model.
            self.optimizer.step() # Applies accumulated gradients.
            self.optimizer.zero_grad() # Zeros out gradients for next accummulation 

            # ---- Train weights ----
            self.train_row_weights(output.router.weights)

            # ---- Step Logs + Tensorboard ----
            processed = ((batch_idx + 1) * self.config.miner.batch_size_train)
            progress = (100. * processed) / len(self.train_data)
            logger.info('GS: {}\t Epoch: {} [{}/{} ({})]\tLoss: {}\tAcc: {}\tAxon: {}\tDendrite: {}', 
                    colored('{}'.format(self.global_step), 'blue'), 
                    colored('{}'.format(self.epoch), 'blue'), 
                    colored('{}'.format(processed), 'green'), 
                    colored('{}'.format(len(self.train_data)), 'red'),
                    colored('{:.2f}%'.format(progress), 'green'),
                    colored('{:.4f}'.format(output.local_target_loss.item()), 'green'),
                    colored('{:.4f}'.format(output.local_accuracy.item()), 'green'),
                    self.neuron.axon,
                    self.neuron.dendrite)
            
            self.update_tensorboard(output.remote_target_loss.item(), output.local_target_loss.item(), output.distillation_loss.item())



    # --- Test epoch ----
    def test (self):
        with torch.no_grad(): # Turns off gradient computation for inference speed up.
            self.model.eval() # Turns off Dropoutlayers, BatchNorm etc.
            loss = 0.0; accuracy = 0.0
            for _, (images, labels) in enumerate(self.testloader):

                # ---- Local Forward pass ----
                outputs = self.model.local_forward(
                    images = images.to(self.device), 
                    targets = torch.LongTensor(labels).to(self.device), 
                )
                loss += outputs.local_target_loss.item()
                accuracy += outputs.local_accuracy.item()
                
            return loss / len(self.testloader), accuracy / len(self.testloader) 

        
if __name__ == "__main__":
    # ---- Build and Run ----
    session = Session(FFNNSynapse)
    logger.info(bittensor.config.Config.toString(session.config))
    session.run()
