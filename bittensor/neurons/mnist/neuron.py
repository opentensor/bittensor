"""Training a MNIST Neuron.
This file demonstrates a training pipeline for an MNIST Neuron.
Example:
        $ python examples/mnist/main.py
"""
import argparse
import math
import numpy as np
import time
import torch
from termcolor import colored
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from munch import Munch
from loguru import logger

from bittensor import Session
from bittensor.config import Config
from bittensor.neuron import NeuronBase
from bittensor.synapse import Synapse
from bittensor.synapses.ffnn import FFNNSynapse

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
        parser.add_argument('--neuron.batch_size_train', default=64, type=int, 
                            help='Training batch size.')
        parser.add_argument('--neuron.batch_size_test', default=64, type=int, 
                            help='Testing batch size.')
        parser.add_argument('--neuron.log_interval', default=10, type=int, 
                            help='Batches until neuron prints log statements.')
        # Load args from FFNNSynapse.
        parser = FFNNSynapse.add_args(parser)
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        assert config.neuron.log_interval > 0, "log_interval dimension must positive"
        assert config.neuron.momentum > 0 and config.neuron.momentum < 1, "momentum must be a value between 0 and 1"
        assert config.neuron.batch_size_train > 0, "batch_size_train must a positive value"
        assert config.neuron.batch_size_test > 0, "batch_size_test must a positive value"
        assert config.neuron.learning_rate > 0, "learning rate must be a positive value."
        Config.validate_path_create('neuron.datapath', config.neuron.datapath)
        config = FFNNSynapse.check_config(config)
        return config

    def start(self, session: Session): 
        """ Starts up the neuron.

        Args:
            session (Session): Session object containing bittensor session configuration.

        """
        epoch = 0
        best_test_loss = math.inf
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # Build local synapse to serve on the network.
        model = FFNNSynapse(self.config, session)

        try:
            if self.config.session.checkout_experiment:
                model = session.replicate_util.checkout_experiment(model, best=False)
        except Exception as e:
            logger.warning("Something happened checking out the model. {}".format(e))
            logger.info("Using new model")

        model.to( device ) # Set model to device.
        session.serve( model.deepcopy() )

        # Build the optimizer.
        optimizer = optim.SGD(model.parameters(), lr=self.config.neuron.learning_rate, momentum=self.config.neuron.momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10.0, gamma=0.1)

        train_data = torchvision.datasets.MNIST(root = self.config.neuron.datapath + "datasets/", train=True, download=True, transform=transforms.ToTensor())
        trainloader = torch.utils.data.DataLoader(train_data, batch_size = self.config.neuron.batch_size_train, shuffle=True, num_workers=2)
        test_data = torchvision.datasets.MNIST(root = self.config.neuron.datapath + "datasets/", train=False, download=True, transform=transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(test_data, batch_size = self.config.neuron.batch_size_test, shuffle=False, num_workers=2)
    
        # Train loop: Single threaded training of MNIST.
        def train(model, epoch):
            """ Training routine of the synapse model

            Args:
                model (torch.nn): Model to be trained.
                epoch (int): Present training epoch
            """
            # Turn on Dropoutlayers BatchNorm etc.
            model.train()
            last_log = time.time()
            for batch_idx, (images, targets) in enumerate(trainloader):
                # Clear gradients.
                optimizer.zero_grad()

                # Emit and sync.
                if (session.metagraph.block() - session.metagraph.state.block) > 15:
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

                    logger.info('Epoch: {} [{}/{} ({})] | Loss: {} | Acc: {} ', 
                        epoch, processed_str, n_str, progress_str, loss_item_str, accuracy_str)

                    # Log weights.
                    np.set_printoptions(precision=2, suppress=True, linewidth=500, sign=' ')
                    numpy_uids = np.array(session.metagraph.state.uids.tolist())
                    numpy_weights = np.array(session.metagraph.state.weights.tolist())
                    numpy_stack = np.stack((numpy_uids, numpy_weights), axis=0)
                    stack_str = colored(numpy_stack, 'green')
                    logger.info('Weights: \n {}', stack_str)
                    
                    session.tbwriter.write_loss('train remote target loss', output.remote_target_loss.item())
                    session.tbwriter.write_loss('train local target loss', output.local_target_loss.item())
                    session.tbwriter.write_loss('train distilation loss', output.distillation_loss.item())
                    session.tbwriter.write_loss('train loss', output.loss.item())
                    session.tbwriter.write_accuracy('train accuracy', accuracy)
                    session.tbwriter.write_custom('global step/global step v.s. time', self.config.neuron.log_interval / (time.time() - last_log))
                    last_log = time.time()

        # Test loop.
        # Evaluates the local model on the hold-out set.
        # Returns the test_accuracy and test_loss.
        def test( model: Synapse):
            """ Tests given synapse model.

            Args:
                model (Synapse): Synapse model to be tested

            Returns:
                int: Loss
                int: Accuracy
            """
            
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
                    outputs = model.forward(images, labels, remote = False)
                    loss = loss + outputs.loss
                    
                    # Count accurate predictions.
                    max_logit = outputs.local_target.data.max(1, keepdim=True)[1]
                    correct = correct + max_logit.eq( labels.data.view_as(max_logit) ).sum()
            
            # # Log results.
            n = len(test_data)
            loss /= n
            accuracy = (100. * correct) / n
            logger.info('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, n, accuracy))  
            session.tbwriter.write_loss('test loss', loss)
            session.tbwriter.write_accuracy('test accuracy', accuracy)
            return loss, accuracy
    
        while True:
            # Train model
            train( model, epoch )
            scheduler.step()

            # Test model.
            test_loss, test_accuracy = test( model )
        
            # Save best model. 
            if test_loss < best_test_loss:
                # Update best loss.
                best_test_loss = test_loss
                
                # Save and serve the new best local model.
                logger.info( 'Saving/Serving model: epoch: {}, loss: {}, path: {}/{}/model.torch', epoch, test_loss, self.config.neuron.datapath, self.config.neuron.neuron_name)
                torch.save( {'epoch': epoch, 'model': model.state_dict(), 'test_loss': test_loss},"{}/{}/model.torch".format(self.config.neuron.datapath , self.config.neuron.neuron_name))
                # Save experiment metrics
                session.replicate_util.checkpoint_experiment(epoch, loss=test_loss, accuracy=test_accuracy)
                session.serve( model.deepcopy() )

            epoch += 1