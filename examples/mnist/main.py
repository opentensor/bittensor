"""BERT Next Sentence Prediction Synapse

This file demonstrates a bittensor.Synapse trained for Next Sentence Prediction.

Example:
        $ python examples/bert/main.py

"""

import bittensor
from bittensor.utils.model_utils import ModelToolbox
from bittensor import bittensor_pb2

import argparse
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
from typing import List, Tuple, Dict, Optional

class MnistSynapse(bittensor.Synapse):
    """ Bittensor endpoint trained on PIL images to detect handwritten characters.
    """
    def __init__(self):
        super(MnistSynapse, self).__init__()
        # Image encoder
        self._transform = bittensor.utils.batch_transforms.Normalize((0.1307,), (0.3081,))
        self._adaptive_pool = nn.AdaptiveAvgPool2d((28, 28))
        
        # Forward Network
        self.forward_layer1 = nn.Linear((784 + 1024), 1024)
        self.forward_layer2 = nn.Linear(1024, 1024)
        
        # Distillation Network
        self.student_layer1 = nn.Linear(784, 1024)
        self.student_layer2 = nn.Linear(1024, 1024)
        
        # Logit Network 
        self.target_layer1 = nn.Linear(1024, 512)
        self.target_layer2 = nn.Linear(512, 256)
        self.target_layer3 = nn.Linear(256, 10)

    def forward_image(self, images: torch.Tensor):
        r""" Forward pass inputs and labels through the NSP BERT module.

            Args:
                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, -1, -1, -1)`, `required`): 
                    batch_size length list of image tensors. (batch index, channel, row, col) produced 
                    by calling PIL.toTensor()
            
            Returns:
                local_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.hidden_size)`, `required`): 
                    Output encoding of inputs produced using the local student distillation model as context.
        """
        return self.forward (images = images, labels = None, network = None) ['local_output']

    def forward (   self, 
                    images: torch.Tensor,
                    labels: torch.Tensor = None,
                    network: torch.Tensor = None):

        r""" Forward pass inputs and labels through the NSP BERT module.

            Args:
                images (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, -1, -1, -1)`, `required`): 
                    PIL.toTensor() encoded images.

                labels (:obj:`torch.FloatTensor`  of shape :obj:`(batch_size, 10)`, `optional`): 
                    Mnist labels.

                network (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.hidden_size)`, `optional`):
                    response context from a bittensor dendrite query. 

            Returns:
                dictionary with { 
                    loss  (:obj:`List[str]` of shape :obj:`(batch_size)`, `required`):
                        Total loss acumulation to be used by loss.backward()

                    local_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.hidden_dim)`, `required`):
                        Output encoding of image inputs produced by using the local student distillation model as 
                        context rather than the network. 

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        MNIST Classification loss computed using the local_output and passed labels.

                    network_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.hidden_dim)`, `optional`): 
                        Output encoding of inputs produced by using the network inputs as context to the local model rather than 
                        the student.

                    network_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                        MNIST Classification loss computed using the local_output and passed labels.

                    distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        Distillation loss produced by the student with respect to the network context.
                }
        """
        # Return vars.
        loss = torch.tensor(0.0)
        local_output = None
        network_output = None
        network_target_loss = None
        local_target_loss = None
        distillation_loss = None

        # Encode images into standard shape. Images could any size PILs.
        images = self._transform(images)
        images = self._adaptive_pool(images).to(self.device)
        images = torch.flatten(images, start_dim = 1)

        # student inputs:
        student = F.relu(self.student_layer1 (images))
        student = F.relu(self.student_layer2 (student))

        # If there is a network context, use it to train the student network.
        if network is not None:
            distillation_loss = F.mse_loss(student, network.detach())
            loss += distillation_loss

        # Build student_y
        local_output = torch.cat((images, student), dim=1)
        local_output = F.relu(self.forward_layer1 (local_output))
        local_output = F.relu(self.forward_layer2 (local_output))
        if labels is not None:
            # Compute the target loss using the student_y and passed labels.
            local_target = F.relu(self.target_layer1 (local_output))
            local_target = F.relu(self.target_layer2 (local_target))
            local_target = F.log_softmax(local_target, dim=1)
            local_target_loss = F.nll_loss(local_target, labels)
            loss += local_target_loss

        # Compute the synapse head using the network inputs.
        # Only compute this when there is a network context.
        if network is not None:
            network_output = torch.cat((images, network), dim=1)
            network_output = F.relu(self.forward_layer1 (network_output))
            network_output = F.relu(self.forward_layer2 (network_output))

        # Compute a target loss using network_y and the passed labels.
        if network is not None and labels is not None:
            network_target = F.relu(self.target_layer1 (network_output))
            network_target = F.relu(self.target_layer2 (network_target))
            network_target = F.log_softmax(network_target, dim=1)
            network_target_loss = F.nll_loss(network_target, labels)
            loss += network_target_loss

        return {
            'loss': loss,
            'local_output': local_output,
            'network_output': network_output,
            'network_target_loss': network_target_loss,
            'local_target_loss': local_target_loss,
            'distillation_loss': distillation_loss
        }
            
        
def main(hparams):
     
    # Load bittensor config from hparams.
    config = bittensor.Config( hparams )

    # Additional training params.
    batch_size_train = 64
    batch_size_test = 64
    learning_rate = 0.01
    momentum = 0.9
    log_interval = 10
    epoch = 0
    global_step = 0
    best_test_loss = math.inf
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate toolbox to load/save model
    model_toolbox = ModelToolbox('mnist')

    # Load (Train, Test) datasets into memory.
    train_data = torchvision.datasets.MNIST(root=model_toolbox.data_path, train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size_train, shuffle=True, num_workers=2)
    
    test_data = torchvision.datasets.MNIST(root=model_toolbox.data_path, train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(test_data, batch_size = batch_size_test, shuffle=False, num_workers=2)
    
    # Build summary writer for tensorboard.
    writer = SummaryWriter(log_dir=model_toolbox.log_dir)
    
    # Build local synapse to serve on the network.
    model = MnistSynapse() # Synapses take a config object.
    model.to( device ) # Set model to device.
    # Build and start the metagraph background object.
    # The metagraph is responsible for connecting to the blockchain
    # and finding the other neurons on the network.
    metagraph = bittensor.Metagraph( config )
    metagraph.subscribe( model ) # Adds the synapse to the metagraph.
    metagraph.start() # Starts the metagraph gossip threads.
    
    # Build and start the Axon server.
    # The axon server serves synapse objects (models) 
    # allowing other neurons to make queries through a dendrite.
    axon = bittensor.Axon( config )
    axon.serve( model )
    axon.start() # Starts the server background threads. Must be paired with axon.stop().
    
    # Build the dendrite and router. 
    # The dendrite is a torch object which makes calls to synapses across the network
    # The router is responsible for learning which synapses to call.
    dendrite = bittensor.Dendrite( config ).to(device)
    router = bittensor.Router(x_dim = 1024, key_dim = 100, topk = 10)
        
    # Build the optimizer.
    params = list(router.parameters()) + list(model.parameters())
    optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum)
    
    # Load previously trained model if it exists
    if config._hparams.load_model is not None:
        present_model, optimizer, epoch, best_test_loss = model_toolbox.load_model(model, config._hparams.load_model, optimizer)
        logger.info("Loaded model stored in {} with test loss {} at epoch {}".format(config._hparams.load_model, best_test_loss, epoch-1))

    # Train loop: Single threaded training of MNIST.
    # 1. Makes calls to the network using the bittensor.dendrite
    # 2. Trains the local model using inputs from network + raw_features.
    # 3. Trains the distillation model to emulate network inputs.
    # 4. Trains the local model and passes gradients through the network.
    def train(model, epoch, global_step):
        # Turn on Dropoutlayers BatchNorm etc.
        model.train()
        for batch_idx, (images, labels) in enumerate(trainloader):
            
            # Clear gradients on model parameters.
            optimizer.zero_grad()
            
            # Encode inputs for network contect used to query.
            context = model.forward_image(images).to(device)
            
            # Targets to Tensor
            labels = torch.LongTensor(labels).to(device)
            
            # Query the remote network.
            # Flatten mnist inputs for routing.
            synapses = metagraph.get_synapses( 1000 ) # Returns a list of synapses on the network (max 1000).
            requests, scores = router.route( synapses, context, images ) # routes inputs to network.
            responses = dendrite.forward_image( synapses, requests ) # Makes network calls.
            network = router.join( responses ) # Joins responses based on scores..
            
            # Compute full pass and get loss.
            output = model.forward(images, labels, network)

            # Loss and step.
            loss = output['loss']
            torch.nn.utils.clip_grad_norm_(router.parameters(), 0.5)
            loss.backward()
            optimizer.step()
            global_step += 1
            
            # Set network weights.
            weights = metagraph.getweights(synapses).to(device)
            weights = (0.99) * weights + 0.01 * torch.mean(scores, dim=0)
            metagraph.setweights(synapses, weights)
                
            # Logs:
            if batch_idx % log_interval == 0:
                n_peers = len(metagraph.peers)
                n_synapses = len(metagraph.synapses)
                writer.add_scalar('n_peers', n_peers, global_step)
                writer.add_scalar('n_synapses', n_synapses, global_step)
                writer.add_scalar('train_loss', float(loss.item()), global_step)
            
                n = len(train_data)
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLocal Loss: {:.6f}\tTarget Loss: {:.6f}\tDistillation Loss: {:.6f}\tnP|nS: {}|{}'.format(
                    epoch, (batch_idx * batch_size_train), n, (100. * batch_idx * batch_size_train)/n, output['local_target_loss'].item(), output['network_target_loss'].item(), output['distillation_loss'].item(), len(metagraph.peers), 
                            len(metagraph.synapses)))

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
                outputs = model.forward(images, labels)
                            
                # Count accurate predictions.
                max_logit = outputs['local_output'].data.max(1, keepdim=True)[1]
                correct += max_logit.eq( labels.data.view_as(max_logit) ).sum()
        
        # # Log results.
        n = len(test_data)
        loss /= n
        accuracy = (100. * correct) / n
        logger.info('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(loss, correct, n, accuracy))        
        return loss, accuracy
    
    try:
        while True:
            # Train model
            train( model, epoch, global_step )
            
            # Test model.
            test_loss, _ = test( model )
       
            # Save best model. 
            if test_loss < best_test_loss:
                # Update best loss.
                best_test_loss = test_loss
                
                # Save the best local model.
                logger.info('Saving model: epoch: {}, loss: {}, path: {}', epoch, test_loss, model_toolbox.model_path)
                model_toolbox.save_model(model, epoch, optimizer, test_loss)
                
            epoch += 1
            
    except Exception as e:
        logger.error(e)
        metagraph.stop()
        axon.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    hparams = bittensor.Config.add_args(parser)
    hparams = parser.parse_args()
    main(hparams)
