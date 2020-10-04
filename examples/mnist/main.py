from loguru import logger

import argparse
import math
import time
import torch
import torchvision
from typing import List, Tuple, Dict, Optional

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from bittensor.utils.model_utils import ModelToolbox

import bittensor
from bittensor import bittensor_pb2

class MnistSynapse(bittensor.Synapse):
    """ Bittensor endpoint trained on 28, 28 pixel images to detect handwritten characters.
    """
    def __init__(self, config: bittensor.Config):
        super(MnistSynapse, self).__init__(config)
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
        self.logit_layer1 = nn.Linear(1024, 512)
        self.logit_layer2 = nn.Linear(512, 256)
        self.logit_layer3 = nn.Linear(256, 10)

    def forward_image(self, inputs: torch.Tensor):
        return self.forward (inputs = inputs, labels = None, network = None) ['student_y']

    def forward (   self, 
                    inputs: torch.Tensor,
                    labels: torch.Tensor = None,
                    network: torch.Tensor = None):

        student_y = None
        network_y = None
        student_target_loss = None
        network_target_loss = None
        student_distillation_loss = None

        # Encode images into standard shape. Images could any size PILs.
        inputs = self._transform(inputs)
        inputs = self._adaptive_pool(inputs).to(self.device)
        inputs = torch.flatten(inputs, start_dim = 1)

        # student inputs:
        student = F.relu(self.student_layer1 (inputs))
        student = F.relu(self.student_layer2 (student))

        # If there is a network context, use it to train the student network.
        if network is not None:
            student_distillation_loss = F.mse_loss(student, network.detach())

        # Build student_y
        student_y = torch.cat((inputs, student), dim=1)
        student_y = F.relu(self.forward_layer1 (student_y))
        student_y = F.relu(self.forward_layer2 (student_y))

        # Compute the synapse head using the network inputs.
        # Only compute this when there is a network context.
        if network is not None:
            network_y = torch.cat((inputs, network), dim=1)
            network_y = F.relu(self.forward_layer1 (network_y))
            network_y = F.relu(self.forward_layer2 (network_y))

        # Compute the target loss using the student_y and passed labels.
        if labels is not None:
            student_target = F.relu(self.logit_layer1 (student_y))
            student_target = F.relu(self.logit_layer2 (student_target))
            student_target = F.log_softmax(student_target, dim=1)
            student_target_loss = F.nll_loss(student_target, labels)

        # Compute a target loss using network_y and the passed labels.
        if network is not None and labels is not None:
            network_target = F.relu(self.logit_layer1 (network_y))
            network_target = F.relu(self.logit_layer2 (network_target))
            network_target = F.log_softmax(network_target, dim=1)
            network_target_loss = F.nll_loss(network_target, labels)

        return {
            'student_y': student_y,
            'network_y': network_y,
            'network_target_loss': network_target_loss,
            'student_target_loss': student_target_loss,
            'student_distillation_loss': student_distillation_loss
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
    model = MnistSynapse(config) # Synapses take a config object.
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
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            
            # Clear gradients on model parameters.
            optimizer.zero_grad()
            
            # Encode inputs for network contect used to query.
            context = model.forward_image(inputs).to(device)
            
            # Targets to Tensor
            targets = torch.LongTensor(targets).to(device)
            
            # Query the remote network.
            # Flatten mnist inputs for routing.
            synapses = metagraph.get_synapses( 1000 ) # Returns a list of synapses on the network (max 1000).
            requests, scores = router.route( synapses, context, inputs ) # routes inputs to network.
            responses = dendrite.forward_image( synapses, requests ) # Makes network calls.
            network = router.join( responses ) # Joins responses based on scores..
            
            # Compute full pass and get loss.
            output = model.forward(inputs, targets, network)

            # Loss and step.
            loss = output['student_target_loss'] + output['student_distillation_loss'] + output['network_target_loss']
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
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tDistill Loss: {:.6f}\tStudent Loss: {:.6f}\tnP|nS: {}|{}'.format(
                    epoch, (batch_idx * batch_size_train), n, (100. * batch_idx * batch_size_train)/n, output['network_target_loss'].item(), output['student_distillation_loss'].item(), output['student_target_loss'].item(), len(metagraph.peers), 
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
            for _, (inputs, targets) in enumerate(testloader):                
               
                # Targets to Tensor
                targets = torch.LongTensor(targets).to(device)

                # Compute full pass and get loss.
                outputs = model.forward(inputs, targets)
                            
                # Count accurate predictions.
                max_logit = outputs['student_y'].data.max(1, keepdim=True)[1]
                correct += max_logit.eq( targets.data.view_as(max_logit) ).sum()
        
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
