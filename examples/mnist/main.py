"""BERT Next Sentence Prediction Synapse

This file demonstrates a bittensor.Synapse trained for Next Sentence Prediction.

Example:
        $ python examples/bert/main.py

"""

import bittensor
from bittensor.utils.model_utils import ModelToolbox
from bittensor import bittensor_pb2

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

class MnistSynapse(bittensor.Synapse):
    """ Bittensor endpoint trained on PIL images to detect handwritten characters.
    """
    def __init__(self):
        super(MnistSynapse, self).__init__()

        # Set up device.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Image encoder: transforms variable shaped PIL tensors to a common shape.
        # Image.PIL.toTensor() -> [Image Encoder]
        self._transform = bittensor.utils.batch_transforms.Normalize((0.1307,), (0.3081,))
        self._adaptive_pool = nn.AdaptiveAvgPool2d((28, 28))
        
        # Forward Network: Transforms inputs and (student or network) context into 
        # a (batch_size, bittensor.network_shape) output. 
        # [Image Encoder + (Student or Network)] -> [Forward Net] -> [Target Net]
        self.forward_layer1 = nn.Linear((784 + 1024), 1024)
        self.forward_layer2 = nn.Linear(1024, 1024)
        
        # Student Network: Learns a mapping from inputs to network context.
        # [Image Encoder] -> [Student Net] -> [Forward Network]
        self.student_layer1 = nn.Linear(784, 1024)
        self.student_layer2 = nn.Linear(1024, 1024)
        
        # Target Network: Transforms the model output to targets and loss.
        # [Image Encoder] -> [Student Net] -> [Forward Net] -> [Target Net]
        self.target_layer1 = nn.Linear(1024, 512)
        self.target_layer2 = nn.Linear(512, 256)
        self.target_layer3 = nn.Linear(256, 10)

    def forward_image(self, images: torch.Tensor):
        r""" Forward pass inputs and labels through the NSP BERT module.

            Args:
                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, -1, -1, -1)`, `required`): 
                    batch_size list of image tensors. (batch index, channel, row, col) produced for images
                    by calling PIL.toTensor()
            
            Returns:
                local_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, bittensor.network_size)`, `required`): 
                    Output encoding of inputs produced using the student model as context.
        """
        return self.forward (images = images) ['local_output']

    def forward (   self, 
                    images: torch.Tensor,
                    labels: torch.Tensor = None,
                    network: torch.Tensor = None):

        r""" Forward pass inputs and labels through the MNIST model.

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

                    local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 10)`, `optional`):
                        MNIST Target predictions using student model as context. 

                    local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                        MNIST Classification loss computed using the local_output, student model and passed labels.

                    network_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 10)`, `optional`):
                        MNIST Target predictions using the network as context. 

                    network_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, bittensor.network_dim)`, `optional`): 
                        Output encoding of inputs produced by using the network inputs as context to the model rather than 
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
        local_target = None
        network_output = None
        network_target = None
        network_target_loss = None
        local_target_loss = None
        distillation_loss = None

        # images: torch.Tensor(batch_size, 784)
        # The images are encoded to a standard shape 784 
        # using an adaptive pooling layer and our normalization
        # transform.
        images = self._transform(images)
        images = self._adaptive_pool(images).to(self.device)
        images = torch.flatten(images, start_dim = 1)

        # student: torch.Tensor(batch_size, bittensor.network_dim)
        # The student model distills from the network and is used
        # to compute the local_outputs when there is no network
        # context.
        student = F.relu(self.student_layer1 (images))
        student = F.relu(self.student_layer2 (student))
        if network is not None:
            # Use the network context to train the student network.
            distillation_loss = F.mse_loss(student, network.detach())
            loss += distillation_loss

        # local_output: torch.Tensor(batch_size, bittensor.network_dim)
        # The local_output is a non-target output of this synapse.
        # Outputs are used by other models as training signals.
        # This output is local because it uses the student inputs to 
        # condition the outputs rather than the network context.
        local_output = torch.cat((images, student.detach()), dim=1)
        local_output = F.relu(self.forward_layer1 (local_output))
        local_output = F.relu(self.forward_layer2 (local_output))
        if labels is not None:
            # local_target = torch.Tensor(batch_size, 10)
            # Compute the target loss using the student and passed labels.
            labels.to(self.device)
            local_target = F.relu(self.target_layer1 (local_output))
            local_target = F.relu(self.target_layer2 (local_target))
            local_target = F.relu(self.target_layer3 (local_target))
            local_target = F.log_softmax(local_target, dim=1)
            local_target_loss = F.nll_loss(local_target, labels)
            loss += local_target_loss

        # network_output = torch.Tensor(batch_size, bittensor.network_dim)
        # The network_output is a non-target output of this synapse.
        # This output is remote because it requries inputs from the network.
        if network is not None:
            network_output = torch.cat((images, network.detach()), dim=1)
            network_output = F.relu(self.forward_layer1 (network_output))
            network_output = F.relu(self.forward_layer2 (network_output))

        # network_target = torch.Tensor(batch_size, 10)
        # Compute a target loss using the network_output and passed labels.
        if network is not None and labels is not None:
            network_target = F.relu(self.target_layer1 (network_output))
            network_target = F.relu(self.target_layer2 (network_target))
            network_target = F.relu(self.target_layer3 (network_target))
            network_target = F.log_softmax(network_target, dim=1)
            network_target_loss = F.nll_loss(network_target, labels)
            loss += network_target_loss

        return {
            'loss': loss,
            'local_output': local_output,
            'network_output': network_output,
            'local_target': local_target,
            'network_target': network_target,
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
    axon.serve( copy.deepcopy(model) )
    axon.start() # Starts the server background threads. Must be paired with axon.stop().
    
    # Build the dendrite and router. 
    # The dendrite is a torch nn.Module object which makes calls to synapses across the network
    # The router is responsible for learning which synapses to call.
    dendrite = bittensor.Dendrite( config ).to(device)
    router = bittensor.Router(x_dim = 1024, key_dim = 100, topk = 10)
        
    # Build the optimizer.
    params = list(router.parameters()) + list(model.parameters())
    optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum)
    
    # Load previously trained model if it exists
    if config._hparams.load_model is not None:
        model, optimizer, epoch, best_test_loss = model_toolbox.load_model(model, config._hparams.load_model, optimizer)
        logger.info("Loaded model stored in {} with test loss {} at epoch {}".format(config._hparams.load_model, best_test_loss, epoch-1))

    # Train loop: Single threaded training of MNIST.
    def train(model, epoch, global_step):
        # Turn on Dropoutlayers BatchNorm etc.
        model.train()
        for batch_idx, (images, labels) in enumerate(trainloader):
            
            # Clear gradients on model parameters.
            optimizer.zero_grad()

            # Targets and images to correct device.
            labels = torch.LongTensor(labels).to(device)
            images = images.to(device)
            
            # Encode inputs for the router context.
            context = model.forward_image(images).to(device)
            
            # Query the remote network.
            # [images] -> TOPK(scores, synapses) -> JOIN(scores, responses)
            synapses = metagraph.get_synapses( 1000 ) # Returns a list of synapses on the network (max 1000).
            requests, scores = router.route( synapses, context, images ) # routes inputs to network.
            responses = dendrite.forward_image( synapses, requests ) # Makes network calls.
            network = router.join( responses ) # Joins responses based on scores..
            
            # Computes model outputs and loss.
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
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLocal Loss: {:.6f}\nNetwork Loss: {:.6f}\tDistillation Loss: {:.6f}\tnP|nS: {}|{}'.format(
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
                logger.info('Serving / Saving model: epoch: {}, loss: {}, path: {}', epoch, test_loss, model_toolbox.model_path)
                axon.serve( copy.deepcopy(model) ) # Save a model copy to the axon, replaces the prvious model.
                model_toolbox.save_model(model, epoch, optimizer, test_loss) # Saves the model to local storage.
            epoch += 1

        except Exception as e:
            traceback.print_exc()
            logger.error(e)
            metagraph.stop()
            axon.stop()
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    hparams = bittensor.Config.add_args(parser)
    hparams = parser.parse_args()
    main(hparams)
