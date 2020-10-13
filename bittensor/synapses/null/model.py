"""Null Synapse for testing.
"""

import bittensor

from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Optional

class NullSynapse(bittensor.Synapse):
    """ Bittensor endpoint trained on PIL images to detect handwritten characters.
    """
    def __init__(self, metagraph, dendrite):
        super(NullSynapse, self).__init__()
        self.router = bittensor.Router(x_dim = bittensor.__network_dim__, key_dim = 100, topk = 10)
        self.metagraph = metagraph
        self.dendrite = dendrite

    def forward_tensor(self, tensor: torch.LongTensor):
        logger.info("accept forward tensor {}", tensor)
        return self.forward(inputs = tensor, query = False)

    def forward (   self, 
                    inputs: torch.Tensor,
                    query: bool = False):

        logger.info('Inputs: {} {}', inputs.shape, inputs)
        batch_size = inputs.shape[0]
        sequence_dim = inputs.shape[1]
        network_dim = bittensor.__network_dim__
        if query:
            logger.info('do query')
            context = torch.ones((batch_size, network_dim)) 
            synapses = self.metagraph.synapses() 
            logger.info('synapses: {} {}', len(synapses), synapses)
            requests, _ = self.router.route( synapses, context, inputs )
            responses = self.dendrite.forward_tensor( synapses, requests )
            network = self.router.join( responses )

        output = inputs + torch.ones((batch_size, sequence_dim, network_dim))
        return output

        

        
        # Return vars.
        loss = torch.tensor(0.0)
        local_output = None
        local_target = None
        network_output = None
        network_target = None
        network_target_loss = None
        local_target_loss = None
        distillation_loss = None

        # images: torch.Tensor(batch_size, -1, -1, -1)
        # transform: torch.Tensor(batch_size, 784)
        # The images are encoded to a standard shape 784 
        # using an adaptive pooling layer and our normalization
        # transform.
        transform = self._transform(images)
        transform = self._adaptive_pool(transform).to(self.device)
        transform = torch.flatten(transform, start_dim = 1)

        # If query == True make a remote network call.
        # network: torch.Tensor(batch_size, bittensor.__network_dim__)
        if query:
            synapses = bittensor.metagraph.synapses() # Returns a list of synapses on the network.
            requests, _ = self.router.route( synapses, transform, images ) # routes inputs to network.
            responses = bittensor.dendrite.forward_image( synapses, requests ) # Makes network calls.
            network = self.router.join( responses ) # Joins responses based on scores..

        # student: torch.Tensor(batch_size, bittensor.network_dim)
        # The student model distills from the network and is used
        # to compute the local_outputs when there is no network
        # context.
        student = F.relu(self.student_layer1 (transform))
        student = F.relu(self.student_layer2 (student))
        if query:
            # Use the network context to train the student network.
            distillation_loss = F.mse_loss(student, network.detach())
            loss += distillation_loss

        # local_output: torch.Tensor(batch_size, bittensor.network_dim)
        # The local_output is a non-target output of this synapse.
        # Outputs are used by other models as training signals.
        # This output is local because it uses the student inputs to 
        # condition the outputs rather than the network context.
        local_output = torch.cat((transform, student.detach()), dim=1)
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
        if query:
            network_output = torch.cat((transform, network), dim=1)
            network_output = F.relu(self.forward_layer1 (network_output))
            network_output = F.relu(self.forward_layer2 (network_output))

        # network_target = torch.Tensor(batch_size, 10)
        # Compute a target loss using the network_output and passed labels.
        if query and labels is not None:
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