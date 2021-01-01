import argparse
from loguru import logger
from munch import Munch
import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING

import bittensor
from bittensor import bittensor_pb2
from bittensor.exceptions.handlers import rollbar

class SynapseOutput(object):
    """ Synapse output container.
            loss  (:obj:`List[str]` of shape :obj:`(batch_size)`, `required`):
                Total loss acumulation to be used by loss.backward()

            local_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, bittensor.__network_dim__)`, `required`):
                Hidden layer encoding produced using the local_context.

            local_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_dim)`, `optional`):
                Target predictions using local_context. 

            local_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                Target loss using the local_context.

            remote_hidden (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, bittensor.__network_dim__)`, `optional`): 
                Hidden layer encoding produced using the remote_context.

            remote_target (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_dim)`, `optional`):
                FFNN Target predictions using the remote_context.

            remote_target_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`):
                FFNN Classification loss using the remote_context.

            distillation_loss (:obj:`torch.FloatTensor` of shape :obj:`(1)`, `optional`): 
                Distillation loss between local_context and remote_context.

            weights (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, metagraph.state.n)`, `optional`): 
                weights for each active neuron.

            requests_sizes (:obj:`torch.LongTensor` of shape :obj:`(metagraph.state.n)`, `optional`): 
                number of requests sent to each uid in this batch.

            return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                dendrite return codes. 0 for success.

            metadata (:obj:`dict {'key_val', torch.FloatTensor} ` of shape :obj:`(1)`, `optional`):
                additional metadata output, for instance accuracy.

    """
    def __init__(   
                self,
                loss: torch.Tensor = None,
                local_hidden: torch.Tensor  = None,
                local_target: torch.Tensor = None,
                local_target_loss: torch.Tensor = None,
                remote_hidden: torch.Tensor = None, 
                remote_target: torch.Tensor = None,
                remote_target_loss: torch.Tensor = None,
                distillation_loss: torch.Tensor = None,
                weights: torch.Tensor = None,
                metadata: dict = None
        ):
        self.loss = loss
        self.local_hidden = local_hidden
        self.local_target = local_target
        self.local_target_loss = local_target_loss
        self.remote_hidden = remote_hidden
        self.remote_target = remote_target
        self.remote_target_loss = remote_target_loss
        self.distillation_loss = distillation_loss
        self.weights = weights
        if metadata == None:
            self.metadata = {}
        else:
            self.metadata = metadata

class Synapse(nn.Module):
    """ Bittensor synapse class. 

    """

    def __init__(   self,
                    config: Munch,
                    neuron):
        r""" Init synapse module.

            Args:
                config (:obj:`SynapseConfig`, `required`): 
                    Base synapse config configuration class.

                neuron (:obj:`bittensor.Neuron`, `optional`): 
                    bittensor training neuron.
        """
        super().__init__()

        self.config = config
        self.neuron = neuron
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def add_args(parser: argparse.ArgumentParser): 
        pass

    @staticmethod   
    def check_config(config: Munch):
        pass

    def deepcopy(self):
        """ Returns a copy of this synapse by passing the model params to load_state_dict.

            Returns:
                synapse_copy (:obj:`self.__class__`, `required`): 
                    Deep copy synapse object.
        """
        SynapseClass = self.__class__
        synapse_copy = SynapseClass(self.config, self.neuron)
        synapse_copy.load_state_dict(self.state_dict())
        return synapse_copy

    def call_forward(self, inputs: torch.Tensor, modality: bittensor_pb2.Modality, no_grad=True) -> torch.Tensor:
        """
        Apply forward pass to the bittensor.synapse given inputs and modality.
        """
        if no_grad:
            with torch.no_grad():
                if modality == bittensor_pb2.Modality.TEXT:
                    outputs = self.forward_text(inputs)
                elif modality == bittensor_pb2.Modality.IMAGE:
                    outputs = self.forward_image(inputs)
                elif modality == bittensor_pb2.Modality.TENSOR:
                    outputs = self.forward_tensor(inputs)
                else:
                    raise NotImplementedError
        else:
            if modality == bittensor_pb2.Modality.TEXT:
                outputs = self.forward_text(inputs)
            elif modality == bittensor_pb2.Modality.IMAGE:
                outputs = self.forward_image(inputs)
            elif modality == bittensor_pb2.Modality.TENSOR:
                outputs = self.forward_tensor(inputs)
            else:
                raise NotImplementedError

        return outputs

    def grad(self, inputs_x: torch.Tensor, grads_dy: torch.Tensor, modality: bittensor_pb2.Modality) -> torch.Tensor:
        """
            Returns gradients for the inputs given inputs and output grads.
        """
        with torch.enable_grad():
            outputs_y = self.call_forward(inputs = inputs_x.to(self.device), modality = modality, no_grad=False)
            grads_dx = torch.autograd.grad(
                outputs = outputs_y.to(self.device), 
                inputs = inputs_x.to(self.device), 
                grad_tensors = grads_dy.to(self.device), 
                only_inputs = True,
                create_graph = False, 
                retain_graph = False
            )
        return grads_dx

    def backward(self, inputs_x: torch.Tensor, grads_dy: torch.Tensor, modality: bittensor_pb2.Modality):
        """
        Apply a backward pass to the nn.module given grads and inputs.
        """
        with torch.enable_grad():
            outputs_y = self.call_forward(inputs = inputs_x.to(self.device), modality = modality, no_grad=False)
            torch.autograd.backward(
                tensors = [outputs_y.to(self.device)], 
                grad_tensors = [grads_dy.to(self.device)], 
                create_graph = False, 
                retain_graph=False
            )
        
    def forward_text(self, inputs: torch.Tensor) -> SynapseOutput:
        """ Local forward inputs through the bittensor.Synapse. To be implemented by sub-classes.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.
            
            Returns:
                local_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Output encoding of inputs produced by the synapse.
        """
        raise NotImplementedError

    def forward_image(self, inputs: torch.Tensor) -> SynapseOutput:
        r""" Forward pass inputs through the bittensor.synapse. To be implemented by sub-classes.

            Args:
                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, channels, rows, cols)`, `required`): 
                    batch_size list of image tensors. (batch index, sequence_len, channel, row, col) produced for images
                    by calling PIL.toTensor().
            
            Returns:
                local_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.network_size)`, `required`): 
                    Output encoding of inputs produced by the synapse.
        """
        raise NotImplementedError

    def forward_tensor(self, inputs: torch.Tensor) -> SynapseOutput:
        """ Forward tensor inputs through the bittensor.synapse. To be implemented by sub-classes.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Batch_size length sequences of tensors.
            
            Returns:
                local_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Output encoding of inputs produced by the synapse.
        """
        raise NotImplementedError

