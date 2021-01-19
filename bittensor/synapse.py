import argparse
from loguru import logger
from munch import Munch
import random
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING

import bittensor

class Synapse(nn.Module):
    """ Bittensor synapse class. 
    """

    def __init__(self, config: Munch = None):
        r""" Init synapse module.

            Args:
                config (:obj:`Munch`, `required`): 
                    synapse.config()
        """
        super().__init__()
        if config == None:
            config = Synapse.build_config()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod   
    def build_config() -> Munch:
        return Munch()

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
        synapse_copy = SynapseClass(self.config)
        synapse_copy.load_state_dict(self.state_dict())
        return synapse_copy

    def call_forward(self, inputs: torch.Tensor, modality: bittensor.bittensor_pb2.Modality, no_grad=True) -> torch.FloatTensor:
        """
        Apply forward pass to the bittensor.synapse given inputs and modality.
        """
        if no_grad:
            with torch.no_grad():
                if modality == bittensor.bittensor_pb2.Modality.TEXT:
                    outputs = self.forward_text(inputs)
                elif modality == bittensor.bittensor_pb2.Modality.IMAGE:
                    outputs = self.forward_image(inputs)
                elif modality == bittensor.bittensor_pb2.Modality.TENSOR:
                    outputs = self.forward_tensor(inputs)
                else:
                    raise NotImplementedError
        else:
            if modality == bittensor.bittensor_pb2.Modality.TEXT:
                outputs = self.forward_text(inputs)
            elif modality == bittensor.bittensor_pb2.Modality.IMAGE:
                outputs = self.forward_image(inputs)
            elif modality == bittensor.bittensor_pb2.Modality.TENSOR:
                outputs = self.forward_tensor(inputs)
            else:
                raise NotImplementedError

        return outputs
        
    def forward_text(self, inputs: torch.Tensor) -> torch.FloatTensor:
        """ Local forward inputs through the bittensor.Synapse. To be implemented by sub-classes.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of text sentences.
            
            Returns:
                hidden_units (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Output encoding of the text produced by the synapse.
        """
        raise NotImplementedError

    def forward_image(self, inputs: torch.Tensor) -> torch.FloatTensor:
        r""" Forward pass inputs through the bittensor.synapse. To be implemented by sub-classes.

            Args:
                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, channels, rows, cols)`, `required`): 
                    batch_size list of image tensors. (batch index, sequence_len, channel, row, col) produced for images
                    by calling PIL.toTensor().
            
            Returns:
                hidden_units (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.network_size)`, `required`): 
                    Output encoding of the images produced by the synapse.
        """
        raise NotImplementedError

    def forward_tensor(self, inputs: torch.Tensor) -> torch.FloatTensor:
        """ Forward tensor inputs through the bittensor.synapse. To be implemented by sub-classes.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Batch_size length sequences of tensors.
            
            Returns:
                hidden_units (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Output encoding of the tensors produced by the synapse.
        """
        raise NotImplementedError


    def grad(self, inputs_x: torch.Tensor, grads_dy: torch.Tensor, modality: bittensor.bittensor_pb2.Modality) -> torch.Tensor:
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

    def backward(self, inputs_x: torch.Tensor, grads_dy: torch.Tensor, modality: bittensor.bittensor_pb2.Modality):
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

