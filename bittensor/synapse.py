'''
The MIT License (MIT)
Copyright © 2021 Opentensor.ai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
'''
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim

from loguru import logger
from munch import Munch
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

    def forward_text(self, text: torch.LongTensor) -> torch.FloatTensor:
        r"""Forward tokenized text inputs through this synapse.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len)`, `required`): 
                    Batch_size length list of tokenized sentences.
            
            Returns:
                outputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Output representations produced by this synapse for passed text.
        """
        raise NotImplementedError('Must be overriden in synapse implementation')

    def forward_image(self, images: torch.FloatTensor) -> torch.FloatTensor:
        r"""Forward sequential image inputs through this synapse.

            Args:
                inputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, channels, rows, cols)`, `required`): 
                    Sequenced Image tensors i.e. created using PIL.toTensor() and an added sequence dimension.
            
            Returns:
                outputs (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_dim, bittensor.__network_dim__)`, `required`): 
                    Output representations produced by this synapse for passed images.
        """
        raise NotImplementedError('Must be overriden in synapse implementation')

    def forward_tensor(self, tensors: torch.FloatTensor) -> torch.FloatTensor:
        r"""Forward raw float encoded tensors through this synapse.

            Args:
                inputs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Sequenced float tensors to be passed through this synapse.
            
            Returns:
                output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`): 
                    Output representations produced by this synapse for passed tensors.
        """
        raise NotImplementedError('Must be overriden in synapse implementation')

    def call_forward(self, inputs: torch.Tensor, modality: bittensor.proto.Modality, no_grad=True) -> torch.FloatTensor:
        """
        Apply forward pass to the bittensor.synapse given inputs and modality.
        """
        if no_grad:
            with torch.no_grad():
                if modality == bittensor.proto.Modality.TEXT:
                    outputs = self.forward_text(inputs)
                elif modality == bittensor.proto.Modality.IMAGE:
                    outputs = self.forward_image(inputs)
                elif modality == bittensor.proto.Modality.TENSOR:
                    outputs = self.forward_tensor(inputs)
                else:
                    raise NotImplementedError
        else:
            if modality == bittensor.proto.Modality.TEXT:
                outputs = self.forward_text(inputs)
            elif modality == bittensor.proto.Modality.IMAGE:
                outputs = self.forward_image(inputs)
            elif modality == bittensor.proto.Modality.TENSOR:
                outputs = self.forward_tensor(inputs)
            else:
                raise NotImplementedError

        return outputs

    def grad(self, inputs_x: torch.Tensor, grads_dy: torch.Tensor, modality: bittensor.proto.Modality) -> torch.Tensor:
        """
            Returns gradients for the inputs given outputs and output grads.
        """
        with torch.enable_grad():
            outputs_y = self.call_forward(inputs = inputs_x.to(self.device), modality = modality, no_grad=False)
            grads_dx = torch.autograd.grad(
                outputs = outputs_y.to(self.device), 
                inputs = inputs_x.to(self.device), 
                grad_outputs = grads_dy.to(self.device), 
                only_inputs = True,
                create_graph = False, 
                retain_graph = False
            )
        return grads_dx[0]

    def backward(self, inputs_x: torch.Tensor, grads_dy: torch.Tensor, modality: bittensor.proto.Modality):
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

