from typing import List, Tuple, Dict, Optional

from loguru import logger
import torch
import torch.nn as nn
import torch.optim as optim

import bittensor
from bittensor import bittensor_pb2
    
class Synapse(nn.Module):
    """ Bittensor synapse class.
    """
    def __init__(self):
        super().__init__()
        self._synapse_key = bittensor.Crypto.public_key_to_string(bittensor.Crypto.generate_private_ed25519().public_key())
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loss = .0
        self._accuracy = .0
            
    def synapse_key(self) -> str:
        return self._synapse_key

    def set_synapse_stats(self, loss, accuracy):
        self.loss = loss
        self.accuracy = accuracy
    
    def loss(self) -> float:
        return self._loss

    def accuracy(self) -> float:
        return self._accuracy

    def forward_text(self, inputs: torch.Tensor):
        """
            Apply forward pass to the bittensor.synapse given inputs text inputs.
        """
        raise NotImplementedError
        
    def forward_image(self, inputs: torch.Tensor):
        """
            Apply forward pass to the bittensor.synapse given image inputs.
        """
        raise NotImplementedError

    def forward_tensor(self, inputs: torch.Tensor):
        """
            Apply forward pass to the bittensor.synapse given tensor inputs.
        """
        raise NotImplementedError
           
    def call_forward(self, inputs: torch.Tensor, modality: bittensor_pb2.Modality) -> torch.Tensor:
        """
        Apply forward pass to the bittensor.synapse given inputs and modality.
        """
        # TODO(const): check schema (inputs, input_schema)
        with torch.no_grad():
            try:
                if modality == bittensor_pb2.Modality.TEXT:
                    outputs = self.forward_text(inputs)
                elif modality == bittensor_pb2.Modality.IMAGE:
                    outputs = self.forward_image(inputs)
                elif modality == bittensor_pb2.Modality.TENSOR:
                    outputs = self.forward_tensor(inputs)  
                else:
                    raise NotImplementedError
            except NotImplementedError:
                # Input modality not implemented.
                # Returns None.
                return None
            except Exception as e:
                logger.error(e)
                return None
        return outputs
    
    def call_backward(self, inputs: object, grads: torch.Tensor)-> torch.Tensor:
        """
        Apply a backward pass to the nn.module given grads and inputs.
        """
        # NOTE(const): removing gradient application here, needs to be replaced with gradient queueing.
        # with torch.enable_grad():
        #    outputs = self.forward(inputs)
        #    torch.autograd.backward(outputs, grad_tensors=grads.to(self.device), create_graph=False, retain_graph=False)
        #    self.apply_gradients()
        # TODO(const): check instance type.
        return torch.zeros((1,1))

    def apply_gradients(self) -> None:
        """
        Train the expert for one step.
        """
        pass
        #self.optimizer.step()
        #self.optimizer.zero_grad()
