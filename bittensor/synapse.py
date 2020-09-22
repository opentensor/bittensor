from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from bittensor import bittensor_pb2
import bittensor
    
class Synapse(nn.Module):
    """
    """
    def __init__(self):
        super().__init__()
        self._synapse_key = bittensor.Crypto.public_key_to_string(bittensor.Crypto.generate_private_ed25519().public_key())
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def indef(self) -> bittensor_pb2.TensorDef:
        raise NotImplementedError
    
    def outdef(self) -> bittensor_pb2.TensorDef:
        return NotImplementedError  
    
    def synapse_key(self) -> str:
        return self._synapse_key
    
    def setup_optimizer(self):
        if not self.optimizer:
            self.optimizer = optim.SGD(self.parameters(),
                          lr=0.1,
                          momentum=0.9)
        
    def call_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply forward pass to the nn.module given inputs.
        """
        # TODO(const): check schema (inputs, input_schema)
        with torch.no_grad():
            outputs = self.forward(inputs)
        return outputs
    
    def call_backward(self, inputs: torch.Tensor, grads: torch.Tensor)-> torch.Tensor:
        """
        Apply a backward pass to the nn.module given grads and inputs.
        """
        #with torch.enable_grad():
        #    outputs = self.forward(inputs)
        #    torch.autograd.backward(outputs, grad_tensors=grads.to(self.device), create_graph=False, retain_graph=False)
        #    self.apply_gradients()
        # NOTE(const): removing gradient application here, needs to be replaced with gradient queueing.
        return torch.zeros_like(inputs)

    def apply_gradients(self) -> None:
        """
        Train the expert for one step.
        """
        pass
        #self.optimizer.step()
        #self.optimizer.zero_grad()