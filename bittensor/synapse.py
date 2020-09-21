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
        # Build the optimizer.
        #self.opt = optim.SGD(params,
        #                  lr=0.01,
        #                  momentum=0.9)
        
    def indef(self) -> bittensor_pb2.TensorDef:
        raise NotImplementedError
    
    def outdef(self) -> bittensor_pb2.TensorDef:
        return NotImplementedError  
    
    def synapse_key(self) -> str:
        return self._synapse_key      
    
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
        with torch.enable_grad():
            #pass
            # TODO(const): fix this.
            outputs = self.forward(inputs)
            torch.autograd.backward(outputs, grad_tensors=grads, create_graph=False, retain_graph=False)
            self.apply_gradients()
        #print ('return none')
        return torch.zeros_like(inputs)

    def apply_gradients(self) -> None:
        """
        Train the expert for one step.
        """
        self.optimizer.step()
        self.optimizer.zero_grad()