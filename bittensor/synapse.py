from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn

from opentensor import opentensor_pb2
import bittensor
    
class Synapse(nn.Module):
    """
    """
    def __init__(self):
        super().__init__()
        self._synapse_key = bittensor.Identity().public_key()
        
    def indef(self) -> opentensor_pb2.TensorDef:
        raise NotImplementedError
    
    def outdef(self) -> opentensor_pb2.TensorDef:
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
            pass
            # TODO(const): fix this.
            #outputs = self.forward(inputs)
            #torch.autograd.backward(outputs, grad_tensors=grads, create_graph=False, retain_graph=False)
            #self.apply_gradients()
        #print ('return none')
        return torch.zeros_like(inputs)

    def apply_gradients(self) -> None:
        """
        Train the expert for one step.
        """
        self.opt.step()
        self.opt.zero_grad()