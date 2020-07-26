from opentensor import opentensor_pb2

import torch
import torch.nn as nn
    
class Synapse(nn.Module):
    """
    """

    def __init__(self, synapse: Synapse):
        super().__init__()
        
        self.input_schema = self.input_schema()
        self.output_schema = self.output_schema()
        
    def indef(self) -> List[opentensor.TensorDef]:
        raise NotImplementedError
    
    def outdef(self) -> List[opentensor.TensorDef]:
        return NotImplementedError  
    
    def call_forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply forward pass to the nn.module given inputs.
        """
        # TODO(const): check schema (inputs, input_schema)
        with torch.no_grad():
            outputs = self.forward(inputs)
        return outputs
    
    def call_backward(self, inputs_and_grads: List[torch.Tensor])-> List[torch.Tensor]:
        """
        Apply a backward pass to the nn.module given grads and inputs.
        """
        # TODO (const): check input schema is same as inputs.
        # TODO (const): check grads schema is same as outputs
        grad_outputs = inputs_and_grads[1]
        inputs = inputs_and_grads[0] 
        with torch.enable_grad():
            outputs = self.forward(inputs)
            torch.autograd.backward(outputs, grad_tensors=grad_outputs, create_graph=False, retain_graph=False)
            self.apply_gradients()
        return None

    def apply_gradients(self) -> None:
        """
        Train the expert for one step.
        """
        self.opt.step()
        self.opt.zero_grad()