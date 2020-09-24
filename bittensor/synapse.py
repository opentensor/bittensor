from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim

import bittensor
from bittensor import bittensor_pb2
    
class Synapse(nn.Module):
    """
    """
    def __init__(self, config: bittensor.Config):
        super().__init__()
        self._config = config
        self._synapse_key = bittensor.Crypto.public_key_to_string(bittensor.Crypto.generate_private_ed25519().public_key())
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def to_proto(self):
        synapse_proto = bittensor_pb2.Synapse(
            version = bittensor.__version__, 
            neuron_key = self._config.neuron_key, 
            synapse_key = self.synapse_key(), 
            address = self._config.remote_ip, 
            port = self._config.axon_port, 
            indef = self.indef(), 
            outdef = self.outdef()
        )
        return synapse_proto
    
    @property
    def input_shape(self):
        return NotImplementedError
    
    @property
    def output_shape(self):
        return NotImplementedError
    
    @property
    def input_dtype (self):
        return bittensor_pb2.FLOAT32
    
    @property
    def output_dtype (self):
        return bittensor_pb2.FLOAT32
    
    def indef(self):
        x_def = bittensor.bittensor_pb2.TensorDef(
                    version = bittensor.__version__,
                    shape = self.input_shape,
                    dtype = self.input_dtype,
                    requires_grad = True,
                )
        return [x_def]
    
    def outdef(self):
        y_def = bittensor.bittensor_pb2.TensorDef(
                    version = bittensor.__version__,
                    shape = self.output_shape,
                    dtype = self.output_dtype,
                    requires_grad = True,
                )
        return [y_def]
    
    def synapse_key(self) -> str:
        return self._synapse_key
    
    def setup_optimizer(self):
        if not self.optimizer:
            self.optimizer = optim.SGD(self.parameters(),
                          lr=0.1,
                          momentum=0.9)

    def encode_tensor(self, inputs: torch.Tensor) -> torch.Tensor:
        return NotImplementedError    
 
    def encode_string(self, inputs: List[str]) -> torch.Tensor:
        return NotImplementedError    
    
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
        # NOTE(const): removing gradient application here, needs to be replaced with gradient queueing.
        # with torch.enable_grad():
        #    outputs = self.forward(inputs)
        #    torch.autograd.backward(outputs, grad_tensors=grads.to(self.device), create_graph=False, retain_graph=False)
        #    self.apply_gradients()
        return torch.zeros_like(inputs)

    def apply_gradients(self) -> None:
        """
        Train the expert for one step.
        """
        pass
        #self.optimizer.step()
        #self.optimizer.zero_grad()
