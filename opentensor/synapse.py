from opentensor import opentensor_pb2

import torch

class Synapse():
    """ Implementation of an synapse. A single ip/port tensor processing unit """
    def __init__(self):
        pass

    def indef(self) -> opentensor_pb2.TensorDef:
        """ Returns the opentensor_pb2.TensorDef for the input """
        raise NotImplementedError

    def outdef(self) -> opentensor_pb2.TensorDef:
        """ Returns the opentensor_pb2.TensorDef for the output """
        raise NotImplementedError

    def forward(self, key, tensor) -> torch.Tensor:
        """ Processes the tensor from the sent key """
        raise NotImplementedError

    def backward(self, key, tensor) -> torch.Tensor:
        """ Processes the gradient from the sent key """
        raise NotImplementedError
