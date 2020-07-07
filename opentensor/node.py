from opentensor import opentensor_pb2
import torch

class Node():
    def __init__(self):
        pass
    
    def definition (self) -> str:
        raise NotImplementedError

    def indef (self) -> torch.Tensor:
        raise NotImplementedError

    def outdef (self) -> torch.Tensor:
        raise NotImplementedError

    def fwd (self, key, tensor) -> torch.Tensor:
        raise NotImplementedError

    def bwd (self, key, tensor) -> torch.Tensor:
        raise NotImplementedError
 
