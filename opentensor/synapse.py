from opentensor import opentensor_pb2

import torch

class Synapse(torch.nn.Module):
    """ Implementation of a opentensor.synapse. A single ip/port tensor processing unit """
    def __init__(self):
        super(Synapse, self).__init__()

    def indef(self) -> opentensor_pb2.TensorDef:
        """ Returns the opentensor_pb2.TensorDef for the input """
        raise NotImplementedError

    def outdef(self) -> opentensor_pb2.TensorDef:
        """ Returns the opentensor_pb2.TensorDef for the output """
        raise NotImplementedError
