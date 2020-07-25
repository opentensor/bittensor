from typing import List, Tuple

import torch

from opentensor import opentensor_pb2
import opentensor

class Router ():
    def __init__(self, x_dim, key_dim, topk):
        self.x_dim = x_dim
        self.key_dim = key_dim
        self.topk = topk
        
        # Keys object.
        # projects from/to opentensor_pb2.Synapse to a variable sized key tensor.
        self.keymap = opentensor.Keys(self.key_dim)

        # Trainable gating object.
        self.gate = opentensor.Gate(self.x_dim, self.topk, self.key_dim)

        # Object for dispatching / combining gated inputs
        self.dispatcher = opentensor.Dispatcher()
        
    def route (self, inputs: torch.Tensor, synapses: List[opentensor_pb2.Synapse]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # Get synapses from the metagraph.
        # and map synapses to torch keys.
        keys = self.keymap.toKeys(synapses)  # (n_keys, key_dim)

        # Learning a map from the gate_inputs to keys
        # scores[i, j] = score for the jth key for input i
        self.scores = self.gate(inputs, keys, topk = min(len(keys), self.topk))

        # Dispatch data to inputs for each key.
        # when scores[i, j] == 0, the key j does not recieve input i
        requests = self.dispatcher.dispatch(inputs, self.scores)  # List[(?, 784)]
        
        return requests, self.scores
    
    def join (self, responses: List[torch.Tensor]) -> torch.Tensor:
        return self.dispatcher.combine (responses, self.scores)