from typing import List, Tuple

import torch
import torch.nn as nn

import bittensor
from bittensor.utils.keys import Keys
from bittensor.utils.gate import Gate
from bittensor.utils.dispatcher import Dispatcher
from bittensor import bittensor_pb2
from loguru import logger
class Router(nn.Module):

    def __init__(self, x_dim, key_dim, topk):
        super().__init__()
        self.x_dim = x_dim
        self.key_dim = key_dim
        self.topk = topk

        # Keys object.
        # projects from/to bittensor_pb2.Neuron to a variable sized key tensor.
        self.keymap = Keys(self.key_dim)
        logger.info("Keymap: {}".format(self.keymap))

        # Trainable gating object.
        self.gate = Gate(self.x_dim, self.topk, self.key_dim)
        logger.info("Gate shape: {}".format(self.gate))

        # Object for dispatching / combining gated inputs
        self.dispatcher = Dispatcher()

    def route(self, neurons: List[bittensor_pb2.Neuron],
              gate_inputs: torch.Tensor,
              raw_inputs: object) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # Get neurons from the metagraph.
        # and map neurons to torch keys.
        keys = self.keymap.toKeys(neurons)  # (n_keys, key_dim)
        logger.info("KEYS_SHAPE: {}".format(keys.shape))

        # Learning a map from the gate_inputs to keys
        # scores[i, j] = score for the jth key for input i
        self.scores = self.gate(gate_inputs,
                                keys,
                                topk=min(len(keys), self.topk))

        logger.info("SCORES_SHAPE: {}".format(self.scores.shape))
        # Dispatch data to inputs for each key.
        # when scores[i, j] == 0, the key j does not recieve input i
        requests = self.dispatcher.dispatch(raw_inputs,
                                            self.scores)  # List[(?, 784)]

        logger.info("requests length: {}".format(len(requests)))
        logger.info("requests: {}".format(requests))

        return requests, self.scores

    def join(self, responses: List[torch.Tensor]) -> torch.Tensor:
        return self.dispatcher.combine(responses, self.scores)
