#!/bin/python3
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
""" Base neuron version 1.

Example:
    $ import neurons
    $ neurons.text.base_neuron_v1().run()

"""

from re import I
import pandas
from pandas.core.frame import DataFrame
import bittensor
import math
import torch
import traceback
import sys
import wandb
from termcolor import colored
from qqdm import qqdm, format_str
from loguru import logger

from bittensor._metagraph import metagraph
logger = logger.opt(colors=True)

from types import SimpleNamespace
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
from functools import partial

import torch.nn.functional as F

class BaseServer():

    def __init__(self):
        assert self.config != None, 'self.config was not initialized.'
        assert self.metagraph != None, 'self.metagraph was not initialized.'
        assert self.nucleus != None, 'self.nucleus wa not initialized.'
        assert self.device != None, 'self.device was not initialized.'
        assert self.axon != None, 'self.axon was not initialized.'

    def __enter__(self):
        self.axon_start()

    def __exit__(self):
        self.axon_stop() 
    
    def axon_start(self):
        self.axon = bittensor.axon (
            config = self.config,
            wallet = self.wallet,
            forward_text = self.forward_text,
            backward_text = self.backward_text,
            blacklist = self.blacklist,
        )

        self.axon.start().serve (
            use_upnpc = self.config.neuron.use_upnpc, 
            subtensor = self.subtensor
        )

    def axon_stop(self):
        self.axon.stop()

    # ---- Axon Forward call ----
    def forward_text ( self, inputs_x: torch.FloatTensor) -> torch.FloatTensor:
        r""" Subscribed to an axon servicing endpoint: processes forward messages from the wire.
            The arguments reflect an RPC request from another miner in the network, the response tensor
            should be the hidden units computed using the local context and with shape: [batch_size, sequence_len, __network_dim__].

            Args:
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.

            Returns:
                outputs (:obj:`torch.FloatTensor`):
                    The nucleus's outputs as a torch tensor of shape [batch_size, sequence_len, __network_dim__]
        """
        output = self.nucleus.forward(
            forward_type = 'local',
            inputs = inputs_x.to( self.device )
        )
        return output.local_hidden

    # ---- Axon Backward call ----
    def backward_text ( self, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor ):
        r""" Subscribed to an axon servicing endpoint: Processes backward messages from the wire.
            Arguments reflect an RPC backward request from another miner in the network. No response
            needed for tokenized text inputs (uint64s have no gradient).

            Args:
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs from previous forward call.
                grads_dy ( :obj:`torch.Tensor`, `required`):
                    torch grads of forward output.                    
        """
        if self.config.neuron.accumulate_remote_gradients:
            with torch.enable_grad():
                # ---- Set up inputs for gradient computations.
                outputs_y = self.nucleus.local_forward( inputs = inputs_x.to( self.device ) ).local_context.to( self.device )
                # ---- The backward call will accumulate gradients on our parameters.
                torch.autograd.backward (
                    tensors = [outputs_y],
                    grad_tensors = [grads_dy.to( self.device )]
                )
    
    def priority(self, pubkey:str, request_type:bittensor.proto.RequestType, inputs_x: torch.FloatTensor) -> float:
        r"""Return the request priority based on stake and size of input. 
            Used by the Axon to order requests.
            Args:
                pubkey ( str, `required`):
                    The public ss58 address of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                request_type ( bittensor.proto.RequestType, `required`):
                    the request type ('FORWARD' or 'BACKWARD').
        """        
        # Priority = stake / request_size 
        priority = self.metagraph.S[ self.metagraph.hotkeys.index(pubkey) ] / sys.getsizeof(inputs_x)
        return priority

    def blacklist(self, pubkey:str, request_type:bittensor.proto.RequestType) -> bool:
        r"""Axon security blacklisting, used to blacklist message from low stake members
            Currently, this is not turned on.
            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                request_type ( bittensor.proto.RequestType, `required`):
                    the request type ('FORWARD' or 'BACKWARD').
        """
        # Blacklist requests from peers who are not subscribed or have stake less that black_list
        is_registered = pubkey in self.metagraph.hotkeys

        # If we allow non-registered requests return False = not blacklisted.
        if not is_registered:
            if self.config.neuron.blacklist_allow_non_registered:
                return False
            else:
                return True
        else:
            # Else, get stake and check is above blacklist stake min.
            uid = self.metagraph.hotkeys.index( pubkey )
            if self.metagraph.S[uid].item() >= self.config.neuron.blacklist:
                return False
            else:
                return True