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
import time

class NeuronServe:
    def __init__( self, config: 'bittensor.config', wallet: 'bittensor.wallet'):
        self.config = config
        self.wallet = wallet
        self.subtensor = bittensor.subtensor ( config = self.config )
        self.axon = bittensor.axon (
            config = self.config,
            wallet = self.wallet,
            forward_text = self.forward_text,
            backward_text = self.backward_text,
            blacklist = self.blacklist,
        )
    
    def __enter__(self):
        self.subtensor.register( self.wallet )
        self.axon.start().serve (
            use_upnpc = self.config.neuron.use_upnpc, 
            subtensor = self.subtensor
        )

    def __exit__ ( self ):
        self.axon.stop()
    
    def run(self):
        with self:
            while (1):
                time.sleep(10)

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
        output = self.nucleus.forward (
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
                outputs_y = self.nucleus.forward(
                    forward_type = 'local',
                    inputs = inputs_x.to( self.device ) ).local_context.to( self.device )
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
