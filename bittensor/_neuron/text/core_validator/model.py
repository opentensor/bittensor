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
""" The bittensor base validator

Example:
    $ python3 miners/text/core_validator.py --logging.debug

"""
import argparse
import time
import datetime
import bittensor
import torch
import os
import wandb
import math
import random
import sys
import pandas
import traceback
from rich import print
from rich.console import Console
from rich.traceback import install
from typing import List, Tuple, Callable, Dict, Any, Union, Set

from bittensor.utils.tokenizer_utils import phrase_cross_entropy, topk_tokens_to_vocab_size, prune_tokens
from bittensor._neuron.text.neuron_utilities import ThreadQueue, PositionalEncoding, calc_loss_fct
from bittensor._neuron.text.log_utilities import ValidatorLogger
from torch.nn.functional import kl_div
from torch.nn.utils import clip_grad_norm_
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from loguru import logger
from threading import Lock
from prometheus_client import Counter, Gauge, Histogram, Summary, Info
import asyncio

logger = logger.opt( colors=True )
console = Console()
install(show_locals=True)

class nucleus( torch.nn.Module ):
    """ Nucleus class which holds the validator model.
    """
    def __init__( self, config, device, subtensor, vlogger ):
        super(nucleus, self).__init__()
        self.config = config
        self.vlogger = vlogger

        self.max_n = subtensor.max_n(netuid=self.config.netuid)

        self.device = device
        self.permute_uids = []  # iterable of next UIDs to query, reset to permuted UIDs when empty

        tokenizer = bittensor.tokenizer()
        self.pad_token = tokenizer(tokenizer.pad_token)['input_ids'][0]

        # Token embeddings project int64 tokens onto representations.
        self.token_embedding = torch.nn.Embedding( bittensor.__vocab_size__,  bittensor.__network_dim__ )
        
        # Routing encoder, projects token embeddings onto context for routing inputs.
        self.routing_encoder_layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout, batch_first=True)
        self.routing_encoder = TransformerEncoder( self.routing_encoder_layers, 1 )

        # Encoder projects response representations onto hidden units.
        self.encoder_layers = TransformerEncoderLayer( bittensor.__network_dim__, config.nucleus.nhead, config.nucleus.nhid, config.nucleus.dropout, batch_first=True)
        self.encoder = TransformerEncoder( self.encoder_layers, config.nucleus.nlayers )

        # Decoder which projects hidden unit representations on to the token dimension.
        self.decoder = torch.nn.Linear( bittensor.__network_dim__, bittensor.__vocab_size__ , bias=False)

        # Positional Encoding
        self.local_pos_encoder = PositionalEncoding( bittensor.__network_dim__, self.config.nucleus.dropout )

        # Crosss entropy loss for NTP.    
        self.loss_fct = torch.nn.CrossEntropyLoss()
    
        # SGMOE Gates: Instantiating the gates per expert.
        self.gates = torch.nn.Linear( bittensor.__network_dim__, self.max_n, bias=True ).to( self.device )

        self.sigmoid = torch.nn.Sigmoid()

        self.reset_weights()

    @classmethod
    def add_args( cls, parser ):
        parser.add_argument('--nucleus.nhid', type=int, help='the dimension of the feedforward network model in nn.TransformerEncoder', default=200 )
        parser.add_argument('--nucleus.nhead', type=int, help='the number of heads in the multiheadattention models', default = 2 )
        parser.add_argument('--nucleus.nlayers', type=int, help='the number of nn.TransformerEncoderLayer in nn.TransformerEncoder', default=2 )
        parser.add_argument('--nucleus.dropout', type=float, help='the dropout value', default=0.2)
        parser.add_argument('--nucleus.importance', type=float, help='hyperparameter for the importance loss', default=3)
        parser.add_argument('--nucleus.noise_multiplier', type=float, help='Standard deviation multipler on weights', default=2 )
        parser.add_argument('--nucleus.no_dendrite_backward', action='store_true', help='Pass backward request to the server side or not', default=False )
        parser.add_argument('--nucleus.logits_divergence', type=float, help=' the divergence value for logit anomaly detection (default value: -1, pulling from subtensor directly)', default=-1)

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        return bittensor.config( parser )

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    def reset_weights ( self ):
        r""" Resets the validator weights.
        """
        # === Resets all the weights using xavier initialization. ===
        torch.nn.init.xavier_uniform_ ( self.token_embedding.weight )
        torch.nn.init.xavier_uniform_ ( self.decoder.weight )
        torch.nn.init.xavier_uniform_( self.gates.weight )
        def init_xavier( component ):
            try:
                torch.nn.init.xavier_uniform_( component.weight )
            except: pass
        self.routing_encoder.apply( init_xavier )
        self.encoder.apply( init_xavier )
        torch.nn.init.xavier_uniform_( self.gates.weight )

    def get_loss(self, stat, text_input, call_response):
        # inputs_nxt = text_input[..., -self.config.neuron.validation_len:]  # input validation with next token target phrase [batch_size, val_len]
        # _losses_val, _losses = phrase_cross_entropy(inputs_nxt, call_response, reduce=False)
        # _losses_val[_losses_val.isnan()] = 20  # assign large loss
        # _losses[_losses.isnan()] = 20  # assign large loss
        # _loss_val = _losses_val.mean()
        # _loss = _losses.mean()
        # stat.update({'loss_val_nxt': _loss_val, 'losses_nxt': _losses, 'loss_nxt': _loss})
        stat.update({
            'loss_val_nxt': torch.rand(1)[0], 
            'losses_nxt': torch.rand(1)[0], 
            'loss_nxt': torch.rand(1)[0]
        })
        
        return stat 
    
    def build_stats(self, stats, step_status, text_input, call_responses, dendrites):
        start_time = time.time()
        for response, dendrite in zip(call_responses, dendrites):
            code = response.response_code if isinstance(response.response_code, int) else response.response_code.value[0]  
            stats[dendrite.endpoint.uid] = {
                'response_time': response.end_time - response.start_time, 
                'return_code': code  
            }
            if True or code == bittensor.proto.ReturnCode.Success:
                self.get_loss(stats[dendrite.endpoint.uid], text_input, response)

        step_status.base_loss_start_time = time.time() - start_time
        return stats, step_status 

    async def async_forward(self, text_input, dendrites):
        # Make calls.
        calls = []
        for dendrite in dendrites:
            calls.append( 
                dendrite.async_forward(
                    forward_call = bittensor.TextLastHiddenStateForwardCall(
                        text_inputs = text_input,
                        mask = torch.ones_like(text_input), 
                        timeout = bittensor.__blocktime__,
                    )
                )
            )
        
        return await asyncio.gather(*calls)

    def forward(
            self,
            stats,
            step_status,
            text_input: torch.FloatTensor,
            dendrites: 'bittensor.Dendrite',
            validation_len: int,
        ):
        try:
            loop = asyncio.get_event_loop()
        
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        start_time = time.time() 
        prune_len = self.config.neuron.prune_len  # Number of tokens to prune from each validation input sequence
        text_input = prune_tokens(text_input.to(self.device), prune_len=prune_len, margin=validation_len+3)  # prune input sequence without last validation tokens [batch_size, sequence_len]

        call_responses = loop.run_until_complete ( 
            self.async_forward(
                text_input = text_input[..., :-validation_len],
                dendrites = dendrites
            ) 
        )
        step_status.forward_start_time = time.time() - start_time

        stats, step_status = self.build_stats(stats, step_status, text_input, call_responses, dendrites)
        return stats, step_status