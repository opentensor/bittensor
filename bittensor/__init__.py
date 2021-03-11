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


import sys
import random
import torch
from loguru import logger
from typing import Tuple, List, Any, Optional
from torch.autograd.function import once_differentiable

import bittensor.bittensor_pb2 as proto
import bittensor.bittensor_pb2_grpc as grpc

# Bittensor code and protocol version.
__version__ = '1.0.4'

# Tensor dimension.
# NOTE (const): if/when this increases peers must be responsible for trimming or expanding output to this size.
__network_dim__ = 512 # All network responses have shape = [ __batch_size__, __sequence_dim__, __network_dim__ ]

# Substrate chain block time (seconds).
__blocktime__ = 6

# Load components.
from bittensor.axon import Axon as Axon
from bittensor.config import Config as Config
from bittensor.executor import Executor as Executor
from bittensor.dendrite import Dendrite as Dendrite
from bittensor.metagraph import Metagraph as Metagraph
from bittensor.metagraph import ChainState as ChainState
from bittensor.metagraph import TorchChainState as TorchChainState
from bittensor.nucleus import Nucleus as Nucleus
from bittensor.receptor import Receptor as Receptor
from bittensor.subtensor import Subtensor as Subtensor
from bittensor.wallet import Wallet as Wallet
import bittensor.substrate

# Create instance components
# TODO(const) these should be protected with a warning if they do not exist
neuron = None
metagraph = None
dendrite = None
axon = None
subtensor = None

# Default logger
logger_config = {
    "handlers": [{
        "sink":
            sys.stdout,
        "format":
            "<level>{level: <8}</level>|<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    }]
}
logger.configure(**logger_config)

# Tokenizer
# NOTE (const): tokenizers are guaranteed to improve and expand as time progresses. We version the tokenizer here.
# neurons must be aware that versions will increase and be ready to convert between tokenizers.
# TODO (const): Add functionality to allow tokenizer conversion. i.e. for input token conversion.
__vocab_size__ = (50278 + 100)  # Must match the __tokenizer__() vocab size.
def __tokenizer__(  version = __version__ ):
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=False)
    tokenizer.padding_side = "left"
    tokenizer.add_prefix_space = False
    tokenizer.add_special_tokens({'bos_token': "[BOS]"}) # A special token representing the beginning of a sentence.
    tokenizer.add_special_tokens({'eos_token': "[EOS]"}) # A special token representing the end of a sentence.
    tokenizer.add_special_tokens({'unk_token': "[UNK]"}) # A special token representing an out-of-vocabulary token.
    tokenizer.add_special_tokens({'sep_token': "[SEP]"}) # A special token separating two different sentences in the same input (used by BERT for instance)
    tokenizer.add_special_tokens({'pad_token': "[PAD]"}) # A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation.
    tokenizer.add_special_tokens({'cls_token': "[CLS]"}) # A special token representing the class of the input (used by BERT for instance).
    tokenizer.add_special_tokens({'mask_token': "[MASK]"}) # A special token representing a masked token (used by masked-language modeling pretraining objectives, like BERT).
    additional_special_tokens = [
        "<s>NOTUSED",  # Used by BARThez
        "</s>NOTUSED", # Used by BARThez
        "<eop>", # Used by MarianMT
        "<eod>", # Used by MarianMT
        "<formula>", # Used by Transformer XL
        "<mask_1>" # Used by Pegasus
        "<special0>", # Used by XLM
        "<special1>", # Used by XLM
        "<special2>", # Used by XLM
        "<special3>", # Used by XLM
        "<special4>", # Used by XLM
        "<special5>", # Used by XLM
        "<special6>", # Used by XLM
        "<special7>", # Used by XLM
        "<special8>", # Used by XLM
        "<special9>", # Used by XLM
    ]
    tokenizer.additional_special_tokens = additional_special_tokens
    global __vocab_size__
    __vocab_size__ = len(tokenizer) + len(additional_special_tokens) + 100 # Plus 100 for eventual token size increase.

    return tokenizer

# Hardcoded entry point nodes. 
__akira_entrypoints__ = [
    '104.248.52.148:9944',
    '142.93.194.110:9944',
    '162.243.175.73:9944',
    '165.227.92.237:9944',
    '167.172.141.223:9944',
    '174.138.32.166:9944',
    '206.189.194.236:9944',
    '68.183.130.145:9944',
    '68.183.140.221:9944',
    '68.183.140.251:9944'
]
__kusanagi_entrypoints__ = [
    '142.93.203.149:9944',
    '157.230.11.1:9944',
    '157.230.11.116:9944',
    '157.230.11.31:9944',
    '157.230.11.36:9944',
    '157.230.11.53:9944',
    '157.230.3.108:9944',
    '159.65.236.189:9944',
    '165.227.81.42:9944',
    '206.189.207.173:9944'
]
__boltzmann_entrypoints__ = [
    'feynman.boltzmann.bittensor.com:9944',
    '157.230.223.68:9944'
]
__local_entrypoints__ = [
    '127.0.0.1:9944'
]

import argparse
import json
import os
import re
import stat
import types
import traceback as tb

from io import StringIO
from munch import Munch
from termcolor import colored
from loguru import logger

from multiprocessing import Process, Manager
import multiprocessing.managers
from multiprocessing.managers import BaseManager, NamespaceProxy, BaseProxy, AutoProxy
import time
import types

def AutoProxy(token, serializer, manager=None, authkey=None,
              exposed=None, incref=True, manager_owned=False):
    '''
    Return an auto-proxy for `token`
    '''
    _Client = multiprocessing.managers.listener_client[serializer][1]

    if exposed is None:
        conn = _Client(token.address, authkey=authkey)
        try:
            exposed = dispatch(conn, None, 'get_methods', (token,))
        finally:
            conn.close()

    if authkey is None and manager is not None:
        authkey = manager._authkey
    if authkey is None:
        authkey = multiprocessing.process.current_process().authkey

    ProxyType = multiprocessing.managers.MakeProxyType('AutoProxy[%s]' % token.typeid, exposed)
    proxy = ProxyType(token, serializer, manager=manager, authkey=authkey,
                      incref=incref, manager_owned=manager_owned)
    proxy._isauto = True
    return proxy
multiprocessing.managers.AutoProxy = AutoProxy

class Neuron:

    def __init__( self, config: Munch = None, wallet: 'bittensor.Wallet' = None, **kwargs ):
        if config == None:
            config = Neuron.default_config()
        bittensor.Config.update_with_kwargs(config.neuron, kwargs) 
        Neuron.check_config(config)
        self.config = config

        if wallet == None:
            wallet = bittensor.Wallet ( config )
        else:
            config.wallet = wallet.config.wallet
        self.wallet = wallet
        print ( bittensor.Config.toString(config) )
        
        if self.config.neuron.multiprocessing:
            BaseManager.register('Subtensor', bittensor.Subtensor)
            BaseManager.register('Metagraph', bittensor.Metagraph)
            BaseManager.register('Dendrite', bittensor.Dendrite)
            BaseManager.register('Axon', bittensor.Axon)
            manager = BaseManager()
            manager.start()

            self.subtensor = manager.Subtensor( config = self.config, wallet = self.wallet )
            self.metagraph = manager.Metagraph( config = self.config, wallet = self.wallet )
            self.dendrite = manager.Dendrite( config = self.config, walelt = self.wallet )
            self.axon = manager.Axon( config = self.config, wallet = self.wallet )
        else:
            self.subtensor = bittensor.Subtensor( config = self.config, wallet = self.wallet )
            self.metagraph = bittensor.Metagraph( config = self.config, wallet = self.wallet )
            self.dendrite = bittensor.Dendrite( config = self.config, walelt = self.wallet )
            self.axon = bittensor.Axon( config = self.config, wallet = self.wallet )

    @staticmethod       
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        Neuron.add_args(parser) 
        config = bittensor.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        bittensor.Wallet.add_args( parser )
        bittensor.Subtensor.add_args( parser )
        bittensor.Metagraph.add_args( parser )
        bittensor.Axon.add_args(parser)
        bittensor.Dendrite.add_args( parser )
        try:
            parser.add_argument('--neuron.modality', default=0, type=int, 
                                help='''Neuron network modality. TEXT=0, IMAGE=1. Currently only allowed TEXT''')
            parser.add_argument('--neuron.multiprocessing', default=True, type=bool, 
                                help='''Are bittensor components process safe objects or run from a single thread.''')
            parser.add_argument('--neuron.debug', default=False, type=bool, 
                                help='''Are bittensor components process safe objects or run from a single thread.''')
        except:
            pass

    @staticmethod   
    def check_config(config: Munch):
        bittensor.Axon.check_config( config )
        bittensor.Subtensor.check_config( config )
        bittensor.Metagraph.check_config( config )
        bittensor.Dendrite.check_config( config )
        assert config.neuron.modality == bittensor.proto.Modality.TEXT, 'Only TEXT modalities are allowed at this time.'

def init( config: Munch = None,  wallet: 'bittensor.Wallet' = None, **kwargs ):
    global neuron
    global subtensor
    global metagraph
    global dendrite
    global axon
    neuron = Neuron(config = config, wallet = wallet, **kwargs)
    axon = neuron.axon
    metagraph = neuron.metagraph
    dendrite = neuron.dendrite
    subtensor = neuron.subtensor

# dummy tensor that triggers autograd in a RemoteExpert
DUMMY = torch.empty(0, requires_grad=True)

class _ForwardCall(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, 
            dummy: torch.Tensor, 
            neurons: List[bittensor.proto.Neuron], 
            mode: bittensor.proto.Modality,
            *inputs: torch.Tensor
        ) -> Tuple[ torch.Tensor, ... ] :
        """ Internal autograd-friendly Forward RPC call to a remote neurons.

            Args:
                ctx: (:obj:`torch.autograd.ctx`, `required`):
                    Autograd context, saves state information between forward and backward calls. i.e. inputs for gradient computation.

                dummy: (:obj:`torch.Tensor`, `required`):
                    Dummy torch tensor used to ensure that torch.backward computation is called on this function 
                    regardless of the input types.

                neurons (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(shape)`, `required`):
                    List of remote neurons which match length of inputs. Tensors inputs are sent forward to these neurons.

                mode (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    List of torch tensors to be sent to the associated endpoint neurons.

            Returns:
                codes (:obj:`torch.LongTensor, `required`):
                    Return code associated with forward call.
                
                outputs (:obj:`List[torch.FloatTensor]`, torch.LongTensor]`, `required`):
                    Results from each endpoint.
        """
        ctx.neurons, ctx.inputs, ctx.mode = neurons, inputs, mode
        inputs = [ x.clone().detach() for x in inputs]
        outputs, forward_codes, messages = neuron.dendrite.forward(
            neurons = neurons, 
            inputs = inputs, 
            mode = mode
        )
        if neuron.config.neuron.debug:
            logger.info('forward messages {}', messages)
        ctx.forward_codes = forward_codes
        return (torch.tensor(forward_codes, dtype=torch.int64), *outputs)

    @staticmethod
    @once_differentiable
    def backward( 
            ctx, 
            code_grads: torch.FloatTensor,
            *output_grads: torch.FloatTensor
        ) -> Tuple[ Optional[torch.Tensor], ... ]:
        """ Internal autograd-friendly Backward RPC call to a remote neurons.

            Args:
                ctx: (:obj:`torch.autograd.ctx`, `required`):
                    Autograd context, saves state information between forward and backward calls. i.e. inputs for gradient computation.
  
                grads (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Gradients of this function's outputs computed during the loss.backward() call.
            
            Returns:
                DUMMY, None, None,
                outputs (:obj:`List[torch.FloatTensor], `optional`):
                    Gradient results for each input.

        """
        grads_cpu = [ x.clone().detach() for x in output_grads ]
        input_grads, codes, messages = neuron.dendrite.backward (
            neurons = ctx.neurons, 
            inputs = ctx.inputs, 
            grads = grads_cpu, 
            codes = ctx.forward_codes, 
            mode = ctx.mode
        )
        if neuron.config.neuron.debug:
            logger.info('backward messages {}', messages)
        return (None, None, None, *input_grads)

def _internal_forward(
            neurons: List[bittensor.proto.Neuron],
            inputs: List[torch.Tensor],
            mode: bittensor.proto.Modality
        ) -> Tuple[List[torch.Tensor], torch.LongTensor]:
        r""" Internal Forward tensor inputs to neurons.

            Args:
                neurons (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of inputs. Tensors from inputs are sent forward to these neurons.

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [shape])`, `required`):
                    List of tensors to send to corresponsing neurons. Tensors are of arbitrary type and shape depending on the
                    modality.

                mode (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                forward_outputs (:obj:`List[torch.FloatTensor]` of shape :obj:`num_neurons * (batch_size, sequence_len, bittensor.network_size)]`, `required`):
                    Output encodings of tensors produced by remote neurons. Non-responses are zeroes of common shape.

                return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                    dendrite call return ops.
        """
        forward_response = _ForwardCall.apply(
                DUMMY, 
                neurons, 
                mode,
                *inputs
            )
        codes = forward_response[0]
        tensors = forward_response[1:]
        return codes, tensors

def forward_text(
            neurons: List[bittensor.proto.Neuron],
            inputs: List[torch.Tensor]
        ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        r""" Forward text inputs to neurons.

            Args:
                neurons (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from inputs are sent forward to these neurons.

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [batch_size, sequence_len])`, `required`):
                    List of tensors to send to corresponsing neurons. Tensors are text input_ids encoded using the
                    bittensor tokenizer of shape [batch_size, sequence_len].

            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                    Output encodings of inputs produced by remote neurons. Non-responses are zeroes of common shape.

                return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                    dendrite call return ops.
        """
        if len(inputs[0].shape) != 2:
            error_msg = 'Text inputs should rank 2 with semantic shape: [batch_size, sequence_len]'
            raise ValueError(error_msg)
        if len(inputs) != len(neurons):
            error_msg = 'List of text inputs should have the same length as passed destination neurons, got {} and {}'.format(len(inputs), len(neurons))
            raise ValueError(error_msg)
        if len(inputs) < 1:
            error_msg = 'Must pass more than 0 inputs, got {}'.format(len(inputs))
            raise ValueError(error_msg)

        return _internal_forward (
            neurons = neurons, 
            mode = bittensor.proto.Modality.TEXT,
            inputs = inputs
        )

def forward_image(
            neurons: List[bittensor.proto.Neuron],
            inputs: List[torch.Tensor]
        ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    r""" Forward image inputs to neurons.

        Args:
            neurons (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(num_neurons)`, `required`):
                List of remote neurons which match length of x. Tensors from inputs are sent forward to these neurons.

            inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [batch_size, sequence_len, channels, rows, cols])`, `required`):
                List of image-tensors to send to corresponsing neurons. Tensors are images encoded using the
                torch.toTensor() or other encoding which produces the shape [batch_size, channels, rows, cols].

        Returns:
            forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.network_size)`, `required`):
                Output encodings of images produced by remote neurons. Non-responses are zeroes of common shape.

            return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                dendrite call return ops.
    """
    # TODO(const): Checks across all tensors and other shape checks.
    if len(inputs[0].shape) != 5:
        error_msg = 'Image inputs should be rank 5 with semantic shape: [batch_size, sequence_dim, channels, rows, cols]'
        raise ValueError(error_msg)
    if len(inputs) != len(neurons):
        error_msg = 'List of image inputs should have the same length as passed destination neurons, got {} and {}'.format(len(inputs), len(neurons))
        raise ValueError(error_msg)
    if len(inputs) < 1:
        error_msg = 'Must pass more than 0 inputs, got {}'.format(len(inputs))
        raise ValueError(error_msg)

    return _internal_forward (
            neurons = neurons, 
            mode = bittensor.proto.Modality.TEXT,
            inputs = inputs
    )

def forward_tensor(
            neurons: List[bittensor.proto.Neuron],
            inputs: List[torch.Tensor]
        ) -> Tuple[List[torch.Tensor], torch.Tensor]:
    r""" Forward tensor inputs to neurons.

        Args:
            neurons (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(num_neurons)`, `required`):
                List of remote neurons which match length of x. Tensors from inputs are sent forward to these neurons.

            inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [batch_size, sequence_len, bittensor.__network_dim__])`, `required`):
                List of tensors to send to corresponsing neurons. Tensors are of arbitrary type and
                with shape [batch_size, sequence_len, bittensor.__network_dim__].

        Returns:
            forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`num_neurons * (batch_size, sequence_len, bittensor.__network_dim__)]`, `required`):
                Output encodings of tensors produced by remote neurons. Non-responses are zeroes of common shape.

            return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                dendrite call return ops.
    """
    if len(inputs[0].shape) != 3:
        error_msg = 'Tensor inputs should be rank 3 with semantic shape: [batch_size, sequence_len, feature_len]'
        raise ValueError(error_msg)
    if len(inputs) != len(neurons):
        error_msg = 'List of tensor inputs should have the same length as passed destination neurons, got {} and {}'.format(len(inputs), len(neurons))
        raise ValueError(error_msg)
    if inputs[0].shape[2] != bittensor.__network_dim__:
        error_msg = 'Passed tensors must have last dimension {} got {}'.format(bittensor.__network_dim__, inputs[0].shape[2])
        raise ValueError(error_msg)
    if len(inputs) == 0:
        error_msg = 'Must pass more than 0 inputs, got {}'.format(len(inputs))
        raise ValueError(error_msg)

    return _internal_forward (
            neurons = neurons, 
            mode = bittensor.proto.Modality.TEXT,
            inputs = inputs
    )