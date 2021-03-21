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

import argparse
import asyncio
import grpc
import math
import sys
import time
import torch
import pandas as pd
import torch.nn as nn
import traceback

from termcolor import colored
from types import SimpleNamespace
from typing import Tuple, List, Optional
from torch.autograd.function import once_differentiable
from loguru import logger
from munch import Munch

import bittensor
import bittensor.utils.stats as stat_utils
import bittensor.serialization as serialization
from bittensor.exceptions.handlers import rollbar

import multiprocessing.managers
from multiprocessing.managers import BaseManager


# dummy tensor that triggers autograd 
DUMMY = torch.empty(0, requires_grad=True)

class Dendrite(torch.autograd.Function):

    def __init__(
            self, 
            config: 'Munch' = None, 
            wallet: 'bittensor.Wallet' = None,
            **kwargs
        ):
        r""" Initializes a new Dendrite network entry point.
        """
        super().__init__()
        if config == None:
            config = Dendrite.default_config()
        print (config)
        bittensor.Config.update_with_kwargs(config.dendrite, kwargs) 
        _Dendrite.check_config(config)
        self.config = config

        if wallet == None:
            wallet = bittensor.Wallet( self.config )
        self.wallet = wallet

        # Create shared memory Dendrite.
        BaseManager.register( '_Dendrite', bittensor.dendrite._Dendrite )
        self._manager = BaseManager()
        self._manager.start()
        self._dendrite = self._manager._Dendrite( config = self.config, wallet = self.wallet )

    @staticmethod   
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        Dendrite.add_args(parser) 
        config = bittensor.Config.to_config(parser); 
        return config

    @staticmethod   
    def check_config(config: Munch):
        pass

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        bittensor.Wallet.add_args( parser )
        _Dendrite.add_args( parser )
        parser.add_argument('--dendrite.multiprocessing', default=False, type=bool, 
            help='''Dendrite object is created in shared memory and can be concurrently accessed from multiple processes.''')

        parser.add_argument('--dendrite.debug', default=False, type=bool, 
            help='''If true, request information is logged. ''')

    def __str__(self) -> str:
        return self._dendrite.toString()
    
    def __full_str__(self) -> str:
        return self._dendrite.fullToString()

    def toString(self) -> str:
        return self._dendrite.toString()

    def fullToString(self) -> str:
        return self._dendrite.fullToString()

    def getReceptors(self) -> List['bittensor.Receptor']:
        return self._dendrite.getReceptors()
    
    @staticmethod
    def forward(
            ctx, 
            dendrite: 'bittensor.Dendrite',
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
        ctx.dendrite = dendrite
        ctx.neurons, ctx.inputs, ctx.mode = neurons, inputs, mode
        inputs = [ x.clone().detach() for x in inputs]
        outputs, forward_codes, messages = ctx.dendrite._dendrite.forward(
            neurons = neurons, 
            inputs = inputs, 
            mode = mode
        )
        if ctx.dendrite.config.dendrite.debug:
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
        input_grads, codes, messages =  ctx.dendrite._dendrite.backward (
            neurons = ctx.neurons, 
            inputs = ctx.inputs, 
            grads = grads_cpu, 
            codes = ctx.forward_codes, 
            mode = ctx.mode
        )
        if ctx.dendrite.config.dendrite.debug:
            logger.info('backward messages {}', messages)
        return (None, None, None, None, *input_grads)

    def _forward(
                self,
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
                    codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                        dendrite call return ops.

                    responses (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Output encodings of inputs produced by remote neurons. Non-responses are zeroes of common shape.
            """
            forward_response = Dendrite.apply(
                self,
                DUMMY, 
                neurons, 
                mode,
                *inputs
            )
            codes = forward_response[0]
            tensors = forward_response[1:]
            return codes, tensors

    def forward_text(
                self,
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
                    codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                        dendrite call return ops.

                    responses (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                        Output encodings of inputs produced by remote neurons. Non-responses are zeroes of common shape.
            """
            if len(inputs[0].shape) != 2:
                error_msg = 'Text inputs should have rank 2 with semantic shape: [batch_size, sequence_len], got {}'.format(inputs[0].shape)
                raise ValueError(error_msg)
            if len(inputs) != len(neurons):
                error_msg = 'List of text inputs should have the same length as passed destination neurons, got {} and {}'.format(len(inputs), len(neurons))
                raise ValueError(error_msg)
            if len(inputs) < 1:
                error_msg = 'Must pass more than 0 inputs, got {}'.format(len(inputs))
                raise ValueError(error_msg)

            return self._forward (
                neurons = neurons, 
                mode = bittensor.proto.Modality.TEXT,
                inputs = inputs
            )

    def forward_image(
                self,
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
                codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                    dendrite call return ops.

                responses (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.network_size)`, `required`):
                    Output encodings of images produced by remote neurons. Non-responses are zeroes of common shape.
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

        return self._forward (
                neurons = neurons, 
                mode = bittensor.proto.Modality.TEXT,
                inputs = inputs
        )

    def forward_tensor(
                self,
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
                codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                    dendrite call return ops.

                response (:obj:`List[torch.FloatTensor]` of shape :obj:`num_neurons * (batch_size, sequence_len, bittensor.__network_dim__)]`, `required`):
                    Output encodings of tensors produced by remote neurons. Non-responses are zeroes of common shape.
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

        return self._forward (
                neurons = neurons, 
                mode = bittensor.proto.Modality.TEXT,
                inputs = inputs
        )

class _Dendrite:
    r"""
        Process Safe Dendrite object, holds RPC connections to other nodes.
    """
    def __init__(self, config: Munch = None, wallet: 'bittensor.Wallet' = None, **kwargs):
        r""" Initializes a new shared dendrite object.
            Args:
                config (:obj:`Munch`, `optional`): 
                    dendrite.Dendrite.config()
                wallet (:obj:`bittensor.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
        """
        # Config: Holds all config items for this items and those that are recursively defined. Specifically
        # config for you wallet and metagraph.
        if config == None:
            config = _Dendrite.default_config()
        _Dendrite.check_config( config )
        self.config = config

        # Wallet: Holds you hotkey keypair and coldkey pub, which can be used to sign messages 
        # and subscribe to the chain.
        if wallet == None:
            wallet = bittensor.Wallet(self.config)
        self.wallet = wallet

        # Receptors: Holds a map from publickey -> receptor objects. Receptors encapsulate a TCP connection between
        # this dendrite and an upstream neuron (i.e. a peer we call for representations)
        self.receptors = {}

        # Stats: hold statistics for this dendrite.
        self.stats = SimpleNamespace(
            qps = stat_utils.timed_rolling_avg(0.0, 0.01),
        )

    @staticmethod   
    def default_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        _Dendrite.add_args(parser) 
        config = bittensor.Config.to_config(parser); 
        return config

    @staticmethod   
    def check_config(config: Munch):
        pass

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        bittensor.Receptor.add_args(parser)
        pass

    def get_receptor_for_neuron( self, neuron: bittensor.proto.Neuron ) -> 'bittensor.Receptor':
        # ---- Find receptor or create one ---- 
        if neuron.public_key not in self.receptors:
            self.receptors[neuron.public_key] = bittensor.Receptor(
                neuron = neuron, 
                config = self.config, 
                wallet = self.wallet
            )
        return self.receptors[neuron.public_key]

    def forward(self, 
                neurons: List[bittensor.proto.Neuron],
                inputs: List[torch.Tensor],
                mode: bittensor.proto.Modality
        ) -> Tuple[List[torch.Tensor], List[int], List[str]]:
        r""" Forward tensor inputs to neurons.

            Args:
                neurons (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [shape])`, `required`):
                    List of tensors to send to corresponsing neurons. Tensors are of arbitrary type and shape depending on the
                    modality.

                mode (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                forward_outputs (:obj:`List[torch.FloatTensor]` of shape :obj:`num_neurons * (batch_size, sequence_len, bittensor.network_size)]`, `required`):
                    Output encodings of tensors produced by remote neurons. Non-responses are zeroes of common shape.

                return_codes (:obj:`List[bittensor.proto.ReturnCode]` of shape :obj:`[num_neurons]`, `required`):
                    dendrite call return ops.
        """
        # ---- Stats ---
        self.stats.qps.update(1)

        # ---- Run async calls ----
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete( self._forward_gather (
            loop = loop, 
            neurons = neurons, 
            inputs = inputs, 
            mode = mode
        ))
        loop.stop()

        # ---- Process results and return ----
        tensor_results = [res[0] for res in results]
        return_codes = [res[1] for res in results]
        messages = [res[2] for res in results]
        return tensor_results, return_codes, messages

    def backward(
                self, 
                neurons: List[bittensor.proto.Neuron],
                inputs: List[torch.Tensor],
                grads: List[torch.Tensor],
                codes: List[int],
                mode: bittensor.proto.Modality
            ) -> Tuple[List[torch.Tensor], List[int], List[str]]:
        r""" Forward tensor inputs to neurons.

            Args:
                neurons (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [shape])`, `required`):
                    List of tensors to send to corresponsing neurons. Tensors are of arbitrary type and shape depending on the
                    modality.

                grads (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [shape])`, `required`):
                    List of grad tensors to send to corresponsing inputs. 

                codes (:obj:`List[bittensor.proto.ReturnCode]` of shape :obj:`[num_neurons]`, `required`):
                    dendrite call return ops from previous forward call.

                mode (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                forward_outputs (:obj:`List[torch.FloatTensor]` of shape :obj:`num_neurons * (batch_size, sequence_len, bittensor.network_size)]`, `required`):
                    Output encodings of tensors produced by remote neurons. Non-responses are zeroes of common shape.

                return_codes (:obj:`List[bittensor.proto.ReturnCodes]` of shape :obj:`[num_neurons]`, `required`):
                    dendrite call return ops.

                messages (:obj:`List[str]` of shape :obj:`[num_neurons]`, `required`):
                    messages associated with return codes
        """
        # ---- Run async calls ----
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete( self._backward_gather (
            loop = loop, 
            neurons = neurons, 
            inputs = inputs, 
            grads = grads, 
            codes = codes, 
            mode = mode
        ))
        loop.stop()

        # ---- Process results and return ----
        tensor_results = [res[0] for res in results]
        return_codes = [res[1] for res in results]
        messages = [res[2] for res in results]
        return tensor_results, return_codes, messages

    async def _forward_gather(
            self, 
            loop: asyncio.base_events.BaseEventLoop, 
            neurons: List[bittensor.proto.Neuron],
            inputs: List[torch.Tensor],
            mode
        ) -> List[Tuple[torch.FloatTensor, int]]:
        r""" Creates and returns the results from len(neurons) torch forward requests. Uses asyncio for concurrency.

            Args:
                loop (:obj:`asyncio.base_events.BaseEventLoop`, `required`):
                    The asyncio concurrency loop to use while making the n calls.

                neurons (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [shape])`, `required`):
                    List of tensors to send to corresponsing neurons. Tensors are of arbitrary type and shape depending on the
                    modality.

                mode (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                results (:obj:`List[Tuple[torch.FloatTensor, torch.LongTensor]]`, `required`):
                    result tuples from the forward call on a Receptor class.
        """
            
        # ---- Calls to fill ---- 
        calls = []
        for (inputs_i, neuron_i) in list(zip(inputs, neurons)):
            receptor = self.get_receptor_for_neuron( neuron_i )
            calls.append( loop.run_in_executor( None, receptor.forward, 
                inputs_i, 
                mode
            ))

        # ---- Gather results and return ---- 
        results = await asyncio.gather(*calls)
        return results

    async def _backward_gather(
            self, 
            loop: asyncio.base_events.BaseEventLoop, 
            neurons: List[bittensor.proto.Neuron],
            inputs: List[torch.Tensor],
            grads: List[torch.Tensor],
            codes: List[int],
            mode) -> List[Tuple[torch.FloatTensor, int, str]]:
        r""" Creates and returns the results from len(neurons) torch forward requests. Uses asyncio for concurrency.

            Args:
                loop (:obj:`asyncio.base_events.BaseEventLoop`, `required`):
                    The asyncio concurrency loop to use while making the n calls.

                neurons (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [shape])`, `required`):
                    List of tensors to send to corresponsing neurons. Tensors are of arbitrary type and shape depending on the
                    modality.

                grads (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    List of grad-tensors to send to corresponsing neurons. 

                codes (:obj:`List[bittensor.proto.ReturnCode]` of shape :obj:`[num_neurons]`, `required`):
                    dendrite call return ops from previous forward call.

                mode (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                results (:obj:`List[Tuple[torch.FloatTensor, torch.LongTensor]]`, `required`):
                    result tuples from the forward call on a Receptor class.
        """
            
        # ---- Calls to fill ---- 
        calls = []
        for (inputs_i, grads_i, code_i, neuron_i) in list(zip(inputs, grads, codes, neurons)):

            # ---- Append async calls ----
            receptor = self.get_receptor_for_neuron( neuron_i )
            calls.append( loop.run_in_executor( None, receptor.backward, 
                inputs_i, 
                grads_i, 
                code_i, 
                mode
            ))

        # ---- Gather results and return ---- 
        results = await asyncio.gather(*calls)
        return results

    def getReceptors(self):
        return self.receptors
    
    def toString(self):
        total_bytes_out = 0
        total_bytes_in = 0
        for receptor in self.receptors.values():
            total_bytes_out += receptor.stats.forward_bytes_out.value
            total_bytes_in += receptor.stats.forward_bytes_in.value
        qps_str = colored('{:.3f}'.format(self.stats.qps.value), 'blue')
        total_in_bytes_str = colored('\u290A {:.1f}'.format((total_bytes_out*8)/1000), 'green')
        total_out_bytes_str = colored('\u290B {:.1f}'.format((total_bytes_in*8)/1000), 'red')
        return "(" + qps_str + "q/s|" + total_in_bytes_str + "/" + total_out_bytes_str + "kB/s" + ")"

    def fullToString(self):
        uids = [receptor.neuron.uid for receptor in self.receptors.values()]
        bytes_out = [receptor.stats.forward_bytes_out.value * (8/1000) for receptor in self.receptors.values()]
        bytes_in = [receptor.stats.forward_bytes_in.value * (8/1000) for receptor in self.receptors.values()]
        qps = [receptor.stats.forward_qps.value + receptor.stats.backward_qps.value for receptor in self.receptors.values()]
        rows = [bytes_out, bytes_in, qps]
        df = pd.DataFrame(rows, columns=uids)
        df = df.rename(index={df.index[0]: colored('\u290A kB/s', 'green')})
        df = df.rename(index={df.index[1]: colored('\u290B kB/s', 'red')})
        df = df.rename(index={df.index[2]: colored('Q/s', 'blue')})
        return '\nDendrite:\n' + df.to_string(max_rows=5000, max_cols=25, line_width=1000, float_format = lambda x: '%.2f' % x, col_space=1, justify='left')
       
    def toTensorboard(self, tensorboard, global_step):
        total_bytes_out = 0
        total_bytes_in = 0
        for receptor in self.receptors.values():
            total_bytes_out += receptor.stats.forward_bytes_out.value
            total_bytes_in += receptor.stats.forward_bytes_in.value
        total_in_bytes = (total_bytes_in*8)/1000
        total_out_bytes = (total_bytes_out*8)/1000
        tensorboard.add_scalar('Dendrite/Incoming bytes', total_in_bytes, global_step)
        tensorboard.add_scalar('Dendrite/Outgoing bytes', total_out_bytes, global_step)

