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
from loguru import logger
from munch import Munch

import bittensor
import bittensor.utils.stats as stat_utils
import bittensor.serialization as serialization
from bittensor.exceptions.handlers import rollbar

class Dendrite(nn.Module):
    r"""
        Creates Forward and Backward calls to other neurons on the network. Maintains receptor TCP-RPC connections into the network.
    """

    def __init__(self, config: Munch = None, wallet: 'bittensor.wallet.Wallet' = None, **kwargs):
        r""" Initializes a new Dendrite entry point.
            Args:
                config (:obj:`Munch`, `optional`): 
                    dendrite.Dendrite.config()
                wallet (:obj:`bittensor.wallet.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
        """
        super().__init__()
        # Config: Holds all config items for this items and those that are recursively defined. Specifically
        # config for you wallet and metagraph.
        if config == None:
            config = Dendrite.default_config()
        Dendrite.check_config( config )
        self.config = config

        # Wallet: Holds you hotkey keypair and coldkey pub, which can be used to sign messages 
        # and subscribe to the chain.
        if wallet == None:
            wallet = bittensor.wallet.Wallet(self.config)
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
        Dendrite.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod   
    def check_config(config: Munch):
        pass

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        bittensor.receptor.Receptor.add_args(parser)
        return parser

    def forward(self, 
                neurons: List[bittensor.proto.Neuron],
                inputs: List[torch.Tensor],
                mode: bittensor.proto.Modality
        ) -> Tuple[List[torch.Tensor], List[bittensor.proto.ReturnCode]]:
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
            modes = mode
        ))
        loop.stop()

        # ---- Process results and return ----
        tensor_results = [res[0] for res in results]
        return_codes = [res[1] for res in results]
        return tensor_results, return_codes

    def backward(
                self, 
                neurons: List[bittensor.proto.Neuron],
                inputs: List[torch.Tensor],
                grads: List[torch.Tensor],
                codes: List[bittensor.proto.ReturnCode],
                mode: bittensor.proto.Modality
            ) -> Tuple[List[torch.Tensor], List[bittensor.proto.ReturnCode]]:
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

                return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                    dendrite call return ops.
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
        return tensor_results, return_codes

    async def _forward_gather(
            self, 
            loop: asyncio.base_events.BaseEventLoop, 
            neurons: List[bittensor.proto.Neuron],
            inputs: List[torch.Tensor],
            mode) -> List[Tuple[torch.FloatTensor, torch.LongTensor]]:
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

            # ---- Find receptor or create one ---- 
            if neuron_i.public_key not in self.receptors:
                self.receptors[neuron_i.public_key] = bittensor.receptor.Receptor(
                    neuron = neuron_i, 
                    config = self.config, 
                    wallet = self.wallet
                )
            receptor = self.receptors[neuron_i.public_key]

            # ---- Append async calls ---- 
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
            codes: List[bittensor.proto.ReturnCode],
            mode) -> List[Tuple[torch.FloatTensor, torch.LongTensor]]:
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

            # ---- Find receptor or create one ---- 
            if neuron_i.public_key not in self.receptors:
                self.receptors[neuron_i.public_key] = bittensor.receptor.Receptor(
                    neuron = neuron_i, 
                    config = self.config, 
                    wallet = self.wallet
                )
            receptor = self.receptors[neuron_i.public_key]

            # ---- Append async calls ---- 
            calls.append( loop.run_in_executor( None, receptor.backward, 
                inputs_i, 
                grads_i, 
                code_i, 
                mode
            ))

        # ---- Gather results and return ---- 
        results = await asyncio.gather(*calls)
        return results
    
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

