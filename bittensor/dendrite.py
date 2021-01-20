'''
The MIT License (MIT)
Copyright © 2021 Opentensor.ai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
'''
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
        Creates Forward and Backward calls to other neurons on the network. It behaves like a normal torch nn.Module and is differentiable. 
    
        Messages passed through this module will be sent to other neuron objects, either remote or local, and return responses as torch tensors. 
        Gradients passing through this module on a .backward() call will trigger Backward rpc calls to the axon terminals of the downstream neurons 
        called during associated Forward operation.
    """

    def __init__(self, config: Munch = None, wallet: 'bittensor.wallet.Wallet' = None, metagraph: 'bittensor.metagraph.Metagraph' = None):
        r""" Initializes a new Dendrite entry point.
            Args:
                config (:obj:`Munch`, `optional`): 
                    dendrite.Dendrite.config()
                wallet (:obj:`bittensor.nucleus.Nucleus`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                metagraph (:obj:`bittensor.metagraph.Metagraph`, `optional`):
                    bittensor network metagraph.
        """
        super().__init__()
        # Config: Holds all config items for this items and those that are recursively defined. Specifically
        # config for you wallet and metagraph.
        if config == None:
            config = Dendrite.build_config()
        self.config = config

        # Wallet: Holds you hotkey keypair and coldkey pub, which can be used to sign messages 
        # and subscribe to the chain.
        if wallet == None:
            wallet = bittensor.wallet.Wallet(self.config)
        self.wallet = wallet

        # Metagraph: Maintains a connection to the subtensor chain and can be queried for the latest state.
        if metagraph == None:
            metagraph = bittensor.metagraph.Metagraph(self.config, self.wallet)
        self.metagraph = metagraph

        # Receptors: Holds a set map of publickey -> receptor objects. Receptors encapsulate a TCP connection between
        # this dendrite and an upstream neuron (i.e. a peer we call for representations)
        self._receptors = {}

        # Stats: hold statistics for this dendrite.
        self.stats = SimpleNamespace(
            qps = stat_utils.timed_rolling_avg(0.0, 0.01),
        )

    def forward_text(self, neurons: List[bittensor.proto.Neuron],
                     x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        r""" Forward text inputs to neurons.

            Args:
                neurons (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [batch_size, sequence_len])`, `required`):
                    List of tensors to send to corresponsing neurons. Tensors are text input_ids encoded using the
                    bittensor tokenizer of shape [batch_size, sequence_len].

            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                    Output encodings of inputs produced by remote neurons. Non-responses are zeroes of common shape.

                return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                    dendrite call return ops.
        """
        if len(x[0].shape) != 2:
            error_msg = 'Text inputs should rank 2 with semantic shape: [batch_size, sequence_len]'
            raise ValueError(error_msg)
        if len(x) != len(neurons):
            error_msg = 'List of text inputs x should have the same length as passed destination neurons, got {} and {}'.format(len(x), len(neurons))
            raise ValueError(error_msg)
        if len(x) < 1:
            error_msg = 'Must pass more than 0 input for argument x, got {}'.format(len(x))
            raise ValueError(error_msg)
        return self.forward(neurons, x, bittensor.proto.Modality.TEXT)

    def forward_image(self, neurons: List[bittensor.proto.Neuron],
                      x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        r""" Forward image inputs to neurons.

            Args:
                neurons (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [batch_size, sequence_len, channels, rows, cols])`, `required`):
                    List of image-tensors to send to corresponsing neurons. Tensors are images encoded using the
                    torch.toTensor() or other encoding which produces the shape [batch_size, channels, rows, cols].

            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.network_size)`, `required`):
                    Output encodings of images produced by remote neurons. Non-responses are zeroes of common shape.

                return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                    dendrite call return ops.
        """
        # TODO(const): Checks across all tensors and other shape checks.
        if len(x[0].shape) != 5:
            error_msg = 'Image inputs should be rank 5 with semantic shape: [batch_size, sequence_dim, channels, rows, cols]'
            raise ValueError(error_msg)
        if len(x) != len(neurons):
            error_msg = 'List of image inputs x should have the same length as passed destination neurons, got {} and {}'.format(len(x), len(neurons))
            raise ValueError(error_msg)
        if len(x) < 1:
            error_msg = 'Must pass more than 0 input for argument x, got {}'.format(len(x))
            raise ValueError(error_msg)
        return self.forward(neurons, x, bittensor.proto.Modality.IMAGE)

    def forward_tensor(self, neurons: List[bittensor.proto.Neuron],
                       x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        r""" Forward tensor inputs to neurons.

            Args:
                neurons (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [batch_size, sequence_len, bittensor.__network_dim__])`, `required`):
                    List of tensors to send to corresponsing neurons. Tensors are of arbitrary type and
                    with shape [batch_size, sequence_len, bittensor.__network_dim__].

            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`num_neurons * (batch_size, sequence_len, bittensor.__network_dim__)]`, `required`):
                    Output encodings of tensors produced by remote neurons. Non-responses are zeroes of common shape.

                return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                    dendrite call return ops.
        """
        if len(x[0].shape) != 3:
            error_msg = 'Tensor inputs should be rank 3 with semantic shape: [batch_size, sequence_len, feature_len]'
            raise ValueError(error_msg)
        if len(x) != len(neurons):
            error_msg = 'List of tensor inputs x should have the same length as passed destination neurons, got {} and {}'.format(len(x), len(neurons))
            raise ValueError(error_msg)
        if x[0].shape[2] != bittensor.__network_dim__:
            error_msg = 'Passed tensor must have last dimension {} got {}'.format(bittensor.__network_dim__, x[0].shape[2])
            raise ValueError(error_msg)
        if len(x) == 0:
            error_msg = 'Must pass more than 0 input for argument x, got {}'.format(len(x))
            raise ValueError(error_msg)
        return self.forward(neurons, x, bittensor.proto.Modality.TENSOR)

    def forward(self, neurons: List[bittensor.proto.Neuron],
                x: List[torch.Tensor],
                mode: bittensor.proto.Modality) -> Tuple[List[torch.Tensor], torch.LongTensor]:
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

                return_codes (:obj:`List[torch.LongTensor]` of shape :obj:`[num_neurons]`, `required`):
                    dendrite call return ops.
        """
        if len(x) != len(neurons):
            error_msg = 'List of inputs x should have the same length as passed destination neurons, got {} and {}'.format(len(x), len(neurons))
            raise ValueError(error_msg)
        if len(x) < 1:
            error_msg = 'Must pass more than 0 input for argument x, got {}'.format(len(x))
            raise ValueError(error_msg)

        # ---- Stats ---
        self.stats.qps.update(1)

        # ---- Run async calls ----
        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(self._gather(loop, x, neurons, mode))
        loop.stop()

        # ---- Process results and return ----
        tensor_results = [res[0] for res in results]
        return_codes = torch.tensor([res[1] for res in results])
        return tensor_results, return_codes

    async def _gather(self, loop: asyncio.base_events.BaseEventLoop, inputs, neurons, mode) -> List[Tuple[torch.FloatTensor, torch.LongTensor]]:
        r""" Creates and returns the results from len(neurons) torch forward requests. Uses asyncio for concurrency.

            Args:
                loop (:obj:`asyncio.base_events.BaseEventLoop`, `required`):
                    The asyncio concurrency loop to use while making the n calls.

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [shape])`, `required`):
                    List of tensors to send to corresponsing neurons. Tensors are of arbitrary type and shape depending on the
                    modality.

                neurons (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

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
            if neuron_i.public_key not in self._receptors:
                self._receptors[neuron_i.public_key] = bittensor.receptor.Receptor(neuron_i, self.config, self.wallet)
            receptor = self._receptors[neuron_i.public_key]

            # ---- Append async calls ---- 
            calls.append( loop.run_in_executor(None, receptor.forward, inputs_i, mode) )

        # ---- Gather results and return ---- 
        results = await asyncio.gather(*calls)
        return results


    @staticmethod   
    def build_config() -> Munch:
        parser = argparse.ArgumentParser(); 
        Dendrite.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        Dendrite.check_config(config)
        return config

    @staticmethod   
    def check_config(config: Munch):
        bittensor.metagraph.Metagraph.check_config(config) # Also checks wallet
        bittensor.receptor.Receptor.check_config(config)

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        bittensor.metagraph.Metagraph.add_args(parser) # Also adds for wallet.
        bittensor.receptor.Receptor.add_args(parser)
        return parser

    def __str__(self):
        total_bytes_out = 0
        total_bytes_in = 0
        for receptor in self._receptors.values():
            total_bytes_out += receptor.stats.forward_bytes_out.value
            total_bytes_in += receptor.stats.forward_bytes_in.value
        qps_str = colored('{:.3f}'.format(self.stats.qps.value), 'blue')
        total_in_bytes_str = colored('\u290A {:.1f}'.format((total_bytes_out*8)/1000), 'green')
        total_out_bytes_str = colored('\u290B {:.1f}'.format((total_bytes_in*8)/1000), 'red')
        return "(" + qps_str + "q/s|" + total_in_bytes_str + "/" + total_out_bytes_str + "kB/s" + ")"

    def __full_str__(self):
        uids = [receptor.neuron.uid for receptor in self._receptors.values()]
        bytes_out = [receptor.stats.forward_bytes_out.value * (8/1000) for receptor in self._receptors.values()]
        bytes_in = [receptor.stats.forward_bytes_in.value * (8/1000) for receptor in self._receptors.values()]
        qps = [receptor.stats.forward_qps.value + receptor.stats.backward_qps.value for receptor in self._receptors.values()]
        rows = [bytes_out, bytes_in, qps]
        df = pd.DataFrame(rows, columns=uids)
        df = df.rename(index={df.index[0]: colored('\u290A kB/s', 'green')})
        df = df.rename(index={df.index[1]: colored('\u290B kB/s', 'red')})
        df = df.rename(index={df.index[2]: colored('Q/s', 'blue')})
        return '\nDendrite:\n' + df.to_string(max_rows=5000, max_cols=25, line_width=1000, float_format = lambda x: '%.2f' % x, col_space=1, justify='left')
    
    def __to_tensorboard__(self, tensorboard, global_step):
        total_bytes_out = 0
        total_bytes_in = 0
        for receptor in self._receptors.values():
            total_bytes_out += receptor.stats.forward_bytes_out.value
            total_bytes_in += receptor.stats.forward_bytes_in.value
        total_in_bytes = (total_bytes_in*8)/1000
        total_out_bytes = (total_bytes_out*8)/1000
        tensorboard.add_scalar('Dendrite/Incoming bytes', total_in_bytes, global_step)
        tensorboard.add_scalar('Dendrite/Outgoing bytes', total_out_bytes, global_step)

    @property
    def receptors(self):
        return self._receptors.values()