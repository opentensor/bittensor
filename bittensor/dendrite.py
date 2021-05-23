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
from bittensor import receptor
import copy
import grpc
import math
import torch
import pandas as pd
import torch.nn as nn
import traceback

from concurrent.futures import ThreadPoolExecutor, as_completed
from termcolor import colored
from types import SimpleNamespace
from typing import Tuple, List, Optional
from munch import Munch

import bittensor
import bittensor.utils.stats as stat_utils

from loguru import logger
logger = logger.opt(colors=True)

class Dendrite(nn.Module):
    r"""
        Creates Forward and Backward calls to other neurons on the network. It behaves like a normal torch nn.Module and is differentiable. 
    
        Messages passed through this module will be sent to other neuron objects, either remote or local, and return responses as torch tensors. 
        Gradients passing through this module on a .backward() call will trigger Backward rpc calls to the axon terminals of the downstream neurons 
        called during associated Forward operation.
    """

    def __init__(
            self, 
            config: Munch = None, 
            wallet: 'bittensor.wallet.Wallet' = None,
            receptor_pass_gradients: bool = None,
            receptor_timeout: int = None,
            receptor_do_backoff: bool = None,
            receptor_max_backoff:int = None
        ):
        r""" Initializes a new Dendrite entry point.
            Args:
                config (:obj:`Munch`, `optional`): 
                    dendrite.Dendrite.config()
                wallet (:obj:`bittensor.wallet.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                receptor_pass_gradients (default=True, type=bool)
                    Switch to true if the neuron passes gradients to downstream peers.
                        By default the backward call i.e. loss.backward() triggers passing gradients on the wire.
                receptor_timeout (default=0.5, type=float):
                    The per request RPC timeout. a.k.a the maximum request time.
                receptor_do_backoff (default=True, type=bool)
                    Neurons who return non successful return codes are
                        periodically not called with a multiplicative backoff.
                        The backoff doubles until max_backoff and then halves on ever sequential successful request.
                receptor_max_backoff (default=100, type=int)
                    The backoff doubles until this saturation point.
        """
        super().__init__()
        # Config: Holds all config items for this items and those that are recursively defined. Specifically
        # config for you wallet and metagraph.
        if config == None:
            config = Dendrite.default_config()
        config.receptor.pass_gradients = receptor_pass_gradients if receptor_pass_gradients != None else config.receptor.pass_gradients
        config.receptor.timeout = receptor_timeout if receptor_timeout != None else config.receptor.timeout
        config.receptor.do_backoff = receptor_do_backoff if receptor_do_backoff != None else config.receptor.do_backoff
        config.receptor.max_backoff = receptor_max_backoff if receptor_max_backoff != None else config.receptor.max_backoff
        Dendrite.check_config( config )
        self.config = copy.deepcopy(config)

        # Wallet: Holds you hotkey keypair and coldkey pub, which can be used to sign messages 
        # and subscribe to the chain.
        if wallet == None:
            wallet = bittensor.wallet.Wallet(self.config)
        self.wallet = wallet

        # Threadpool executor for making queries across the line.
        self._executor = ThreadPoolExecutor( max_workers = self.config.dendrite.max_worker_threads )

        # Receptors: Holds a set map of hotkey -> receptor objects. Receptors encapsulate a TCP connection between
        # this dendrite and an upstream neuron (i.e. a peer we call for representations)
        self._receptors = {}

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
    def add_args( parser: argparse.ArgumentParser ):
        bittensor.receptor.Receptor.add_args(parser)
        parser.add_argument('--dendrite.max_worker_threads', default=20, type=int, 
                help='''Max number of concurrent threads used for sending RPC requests.''')
        parser.add_argument('--dendrite.max_active_tcp_connections', default=150, type=int, 
                help='''Max number of concurrently active receptors / tcp-connections''')
        return parser

    def forward_text(self, neurons: List[bittensor.utils.neurons.NeuronEndpoint],
                     x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        r""" Forward text inputs to neurons.

            Args:
                neurons (:obj:`List[bittensor.utils.neurons.NeuronEndpoint]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [batch_size, sequence_len])`, `required`):
                    List of tensors to send to corresponsing neurons. Tensors are text input_ids encoded using the
                    bittensor tokenizer of shape [batch_size, sequence_len].

            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.__network_dim__)`, `required`):
                    Output encodings of inputs produced by remote neurons. Non-responses are zeroes of common shape.

                return_codes (:obj:`torch.LongTensor` of shape :obj:`[num_neurons]`, `required`):
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

    def forward_image(self, neurons: List[bittensor.utils.neurons.NeuronEndpoint],
                      x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        r""" Forward image inputs to neurons.

            Args:
                neurons (:obj:`List[bittensor.utils.neurons.NeuronEndpoint]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [batch_size, sequence_len, channels, rows, cols])`, `required`):
                    List of image-tensors to send to corresponsing neurons. Tensors are images encoded using the
                    torch.toTensor() or other encoding which produces the shape [batch_size, channels, rows, cols].

            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`(batch_size, sequence_len, bittensor.network_size)`, `required`):
                    Output encodings of images produced by remote neurons. Non-responses are zeroes of common shape.

                return_codes (:obj:`torch.LongTensor` of shape :obj:`[num_neurons]`, `required`):
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

    def forward_tensor(self, neurons: List[bittensor.utils.neurons.NeuronEndpoint],
                       x: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        r""" Forward tensor inputs to neurons.

            Args:
                neurons (:obj:`List[bittensor.utils.neurons.NeuronEndpoint]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [batch_size, sequence_len, bittensor.__network_dim__])`, `required`):
                    List of tensors to send to corresponsing neurons. Tensors are of arbitrary type and
                    with shape [batch_size, sequence_len, bittensor.__network_dim__].

            Returns:
                forwad_output (:obj:`List[torch.FloatTensor]` of shape :obj:`num_neurons * (batch_size, sequence_len, bittensor.__network_dim__)]`, `required`):
                    Output encodings of tensors produced by remote neurons. Non-responses are zeroes of common shape.

                return_codes (:obj:`torch.LongTensor` of shape :obj:`[num_neurons]`, `required`):
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

    def forward(self, neurons: List[bittensor.utils.neurons.NeuronEndpoint],
                x: List[torch.Tensor],
                mode: bittensor.proto.Modality) -> Tuple[List[torch.Tensor], torch.LongTensor]:
        r""" Forward tensor inputs to neurons.

            Args:
                neurons (:obj:`List[bittensor.utils.neurons.NeuronEndpoint]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                x (:obj:`List[torch.Tensor]` of shape :obj:`(num_neurons * [shape])`, `required`):
                    List of tensors to send to corresponsing neurons. Tensors are of arbitrary type and shape depending on the
                    modality.

                mode (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

            Returns:
                forward_outputs (:obj:`List[torch.FloatTensor]` of shape :obj:`num_neurons * (batch_size, sequence_len, bittensor.network_size)]`, `required`):
                    Output encodings of tensors produced by remote neurons. Non-responses are zeroes of common shape.

                return_codes (:obj:`torch.LongTensor` of shape :obj:`[num_neurons]`, `required`):
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

        # ---- Run threaded calls with executor ----
        tensor_results = []
        return_codes = []
        
        # --- Create calls ----
        def _call_receptor_with_args( receptor, inputs, mode ):
            return receptor.forward( inputs = inputs, mode = mode )

        # ---- Fill calls ----
        call_args = [ (self._get_or_create_receptor_for_neuron( neuron ), inputs, mode) for (inputs, neuron) in list(zip( x, neurons )) ]
        for result in self._executor.map( lambda args: _call_receptor_with_args(*args), call_args ):
            tensor_results.append( result[0] )
            return_codes.append( result[1] )

        # ---- Kill receptors ----
        self._destroy_receptors_over_max_allowed()
        
        # ---- Return ----
        return_codes = torch.tensor(return_codes, dtype=torch.int64)
        return tensor_results, return_codes

    def _destroy_receptors_over_max_allowed( self ):
        r""" Destroys receptors based on QPS until there are no more than max_active_tcp_connections.
        """

        # ---- Finally: Kill receptors over max allowed ----
        while len(self._receptors) > self.config.dendrite.max_active_tcp_connections:
            min_receptor_qps = math.inf
            receptor_to_remove = None
            for next_receptor in self._receptors.values():
                next_qps = next_receptor.stats.forward_qps.value
                if min_receptor_qps > next_qps:
                    receptor_to_remove = next_receptor
            if receptor_to_remove != None:
                logger.debug('<white>Destroy receptor for neuron:</white> {}', receptor_to_remove.neuron )
                del self._receptors[ receptor_to_remove.neuron.hotkey ]

    def _get_or_create_receptor_for_neuron( self, neuron: bittensor.utils.neurons.NeuronEndpoint ) -> 'bittensor.receptor.Receptor':
        r""" Finds or creates a receptor TCP connection associated with the passed Neuron Endpoint
            Returns
                receptor: (bittensor.receptor.Receptor):
                    receptor with tcp connection endpoint at neuron.ip:neuron.port
        """

        # ---- Find the active receptor for this neuron ----
        if neuron.hotkey in self._receptors:
            receptor = self._receptors[ neuron.hotkey ]

            # Change receptor address.
            if receptor.neuron.ip != neuron.ip or receptor.neuron.port != neuron.port:
                del receptor
                logger.debug('<white>Update receptor for neuron:</white> {}', neuron )
                receptor = bittensor.receptor.Receptor (
                    neuron = neuron, 
                    config = self.config, 
                    wallet = self.wallet
                )            
                self._receptors[ receptor.neuron.hotkey ] = receptor

        # ---- Or: Create a new receptor ----
        else:
            logger.debug('<white>Create receptor for neuron:</white> {}', neuron )
            receptor = bittensor.receptor.Receptor (
                    neuron = neuron, 
                    config = self.config, 
                    wallet = self.wallet
            )
            self._receptors[ receptor.neuron.hotkey ] = receptor

        return receptor

    def __del__(self):
        # Close down executor.
        self._executor.shutdown()

    def __str__(self):
        total_bytes_out = 0
        total_bytes_in = 0
        for receptor in self._receptors.values():
            total_bytes_out += receptor.stats.forward_bytes_out.value
            total_bytes_in += receptor.stats.forward_bytes_in.value
        qps_str = colored('{:.3f}'.format(self.stats.qps.value), 'blue')
        total_out_bytes_str = colored('\u290A{:.1f}'.format((total_bytes_out*8)/1000), 'green')
        total_in_bytes_str = colored('\u290B{:.1f}'.format((total_bytes_in*8)/1000), 'red')
        return "(" + qps_str + "q/s|" + total_in_bytes_str + "/" + total_out_bytes_str + "kB/s" + ")"

    def __rich__(self): 
        total_bytes_out = 0
        total_bytes_in = 0
        for receptor in self._receptors.values():
            total_bytes_out += receptor.stats.forward_bytes_out.value
            total_bytes_in += receptor.stats.forward_bytes_in.value
        total_out_bytes_str = '[green]\u290A{:.1f}[/green]'.format((total_bytes_out * 8)/1000)
        total_in_bytes_str = '[red]\u290B{:.1f}[/red]'.format((total_bytes_in * 8)/1000)
        qps_str = "[blue]{:.3f}[/blue]".format(float(self.stats.qps.value))
        return "(" + qps_str + "q/s|" + total_out_bytes_str + "/" + total_in_bytes_str + "kB/s" + ")"

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