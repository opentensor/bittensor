""" Manages a pool of grpc connections as receptors
"""
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

import math
from typing import Tuple, List, Union
from threading import Lock

import torch
import asyncio
from loguru import logger
import concurrent
import bittensor
from bittensor._endpoint import endpoint
import bittensor.utils.networking as net
from concurrent.futures import ThreadPoolExecutor

logger = logger.opt(colors=True)

class ReceptorPool ( torch.nn.Module ):
    """ Manages a pool of grpc connections as receptors
    """
    def __init__(
        self, 
        wallet: 'bittensor.Wallet',
        max_active_receptors: int,
        compression: str,
    ):
        super().__init__()
        self.wallet = wallet
        self.max_active_receptors = max_active_receptors
        self.receptors = {}
        self.cull_mutex = Lock()
        self.max_processes = 10
        self.compression = compression
        self.total_requests = 0

        try:
            self.external_ip = str(net.get_external_ip())
        except Exception:
            self.external_ip = None

    def __str__(self):
        return "ReceptorPool({},{})".format(len(self.receptors), self.max_active_receptors)

    def __repr__(self):
        return self.__str__()
    
    def __exit__(self):
        for receptor in self.receptors:
            receptor.__del__()

    def get_total_requests(self):
        return self.total_requests
    def get_receptors_state(self):
        r""" Return the state of each receptor.
            Returns:
                states (:obj:`List[grpc.channel.state]`)
                    The state of receptor.
        """
        return {hotkey: v.state() for hotkey, v in self.receptors.items()}

    def forward (
            self, 
            endpoints: List [ 'bittensor.Endpoint' ],
            synapses: List[ 'bittensor.Synapse' ],
            inputs: List [ torch.Tensor ],
            timeout: int,
        ) -> Tuple[List[torch.Tensor], List[int], List[float]]:
        r""" Forward tensor inputs to endpoints.

            Args:
                endpoints (:obj:`List[ bittensor.Endpoint ]` of shape :obj:`(num_endpoints)`, `required`):
                    List of remote endpoints which match length of inputs. Tensors from x are sent forward to these endpoints.

                synapses (:obj:`List[ 'bittensor.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Bittensor synapse objects with arguments. Each corresponds to a synapse function on the axon.
                    Responses are packed in this ordering. 

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    TODO(const): Allow multiple tensors.
                    List of tensors to send to corresponsing endpoints. Tensors are of arbitrary type and shape depending on the
                    modality.

                timeout (int):
                    Request timeout.

            Returns:
                forward_outputs (:obj:`List[ List[ torch.FloatTensor ]]` of shape :obj:`(num_endpoints * (num_synapses * (shape)))`, `required`):
                    Output encodings of tensors produced by remote endpoints. Non-responses are zeroes of common shape.

                forward_codes (:obj:`List[ List[bittensor.proto.ReturnCodes] ]` of shape :obj:`(num_endpoints * ( num_synapses ))`, `required`):
                    dendrite backward call return ops.

                forward_times (:obj:`List[ List [float] ]` of shape :obj:`(num_endpoints * ( num_synapses ))`, `required`):
                    dendrite backward call times
        """
        if len(endpoints) != len(inputs):
            raise ValueError('Endpoints must have the same length as passed inputs. Got {} and {}'.format(len(endpoints), len(inputs)))
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete ( 
            self.async_forward(
                endpoints = endpoints,
                synapses = synapses,
                inputs = inputs,
                timeout = timeout
            ) 
        )


    def backward(
                self, 
                endpoints: List [ 'bittensor.Endpoint' ],
                synapses: List[ 'bittensor.Synapse' ],
                inputs: List [ torch.Tensor ],
                grads: List [ List[ torch.FloatTensor ] ],
                timeout: int
            ) -> Tuple[List[torch.Tensor], List[int], List[float]]:
        r""" Backward tensor inputs to endpoints.

            Args:
                endpoints (:obj:`List['bittensor.Endpoint']` of shape :obj:`(num_endpoints)`, `required`):
                    List of remote endpoints which match length of x. Tensors from x are sent backward to these endpoints.

                synapses (:obj:`List[ 'bittensor.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Bittensor synapse objects with arguments. Each corresponds to a synapse function on the axon.
                    Responses are packed in this ordering. 

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    List of tensors to send to corresponsing endpoints. Tensors are of arbitrary type and shape depending on the
                    synapse.

                grads (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    List of list of grad tensors where each grad corresponds to a synapse call on an endpoint.
                
                timeout (int):
                    request timeout.

            Returns:
                backward_outputs (:obj:`List[ List[ torch.FloatTensor] ]` of shape :obj:`num_endpoints * (batch_size, sequence_len, -1)]`, `required`):
                    Gradients returned from the backward call one per endpoint.

                backward_codes (:obj:`List[ List[ bittensor.proto.ReturnCodes ] ]` of shape :obj:`(num_endpoints)`, `required`):
                    List of list of Backward call return ops, one per endpoint and synapse.

                backward_times (:obj:`List[float]` of shape :obj:`(num_endpoints)`, `required`):
                    List of list of Backward call times one per endpoint and synapse.
        """
        if len(endpoints) != len(inputs):
            raise ValueError('Endpoints must have the same length as passed inputs. Got {} and {}'.format(len(endpoints), len(inputs)))
        if len(endpoints) != len(grads):
            raise ValueError('Endpoints must have the same length as passed grads_dy. Got {} and {}'.format(len(endpoints), len(grads)))
        for grads_per_synapse in grads:
            if len(grads_per_synapse) != len(synapses):
                raise ValueError('Gradients must have the same length as passed synapses. Got {} and {}'.format(len(grads_per_synapse), len(synapses)))
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete (
            self.async_backward(
                endpoints = endpoints,
                synapses = synapses,
                inputs = inputs,
                grads = grads,
                timeout = timeout
            ) 
        )

    async def async_forward (
            self, 
            endpoints: List [ 'bittensor.Endpoint' ],
            synapses: List[ 'bittensor.Synapse' ],
            inputs: List [ torch.Tensor ],
            timeout: int,
        ) -> Tuple[List[torch.Tensor], List[int], List[float]]:
        r""" Forward tensor inputs to endpoints.

            Args:
                endpoints (:obj:`List[ bittensor.Endpoint ]` of shape :obj:`(num_endpoints)`, `required`):
                    List of remote endpoints which match length of inputs. Tensors from x are sent forward to these endpoints.

                synapses (:obj:`List[ 'bittensor.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Bittensor synapse objects with arguments. Each corresponds to a synapse function on the axon.
                    Responses are packed in this ordering. 

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    TODO(const): Allow multiple tensors.
                    List of tensors to send to corresponsing endpoints. Tensors are of arbitrary type and shape depending on the
                    modality.

                timeout (int):
                    Request timeout.

            Returns:
                forward_outputs (:obj:`List[ List[ torch.FloatTensor ]]` of shape :obj:`(num_endpoints * (num_synapses * (shape)))`, `required`):
                    Output encodings of tensors produced by remote endpoints. Non-responses are zeroes of common shape.

                forward_codes (:obj:`List[ List[bittensor.proto.ReturnCodes] ]` of shape :obj:`(num_endpoints * ( num_synapses ))`, `required`):
                    dendrite backward call return ops.

                forward_times (:obj:`List[ List [float] ]` of shape :obj:`(num_endpoints * ( num_synapses ))`, `required`):
                    dendrite backward call times
        """
        # Init receptors.
        receptors = [ self._get_or_create_receptor_for_endpoint( endpoint ) for endpoint in endpoints ]

        # Make calls.
        calls = []
        for index, receptor in enumerate(receptors):
            calls.append( 
                receptor.async_forward(
                    synapses = synapses,
                    inputs = inputs[index], 
                    timeout = timeout
                )
            )

        responses = await asyncio.gather( *calls )

        # Unpack responses
        forward_outputs = []
        forward_codes = []
        forward_times = []
        for response in responses:
            forward_outputs.append( response[0] )
            forward_codes.append( response[1] )
            forward_times.append( response[2] )

        # ---- Kill receptors ----
        self._destroy_receptors_over_max_allowed()
        # ---- Return ----
        return forward_outputs, forward_codes, forward_times

    async def async_backward(
                self, 
                endpoints: List [ 'bittensor.Endpoint' ],
                synapses: List[ 'bittensor.Synapse' ],
                inputs: List [ torch.Tensor ],
                grads: List [ List[ torch.FloatTensor ] ],
                timeout: int
            ) -> Tuple[List[torch.Tensor], List[int], List[float]]:
        r""" Backward tensor inputs to endpoints.

            Args:
                endpoints (:obj:`List['bittensor.Endpoint']` of shape :obj:`(num_endpoints)`, `required`):
                    List of remote endpoints which match length of x. Tensors from x are sent backward to these endpoints.

                synapses (:obj:`List[ 'bittensor.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Bittensor synapse objects with arguments. Each corresponds to a synapse function on the axon.
                    Responses are packed in this ordering. 

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    List of tensors to send to corresponsing endpoints. Tensors are of arbitrary type and shape depending on the
                    synapse.

                grads (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    List of list of grad tensors where each grad corresponds to a synapse call on an endpoint.
                
                timeout (int):
                    request timeout.

            Returns:
                backward_outputs (:obj:`List[ List[ torch.FloatTensor] ]` of shape :obj:`num_endpoints * (batch_size, sequence_len, -1)]`, `required`):
                    Gradients returned from the backward call one per endpoint.

                backward_codes (:obj:`List[ List[ bittensor.proto.ReturnCodes ] ]` of shape :obj:`(num_endpoints)`, `required`):
                    List of list of Backward call return ops, one per endpoint and synapse.

                backward_times (:obj:`List[float]` of shape :obj:`(num_endpoints)`, `required`):
                    List of list of Backward call times one per endpoint and synapse.
        """
        # Init receptors.
        receptors = [ self._get_or_create_receptor_for_endpoint( endpoint ) for endpoint in endpoints ]

        # Make calls.
        calls = []
        for index, receptor in enumerate(receptors):
            calls.append( 
                receptor.async_backward (
                    synapses = synapses,
                    inputs = inputs[index], 
                    grads = grads[index],
                    timeout = timeout
                )
            )
        responses = await asyncio.gather( *calls )
            
        # Unpack responses
        backward_outputs = []
        backward_codes = []
        backward_times = []
        for response in responses:
            backward_outputs.append( response[0] )
            backward_codes.append( response[1] )
            backward_times.append( response[2] )

        # ---- Kill receptors ----
        self._destroy_receptors_over_max_allowed()
        # ---- Return ----
        return backward_outputs, backward_codes, backward_times

    def _destroy_receptors_over_max_allowed( self ):
        r""" Destroys receptors based on QPS until there are no more than max_active_receptors.
        """
        with self.cull_mutex:
            # ---- Finally: Kill receptors over max allowed ----
            while len(self.receptors) > self.max_active_receptors:
                min_receptor_qps = math.inf
                receptor_to_remove = None
                for next_receptor in self.receptors.values():
                    next_qps = next_receptor.stats.forward_qps.value
                    sema_value = next_receptor.semaphore._value
                    if (min_receptor_qps > next_qps) and (sema_value == self.max_processes):
                        receptor_to_remove = next_receptor
                        min_receptor_qps = next_receptor.stats.forward_qps.value
                        
                if receptor_to_remove != None:
                    try:
                        self.receptors[ receptor_to_remove.endpoint.hotkey ].close()
                        del self.receptors[ receptor_to_remove.endpoint.hotkey ]
                    except KeyError:
                        pass
                elif receptor_to_remove == None:
                    break

    def _get_or_create_receptor_for_endpoint( self, endpoint: 'bittensor.Endpoint' ) -> 'bittensor.Receptor':
        r""" Finds or creates a receptor TCP connection associated with the passed Neuron Endpoint
            Returns
                receptor: (`bittensor.Receptor`):
                    receptor with tcp connection endpoint at endpoint.ip:endpoint.port
        """
        # ---- Find the active receptor for this endpoint ----
        if endpoint.hotkey in self.receptors:
            receptor = self.receptors[ endpoint.hotkey ]

            # Change receptor address.
            if receptor.endpoint.ip != endpoint.ip or receptor.endpoint.port != endpoint.port:
                #receptor.close()
                bittensor.logging.update_receptor_log( endpoint )
                receptor = bittensor.receptor (
                    endpoint = endpoint, 
                    wallet = self.wallet,
                    external_ip = self.external_ip,
                    max_processes = self.max_processes
                )            
                self.receptors[ receptor.endpoint.hotkey ] = receptor

        # ---- Or: Create a new receptor ----
        else:
            receptor = bittensor.receptor (
                    endpoint = endpoint, 
                    wallet = self.wallet,
                    external_ip = self.external_ip,
                    max_processes = self.max_processes,
                    compression = self.compression
            )
            self.receptors[ receptor.endpoint.hotkey ] = receptor
            
        return receptor