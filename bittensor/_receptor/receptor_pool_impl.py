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

import bittensor
import math
import torch

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Tuple, List, Union, Optional
import bittensor.utils.stats as stat_utils

from loguru import logger
logger = logger.opt(colors=True)

class ReceptorPool ( torch.nn.Module ):

    def __init__(
        self, 
        wallet: 'bittensor.Wallet',
        thread_pool: 'ThreadPoolExecutor',
        max_active_receptors: int
    ):
        self.wallet = wallet
        self.thread_pool = thread_pool
        self.max_active_receptors = max_active_receptors
        self.receptors = {}

    def forward(
            self, 
            endpoints: List['bittensor.Endpoint'],
            inputs: List[torch.Tensor],
            modality: bittensor.proto.Modality,
            timeout: int
        ) -> Tuple[List[torch.Tensor], List[int]]:
        r""" Forward tensor inputs to endpoints.

            Args:
                endpoints (:obj:`List[bittensor.Endpoint]` of shape :obj:`(num_endpoints)`, `required`):
                    List of remote endpoints which match length of x. Tensors from x are sent forward to these endpoints.

                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    List of tensors to send to corresponsing endpoints. Tensors are of arbitrary type and shape depending on the
                    modality.

                modality (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

                timeout (int):
                    request timeout.

            Returns:
                forward_outputs (:obj:`List[torch.FloatTensor]` of shape :obj:`num_endpoints * (batch_size, sequence_len, bittensor.network_size)]`, `required`):
                    Output encodings of tensors produced by remote endpoints. Non-responses are zeroes of common shape.

                forwad_codes (:obj:`List[bittensor.proto.ReturnCodes]` of shape :obj:`(num_endpoints)`, `required`):
                    dendrite backward call return ops.
        """
        if len(endpoints) != len(inputs):
            raise ValueError('Endpoints must have the same length as passed inputs. Got {} and {}'.format(len(endpoints), len(inputs)))
        # ---- Run threaded calls with executor ----
        forward_outputs = []
        forward_codes = []
        
        # --- Create calls ----
        def _call_receptor_forward_with_args( receptor, inputs, modality ):
            return receptor.forward( inputs = inputs, modality = modality, timeout = timeout )

        # ---- Fill calls ----
        call_args = [ 
            (self._get_or_create_receptor_for_endpoint( endpoint ), inputs, modality) 
            for (inputs, endpoint) 
            in list(zip( inputs, endpoints )) 
        ]
        for result in self.thread_pool.map( lambda args: _call_receptor_forward_with_args(*args), call_args ):
            forward_outputs.append( result[0] )
            forward_codes.append( result[1] )

        # ---- Kill receptors ----
        self._destroy_receptors_over_max_allowed()
        
        # ---- Return ----
        return forward_outputs, forward_codes

    def backward(
                self, 
                endpoints: List['bittensor.Endpoint'],
                inputs_x: List[torch.Tensor],
                grads_dy: List[torch.Tensor],
                modality: bittensor.proto.Modality,
                timeout: int
            ) -> Tuple[List[torch.Tensor], List[int]]:
        r""" Backward tensor inputs to endpoints.

            Args:
                endpoints (:obj:`List['bittensor.Endpoint']` of shape :obj:`(num_endpoints)`, `required`):
                    List of remote endpoints which match length of x. Tensors from x are sent backward to these endpoints.

                inputs_x (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    List of tensors to send to corresponsing endpoints. Tensors are of arbitrary type and shape depending on the
                    modality.

                grads_dy (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    List of grad tensors to send to corresponsing inputs. 

                modality (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]
                
                timeout (int):
                    request timeout.

            Returns:
                backward_outputs (:obj:`List[torch.FloatTensor]` of shape :obj:`num_endpoints * (batch_size, sequence_len, -1)]`, `required`):
                    gradients of returned from backward call.

                backward_codes (:obj:`torch.LongTensor` of shape :obj:`(num_endpoints)`, `required`):
                    dendrite call return ops.
        """
        if len(endpoints) != len(inputs_x):
            raise ValueError('Endpoints and inputs must have the same length. Got {} and {}'.format(len(endpoints), len(inputs_x)))

        # ---- Run threaded calls with executor ----
        backward_outputs = []
        backward_codes = []
        
        # --- Create calls ----
        def _call_receptor_backward_with_args( receptor, inputs_x, grads_dy , modality ):
            return receptor.backward( inputs_x = inputs_x, grads_dy = grads_dy, modality = modality, timeout = timeout )

        # ---- Fill calls ----
        call_args = [
            (self._get_or_create_receptor_for_endpoint( endpoint ), inputs_x, grads_dy, modality) 
            for (inputs_x, grads_dy, endpoint) in 
            list(zip( inputs_x, grads_dy, endpoints )) 
        ]
        for result in self.thread_pool.map( lambda args: _call_receptor_backward_with_args(*args), call_args ):
            backward_outputs.append( result[0] )
            backward_codes.append( result[1] )

        # ---- Kill receptors ----
        self._destroy_receptors_over_max_allowed()
        
        # ---- Return ----
        return backward_outputs, backward_codes

    def _destroy_receptors_over_max_allowed( self ):
        r""" Destroys receptors based on QPS until there are no more than max_active_receptors.
        """

        # ---- Finally: Kill receptors over max allowed ----
        while len(self.receptors) > self.max_active_receptors:
            min_receptor_qps = math.inf
            receptor_to_remove = None
            for next_receptor in self.receptors.values():
                next_qps = next_receptor.stats.forward_qps.value
                if min_receptor_qps > next_qps:
                    receptor_to_remove = next_receptor
            if receptor_to_remove != None:
                bittensor.logging.destroy_receptor_log(receptor_to_remove.endpoint)
                del self.receptors[ receptor_to_remove.endpoint.hotkey ]

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
                del receptor
                bittensor.logging.update_receptor_log( endpoint )
                receptor = bittensor.receptor (
                    endpoint = endpoint, 
                    wallet = self.wallet
                )            
                self.receptors[ receptor.endpoint.hotkey ] = receptor

        # ---- Or: Create a new receptor ----
        else:
            bittensor.logging.create_receptor_log( endpoint )
            receptor = bittensor.receptor (
                    endpoint = endpoint, 
                    wallet = self.wallet
            )
            self.receptors[ receptor.endpoint.hotkey ] = receptor

        return receptor

