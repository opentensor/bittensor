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
from typing import Tuple, List

import torch
from loguru import logger
import concurrent
import bittensor
import bittensor.utils.networking as net
from concurrent.futures import ThreadPoolExecutor

logger = logger.opt(colors=True)

class ReceptorPool ( torch.nn.Module ):
    """ Manages a pool of grpc connections as receptors
    """
    def __init__(
        self, 
        wallet: 'bittensor.Wallet',
        thread_pool: 'ThreadPoolExecutor',
        max_worker_threads: int,
        max_active_receptors: int
    ):
        super().__init__()
        self.wallet = wallet
        self.thread_pool = thread_pool
        self.max_worker_threads = max_worker_threads
        self.max_active_receptors = max_active_receptors
        self.receptors = {}
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

    def forward(
            self, 
            endpoints: List['bittensor.Endpoint'],
            inputs: List[torch.Tensor],
            modality: bittensor.proto.Modality,
            timeout: int
        ) -> Tuple[List[torch.Tensor], List[int], List[float]]:
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

                forward_codes (:obj:`List[bittensor.proto.ReturnCodes]` of shape :obj:`(num_endpoints)`, `required`):
                    dendrite backward call return ops.

                forward_times (:obj:`List[float]` of shape :obj:`(num_endpoints)`, `required`):
                    dendrite backward call times
        """
        
        if len(endpoints) != len(inputs):
            raise ValueError('Endpoints must have the same length as passed inputs. Got {} and {}'.format(len(endpoints), len(inputs)))

        # ---- Fill calls ----
        call_args = [ 
            (self._get_or_create_receptor_for_endpoint( endpoint ), inputs, modality) 
            for (inputs, endpoint) 
            in list(zip( inputs, endpoints )) 
        ]

        # ---- Preprocessing for the forward function, get the request. ---- 
        requests = []
        for arg in call_args:
            receptor, inputs, modality = arg
            requests.append(receptor.preprocess_request ( inputs = inputs, modality = modality ))

        # ---- Send the forward request to peers. ---- 
        request_futures = []
        for arg, request in zip(call_args, requests):
            receptor = arg[0]
            request_futures.append(receptor.make_request_call(request = request, timeout = timeout))

        # ---- Collect the futures. ---- 
        thread_pool = ThreadPoolExecutor(max_workers=self.max_worker_threads)    
        results = thread_pool.map(lambda arg, request_future: arg[0].handle_request_response(request = request_future), call_args, request_futures)
        try:
            forward_outputs, forward_codes, forward_times = zip(*results)

        except concurrent.futures._base.TimeoutError:
            forward_outputs= [torch.zeros( (inputs[0].size(0), inputs[0].size(1), bittensor.__network_dim__), dtype=torch.float32)] * len(endpoints) 
            forward_codes= [bittensor.proto.ReturnCode.Timeout] * len(endpoints) 
            forward_times= [15] * len(endpoints)
        except Exception as e:
            forward_outputs= [torch.zeros( (inputs[0].size(0), inputs[0].size(1), bittensor.__network_dim__), dtype=torch.float32)] * len(endpoints) 
            forward_codes= [bittensor.proto.ReturnCode.UnknownException] * len(endpoints) 
            forward_times= [15] * len(endpoints)
            logger.exception('Exception encountered: {}'.format(e))

        # ---- Kill receptors ----
        self._destroy_receptors_over_max_allowed()

        # ---- Return ----
        return list(forward_outputs), list(forward_codes), list(forward_times)

    def backward(
                self, 
                endpoints: List['bittensor.Endpoint'],
                inputs_x: List[torch.Tensor],
                grads_dy: List[torch.Tensor],
                modality: bittensor.proto.Modality,
                timeout: int
            ) -> Tuple[List[torch.Tensor], List[int], List[float]]:
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

                backward_codes (:obj:`List[bittensor.proto.ReturnCodes]` of shape :obj:`(num_endpoints)`, `required`):
                    dendrite call return ops.

                backward_times (:obj:`List[float]` of shape :obj:`(num_endpoints)`, `required`):
                    dendrite call times.
        """
        if len(endpoints) != len(inputs_x):
            raise ValueError('Endpoints and inputs must have the same length. Got {} and {}'.format(len(endpoints), len(inputs_x)))

        # ---- Fill calls ----
        call_args = [
            (self._get_or_create_receptor_for_endpoint( endpoint ), inputs_x, grads_dy, modality) 
            for (inputs_x, grads_dy, endpoint) in 
            list(zip( inputs_x, grads_dy, endpoints )) 
        ]

        # ---- Preprocessing for the forward function, get the request. ---- 
        requests = []
        for arg in call_args:
            receptor, inputs, grads_dy, modality = arg
            requests.append(receptor.preprocess_request ( inputs = inputs, modality = modality, grads_dy = grads_dy, backward = True))

        # ---- Send the forward request to peers. ---- 
        request_futures = []
        for arg, request in zip(call_args, requests):
            receptor = arg[0]
            request_futures.append(receptor.make_request_call(request = request, timeout = timeout))

        # ---- Return zeros ----
        backward_outputs= [torch.zeros( (inputs_x[0].size(0), inputs_x[0].size(1), bittensor.__network_dim__), dtype=torch.float32)] * len(endpoints) 
        backward_codes= [bittensor.proto.ReturnCode.Timeout] * len(endpoints) 
        backward_times= [15] * len(endpoints)

        # ---- Kill receptors ----
        self._destroy_receptors_over_max_allowed()
        
        return backward_outputs, backward_codes, backward_times

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
                    min_receptor_qps = next_receptor.stats.forward_qps.value
                    
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
                    wallet = self.wallet,
                    external_ip = self.external_ip,
            )
            self.receptors[ receptor.endpoint.hotkey ] = receptor

        return receptor