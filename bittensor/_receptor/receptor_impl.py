""" Encapsulates a grpc connection to an axon endpoint as a standard auto-grad torch.nn.Module.
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
import traceback

import bittensor
from bittensor._synapse import synapse
import bittensor.utils.stats as stat_utils

import torch
import asyncio
import threading
import uuid
import sys
import torch.nn as nn
import grpc
import time as clock

from types import SimpleNamespace
from typing import Tuple, List, Union
from loguru import logger
from grpc import _common



class Receptor(nn.Module):

    def __init__(
            self, 
            wallet: 'bittensor.wallet',
            endpoint: 'bittensor.Endpoint', 
            channel: 'grpc._Channel',
            stub: 'bittensor.grpc.BittensorStub',
            max_processes: int,
        ):
        r""" Initializes a receptor grpc connection.

            Args:
                wallet (:obj:`bittensor.Wallet`, `required`):
                    bittensor wallet with hotkey and coldkeypub.
                endpoint (:obj:`bittensor.Endpoint`, `required`):
                    neuron endpoint descriptor proto.
                channel (:obj:`grpc._Channel`, `required`):
                    grpc TCP channel.
                endpoint (:obj:`bittensor.grpc.BittensorStub`, `required`):
                    bittensor protocol stub created from channel.
        """
        super().__init__()
        self.wallet = wallet # Keypair information
        self.endpoint = endpoint # Endpoint information.
        self.channel = channel
        self.stub = stub
        self.receptor_uid = str(uuid.uuid1())
        self.semaphore = threading.Semaphore(max_processes)
        self.state_dict = _common.CYGRPC_CONNECTIVITY_STATE_TO_CHANNEL_CONNECTIVITY
        self.stats = SimpleNamespace(
            forward_qps = stat_utils.timed_rolling_avg(0.0, 0.01),
            backward_qps = stat_utils.timed_rolling_avg(0.0, 0.01),
            forward_elapsed_time = stat_utils.timed_rolling_avg(0.0, 0.01),
            forward_bytes_out = stat_utils.timed_rolling_avg(0.0, 0.01),
            forward_bytes_in = stat_utils.timed_rolling_avg(0.0, 0.01),
            backward_bytes_out = stat_utils.timed_rolling_avg(0.0, 0.01),
            backward_bytes_in = stat_utils.timed_rolling_avg(0.0, 0.01),
            codes = {
                bittensor.proto.ReturnCode.NoReturn: 0,
                bittensor.proto.ReturnCode.Success: 0,
                bittensor.proto.ReturnCode.Timeout: 0,
                bittensor.proto.ReturnCode.Backoff: 0,
                bittensor.proto.ReturnCode.Unavailable: 0,
                bittensor.proto.ReturnCode.NotImplemented: 0,
                bittensor.proto.ReturnCode.EmptyRequest: 0,
                bittensor.proto.ReturnCode.EmptyResponse: 0,
                bittensor.proto.ReturnCode.InvalidResponse: 0,
                bittensor.proto.ReturnCode.InvalidRequest: 0,
                bittensor.proto.ReturnCode.RequestShapeException: 0,
                bittensor.proto.ReturnCode.ResponseShapeException: 0,
                bittensor.proto.ReturnCode.RequestSerializationException: 0,
                bittensor.proto.ReturnCode.ResponseSerializationException: 0,
                bittensor.proto.ReturnCode.RequestDeserializationException: 0,
                bittensor.proto.ReturnCode.ResponseDeserializationException: 0,
                bittensor.proto.ReturnCode.NotServingNucleus: 0,
                bittensor.proto.ReturnCode.NucleusTimeout: 0,
                bittensor.proto.ReturnCode.NucleusFull: 0,
                bittensor.proto.ReturnCode.RequestIncompatibleVersion: 0,
                bittensor.proto.ReturnCode.ResponseIncompatibleVersion: 0,
                bittensor.proto.ReturnCode.SenderUnknown: 0,
                bittensor.proto.ReturnCode.UnknownException: 0,
                bittensor.proto.ReturnCode.Unauthenticated: 0,
                bittensor.proto.ReturnCode.BadEndpoint: 0,
            }
        )

    def __str__ ( self ):
        return "Receptor({})".format(self.endpoint) 

    def __repr__ ( self ):
        return self.__str__()

    def __del__ ( self ):
        try:
            result = self.channel._channel.check_connectivity_state(True)
            if self.state_dict[result] != self.state_dict[result].SHUTDOWN: 
                loop = asyncio.get_event_loop()
                loop.run_until_complete ( self.channel.close() )
        except:
            pass
    
    def __exit__ ( self ):
        self.__del__()

    def sign(self):
        nonce = f"{self.nonce()}"
        sender_hotkey = self.wallet.hotkey.ss58_address
        receiver_hotkey = self.endpoint.hotkey
        message = f"{nonce}.{sender_hotkey}.{receiver_hotkey}.{self.receptor_uid}"
        signature = f"0x{self.wallet.hotkey.sign(message).hex()}"
        return ".".join([nonce, sender_hotkey, signature, self.receptor_uid])


    def nonce ( self ):
        r"""creates a string representation of the time
        """
        return clock.monotonic_ns()
        
    def state ( self ):
        try: 
            return self.state_dict[self.channel._channel.check_connectivity_state(True)]
        except ValueError:
            return "Channel closed"

    def close ( self ):
        self.__exit__()

    def forward (
        self, 
        synapses: List[ 'bittensor.Synapse' ],
        inputs: torch.Tensor, 
        timeout: int,
    ) -> Tuple[ List[ torch.FloatTensor ], List['bittensor.proto.ReturnCode'], List[float] ]:
        r""" Triggers the grpc call to the remote endpoint.
            This triggers the synapse calls with arguments.
            Call returns a list of output tensors one per synapse with corresponding time and bittensor.proto.ReturnCode.

            Args:
                synapses (:obj:`List[ 'bittensor.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Bittensor synapse objects with arguments. Each corresponds to a synapse function on the axon.
                    Responses are packed in this ordering. 

                inputs (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Single torch tensor to be sent to the remote endpoint.
                    TODO(const): Make this a multi-forward tensor.

                timeout (:obj:`int`, `required`):
                    Request max timeout
            Returns:
                outputs (:obj:`List[ Union[torch.FloatTensor, torch.LongTensor] ]`, `required`):
                    outputs.shape = [batch_size, synapse_length, response] 
                    List of result tensors from the forward call each corresponding to a passed synapse enum.

                codes (:obj:`bittensor.proto.ReturnCode`, `required`):
                    List of return codes associated with each passed synapse enum.
                    Connection failures return all the same code, otherwise a unique code per synapse. 

                times (:obj:`float`, `required`):
                    List of times for each call associated with each passed synapse enum. 
                    Success responses all get the same time.

        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete( self.async_forward ( synapses = synapses,inputs = inputs, timeout = timeout ) )


    def backward (
        self, 
        synapses: List[ 'bittensor.Synapse' ],
        inputs: torch.Tensor, 
        grads: List[torch.Tensor], 
        timeout: int
    ) -> Tuple[ List[ torch.FloatTensor ], List['bittensor.proto.ReturnCode'], List[float] ]:
        r""" Triggers the grpc backward call to the remote endpoint.
            This triggers the synapse's backward calls with arguments.
            Call returns a list of output gradient tensors one per synapse with corresponding time and bittensor.proto.ReturnCode.

            Args:
                synapses (:obj:`List[ 'bittensor.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Bittensor synapse objects with arguments. Each corresponds to a synapse function on the axon.
                    Responses are packed in this ordering. 

                inputs (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Single torch tensor input corresponding to the linked forward call.
                    TODO(const): Make this multi-forward tensor.

                grads (:obj:`List[torch.FloatTensor]` of shape :obj:`num_synapses * (shape_of_synapse_output_i)`, `required`):
                    List of torch tensor gradients associated with each synapse.
             
                timeout (:obj:`int`, `required`):
                    Request max timeout
            Returns:
                output (:obj:`torch.FloatTensor`, `required`):
                    Result tensors (likely zero) from the backward call each corresponding to a single forward input.
                    NOTE(const) Always zeros because responses are not waited.
                    TODO(const): Make this multi-forward tensor.

                codes (:obj:`bittensor.proto.ReturnCode`, `required`):
                    List of return codes associated with each passed synapse enum.
                    Connection failures return all the same code, otherwise a unique code per synapse. 

                times (:obj:`float`, `required`):
                    List of times for each call associated with each passed synapse enum. 
                    Success responses all get the same time.
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete ( self.async_backward ( synapses = synapses, inputs = inputs, grads = grads, timeout = timeout ) )

    async def async_forward (
        self, 
        synapses: List[ 'bittensor.Synapse' ],
        inputs: torch.Tensor, 
        timeout: int,
    ) -> Tuple[ List[ torch.FloatTensor ], List['bittensor.proto.ReturnCode'], List[float] ]:
        r""" Triggers the grpc call to the remote endpoint.
            This triggers the synapse calls with arguments.
            Call returns a list of output tensors one per synapse with corresponding time and bittensor.proto.ReturnCode.

            Args:
                synapses (:obj:`List[ 'bittensor.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Bittensor synapse objects with arguments. Each corresponds to a synapse function on the axon.
                    Responses are packed in this ordering. 

                inputs (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Single torch tensor to be sent to the remote endpoint.
                    TODO(const): Make this a multi-forward tensor.

                timeout (:obj:`int`, `required`):
                    Request max timeout
            Returns:
                outputs (:obj:`List[ Union[torch.FloatTensor, torch.LongTensor] ]`, `required`):
                    outputs.shape = [batch_size, synapse_length, response] 
                    List of result tensors from the forward call each corresponding to a passed synapse enum.

                codes (:obj:`bittensor.proto.ReturnCode`, `required`):
                    List of return codes associated with each passed synapse enum.
                    Connection failures return all the same code, otherwise a unique code per synapse. 

                times (:obj:`float`, `required`):
                    List of times for each call associated with each passed synapse enum. 
                    Success responses all get the same time.

        """
        # =====================
        # ==== Init params ====        
        # =====================
        # These items are filled through the call and the function returns 
        # when all codes are non-success or the function finishes completely.
        synapse_messages = [ "Success" for _ in synapses ]
        synapse_codes = [ bittensor.proto.ReturnCode.Success for _ in synapses ]
        synapse_responses = [ synapse.nill_forward_response_tensor( inputs ) for synapse in synapses ]
        synapse_is_response = [ False for _ in synapses ]
        synapse_call_times = [ 0 for _ in synapses ]
        start_time = clock.time()

        # ==================================================================
        # ==== Function which returns true if all codes are non success ====
        # ==================================================================
        def check_if_should_return() -> bool:
            for code in synapse_codes:
                if code == bittensor.proto.ReturnCode.Success:
                    return False
            return True

        # ==============================================================
        # ==== Function which prints all log statements per synapse ====
        # ==============================================================
        def finalize_stats_and_logs():
            self.stats.forward_elapsed_time.update( clock.time() - start_time )
            for index, synapse in enumerate( synapses ):
                self.stats.codes[ synapse_codes[ index ] ] += 1
                bittensor.logging.rpc_log ( 
                    axon = False, 
                    forward = True, 
                    is_response = synapse_is_response [index], 
                    code = synapse_codes[ index ], 
                    call_time = synapse_call_times[ index ], 
                    pubkey = self.endpoint.hotkey, 
                    uid = self.endpoint.uid, 
                    inputs = list(inputs.shape), 
                    outputs = None if synapse_codes[ index ] != bittensor.proto.ReturnCode.Success else list( synapse_responses[index].shape ), 
                    message = synapse_messages[ index ],
                    synapse = synapse.synapse_type
                )

        # ===========================
        # ==== Check inputs size ====
        # ===========================
        if torch.numel(inputs) == 0:
            # Inputs are nill.
            code = bittensor.proto.ReturnCode.EmptyRequest
            call_time = clock.time() - start_time
            message = "Empty Request"
            synapse_codes = [ code for _ in synapses ]
            synapse_call_times = [ call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_stats_and_logs()
            return synapse_responses, synapse_codes, synapse_call_times
        
        # ========================
        # ==== Check endpoint ====
        # ========================
        if self.endpoint.hotkey  == 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX':
            # Endpoint is dummy.
            code = bittensor.proto.ReturnCode.BadEndpoint
            call_time = clock.time() - start_time
            message =  "Bad endpoint."
            synapse_call_times = [ call_time for _ in synapses ]
            synapse_codes = [ code for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_stats_and_logs()
            return synapse_responses, synapse_codes, synapse_call_times

        # ==========================
        # ==== Serialize inputs ====
        # ==========================
        serialized_forward_tensors = []
        serialized_synapses = []
        for index, synapse in enumerate( synapses ):
            try:
                serialized_forward_tensors.append( synapse.serialize_forward_request_tensor ( inputs ))
                serialized_synapses.append(synapse.serialize_to_wire_proto())
            except Exception as e:
                synapse_codes [index] = bittensor.proto.ReturnCode.RequestSerializationException
                synapse_call_times [index] = clock.time() - start_time
                synapse_messages [index] = 'Input serialization exception with error:{}'.format(str(e))
        # Check if the call can stop here.
        if check_if_should_return():
            finalize_stats_and_logs()
            return synapse_responses, synapse_codes, synapse_call_times
            
        # ============================
        # ==== Build proto request ====
        # ============================
        try: 
            grpc_request = bittensor.proto.TensorMessage (
                version = bittensor.__version_as_int__,
                hotkey = self.wallet.hotkey.ss58_address,
                tensors = serialized_forward_tensors,
                synapses = serialized_synapses,
                requires_grad = True,
            )
        except Exception as e:
            # Synapse request creation failed.
            code = bittensor.proto.ReturnCode.UnknownException
            call_time = clock.time() - start_time
            message = 'Request proto creation failed with error:{}'.format(str(e)) 
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_stats_and_logs()
            return synapse_responses, synapse_codes, synapse_call_times


        # ===============================
        # ==== Fire Asyncio RPC Call ====
        # ===============================
        try:
            self.stats.forward_qps.update(1)
            self.stats.forward_bytes_out.update( sys.getsizeof( grpc_request ) )
            finalize_stats_and_logs()
            asyncio_future = self.stub.Forward (
                request = grpc_request, 
                timeout = timeout,
                metadata = (
                    ('rpc-auth-header','Bittensor'),
                    ('bittensor-signature',self.sign()),
                    ('bittensor-version',str(bittensor.__version_as_int__)),
                    ('request_type', str(bittensor.proto.RequestType.FORWARD)),
                ))
            grpc_response = await asyncio.wait_for(asyncio_future, timeout=timeout)
            self.stats.forward_bytes_in.update( grpc_response.ByteSize() )
            synapse_is_response = [ True for _ in synapses ]

        # ====================================
        # ==== Handle GRPC Errors ====
        # ====================================
        except grpc.RpcError as rpc_error_call:
            # Request failed with GRPC code.
            call_time = clock.time() - start_time
            grpc_code = rpc_error_call.code()
            if grpc_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                code = bittensor.proto.ReturnCode.Timeout
                message = 'grpc.StatusCode.DEADLINE_EXCEEDED'+': '+ rpc_error_call.details()
            elif grpc_code == grpc.StatusCode.UNAVAILABLE:
                code = bittensor.proto.ReturnCode.Unavailable
                message = 'grpc.StatusCode.UNAVAILABLE'+': '+ rpc_error_call.details()
            elif grpc_code == grpc.StatusCode.UNAUTHENTICATED:
                code = bittensor.proto.ReturnCode.Unauthenticated
                message = 'grpc.StatusCode.UNAUTHENTICATED'+': '+ rpc_error_call.details()
            else:
                code = bittensor.proto.ReturnCode.UnknownException
                message = 'GRPC error code: {}, details: {}'.format( grpc_code, str(rpc_error_call.details()) )
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_stats_and_logs()
            return synapse_responses, synapse_codes, synapse_call_times

        except asyncio.TimeoutError:
            code = bittensor.proto.ReturnCode.Timeout
            call_time = clock.time() - start_time
            message = 'GRPC request timeout after: {}s'.format(timeout)
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_stats_and_logs()
            return synapse_responses, synapse_codes, synapse_call_times

        # ====================================
        # ==== Handle GRPC Unknown Errors ====
        # ====================================
        except Exception as e:
            # Request failed with unknown exception.
            code = bittensor.proto.ReturnCode.UnknownException
            call_time = clock.time() - start_time
            message = 'GRPC request failed with unknown exception:{}'.format(str(e))
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_stats_and_logs()
            return synapse_responses, synapse_codes, synapse_call_times


        # ==========================================
        # ==== Handle Non Success GRPC Response ====
        # ==========================================
        if grpc_response.return_code != bittensor.proto.ReturnCode.Success:
            # Request failed with unknown exception.
            call_time = clock.time() - start_time
            synapse_call_times = [call_time for _ in synapses ]
            if len(grpc_response.synapses) == len(synapses):
                synapse_codes = [synapse.return_code for synapse in grpc_response.synapses ]
                synapse_messages = ['Remote Server Failure: '+ synapse.message for synapse in grpc_response.synapses ]
            finalize_stats_and_logs()
            return synapse_responses, synapse_codes, synapse_call_times



        # ======================================
        # ==== Check response length ====
        # ======================================
        if ( len(grpc_response.tensors) != len(grpc_response.synapses) ) or ( len(grpc_response.tensors) != len(synapses) ):
            # Not enough responses per request.
            code = bittensor.proto.ReturnCode.ResponseShapeException
            call_time = clock.time() - start_time
            message = "Responses dont match synape length"
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_stats_and_logs()
            return synapse_responses, synapse_codes, synapse_call_times

        # ======================================
        # ==== Check for non success response codes ====
        # ======================================
        for index, wire_synapse in enumerate( grpc_response.synapses ):
            if wire_synapse.return_code != bittensor.proto.ReturnCode.Success: 
                synapse_codes[index] = wire_synapse.return_code
                synapse_messages[index] = wire_synapse.message
                synapse_call_times[index] = clock.time() - start_time

        # Check if the call can stop here.
        if check_if_should_return():
            finalize_stats_and_logs()
            return synapse_responses, synapse_codes, synapse_call_times

        # ======================================
        # ==== Deserialize synapse responses ====
        # ======================================
        for index, response_proto in enumerate(grpc_response.tensors):
            try:
                synapse = synapses[index]
                if synapse_codes[index] == bittensor.proto.ReturnCode.Success:
                    synapse_responses[index] = synapse.deserialize_forward_response_proto ( inputs, response_proto ) 
            except Exception as e:
                # Input Serialization failed.
                synapse_codes[index] = bittensor.proto.ReturnCode.ResponseDeserializationException
                synapse_call_times[index] = clock.time() - start_time
                synapse_messages[index] = 'Response deserialization exception with error:{}'.format(str(e))


        # ======================================
        # ==== Finalize forward call times ====
        # ======================================
        for index, _ in enumerate( synapses ):
            if synapse_codes[index] == bittensor.proto.ReturnCode.Success:
                synapse_call_times[index] = clock.time() - start_time
        finalize_stats_and_logs()
        return synapse_responses, synapse_codes, synapse_call_times  

    async def async_backward (
        self, 
        synapses: List[ 'bittensor.Synapse' ],
        inputs: torch.Tensor, 
        grads: List[torch.Tensor], 
        timeout: int
    ) -> Tuple[ List[ torch.FloatTensor ], List['bittensor.proto.ReturnCode'], List[float] ]:
        r""" Triggers the grpc backward call to the remote endpoint.
            This triggers the synapse's backward calls with arguments.
            Call returns a list of output gradient tensors one per synapse with corresponding time and bittensor.proto.ReturnCode.

            Args:
                synapses (:obj:`List[ 'bittensor.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Bittensor synapse objects with arguments. Each corresponds to a synapse function on the axon.
                    Responses are packed in this ordering. 

                inputs (:obj:`torch.Tensor` of shape :obj:`(shape)`, `required`):
                    Single torch tensor input corresponding to the linked forward call.
                    TODO(const): Make this multi-forward tensor.

                grads (:obj:`List[torch.FloatTensor]` of shape :obj:`num_synapses * (shape_of_synapse_output_i)`, `required`):
                    List of torch tensor gradients associated with each synapse.
             
                timeout (:obj:`int`, `required`):
                    Request max timeout
            Returns:
                output (:obj:`torch.FloatTensor`, `required`):
                    Result tensors (likely zero) from the backward call each corresponding to a single forward input.
                    NOTE(const) Always zeros because responses are not waited.
                    TODO(const): Make this multi-forward tensor.

                codes (:obj:`bittensor.proto.ReturnCode`, `required`):
                    List of return codes associated with each passed synapse enum.
                    Connection failures return all the same code, otherwise a unique code per synapse. 

                times (:obj:`float`, `required`):
                    List of times for each call associated with each passed synapse enum. 
                    Success responses all get the same time.
        """
        # =====================
        # ==== Init params ====        
        # =====================
        # These items are filled through the call and the function returns 
        # when all codes are non-success or the function finishes completely.
        synapse_messages = [ "Success" for _ in synapses ]
        synapse_codes = [ bittensor.proto.ReturnCode.Success for _ in synapses ]
        synapse_responses = [ synapse.nill_backward_response_tensor ( inputs ) for synapse in synapses ]
        synapse_is_response = [ False for _ in synapses ]
        synapse_call_times = [ 0 for _ in synapses ]
        start_time = clock.time()

        # ==================================================================
        # ==== Function which returns true if all codes are non success ====
        # ==================================================================
        def check_if_should_return() -> bool:
            for code in synapse_codes:
                if code == bittensor.proto.ReturnCode.Success:
                    return False
            return True

        # ==============================================================
        # ==== Function which prints all log statements per synapse ====
        # ==============================================================
        def finalize_stats_and_logs():
            for index, synapse in enumerate( synapses ):
                self.stats.codes[ synapse_codes[ index ] ] += 1
                bittensor.logging.rpc_log ( 
                    axon = False, 
                    forward = False, 
                    is_response = synapse_is_response [index], 
                    code = synapse_codes[ index ], 
                    call_time = synapse_call_times[ index ], 
                    pubkey = self.endpoint.hotkey, 
                    uid = self.endpoint.uid, 
                    inputs = list(grads[index].shape), 
                    outputs = None, 
                    message = synapse_messages[ index ],
                    synapse = synapse.synapse_type
                )

        # ========================
        # ==== Check endpoint ====
        # ========================
        if self.endpoint.hotkey  == 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX':
            # Endpoint is dummy.
            code = bittensor.proto.ReturnCode.BadEndpoint
            call_time = clock.time() - start_time
            message =  "Bad endpoint."
            synapse_call_times = [ call_time for _ in synapses ]
            synapse_codes = [ code for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_stats_and_logs()
            return synapse_responses, synapse_codes, synapse_call_times

        # ==================================
        # ==== Serialize inputs & grads ====
        # ==================================
        serialized_forward_tensors = []
        serialized_backward_grads = []
        serialized_synapses = []
        for index, synapse in enumerate( synapses ):
            try:
                serialized_forward_tensors.append(synapse.serialize_forward_request_tensor( inputs ))
                serialized_backward_grads.append(synapse.serialize_backward_request_gradient (inputs, grads[index] ))
                serialized_synapses.append(synapse.serialize_to_wire_proto())
            except Exception as e:
                # Input Serialization failed.
                synapse_codes [index] = bittensor.proto.ReturnCode.RequestSerializationException
                synapse_call_times [index] = clock.time() - start_time
                synapse_messages [index] = 'Input serialization exception with error:{}'.format(str(e))
        # Check if the call can stop here.
        if check_if_should_return():
            finalize_stats_and_logs()
            return synapse_responses, synapse_codes, synapse_call_times


        # =============================
        # ==== Build proto request ====
        # =============================
        try: 
            grpc_request = bittensor.proto.TensorMessage (
                version = bittensor.__version_as_int__,
                hotkey = self.wallet.hotkey.ss58_address,
                tensors = serialized_forward_tensors + serialized_backward_grads,
                synapses = serialized_synapses,
                requires_grad = True,
            )

        except Exception as e:
            # Synapse request creation failed.
            code = bittensor.proto.ReturnCode.UnknownException
            call_time = clock.time() - start_time
            message = 'Request proto creation failed with error:{}'.format(str(e)) 
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_stats_and_logs()
            return synapse_responses, synapse_codes, synapse_call_times

        # =======================
        # ==== Make RPC Call ====
        # =======================
        try:
            self.stats.backward_qps.update(1)
            self.stats.backward_bytes_out.update(sys.getsizeof(grpc_request))
            # Fire and forget.
            asyncio_future = self.stub.Backward(
                request = grpc_request, 
                timeout = timeout,
                metadata = (
                    ('rpc-auth-header','Bittensor'),
                    ('bittensor-signature',self.sign()),
                    ('bittensor-version',str(bittensor.__version_as_int__)),
                    ('request_type', str(bittensor.proto.RequestType.BACKWARD)),
                ))
            asyncio_future.cancel()

        # ====================================
        # ==== Handle GRPC Errors ====
        # ====================================
        except grpc.RpcError as rpc_error_call:

            # Request failed with GRPC code.
            call_time = clock.time() - start_time
            grpc_code = rpc_error_call.code()
            if grpc_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                code = bittensor.proto.ReturnCode.Timeout
                message = 'grpc.StatusCode.DEADLINE_EXCEEDED'+': '+ rpc_error_call.details()
            elif grpc_code == grpc.StatusCode.UNAVAILABLE:
                code = bittensor.proto.ReturnCode.Unavailable
                message = 'grpc.StatusCode.UNAVAILABLE'+': '+ rpc_error_call.details()
            elif grpc_code == grpc.StatusCode.UNAUTHENTICATED:
                code = bittensor.proto.ReturnCode.Unauthenticated
                message = 'grpc.StatusCode.UNAUTHENTICATED'+': '+ rpc_error_call.details()
            else:
                code = bittensor.proto.ReturnCode.UnknownException
                message = 'GRPC error code: {}, details: {}'.format( grpc_code, str(rpc_error_call.details()) )
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_stats_and_logs()
            return synapse_responses, synapse_codes, synapse_call_times


        # =======================
        # ==== Timeout Error ====
        # =======================
        except asyncio.TimeoutError:
            code = bittensor.proto.ReturnCode.Timeout
            call_time = clock.time() - start_time
            message = 'GRPC request timeout after: {}s'.format(timeout)
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_stats_and_logs()
            return synapse_responses, synapse_codes, synapse_call_times

        # ====================================
        # ==== Handle GRPC Unknown Errors ====
        # ====================================
        except Exception as e:

            # Request failed with unknown exception.
            code = bittensor.proto.ReturnCode.UnknownException
            call_time = clock.time() - start_time
            message = 'GRPC request failed with unknown exception:{}'.format(str(e))
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]

        # ======================================
        # ==== Finalize backward call times ====
        # ======================================
        for index, _ in enumerate( synapses ):
            if synapse_codes[index] == bittensor.proto.ReturnCode.Success:
                synapse_call_times[index] = clock.time() - start_time
        finalize_stats_and_logs()
        return synapse_responses, synapse_codes, synapse_call_times            
            

        



        
