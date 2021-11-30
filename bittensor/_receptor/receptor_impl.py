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

import sys
import time as clock
from types import SimpleNamespace
from typing import Tuple

import torch
import uuid
import time
import torch.nn as nn
import grpc
from loguru import logger
from grpc import _common

import bittensor
import bittensor.utils.stats as stat_utils

logger = logger.opt(colors=True)

# dummy tensor that triggers autograd in a RemoteExpert
DUMMY = torch.empty(0, requires_grad=True)

# Helper function for filling nill (zero) responses on failures.
def nill_response_for(inputs):
    """ Empty response
    """
    if torch.numel(inputs) == 0:
        return torch.tensor([])
    return torch.zeros( (inputs.size(0), inputs.size(1), bittensor.__network_dim__), dtype=torch.float32)

class Receptor(nn.Module):
    """ Encapsulates a grpc connection to an axon endpoint as a standard auto-grad torch.nn.Module.
    """

    def __init__(
            self, 
            wallet: 'bittensor.wallet',
            endpoint: 'bittensor.Endpoint', 
            channel: 'grpc._Channel',
            stub: 'bittensor.grpc.BittensorStub',
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
        self.backoff = 0 # Number o queries to backoff.
        self.next_backoff = 1 # Next backoff level.
        self.receptor_uid = str(uuid.uuid1())
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
            }
        )

    def __str__(self):
        return "Receptor({})".format(self.endpoint) 

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        try:
            result = self.channel._channel.check_connectivity_state(True)
            if self.state_dict[result] != self.state_dict[result].SHUTDOWN:        
                self.channel.close()
        except:
            pass
        
    def __exit__(self):
        self.__del__()

    def forward (
        self, 
        inputs: torch.Tensor, 
        modality: bittensor.proto.Modality,
        timeout: int,
    ) -> Tuple[torch.Tensor, int]:
        r""" Torch.nn.Module forward call: Triggers the grpc call to the remote endpoint.
            Call returns the output tensor and a bittensor.proto.ReturnCode.

            Args:
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Single torch tensor to be sent to the remote endpoint.
                modality (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]
                timeout (:obj:`int`, `required`)
            Returns:
                output (:obj:`Tuple[torch.FloatTensor, torch.LongTensor]`, `required`):
                    Result tuple from the forward call.
                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Return code associated with forward call.
                time (:obj:`float`, `required`):
                    Time of call.

        """
        request = self.forward_prep( inputs = inputs, modality = modality)
        request = self.forward_call(request, timeout = timeout)
        return self.forward_collect(request)


    def backward(
            self, 
            inputs_x: torch.Tensor, 
            grads_dy: torch.Tensor, 
            modality: bittensor.proto.Modality,
            timeout: int
        ) -> Tuple[ torch.Tensor, int ]:
        r""" Backward call: Triggers the grpc Backward call to the associated endpoint.

            Args:
                inputs_x (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    inputs from previous forward call.
    
                grads_dy (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    gradient outputs.

                modality (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

                timeout (int):
                    request timeout.

            Returns:
                output (:obj:`Tuple[torch.FloatTensor, torch.LongTensor]`, `required`):
                    Result tuple from the forward call.

                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Return code associated with backward call.

                time (:obj:`float`, `required`):
                    Time of call.
        """
        outputs, code, time, _ = self._call_backward( 
            inputs_x = inputs_x, 
            grads_dy = grads_dy, 
            modality = modality,
            timeout = timeout
        )
        try:
            self.stats.codes[code] += 1
        except Exception: 
            pass
        return outputs, code, time

    def forward_log(self, request, is_response = False, inputs = None, outputs = None):
        r""" rpc logging for forward/backward request
            Args:
                request (:type:`SimpleNamespace`):
                    the details of the forward/backward request

                        inputs (torch.tensor): the tensor input for the grpc call 
                        modality (int): the modality of the input
                        start_time (float): the time when the forward request started
                        code (int): the return code, biding to bittensor.proto
                        message (str): the return message
                        request (bittensor.proto.TensorMessage): the tensor message that is ready for the grpc call
                        future (grpc.Future): the future of the forward/backward call
                
                is_response (:type: `bool`):
                    True if we are logging a response from the grpc call, false if it is a request instead

                inputs (:type: `List`): 
                    shape of the tensor input that was being handled
        """

        call_time = clock.time() - request.start_time
        bittensor.logging.rpc_log(
            axon=False, 
            forward=True, 
            is_response=is_response, 
            code=request.code, 
            call_time=call_time, 
            pubkey=self.endpoint.hotkey, 
            uid = self.endpoint.uid, 
            inputs=inputs, 
            outputs=outputs, 
            message=request.message
        )

    def forward_prep(
        self, 
        inputs: torch.Tensor, 
        modality: bittensor.proto.Modality,
    ):  
        r""" Serialize the inputs and format a grpc request. 
            Initialize the forward_request namespace which will hold all the objects regrading the forward request. 
            The resulted request was stored in forward_request.
            
            Args:  
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be sent to this endpoint.

                modality (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality of type Enum: [TEXT, IMAGE, TENSOR]
        """
        # ---- Setup forward request namespace, which will hold all the objects regarding the forward request ----
        forward_request = SimpleNamespace(
            inputs = inputs,
            serialized_inputs = None,
            zeros = nill_response_for(inputs),
            modality = modality,
            start_time = clock.time(),
            code = None,
            message = None,
            request = None,
            future = None
        )

        # ---- Check inputs size ----
        if torch.numel(inputs) == 0:
            forward_request.code = bittensor.proto.ReturnCode.EmptyRequest
            forward_request.message = 'Empty request.'
            self.forward_log(request = forward_request, is_response = False, inputs = list(inputs.shape))
            return forward_request
        
        # ---- Check endpoint----
        elif self.endpoint.uid == -1:
            forward_request.code = bittensor.proto.ReturnCode.EmptyRequest
            forward_request.message = 'Bad endpoint.'
            self.forward_log(request = forward_request, is_response = False, inputs = list(inputs.shape))
            return forward_request

        # ---- Inputs Serialization ----
        try:
            serializer = bittensor.serializer( bittensor.proto.Serializer.MSGPACK )
            forward_request.serialized_inputs = serializer.serialize(inputs, modality = modality, from_type = bittensor.proto.TensorType.TORCH)
            
        except Exception as e:
            forward_request.code =  bittensor.proto.ReturnCode.RequestSerializationException
            forward_request.message = 'Input serialization exception with error:{}'.format(str(e))
            self.forward_log(request = forward_request, is_response = False, inputs = list(inputs.shape))
            return forward_request
            
        try: 
            # ---- Build request ----
            forward_request.request = bittensor.proto.TensorMessage (
                version = bittensor.__version_as_int__,
                hotkey = self.wallet.hotkey.ss58_address,
                tensors = [forward_request.serialized_inputs],
                requires_grad = True,
            )
        except Exception as e:
            forward_request.code = bittensor.proto.ReturnCode.UnknownException
            forward_request.message = str(e)
            self.forward_log(request = forward_request, is_response = False, inputs = list(forward_request.serialized_inputs.shape))
            return forward_request

        forward_request.code = bittensor.proto.ReturnCode.Success
        return forward_request
        
    def forward_call(self, forward_request, timeout):
        r""" Torch.nn.Module forward call: Triggers the grpc call to the remote endpoint. (calls the Forward method on an Axon terminal.)
            The resulted future of forward call was stored in forward_request.

            Args:            
                timeout (:type:`int`, `required`):
                    request timeout.
        """
        # ---- Return if the previous statue was not finished. ----
        if (forward_request.request == None) or (forward_request.code != bittensor.proto.ReturnCode.Success):
            return forward_request
        
        # ---- Make RPC call ----
        try:
            self.stats.forward_qps.update(1)
            self.stats.forward_bytes_out.update(sys.getsizeof(forward_request.request))

            # Forwarding grpc request to the server.
            forward_request.future = self.stub.Forward.future(request = forward_request.request, 
                                            timeout = timeout,
                                            metadata = (
                                                    ('rpc-auth-header','Bittensor'),
                                                    ('bittensor-signature',self.sign()),
                                                    ('bittensor-version',str(bittensor.__version_as_int__)),
                                                    ('request_type', str(bittensor.proto.RequestType.FORWARD)),
                                                    ))
            
            forward_request.code = bittensor.proto.ReturnCode.Success
            self.forward_log(request = forward_request, is_response = False, inputs = list(forward_request.serialized_inputs.shape))
            return forward_request
        
        
        # ---- Catch GRPC Errors ----
        except grpc.RpcError as rpc_error_call:
            forward_request.code, forward_request.message =  self.rpc_exception_handler(forward_request, rpc_error_call)
            self.forward_log(request = forward_request, is_response = False, inputs = list(forward_request.serialized_inputs.shape))
            return forward_request

        # ---- Catch Unknown Errors ----
        except Exception as e:
            forward_request.code = bittensor.proto.ReturnCode.UnknownException
            forward_request.message = str(e)
            self.forward_log(request = forward_request, is_response = False, inputs = list(forward_request.serialized_inputs.shape))
            return forward_request

    def forward_collect(self, forward_request):
        r""" Collect the future.result(), and deserialize the output.
        
            Returns:
                output (:obj:`Tuple[torch.FloatTensor`, torch.LongTensor]`, `optional`):
                    Result from forward call. May be None in the case of failure.

                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Return code associated with forward call.

                time (:type:`float`, `required`):
                    Length of call in seconds.

                message (:type:`str`, `required`): 
                    message associated with forward call, potentially error, or 'success'.
        """ 
        if (forward_request.code != bittensor.proto.ReturnCode.Success) or (forward_request.future == None):
            return forward_request.zeros, forward_request.code, clock.time() - forward_request.start_time

        
        try:
            response = forward_request.future.result()
            self.stats.forward_bytes_in.update(sys.getsizeof(response))
            self.stats.forward_elapsed_time.update((clock.time()-forward_request.start_time))

            # ---- Get response message ----
            try:
                response_message = response.message 
            except Exception:
                response_message = ''

            # ---- Catch non-code ----
            bittensor_code = response.return_code

            if bittensor_code == bittensor.proto.ReturnCode.NoReturn:
                forward_request.message = 'No return code.'
                self.forward_log(request = forward_request, is_response = True, inputs = list(forward_request.inputs.shape))
                return forward_request.zeros, bittensor_code, clock.time() - forward_request.start_time

            # ---- Catch bittensor errors ----
            if bittensor_code == bittensor.proto.ReturnCode.UnknownException:
                forward_request.message = 'Return code unknown exception.'
                self.forward_log(request = forward_request, is_response = True, inputs = list(forward_request.inputs.shape))
                return forward_request.zeros, bittensor_code, clock.time() - forward_request.start_time

            elif bittensor_code != bittensor.proto.ReturnCode.Success:
                self.forward_log(request = forward_request, is_response = True, inputs = list(forward_request.inputs.shape))
                return forward_request.zeros, bittensor_code, clock.time() - forward_request.start_time

        # ---- Catch GRPC Errors ----
        except grpc.RpcError as rpc_error_call:
            forward_request.code, forward_request.message =  self.rpc_exception_handler(forward_request, rpc_error_call)
            return forward_request.zeros, forward_request.code, clock.time() - forward_request.start_time

        # ---- Catch Unknown Errors ----
        except Exception as e:
            forward_request.code = bittensor.proto.ReturnCode.UnknownException
            forward_request.message = str(e)
            self.forward_log(request = forward_request, is_response = True, inputs = list(forward_request.inputs.shape))
            return forward_request.zeros, forward_request.code, clock.time() - forward_request.start_time

        # ---- Check tensor response length ----
        if len(response.tensors) == 0:
            forward_request.code = bittensor.proto.ReturnCode.EmptyResponse
            forward_request.message = 'No tensors in response.'
            self.forward_log(request = forward_request, is_response = True, inputs = list(forward_request.inputs.shape))
            return forward_request.zeros, forward_request.code, clock.time() - forward_request.start_time

        # ---- Deserialize response ----
        try:
            outputs = response.tensors[0]
            deserializer = bittensor.serializer(  outputs.serializer )
            outputs = deserializer.deserialize( outputs, to_type = bittensor.proto.TensorType.TORCH )

        except Exception as e:
            forward_request.code = bittensor.proto.ReturnCode.ResponseDeserializationException
            forward_request.message = 'Deserialziation exception with error:{}'.format(str(e))
            self.forward_log(request = forward_request, is_response = True, inputs = list(forward_request.inputs.shape))
            return forward_request.zeros, forward_request.code, clock.time() - forward_request.start_time
    
        # ---- Check response shape ----
        if  (
            outputs.size(0) != forward_request.inputs.size(0) or
            outputs.size(1) != forward_request.inputs.size(1) or 
            outputs.size(2) != bittensor.__network_dim__
            ):
            forward_request.code = bittensor.proto.ReturnCode.ResponseShapeException
            forward_request.message = "output.shape:{} does not match inputs:{}".format(outputs.shape, forward_request.inputs.shape)
            self.forward_log(request = forward_request, is_response = True, inputs = list(forward_request.inputs.shape), outputs = list(outputs.shape))
            return forward_request.zeros, forward_request.code, clock.time() - forward_request.start_time

        # ---- Safe catch NaNs and replace with 0.0 ----
        outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
        
        # ---- Return ----
        forward_request.code = response.return_code
        forward_request.message = response_message
        self.forward_log(request = forward_request, is_response = True, inputs = list(forward_request.inputs.shape), outputs = list(outputs.shape))
        
        self.stats.codes[forward_request.code] += 1
        call_time = clock.time() - forward_request.start_time
        
        del forward_request
        return outputs, response.return_code, call_time

    def rpc_exception_handler(self, forward_request, rpc_error_call):
        r""" Handle the rpc exception call according to grpc status code.
        """
        grpc_code = rpc_error_call.code()

        if grpc_code == grpc.StatusCode.DEADLINE_EXCEEDED:
            forward_request.code = bittensor.proto.ReturnCode.Timeout
            forward_request.message = 'grpc.StatusCode.DEADLINE_EXCEEDED'+': '+ rpc_error_call.details()
            self.forward_log(request = forward_request, is_response = True, inputs = list(forward_request.inputs.shape))
            return forward_request.code, forward_request.message

        elif grpc_code == grpc.StatusCode.UNAVAILABLE:
            forward_request.code = bittensor.proto.ReturnCode.Unavailable
            forward_request.message = 'grpc.StatusCode.UNAVAILABLE'+': '+ rpc_error_call.details()
            self.forward_log(request = forward_request, is_response = True, inputs = list(forward_request.inputs.shape))
            return forward_request.code, forward_request.message

        elif grpc_code == grpc.StatusCode.UNAUTHENTICATED:
            forward_request.code = bittensor.proto.ReturnCode.Unauthenticated
            forward_request.message = 'grpc.StatusCode.UNAUTHENTICATED'+': '+ rpc_error_call.details()
            self.forward_log(request = forward_request, is_response = True, inputs = list(forward_request.inputs.shape))
            return forward_request.code, forward_request.message
        else:
            forward_request.code = bittensor.proto.ReturnCode.UnknownException
            forward_request.message = 'GRPC error code: {}, details: {}'.format( grpc_code, str(rpc_error_call.details()) )
            self.forward_log(request = forward_request, is_response = True, inputs = list(forward_request.inputs.shape))
            return forward_request.code, forward_request.message

    def _call_backward(
            self,
            inputs_x: torch.Tensor, 
            grads_dy: torch.FloatTensor, 
            modality: bittensor.proto.Modality,
            timeout: int
        ) -> Tuple[torch.Tensor, int, float, str]:
        """ Checks and makes RPC Forward call to a remote neuron (calls the Forward method on an Axon terminal of the endpoint)

            Args:
                inputs_x (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be sent to the caller associated endpoint neurons.
  
                grads_dy (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Gradients of this function's outputs computed during the loss.backward() call.
                
                timeout (int):
                    request timeout.

            Returns:
                outputs (:obj:`Tuple[torch.FloatTensor`, torch.LongTensor]`, `optional`):
                    Gradients of the inputs with respect to the inputs and grads of the outputs.

                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Return code associated with backward call.

                time (:type:`float`, `required`):
                    Length of call in seconds.

                message (:type:`str`, `required`):
                    Message associated with forward call, potentially error, or 'success'.

        """
        start_time = clock.time()
        # ---- Zeros response in the case of failure ----
        zeros = nill_response_for( inputs_x )
 
        # ---- Check inputs size ----
        if torch.numel( inputs_x ) == 0:
            code = bittensor.proto.ReturnCode.EmptyRequest
            message = 'empty request'
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log(axon=False, forward=False, is_response=False, code=code, call_time=call_time, pubkey=self.endpoint.hotkey, uid = self.endpoint.uid, inputs=list(grads_dy.shape), outputs=None, message=message)
            return zeros, code, call_time, message
        
        if self.endpoint.uid == -1:
            code = bittensor.proto.ReturnCode.EmptyRequest
            message = 'bad endpoint'
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log(axon=False, forward=False, is_response=False, code=code, call_time=call_time, pubkey=self.endpoint.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message )
            return zeros, code, call_time, message


        # ---- Check grads size ----
        if torch.numel( grads_dy ) == 0:
            code = bittensor.proto.ReturnCode.EmptyRequest
            message = 'empty request'
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log(axon=False, forward=False, is_response=False, code=code, call_time=call_time, pubkey=self.endpoint.hotkey, uid = self.endpoint.uid, inputs=list(grads_dy.shape), outputs=None, message=message)
            return zeros, code, call_time, message

        # ---- Serialization ----
        try:
            serializer = bittensor.serializer( bittensor.proto.Serializer.MSGPACK )
            serialized_inputs = serializer.serialize (inputs_x, modality = modality, from_type = bittensor.proto.TensorType.TORCH )
            serialized_grads = serializer.serialize (grads_dy, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH )
        except Exception as e:
            code = bittensor.proto.ReturnCode.RequestSerializationException
            message = 'serializer exception with error:{}'.format(e)
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log(axon=False, forward=False, is_response=False, code=code, call_time=call_time, pubkey=self.endpoint.hotkey, uid = self.endpoint.uid, inputs=list(grads_dy.shape), outputs=None, message=message)
            return zeros, code, call_time, message
        
        # ---- Make RPC call ----
        try:
            request = bittensor.proto.TensorMessage(
                version = bittensor.__version_as_int__,
                hotkey = self.wallet.hotkey.ss58_address,
                tensors = [serialized_inputs, serialized_grads],
                requires_grad = True,
            )
            
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log(axon=False, forward=False, is_response=False, code=bittensor.proto.ReturnCode.Success, call_time=call_time, pubkey=self.endpoint.hotkey, uid = self.endpoint.uid, inputs=list(grads_dy.shape), outputs=None, message=None)
            response = self.stub.Backward(request = request, 
                                          timeout = timeout,
                                          metadata = (
                                                    ('rpc-auth-header','Bittensor'),
                                                    ('bittensor-signature',self.sign()),
                                                    ('bittensor-version',str(bittensor.__version_as_int__)),
                                                    ('request_type', str(bittensor.proto.RequestType.BACKWARD)),
                                                    ))

            # Get message
            try:
                response_message = response.message 
            except Exception:
                response_message = ''

        # ---- Catch GRPC Errors ----
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                code = bittensor.proto.ReturnCode.Timeout
                message = 'grpc.StatusCode.DEADLINE_EXCEEDED'+': '+ e.details()
                call_time = clock.time() - start_time
                bittensor.logging.rpc_log(axon=False, forward=False, is_response=True, code=code, call_time=call_time, pubkey=self.endpoint.hotkey, uid = self.endpoint.uid, inputs=list(grads_dy.shape), outputs=None, message=message)
                return zeros, code, call_time, message
            
            elif e.code() == grpc.StatusCode.UNAVAILABLE:
                code = bittensor.proto.ReturnCode.Unavailable
                message = 'grpc.StatusCode.UNAVAILABLE'+': '+ e.details()
                call_time = clock.time() - start_time
                bittensor.logging.rpc_log(axon=False, forward=False, is_response=True, code=code, call_time=call_time, pubkey=self.endpoint.hotkey, uid = self.endpoint.uid, inputs=list(grads_dy.shape), outputs=None, message=message)
                return zeros, code, call_time, message
            
            elif e.code() == grpc.StatusCode.UNAUTHENTICATED:
                code = bittensor.proto.ReturnCode.Unauthenticated
                message = 'grpc.StatusCode.UNAUTHENTICATED'+': '+ e.details()
                call_time = clock.time() - start_time
                bittensor.logging.rpc_log(axon=False, forward=False, is_response=True, code=code, call_time=call_time, pubkey=self.endpoint.hotkey, uid = self.endpoint.uid, inputs=list(grads_dy.shape), outputs=None, message=message)
                return zeros, code, call_time, message

            else:
                code = bittensor.proto.ReturnCode.UnknownException
                message = 'grpc error code:{}, details: {}'.format(str(e.code()), str(e.details()))
                call_time = clock.time() - start_time
                bittensor.logging.rpc_log(axon=False, forward=False, is_response=True, code=code, call_time=call_time, pubkey=self.endpoint.hotkey, uid = self.endpoint.uid, inputs=list(grads_dy.shape), outputs=None, message=message)
                return zeros, code, call_time, message

        # ---- Catch Unknown RPC Errors ----
        except Exception as e:
            code = bittensor.proto.ReturnCode.UnknownException
            message = str(e)
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log(axon=False, forward=False, is_response=True, code=code, call_time=call_time, pubkey=self.endpoint.hotkey, uid = self.endpoint.uid, inputs=list(grads_dy.shape), outputs=None, message=message)
            return zeros, code, call_time, message

        # ---- Catch Code Errors ----
        try:
            bittensor_code = response.return_code
        except:
            bittensor_code = bittensor.proto.ReturnCode.NoReturn

        if bittensor_code == bittensor.proto.ReturnCode.NoReturn:
            code = bittensor.proto.ReturnCode.NoReturn
            message = 'no response code.'
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log(axon=False, forward=False, is_response=True, code=code, call_time=call_time, pubkey=self.endpoint.hotkey, uid = self.endpoint.uid, inputs=list(grads_dy.shape), outputs=None, message=message)
            return zeros, code, call_time, message

        # ---- Catch negative codes ----
        if bittensor_code != bittensor.proto.ReturnCode.Success:
            code = bittensor_code
            message = response_message
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log(axon=False, forward=False, is_response=True, code=code, call_time=call_time, pubkey=self.endpoint.hotkey, uid = self.endpoint.uid, inputs=list(grads_dy.shape), outputs=None, message=response_message)
            return zeros, code, call_time, message

        # ---- Check for empty response ----
        if len(response.tensors) == 0:
            code = bittensor.proto.ReturnCode.EmptyResponse
            message = 'empty response tensor.'
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log(axon=False, forward=False, is_response=True, code=code, call_time=call_time, pubkey=self.endpoint.hotkey, uid = self.endpoint.uid, inputs=list(grads_dy.shape), outputs=None, message=message)
            return zeros, code, call_time, message

        # ---- Post-process request ----
        try:
            outputs = response.tensors[0]
            deserializer = bittensor.serializer( outputs.serializer )
            outputs = deserializer.deserialize( outputs, to_type = bittensor.proto.TensorType.TORCH )
        except Exception as e:
            code = bittensor.proto.ReturnCode.ResponseDeserializationException
            message = 'deserialization exception with error:{}'.format(e)
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log(axon=False, forward=False, is_response=True, code=code, call_time=call_time, pubkey=self.endpoint.hotkey, uid = self.endpoint.uid, inputs=list(grads_dy.shape), outputs=None, message=message)
            return zeros, code, call_time, message
        try:
            # ---- Check response shape is same as inputs ----
            if  outputs.size() != inputs_x.size():

                code = bittensor.proto.ReturnCode.ResponseShapeException 
                message = 'output shape does not match inputs shape'
                call_time = clock.time() - start_time
                bittensor.logging.rpc_log(axon=False, forward=False, is_response=True, code=code, call_time=call_time, pubkey=self.endpoint.hotkey, uid = self.endpoint.uid, inputs=list(grads_dy.shape), outputs=list(outputs.shape), message=message)
                return zeros, code, call_time, message
        except Exception as e:
            code = bittensor.proto.ReturnCode.UnknownException
            message = 'Size Error: {}'.format(e)
            call_time = clock.time() - start_time

            bittensor.logging.rpc_log(axon=False, forward=False, is_response=True, code=code, call_time=call_time, pubkey=self.endpoint.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message )
            return zeros, code, call_time, message
            
        # ---- Safe catch NaNs and replace with 0.0 ----
        outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
   
        # ---- Return ----
        code = bittensor.proto.ReturnCode.Success
        message = 'success'
        call_time = clock.time() - start_time
        bittensor.logging.rpc_log(axon=False, forward=False, is_response=True, code=code, call_time=call_time, pubkey=self.endpoint.hotkey, uid = self.endpoint.uid, inputs=list(grads_dy.shape), outputs=list(outputs.shape), message=response_message)
        return outputs, code, clock.time() - start_time, message

    def sign(self):
        r""" Uses the wallet pubkey to sign a message containing the pubkey and the time
        """
        nounce = self.nounce()
        message  = str(nounce) + str(self.wallet.hotkey.ss58_address) + str(self.receptor_uid)
        spliter = 'bitxx'
        signature = spliter.join([ str(nounce), str(self.wallet.hotkey.ss58_address), self.wallet.hotkey.sign(message), str(self.receptor_uid) ])
        return signature
    
    def nounce(self):
        r"""creates a string representation of the time
        """
        nounce = int(time.time() * 1000)
        return nounce
        
    def state(self):
        try: 
            return self.state_dict[self.channel._channel.check_connectivity_state(True)]
        except ValueError:
            return "Channel closed"