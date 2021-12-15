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

class Request():
    """ Contains all of the inputs, intermediate, and output state of a forward/backward request. 
    """
    def __init__(
        self,
        inputs, 
        modality,
        grads_dy = None,
        backward = False
        ):
        r""" Initialize a forward/backward request.

            Args: 
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    List of tensors to send to corresponsing endpoints. Tensors are of arbitrary type and shape depending on the
                    modality.

                grads_dy (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`):
                    List of grad tensors to send to corresponsing inputs. Only needed when it is a backward request.

                modality (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality type. Enum in [TEXT, IMAGE, TENSOR]

                backward (:type:`Bool`);
                    True if it is a backward request. False when it is a forward request instead.
        """
        # ---- Inputs ----
        self.inputs = inputs
        self.grads_dy = grads_dy
        self.zeros = nill_response_for(inputs)
        
        # ---- Setups ----
        self.modality = modality
        self.backward = backward
        self.start_time = clock.time()
        self.end_time = None

        # ---- Intermediate states ---- 
        self.serialized_inputs = None
        self.grpc_request = None
        self.future = None

        # ---- Outputs ----
        self.code = None
        self.message = None
        self.outputs = None

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
        request = self.preprocess_request ( inputs = inputs, modality = modality)
        request = self.make_request_call(request, timeout = timeout)
        return self.handle_request_response(request)


    def backward(
            self, 
            inputs_x: torch.Tensor, 
            grads_dy: torch.Tensor, 
            modality: bittensor.proto.Modality,
            timeout: int
        ) -> Tuple[ torch.Tensor, int, float, str ]:
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
        request = self.preprocess_request (inputs = inputs_x, modality = modality, grads_dy = grads_dy, backward = True)
        request = self.make_request_call(request, timeout = timeout)
        return self.handle_request_response(request)
            

    def prerequisite_check(self, request):
        r""" Check the input size and endpoint validity.

            Args:
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.

            Returns:
                success: (:type:`bool`, `required`):
                    True if the check has passed.
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.
        """

        # ---- Check inputs size ----
        if torch.numel(request.inputs) == 0 or ( request.backward and torch.numel( request.grads_dy ) == 0):
            request.code = bittensor.proto.ReturnCode.EmptyRequest
            request.message = 'Empty request.'
            self.request_log(request = request, is_response = False, inputs = list(request.inputs.shape))
            return False, request
        
        # ---- Check endpoint----
        if self.endpoint.uid == -1:
            request.code = bittensor.proto.ReturnCode.EmptyRequest
            request.message = 'Bad endpoint.'            
            self.request_log(request = request, is_response = False, inputs = list(request.inputs.shape))
            return False, request
        
        return True, request

    def serialization(self, request):
        r""" Does the serialization to the request inputs and grads(backward request only).
            The result would update request.serialized_inputs and request.serialized_grad.

            Args:
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.

            Returns:
                success: (:type:`bool`, `required`):
                    True if the serialization is successful.
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.
        """
        try:
            serializer = bittensor.serializer( bittensor.proto.Serializer.MSGPACK )
            request.serialized_inputs = serializer.serialize(request.inputs, modality = request.modality, from_type = bittensor.proto.TensorType.TORCH)

            if request.backward:
                request.serialized_grads = serializer.serialize (request.grads_dy, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH )

        except Exception as e:
            request.code =  bittensor.proto.ReturnCode.RequestSerializationException
            request.message = 'Input serialization exception with error:{}'.format(str(e))
            self.request_log(request = request, is_response = False, inputs = list(request.inputs.shape))
            return False, request
        
        return True, request

    def build_grpc_request(self, request):
        r"""Build the grapc call with the serialized_inputs and serialized grad(backward request only). 
            The result would update request.grpc_request.

            Args:
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.

            Returns:
                success: (:type:`bool`, `required`):
                    True if the build is successful.
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.
        """
        try: 
            if not request.backward:
                request.grpc_request = bittensor.proto.TensorMessage (
                    version = bittensor.__version_as_int__,
                    hotkey = self.wallet.hotkey.ss58_address,
                    tensors = [request.serialized_inputs],
                    requires_grad = True,
                )
            else:
                request.grpc_request = bittensor.proto.TensorMessage(
                    version = bittensor.__version_as_int__,
                    hotkey = self.wallet.hotkey.ss58_address,
                    tensors = [request.serialized_inputs, request.serialized_grads],
                    requires_grad = True,
                )

        except Exception as e:
            request.code = bittensor.proto.ReturnCode.UnknownException
            request.message = str(e)
            self.request_log(request = request, is_response = False, inputs = list(request.serialized_inputs.shape))
            return False, request
        
        return True, request

    def collect_future(self, request):
        r"""Get the result of the grpc request. 
            The result would update request.response.

            Args:
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.

            Returns:
                success: (:type:`bool`, `required`):
                    True if getting the result is successful.
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.
        """
        try:
            request.response = request.future.result()
            self.stats.forward_bytes_in.update(sys.getsizeof(request.response))
            self.stats.forward_elapsed_time.update((clock.time()-request.start_time))
            
        # ---- Catch GRPC Errors ----
        except grpc.RpcError as rpc_error_call:
            request.code, request.message =  self.rpc_exception_handler(request, rpc_error_call)
            return False, request

        # ---- Catch Unknown Errors ----
        except Exception as e:
            request.code = bittensor.proto.ReturnCode.UnknownException
            request.message = str(e)
            self.request_log(request = request, is_response = True, inputs = list(request.inputs.shape))
            return False, request

        return True, request

    def check_response(self, request):
        r"""Check the response. 
            This function should not update any part of request.

            Args:
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.

            Returns:
                success: (:type:`bool`, `required`):
                    True if the check is successful.
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.
        """
        # ---- Get response message ----
        try:
            request.message = request.response.message 
        except Exception:
            request.message = ''

        # ---- Catch non-code ----
        request.code = request.response.return_code

        if request.code == bittensor.proto.ReturnCode.NoReturn:
            request.message = 'No return code.'
            self.request_log(request = request, is_response = True, inputs = list(request.inputs.shape))
            return False, request

        # ---- Catch bittensor errors ----
        if request.code == bittensor.proto.ReturnCode.UnknownException:
            request.message = 'Return code unknown exception.'
            self.request_log(request = request, is_response = True, inputs = list(request.inputs.shape))
            return False, request

        elif request.code != bittensor.proto.ReturnCode.Success:
            self.request_log(request = request, is_response = True, inputs = list(request.inputs.shape))
            return False, request

        # ---- Check for empty length ----
        if len(request.response.tensors) == 0:
            request.code = bittensor.proto.ReturnCode.EmptyResponse
            request.message = 'No tensors in response.'
            self.request_log(request = request, is_response = True, inputs = list(request.inputs.shape))
            return False, request
        
        return True, request

    def deserialize_forward_response(self, request):
        r"""Deserialization for the forward request.
            The result would update request.output.

            Args:
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.

            Returns:
                success: (:type:`bool`, `required`):
                    True if the deserialization is successful.
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.
        """

        # ---- Deserialize response ----
        try:
            outputs = request.response.tensors[0]
            deserializer = bittensor.serializer(  outputs.serializer )
            outputs = deserializer.deserialize( outputs, to_type = bittensor.proto.TensorType.TORCH )

        except Exception as e:
            request.code = bittensor.proto.ReturnCode.ResponseDeserializationException
            request.message = 'Deserialziation exception with error:{}'.format(str(e))
            self.request_log(request = request, is_response = True, inputs = list(request.inputs.shape))
            return False, request

        # ---- Check response shape ----
        if  (
            outputs.size(0) != request.inputs.size(0) or
            outputs.size(1) != request.inputs.size(1) or 
            outputs.size(2) != bittensor.__network_dim__
            ):
            request.code = bittensor.proto.ReturnCode.ResponseShapeException
            request.message = "output.shape:{} does not match inputs:{}".format(outputs.shape, request.inputs.shape)
            self.request_log(request = request, is_response = True, inputs = list(request.inputs.shape), outputs = list(outputs.shape))
            return False, request

        # ---- Safe catch NaNs and replace with 0.0 ----
        request.outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
        
        # ---- Return ----
        request.code = request.response.return_code
        self.request_log(request = request, is_response = True, inputs = list(request.inputs.shape), outputs = list(outputs.shape))
        self.stats.codes[request.code] += 1
        
        return True, request 

    def deserialize_backward_response(self, request):
        r"""Deserialization for the backward request.
            The result would update request.output.

            Args:
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.

            Returns:
                success: (:type:`bool`, `required`):
                    True if the deserialization is successful.
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.
        """
        # ---- Post-process request ----
        try:
            outputs = request.response.tensors[0]
            deserializer = bittensor.serializer( outputs.serializer )
            outputs = deserializer.deserialize( outputs, to_type = bittensor.proto.TensorType.TORCH )
        except Exception as e:
            request.code = bittensor.proto.ReturnCode.ResponseDeserializationException
            request.message = 'deserialization exception with error:{}'.format(e)
            self.request_log(request = request, is_response = True, inputs = list(request.inputs.shape))
            return False, request

        try:
            # ---- Check response shape is same as inputs ----
            if  outputs.size() != request.inputs.size():
                request.code = bittensor.proto.ReturnCode.ResponseShapeException 
                request.message = 'output shape does not match inputs shape'
                self.request_log(request = request, is_response = True, inputs = list(request.inputs.shape))
                return False, request
        
        except Exception as e:
            request.code = bittensor.proto.ReturnCode.UnknownException
            request.message = 'Size Error: {}'.format(e)
            self.request_log(request = request, is_response = True, inputs = list(request.inputs.shape))
            return False, request

        # ---- Safe catch NaNs and replace with 0.0 ----
        request.outputs = torch.where(torch.isnan(outputs), torch.zeros_like(outputs), outputs)
   
        # ---- Return ----
        request.code = bittensor.proto.ReturnCode.Success
        request.message = 'Success'
        self.request_log(request = request, is_response = True, inputs = list(request.inputs.shape))
        self.stats.codes[request.code] += 1
        return False, request

    def request_log(self, request, is_response = False, inputs = None, outputs = None):
        r""" rpc logging for forward/backward request
            Args:
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.

                is_response (:type: `bool`):
                    True if we are logging a response from the grpc call, false if it is a request instead

                inputs (:type: `List`): 
                    shape of the tensor input that was being handled
        """

        call_time = clock.time() - request.start_time
        if bittensor.logging.__debug_on__: 
            bittensor.logging.rpc_log(
                axon=False, 
                forward= not request.backward, 
                is_response=is_response, 
                code=request.code, 
                call_time=call_time, 
                pubkey=self.endpoint.hotkey, 
                uid = self.endpoint.uid, 
                inputs=inputs, 
                outputs=outputs, 
                message=request.message
            )

    def preprocess_request (
        self, 
        inputs: torch.Tensor, 
        modality: bittensor.proto.Modality,
        grads_dy: torch.FloatTensor = None,
        backward: str = False 
    ):  
        r""" Does all the checking and preprocessing to build the grpc request.
            
            Args:  
                inputs (:obj:`List[torch.Tensor]` of shape :obj:`(shape)`, `required`):
                    Torch tensor to be sent to this endpoint.

                modality (:obj:`bittensor.proto.Modality` of shape :obj:`(1)`, `required`):
                    Bittensor forward modality of type Enum: [TEXT, IMAGE, TENSOR]
                
                grads_dy (:obj:`List[torch.Tensor]` of shape :obj:`(num_endpoints * [shape])`, `required`):
                    List of grad tensors to send to corresponsing inputs. 

                backward (:type:`Bool`, `required`);
                    If the request is a backward request.

            Returns:
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.
        """
        # ---- Setup forward request namespace, which will hold all the objects regarding the forward request ----
        request = Request(inputs = inputs, modality = modality, grads_dy = grads_dy, backward = backward)

        preprocessing_funs = [self.prerequisite_check, self.serialization, self.build_grpc_request]

        for fun in preprocessing_funs:
            check, request = fun(request)
            if not check:
                return request 

        request.code = bittensor.proto.ReturnCode.Success
        return request
        
    def make_request_call(self, request, timeout):
        r""" Torch.nn.Module forward call: Triggers the grpc call to the remote endpoint. (calls the Forward method on an Axon terminal.)
            The resulted future of forward call was stored in forward_request.

            Args:            
                timeout (:type:`int`, `required`):
                    request timeout.

            Returns:
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.
        """
        # ---- Return if the previous statue was not finished. ----
        if (request.grpc_request == None) or (request.code != bittensor.proto.ReturnCode.Success):
            return request
        
        # ---- Make RPC call ----
        try:
            if not request.backward:
                self.stats.forward_qps.update(1)
                self.stats.forward_bytes_out.update(sys.getsizeof(request.grpc_request))
                request.future = self.stub.Forward.future(request = request.grpc_request, 
                                timeout = timeout,
                                metadata = (
                                        ('rpc-auth-header','Bittensor'),
                                        ('bittensor-signature',self.sign()),
                                        ('bittensor-version',str(bittensor.__version_as_int__)),
                                        ('request_type', str(bittensor.proto.RequestType.FORWARD)),
                                        ))
                request.future.add_done_callback(lambda z : self.handle_request_response(request))
            else:
                self.stats.backward_qps.update(1)
                self.stats.backward_bytes_out.update(sys.getsizeof(request.grpc_request))
                request.future = self.stub.Backward.future(request = request.grpc_request, 
                                timeout = timeout,
                                metadata = (
                                        ('rpc-auth-header','Bittensor'),
                                        ('bittensor-signature',self.sign()),
                                        ('bittensor-version',str(bittensor.__version_as_int__)),
                                        ('request_type', str(bittensor.proto.RequestType.BACKWARD)),
                                        ))
            
            request.code = bittensor.proto.ReturnCode.Success
            self.request_log(request = request, is_response = False, inputs = list(request.serialized_inputs.shape))
            return request
        
        # ---- Catch GRPC Errors ----
        except grpc.RpcError as rpc_error_call:
            request.code, request.message =  self.rpc_exception_handler(request, rpc_error_call)
            self.request_log(request = request, is_response = False, inputs = list(request.serialized_inputs.shape))
            return request

        # ---- Catch Unknown Errors ----
        except Exception as e:
            request.code = bittensor.proto.ReturnCode.UnknownException
            request.message = str(e)
            self.request_log(request = request, is_response = False, inputs = list(request.serialized_inputs.shape))
            return request

    def handle_request_response(self, request):
        r""" Handle all the getting result checking, and processing the response.

            Args:
                request: (:obj:`Request`, required):
                    The request object holds all specifications and processing of the request.

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
        if request.outputs != None:
            return request.outputs, request.code, request.end_time

        if (request.code != bittensor.proto.ReturnCode.Success) or (request.future == None):
            request.end_time = clock.time() - request.start_time
            return request.zeros, request.code, clock.time() - request.start_time

        deserializer = self.deserialize_forward_response if not request.backward else self.deserialize_backward_response
        response_handling_funs = [self.collect_future, self.check_response, deserializer]

        for fun in response_handling_funs:
            check, request = fun(request)
            if not check:
                request.end_time = clock.time() - request.start_time
                return request.zeros, request.code, clock.time() - request.start_time
        
        request.end_time = clock.time() - request.start_time
        return request.outputs if check else request.zeros, request.code, request.end_time
 

    def rpc_exception_handler(self, request, rpc_error_call):
        r""" Handle the rpc exception call according to grpc status code.
        """
        grpc_code = rpc_error_call.code()

        if grpc_code == grpc.StatusCode.DEADLINE_EXCEEDED:
            request.code = bittensor.proto.ReturnCode.Timeout
            request.message = 'grpc.StatusCode.DEADLINE_EXCEEDED'+': '+ rpc_error_call.details()
            self.request_log(request = request, is_response = True, inputs = list(request.inputs.shape))
            return request.code, request.message

        elif grpc_code == grpc.StatusCode.UNAVAILABLE:
            request.code = bittensor.proto.ReturnCode.Unavailable
            request.message = 'grpc.StatusCode.UNAVAILABLE'+': '+ rpc_error_call.details()
            self.request_log(request = request, is_response = True, inputs = list(request.inputs.shape))
            return request.code, request.message

        elif grpc_code == grpc.StatusCode.UNAUTHENTICATED:
            request.code = bittensor.proto.ReturnCode.Unauthenticated
            request.message = 'grpc.StatusCode.UNAUTHENTICATED'+': '+ rpc_error_call.details()
            self.request_log(request = request, is_response = True, inputs = list(request.inputs.shape))
            return request.code, request.message
        else:
            request.code = bittensor.proto.ReturnCode.UnknownException
            request.message = 'GRPC error code: {}, details: {}'.format( grpc_code, str(rpc_error_call.details()) )
            self.request_log(request = request, is_response = True, inputs = list(request.inputs.shape))
            return request.code, request.message


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