""" Implementation of Axon, services Forward and Backward requests from other neurons.
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
from typing import List, Tuple, Callable

import torch
import grpc
from loguru import logger
import torch.nn.functional as F
import concurrent

import bittensor
import bittensor.utils.stats as stat_utils

logger = logger.opt(colors=True)

class Axon( bittensor.grpc.BittensorServicer ):
    r""" Services Forward and Backward requests from other neurons.
    """
    def __init__( 
        self, 
        wallet: 'bittensor.wallet',
        ip: str,
        port: int,
        server: 'grpc._Server',
        forwards: List  = [],
        backwards: List = [],
        priority:  'Callable' = None,
        priority_threadpool: 'bittensor.prioritythreadpool' = None,
        forward_timeout: int = None,
        backward_timeout: int = None,
    ):
        r""" Initializes a new Axon tensor processing endpoint.
            
            Args:
                config (:obj:`bittensor.Config`, `required`): 
                    bittensor.axon.config()
                wallet (:obj:`bittensor.wallet`, `required`):
                    bittensor wallet with hotkey and coldkeypub.
                server (:obj:`grpc._Server`, `required`):
                    Grpc server endpoint.
                forward (:obj:list of `callable`, `optional`):
                    list of functions which is called on forward requests.
                backward (:obj:list of `callable`, `optional`):
                    list of functions which is called on backward requests.
                priority (:obj:`callable`, `optional`):
                    function to assign priority on requests.
                priority_threadpool (:obj:`bittensor.prioritythreadpool`, `optional`):
                    bittensor priority_threadpool.                
        """
        self.ip = ip
        self.port = port
        self.wallet = wallet
        self.server = server
        self.forward_callback = forwards
        self.backward_callback = backwards
        self.forward_timeout = forward_timeout
        self.backward_timeout = backward_timeout
        self.modality = self.find_modality()
        self.stats = SimpleNamespace(
            qps = stat_utils.timed_rolling_avg(0.0, 0.01),
            qps_failed = stat_utils.timed_rolling_avg(0.0, 0.01),
            total_in_bytes = stat_utils.timed_rolling_avg(0.0, 0.01),
            total_out_bytes= stat_utils.timed_rolling_avg(0.0, 0.01),
            in_bytes_per_pubkey = {},
            out_bytes_per_pubkey = {},
            qps_per_pubkey = {},
            qps_failed_per_pubkey = {},
        )
        self.started = None
        
        # -- Priority 
        self.priority = priority 
        self.priority_threadpool= priority_threadpool

    def __str__(self) -> str:
        return "Axon({}, {}, {}, {})".format( self.ip, self.port, self.wallet.hotkey.ss58_address, "started" if self.started else "stopped")

    def __repr__(self) -> str:
        return self.__str__()

    def Forward(self, request: bittensor.proto.TensorMessage, context: grpc.ServicerContext) -> bittensor.proto.TensorMessage:
        r""" The function called by remote GRPC Forward requests from other neurons.
            Forward is equivalent to a 'forward' pass through a neural network.
            After checking request validity, this function passes the request to the nucleus for processing.
            See :obj:`bittensor.proto.ReturnCode` for all possible return codes.
            
            Args:
                request (:obj:`bittensor.proto`, `required`): 
                    Tensor request proto.
                context (:obj:`grpc.ServicerContext`, `required`): 
                    grpc server context.
            
            Returns:
                response (bittensor.proto.TensorMessage): 
                    proto response carring the nucleus forward output or None under failure.
        """
        tensor, code, _, message = self._forward( request )
        response = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__, 
            hotkey = self.wallet.hotkey.ss58_address, 
            return_code = code,
            message = message,
            tensors = [tensor] if tensor is not None else [],
            requires_grad = True,
        )
        # ---- Update stats for this request.
        self.update_stats_for_request( request, response )
        return response

    def Backward(self, request: bittensor.proto.TensorMessage, context: grpc.ServicerContext) -> bittensor.proto.TensorMessage:
        r""" The function called by remote GRPC Backward requests from other neurons.
            Backward is equivalent to a 'backward' gradient descent pass through a neural network.
            After checking request validity, passes the request to the nucleus for processing.
            See :obj:`bittensor.proto.ReturnCode` for all possible return codes.
            
            Args:
                request (:obj:`bittensor.proto`, `required`): 
                    Tensor request proto.
                context (:obj:`grpc.ServicerContext`, `required`): 
                    grpc server context.
            
            Returns:
                response (:obj:`bittensor.proto.TensorMessage`): 
                    proto response carring the nucleus backward output or None under failure.
        """
        tensor, code, _, message = self._backward( request )
        response = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__, 
            hotkey = self.wallet.hotkey.ss58_address, 
            return_code = code,
            message = message,
            tensors = [tensor] if tensor is not None else [],
            requires_grad = True,
        )
        self.update_stats_for_request( request, response )
        return response

    def _call_forward(
            self, 
            public_key: str, 
            inputs_x: torch.Tensor, 
            modality: bittensor.proto.Modality
        ) -> Tuple[ torch.FloatTensor, int, str ]:
        r""" Calls the forward callback served by the nucleus.
            
            Args:
                public_key (str, `required`): 
                    public key of the sender
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                modality ( bittensor.proto.Modality, `required`):
                    modality of inputs.
            
            Returns:
                response (:obj:`torch.FloatTensor, `required`): 
                    Torch tensor response from miner processes.
                code (:obj:`bittensor.proto.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
                message (str, `required`): 
                    message associated with forward call, potentially error, or 'success'.

        """
        # Check forward has been subscribed.
        if self.forward_callback[modality] == None:
            message = "Forward callback is not yet subscribed on this axon."
            return None, bittensor.proto.ReturnCode.NotImplemented, message
        
        # Make forward call.
        try:
            if self.priority != None:
                priority = self.priority(public_key,inputs_x=inputs_x, request_type = bittensor.proto.RequestType.FORWARD)
                future = self.priority_threadpool.submit(self.forward_callback[modality],inputs_x=inputs_x,priority=priority)

                try:
                    response_tensor = future.result(timeout= self.forward_timeout)
                except concurrent.futures.TimeoutError :
                    raise TimeoutError('TimeOutError')
                except Exception as e:
                    logger.error('Error found: {}, with message {}'.format(repr(e), e))

            else:
                response_tensor = self.forward_callback[modality]( inputs_x= inputs_x)

            message = "Success"
            code = bittensor.proto.ReturnCode.Success
            return response_tensor, code, message

        except Exception as e:
            response_tensor = None
            message = "Error calling forward callback: {}".format(e)
            if isinstance(e, TimeoutError):
                code = bittensor.proto.ReturnCode.Timeout
            else:
                code = bittensor.proto.ReturnCode.UnknownException
            return response_tensor, code, message

    def _call_backward(
            self, 
            public_key: str, 
            inputs_x: torch.Tensor, 
            grads_dy: torch.FloatTensor,
            modality: bittensor.proto.Modality
        ) -> Tuple[ torch.FloatTensor, int, str ]:
        r""" Calls the backward callback.
            
            Args:
                public_key (str, `required`): 
                    public key of the sender
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be backward processed.
                grads_dy ( :obj:`torch.Tensor`, `required`):
                    torch gradient inputs to be backward processed with inputs.
                modality ( bittensor.proto.Modality, `required`):
                    modality of inputs.
            
            Returns:
                response (:obj:`torch.FloatTensor, `required`): 
                    Torch tensor response from miner processes.
                code (:obj:`bittensor.proto.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
                message (str, `required`): 
                    message associated with forward call, potentially error, or 'success'.
        """
        # Check backward has been subscribed.
        if self.backward_callback[modality] == None:
            message = "Backward callback is not yet subscribed on this axon."
            return None, bittensor.proto.ReturnCode.NotImplemented, message

        if modality == bittensor.proto.Modality.TEXT:
            if self.priority != None:
                try:
                    priority = self.priority(public_key,inputs_x=inputs_x, request_type = bittensor.proto.RequestType.BACKWARD)
                    future = self.priority_threadpool.submit(self.backward_callback[modality],inputs_x=inputs_x,grads_dy=grads_dy,priority=priority)
                except concurrent.futures.TimeoutError :
                    raise TimeoutError('TimeOutError')
                except Exception as e:
                    logger.error('Error found: {}, with message {}'.format(repr(e), e))
            else:
                self.backward_callback[modality](inputs_x, grads_dy)

            response_tensor = torch.ones(inputs_x.size())
            message = "Success"
            code = bittensor.proto.ReturnCode.Success
            return response_tensor, code, message
            
        # Make backward call.
        try:
            response_tensor = self.backward_callback[modality]( inputs_x, grads_dy)
            message = "Success"
            code = bittensor.proto.ReturnCode.Success
            return response_tensor, code, message

        except Exception as e:
            response_tensor = None
            message = "Error calling backward callback: {}".format(e)
            if isinstance(e, TimeoutError):
                code = bittensor.proto.ReturnCode.Timeout
            else:
                code = bittensor.proto.ReturnCode.UnknownException

            return response_tensor, code, message 
            
    def _forward(self, request):
        r""" Performs validity checks on the grpc request before passing the tensors to the forward queue.
            Returns the output, message and code from the backend forward call.
            
            Args:
                request (:obj:`bittensor.proto`, `required`): 
                    Tensor request proto.
            Returns:
                response (:obj:`bittensor.proto.Tensor, `required`): 
                    serialized tensor response from the nucleus call or None.
                code (:obj:`bittensor.proto.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
                time (:type:`float`, `required`):
                    Length of call in seconds.
                message (str, `required`): 
                    message associated with forward call, potentially error, or 'success'.
        """
        start_time = clock.time()
        try:
            # ---- Check Empty request ----
            if len(request.tensors) == 0:
                code = bittensor.proto.ReturnCode.EmptyRequest
                message = "Forward request contains {} tensors, expected 1 tensor in the forward call".format(len(request.tensors))
                call_time = clock.time() - start_time
                bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, call_time = call_time, pubkey=request.hotkey, inputs=None, outputs=None, message=message  )
                return None, code, call_time, message

            # ---- Check deserialization ----
            tensor_inputs = request.tensors[0]
            modality = tensor_inputs.modality
            try:
                deserializer = bittensor.serializer( serialzer_type = tensor_inputs.serializer )
                torch_inputs = deserializer.deserialize(tensor_inputs, to_type = bittensor.proto.TensorType.TORCH)
            except Exception as e:
                code = bittensor.proto.ReturnCode.RequestDeserializationException
                message = "Request deserialization exception: {}".format(str(e))
                call_time = clock.time() - start_time
                bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, call_time = call_time, pubkey=request.hotkey, inputs=None, outputs=None, message=message  )
                return None, code, call_time, message

            # ---- Check shape and modality ----
            if list(torch_inputs.shape)[0] < 1:
                code = bittensor.proto.ReturnCode.RequestShapeException
                message = "Forward request batch dim exception with batch_size = {} ".format(list(torch_inputs.shape)[0])
                call_time = clock.time() - start_time
                bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message  )
                return None, code, call_time, message

            if list(torch_inputs.shape)[1] < 1:
                code = bittensor.proto.ReturnCode.RequestShapeException
                message = "Forward request sequence dim exception with sequence_dim = {} ".format(list(torch_inputs.shape)[1])
                call_time = clock.time() - start_time
                bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message  )
                return None, code, call_time, message

            if modality == bittensor.proto.Modality.TEXT:
                if len(list(torch_inputs.shape)) != 2:
                    code = bittensor.proto.ReturnCode.RequestShapeException
                    message = "Forward text input shape exception with len(request.shape) = {} must have rank 2.".format(len(list(torch_inputs.shape)))
                    call_time = clock.time() - start_time
                    bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message  )
                    return None, code, call_time, message
          
            if modality == bittensor.proto.Modality.IMAGE:
                if len(list(torch_inputs.shape)) != 5:
                    code = bittensor.proto.ReturnCode.RequestShapeException
                    message =  "Forward image input shape exception for len(shape) = {}  must have rank 5".format(len(list(torch_inputs.shape)))
                    call_time = clock.time() - start_time
                    bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message  )
                    return None, code, call_time, message

            if modality == bittensor.proto.Modality.TENSOR:
                if len(list(torch_inputs.shape)) != 3:
                    code = bittensor.proto.ReturnCode.RequestShapeException
                    message = "Forward message tensor input shape exception len(shape) = {} must have rank 3".format(len(list(torch_inputs.shape)))
                    call_time = clock.time() - start_time
                    bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message  )
                    return None, code, call_time, message

        except Exception as e:
            code = bittensor.proto.ReturnCode.UnknownException
            message = 'exception in preprocessing forward call with error: {}'.format(e)
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message  )
            return None, code, call_time, message

        # Post process.
        try:

            # ---- Make nucleus forward call. ----
            code = bittensor.proto.ReturnCode.Success
            message = None
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message  )
            outputs, code, message = self._call_forward( 
                public_key = request.hotkey, 
                inputs_x = torch_inputs, 
                modality = modality
            )
            if code != bittensor.proto.ReturnCode.Success:
                call_time = clock.time() - start_time
                bittensor.logging.rpc_log( axon=True, forward=True, is_response=True, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message  )
                return None, code, call_time, message

            # ---- Catch empty ----
            if outputs == None:
                code = bittensor.proto.ReturnCode.EmptyResponse
                message = None
                call_time = clock.time() - start_time
                bittensor.logging.rpc_log( axon=True, forward=True, is_response=True, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message  )
                return None, code, call_time, message

            # ---- Serialize response ----
            try:
                serializer = bittensor.serializer ( bittensor.proto.Serializer.MSGPACK )
                outputs_serialized = serializer.serialize ( outputs, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH )
            except Exception as e:
                code = bittensor.proto.ReturnCode.ResponseDeserializationException
                message = e
                call_time = clock.time() - start_time
                bittensor.logging.rpc_log( axon=True, forward=True, is_response=True, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message  )
                return None, code, call_time, message

        except Exception as e:
            code = bittensor.proto.ReturnCode.UnknownException
            message = 'exception in processing forward call: {}'.format(e)
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log( axon=True, forward=True, is_response=True, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message  )
            return None, code, call_time, message

        # ---- Return successful response ----
        call_time = clock.time() - start_time
        bittensor.logging.rpc_log( axon=True, forward=True, is_response=True, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(list(torch_inputs.shape)), outputs=outputs_serialized.shape, message=None  )
        return outputs_serialized, code, call_time, message
 
    def _backward(self, request):
        r""" Performs validity checks on the grpc request before piping the request to backend queue.
            Returns the output, message and code from the call.
            Args:
                request (:obj:`bittensor.proto`, `required`): 
                    Tensor request proto.
            Returns:
                response: (:obj:`bittensor.proto.Tensor, `required`): 
                    serialized tensor response from the nucleus call or None.
                code: (:obj:`bittensor.proto.ReturnCode, `required`)
                    return code associated with backward call i.e. Success of Timeout.
                time (:type:`float`, `required`):
                    Length of call in seconds.
                message: (str, `required`): 
                    message associated with backward call, potentially error, or 'success'.
        """
        start_time = clock.time()
        # ---- Check request inputs ----.
        if len(request.tensors) == 2:
            inputs_x = request.tensors[0]
            grads_dy = request.tensors[1]
            modality_x = inputs_x.modality
        else:
            code = bittensor.proto.ReturnCode.InvalidRequest
            message = "During backward: There are {} tensors in the request, expected 2.".format(len(request.tensors))
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log( axon=True, forward=False, is_response=False, code=code, call_time = call_time, pubkey = request.hotkey, inputs=None, outputs=None, message = message  )
            return None, code, call_time, message

        # ---- Deserialize request ---
        try:
            serializer = bittensor.serializer( inputs_x.serializer )
            inputs_x = serializer.deserialize( inputs_x, to_type = bittensor.proto.TensorType.TORCH )
            grads_dy = serializer.deserialize( grads_dy, to_type = bittensor.proto.TensorType.TORCH )
        except Exception as e:
            code = bittensor.proto.ReturnCode.RequestDeserializationException
            message = "Request serialization exception with error: {}".format(str(e))
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log( axon=True, forward=False, is_response=False, code=code, call_time = call_time, pubkey=request.hotkey, inputs=None, outputs=None, message=message  )
            return None, code, call_time, message
        
        # ---- Check shapes ----
        if modality_x == bittensor.proto.Modality.TEXT:
            if len(inputs_x.shape) != 2:
                code = bittensor.proto.ReturnCode.RequestShapeException
                message = "Forward text input shape exception with len(request.shape) = {} must have rank 2.".format(len(inputs_x.shape))
                call_time = clock.time() - start_time
                bittensor.logging.rpc_log( axon=True, forward=False, is_response=False, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message  )
                return None, code, call_time, message

        if modality_x == bittensor.proto.Modality.IMAGE:
            if len(inputs_x.shape) != 5:
                code = bittensor.proto.ReturnCode.RequestShapeException
                message =  "Forward image input shape exception for len(shape) = {}  must have rank 5".format(len(inputs_x.shape))
                call_time = clock.time() - start_time
                bittensor.logging.rpc_log( axon=True, forward=False, is_response=False, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message  )
                return None, code, call_time, message

        if modality_x == bittensor.proto.Modality.TENSOR:
            if len(inputs_x.shape) != 3:
                code = bittensor.proto.ReturnCode.RequestShapeException
                message = "Forward message tensor input shape exception len(shape) = {} must have rank 3".format(len(inputs_x.shape))
                call_time = clock.time() - start_time
                bittensor.logging.rpc_log( axon=True, forward=False, is_response=False, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message  )
                return None, code, call_time, message

        if len(grads_dy.shape) != 3:
            code = bittensor.proto.ReturnCode.RequestShapeException
            message = "Passed gradients must have rank 3 but got {}".format(len(grads_dy.shape))
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log( axon=True, forward=False, is_response=False, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message  )
            return None, code, call_time, message

        if grads_dy.shape[0] != inputs_x.shape[0] or grads_dy.shape[1] != inputs_x.shape[1]:
            code = bittensor.proto.ReturnCode.RequestShapeException
            message = "Passed gradients must same first and second dimension as passed inputs got shapes {} and {}".format(grads_dy.shape, inputs_x.shape)
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log( axon=True, forward=False, is_response=False, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message  )
            return None, code, call_time, message
 
        # ---- Make nucleus backward call. ----
        call_time = clock.time() - start_time
        bittensor.logging.rpc_log( axon=True, forward=False, is_response=False, code=bittensor.proto.ReturnCode.Success, call_time = call_time, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=None  )
        outputs, code, message = self._call_backward( 
            public_key = request.hotkey, 
            inputs_x = inputs_x, 
            grads_dy = grads_dy, 
            modality = modality_x
        )
        if code != bittensor.proto.ReturnCode.Success:
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log( axon=True, forward=False, is_response=True, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message  )
            return None, code, call_time, message

        # ---- Catch empty ----
        if outputs == None:
            code = bittensor.proto.ReturnCode.EmptyResponse
            message = None
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log( axon=True, forward=False, is_response=True, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message  )
            return None, code, call_time, message

        # ---- Deserialize response ----
        try:
            serializer = bittensor.serializer( bittensor.proto.Serializer.MSGPACK )
            outputs_serialized = serializer.serialize( outputs, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH )
        except Exception as e:
            code = bittensor.proto.ReturnCode.ResponseSerializationException
            message = "Backward request serialization failed with error {} and inputs {}".format(e, outputs)
            call_time = clock.time() - start_time
            bittensor.logging.rpc_log( axon=True, forward=False, is_response=True, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message  )
            return None, code, call_time, message

        # ---- Finaly return ----
        call_time = clock.time() - start_time
        bittensor.logging.rpc_log( axon=True, forward=False, is_response=True, code=code, call_time = call_time, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=list(outputs_serialized.shape), message=None  )
        return outputs_serialized, code, call_time, message

    def attach( self, servicer:object, modality:int):
        """
            Attaches the forward and backward callbacks to the passed object.

            Returns:
                servicer (:object:`object`, `required`): 
                    object with callbacks servicer.forward and servicer.backward
        """
        self.attach_forward_callback( servicer.forward , modality)
        self.attach_backward_callback( servicer.backward , modality)

    def attach_forward_callback(self, forward_callback: Callable[ [str, torch.Tensor, int], torch.Tensor ] , modality: int):
        """ Assigns the forward_callback.

            Returns:
                forward_callback (:callabl:`Callable[ [str, torch.Tensor, int], torch.Tensor `, `required`): 
                    Forward function called on recieving a forward request.
        """
        bittensor.axon.check_forward_callback(forward_callback,modality)
        self.forward_callback[modality] = forward_callback

    def attach_backward_callback(self, backward_callback: Callable[ [str, torch.Tensor, torch.Tensor, int], torch.Tensor ], modality: int ):
        """ Assigns the backward_callback call to this neuron.

            Returns:
                backward_callback (:callabl:`Callable[ [torch.Tensor, torch.Tensor], torch.Tensor `, `required`): 
                     Backward callback called on recieving a backward request.
        """
        bittensor.axon.check_backward_callback(backward_callback,modality)
        self.backward_callback[modality] = backward_callback

    def update_stats_for_request(self, request, response):
        """ Save the in_bytes and out_bytes from request and respond to self.stats 
        """
        self.stats.qps.update(1)
        in_bytes = sys.getsizeof(request)
        out_bytes = sys.getsizeof(response)
        self.stats.total_in_bytes.update(in_bytes)
        self.stats.total_out_bytes.update(out_bytes)

        # ---- Check we have a stats column for this peer
        if request.hotkey in self.stats.in_bytes_per_pubkey:
            self.stats.in_bytes_per_pubkey[request.hotkey].update(in_bytes)
            self.stats.out_bytes_per_pubkey[request.hotkey].update(out_bytes)
            self.stats.qps_per_pubkey[request.hotkey].update(1)

        else:
            self.stats.in_bytes_per_pubkey[request.hotkey] = stat_utils.timed_rolling_avg(in_bytes, 0.01)
            self.stats.out_bytes_per_pubkey[request.hotkey] = stat_utils.timed_rolling_avg(out_bytes, 0.01)
            self.stats.qps_per_pubkey[request.hotkey] = stat_utils.timed_rolling_avg(1, 0.01)
            self.stats.qps_failed_per_pubkey[request.hotkey] = stat_utils.timed_rolling_avg(0.0, 0.01)

        # ---- Adding failed responses to stat  
        if response.return_code != bittensor.proto.ReturnCode.Success:
            self.stats.qps_failed.update(1)
            self.stats.qps_failed_per_pubkey[request.hotkey].update(1)

    def __del__(self):
        r""" Called when this axon is deleted, ensures background threads shut down properly.
        """
        self.stop()

    def serve( 
            self, 
            use_upnpc: bool = False, 
            subtensor: 'bittensor.Subtensor' = None,
            network: str = None,
            chain_endpoint: str = None,
            prompt: bool = False
        ) -> 'Axon':
        r""" Subscribes this Axon servicing endpoint to the passed network using it's wallet.
            Args:
                use_upnpc (:type:bool, `optional`): 
                    If true, serves the axon attempts port forward through your router before 
                    subscribing.
                subtensor (:obj:`bittensor.Subtensor`, `optional`): 
                    Chain connection through which to serve.
                network (default='akatsuki', type=str)
                    If subtensor is not set, uses this network flag to create the subtensor connection.
                chain_endpoint (default=None, type=str)
                    Overrides the network argument if not set.
                prompt (bool):
                    If true, the call waits for confirmation from the user before proceeding.

        """   
        if subtensor == None: subtensor = bittensor.subtensor( network = network, chain_endpoint = chain_endpoint) 
        serv_success = subtensor.serve_axon( axon = self, use_upnpc = use_upnpc, prompt = prompt )
        if not serv_success:
            raise RuntimeError('Failed to serve neuron.')
        return self


    def start(self) -> 'Axon':
        r""" Starts the standalone axon GRPC server thread.
        """
        if self.server != None:
            self.server.stop( grace = 1 )  
            logger.success("Axon Stopped:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))

        self.server.start()
        logger.success("Axon Started:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))
        self.started = True
        return self

    def stop(self) -> 'Axon':
        r""" Stop the axon grpc server.
        """
        if self.server != None:
            self.server.stop( grace = 1 )
            logger.success("Axon Stopped:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))
        self.started = False
        return self

    def find_modality(self):
        r""" Detects modality from forward callbacks
        """
        modality_list= [index for index, v in enumerate(self.forward_callback) if v != None]

        if len(modality_list) > 1:
            raise NotImplementedError('Multiple modality detected. We do not currently support multi-modality miners.')
        elif len(modality_list) == 1:
            if modality_list[0] == 0:
                return bittensor.proto.Modality.TEXT
            if modality_list[0] == 1:
                return bittensor.proto.Modality.IMAGE
            if modality_list[0] == 2:
                return bittensor.proto.Modality.TENSOR
        elif len(modality_list) == 0:
            logger.warning('No modality detected. Defaulting to the text modality')
            return bittensor.proto.Modality.TEXT
    
    def check(self):
        r""" Checks axon's forward and backward callbacks 
        """
        pubkey = self.wallet.hotkey.ss58_address
        for index,forward in enumerate(self.forward_callback):
            if forward != None:
                bittensor.axon.check_forward_callback(forward,index,pubkey)
        for index, backward in  enumerate(self.backward_callback):
            if backward != None:
                bittensor.axon.check_backward_callback(backward,index,pubkey)
        return self

    def to_wandb(self):
        r""" Return a dictionary of axon stat for wandb logging
            
            Return:
                wandb_info (:obj:`Dict`)
        """
        # ---- Axon summary for wandb
        wandb_data = {
            'axon_qps': self.stats.qps.value,
            'axon_qps_failed' : self.stats.qps_failed.value,
            'axon_total_in_bytes' : self.stats.total_in_bytes.value,
            'axon_total_out_bytes' : self.stats.total_out_bytes.value,
        }

        # ---- Axon stats per pubkey for wandb 
        for pubkey in self.stats.in_bytes_per_pubkey.keys():
            wandb_data[f'axon_in_bytes\n{pubkey}'] = self.stats.in_bytes_per_pubkey[pubkey].value
            wandb_data[f'axon_out_bytes\n{pubkey}'] = self.stats.out_bytes_per_pubkey[pubkey].value
            wandb_data[f'axon_qps\n{pubkey}'] = self.stats.qps_per_pubkey[pubkey].value
            wandb_data[f'axon_qps_failed\n{pubkey}'] = self.stats.qps_failed_per_pubkey[pubkey].value
            
        return wandb_data 