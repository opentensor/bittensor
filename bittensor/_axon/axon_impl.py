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

import grpc
import sys
import threading
import torch
import queue
import multiprocessing as mp

from concurrent import futures
from termcolor import colored
from types import SimpleNamespace
from typing import List, Tuple, Optional, Callable

import bittensor
import bittensor.utils.stats as stat_utils
from substrateinterface.utils.ss58 import ss58_encode

from loguru import logger
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
        modality: int = None
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
        """
        self.ip = ip
        self.port = port
        self.wallet = wallet
        self.server = server
        self.forward_callback = forwards
        self.backward_callback = backwards
        self.modality = modality if modality != None else self.find_modality()
        self.stats = SimpleNamespace(
            qps = stat_utils.timed_rolling_avg(0.0, 0.01),
            total_in_bytes = stat_utils.timed_rolling_avg(0.0, 0.01),
            total_out_bytes= stat_utils.timed_rolling_avg(0.0, 0.01),
            in_bytes_per_pubkey = {},
            out_bytes_per_pubkey = {},
            qps_per_pubkey = {},
        )

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
        # TODO(const): check signature
        # TODO(const): black and white listing.
        tensor, code, message = self._forward( request )
        response = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__, 
            hotkey = self.wallet.hotkey.ss58_address, 
            return_code = code,
            message = message,
            tensors = [tensor] if tensor is not None else [],
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
        tensor, code, message = self._backward( request )
        response = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__, 
            hotkey = self.wallet.hotkey.ss58_address, 
            return_code = code,
            message = message,
            tensors = [tensor] if tensor is not None else [],
        )
        self.update_stats_for_request( request, response )
        return response

    def _call_forward(
            self, 
            public_key: str, 
            inputs_x: torch.Tensor, 
            modality: bittensor.proto.Modality
        ) -> Tuple[ torch.FloatTensor, int, str ]:
        r""" Calls the forward callback subscribed by the nucleus.
            
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
            response_tensor = self.forward_callback[modality](pubkey = public_key, inputs_x= inputs_x)
            message = "Success"
            code = bittensor.proto.ReturnCode.Success
            return response_tensor, code, message

        except Exception as e:
            response_tensor = None
            message = "Error calling forward callback: {}".format(e)
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

        # Make backward call.
        try:
            response_tensor = self.backward_callback[modality]( public_key, inputs_x, grads_dy)
            message = "Success"
            code = bittensor.proto.ReturnCode.Success
            return response_tensor, code, message

        except Exception as e:
            response_tensor = None
            message = "Error calling backward callback: {}".format(e)
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
                message (str, `required`): 
                    message associated with forward call, potentially error, or 'success'.
        """
        try:
            # ---- Check Empty request ----
            if len(request.tensors) == 0:
                code = bittensor.proto.ReturnCode.EmptyRequest
                message = "Forward request contains {} tensors, expected 1 tensor in the forward call".format(len(request.tensors))
                bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, pubkey=request.hotkey, inputs=None, outputs=None, message=message )
                return None, code, message

            # ---- Check deserialization ----
            tensor_inputs = request.tensors[0]
            modality = tensor_inputs.modality
            try:
                deserializer = bittensor.serializer( serialzer_type = tensor_inputs.serializer )
                torch_inputs = deserializer.deserialize(tensor_inputs, to_type = bittensor.proto.TensorType.TORCH)
            except Exception as e:
                code = bittensor.proto.ReturnCode.RequestDeserializationException
                message = "Request deserialization exception: {}".format(str(e))
                bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, pubkey=request.hotkey, inputs=None, outputs=None, message=message )
                return None, code, message

            # ---- Check shape and modality ----
            if list(torch_inputs.shape)[0] < 1:
                code = bittensor.proto.ReturnCode.RequestShapeException,
                message = "Forward request batch dim exception with batch_size = {} ".format(list(torch_inputs.shape)[0])
                bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message )
                return None, code, message

            if list(torch_inputs.shape)[1] < 1:
                code = bittensor.proto.ReturnCode.RequestShapeException
                message = "Forward request sequence dim exception with sequence_dim = {} ".format(list(torch_inputs.shape)[1])
                bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message )
                return None, code, message

            if modality == bittensor.proto.Modality.TEXT:
                if len(list(torch_inputs.shape)) != 2:
                    code = bittensor.proto.ReturnCode.RequestShapeException
                    message = "Forward text input shape exception with len(request.shape) = {} must have rank 2.".format(len(list(torch_inputs.shape)))
                    bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message )
                    return None, code, message
          
            if modality == bittensor.proto.Modality.IMAGE:
                if len(list(torch_inputs.shape)) != 5:
                    code = bittensor.proto.ReturnCode.RequestShapeException
                    message =  "Forward image input shape exception for len(shape) = {}  must have rank 5".format(len(list(torch_inputs.shape)))
                    bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message )
                    return None, code, message

            if modality == bittensor.proto.Modality.TENSOR:
                if len(list(torch_inputs.shape)) != 3:
                    code = bittensor.proto.ReturnCode.RequestShapeException
                    message = "Forward message tensor input shape exception len(shape) = {} must have rank 3".format(len(list(torch_inputs.shape)))
                    bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message )
                    return None, code, message

        except Exception as e:
            code = bittensor.proto.ReturnCode.UnknownException
            message = 'exception in preprocessing forward call with error: {}'.format(e)
            bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message )
            return None, code, message

        # Post process.
        try:

            # ---- Make nucleus forward call. ----
            code = bittensor.proto.ReturnCode.Success
            message = None
            bittensor.logging.rpc_log( axon=True, forward=True, is_response=False, code=code, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message )
            outputs, code, message = self._call_forward( 
                public_key = request.hotkey, 
                inputs_x = torch_inputs, 
                modality = modality
            )
            if code != bittensor.proto.ReturnCode.Success:
                bittensor.logging.rpc_log( axon=True, forward=True, is_response=True, code=code, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message )
                return None, code, message

            # ---- Catch empty ----
            if outputs == None:
                code = bittensor.proto.ReturnCode.EmptyResponse
                message = None
                bittensor.logging.rpc_log( axon=True, forward=True, is_response=True, code=code, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message )
                return None, code, message

            # ---- Serialize response ----
            try:
                serializer = bittensor.serializer ( bittensor.proto.Serializer.MSGPACK )
                outputs_serialized = serializer.serialize ( outputs, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH )
            except Exception as e:
                code = bittensor.proto.ReturnCode.ResponseDeserializationException
                message = e
                bittensor.logging.rpc_log( axon=True, forward=True, is_response=True, code=code, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message )
                return None, code, message

        except Exception as e:
            code = bittensor.proto.ReturnCode.UnknownException
            message = 'exception in processing forward call: {}'.format(e)
            bittensor.logging.rpc_log( axon=True, forward=True, is_response=True, code=code, pubkey=request.hotkey, inputs=list(torch_inputs.shape), outputs=None, message=message )
            return None, code, message

        # ---- Return successful response ----
        bittensor.logging.rpc_log( axon=True, forward=True, is_response=True, code=code, pubkey=request.hotkey, inputs=list(list(torch_inputs.shape)), outputs=outputs_serialized.shape, message=None )
        return outputs_serialized, code, message
 
    def _backward(self, request):
        r""" Performs validity checks on the grpc request before piping the request to backend queue.
            Returns the output, message and code from the call.
            Args:
                request (:obj:`bittensor.proto`, `required`): 
                    Tensor request proto.
            Returns:
                response: (:obj:`bittensor.proto.Tensor, `required`): 
                    serialized tensor response from the nucleus call or None.
                message: (str, `required`): 
                    message associated with backward call, potentially error, or 'success'.
                code: (:obj:`bittensor.proto.ReturnCode, `required`)
                    return code associated with backward call i.e. Success of Timeout.
        """
        # ---- Check request inputs ----.
        if len(request.tensors) == 2:
            inputs_x = request.tensors[0]
            grads_dy = request.tensors[1]
            modality_x = inputs_x.modality
        else:
            code = bittensor.proto.ReturnCode.InvalidRequest
            message = "During backward: There are {} tensors in the request, expected 2.".format(len(request.tensors))
            bittensor.logging.rpc_log( axon=True, forward=False, is_response=False, code=code, pubkey = request.hotkey, inputs=None, outputs=None, message = message )
            return None, code, message

        # ---- Deserialize request ---
        try:
            serializer = bittensor.serializer( inputs_x.serializer )
            inputs_x = serializer.deserialize( inputs_x, to_type = bittensor.proto.TensorType.TORCH )
            grads_dy = serializer.deserialize( grads_dy, to_type = bittensor.proto.TensorType.TORCH )
        except Exception as e:
            code = bittensor.proto.ReturnCode.RequestDeserializationException
            message = "Request serialization exception with error: {}".format(str(e))
            bittensor.logging.rpc_log( axon=True, forward=False, is_response=False, code=code, pubkey=request.hotkey, inputs=None, outputs=None, message=message )
            return None, code, message

        # ---- Check shapes ----
        if modality_x == bittensor.proto.Modality.TEXT:
            if len(inputs_x.shape) != 2:
                code = bittensor.proto.ReturnCode.RequestShapeException
                message = "Forward text input shape exception with len(request.shape) = {} must have rank 2.".format(len(inputs_x.shape))
                bittensor.logging.rpc_log( axon=True, forward=False, is_response=False, code=code, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message )
                return None, code, message
            
        if modality_x == bittensor.proto.Modality.IMAGE:
            if len(inputs_x.shape) != 5:
                code = bittensor.proto.ReturnCode.RequestShapeException
                message =  "Forward image input shape exception for len(shape) = {}  must have rank 5".format(len(inputs_x.shape))
                bittensor.logging.rpc_log( axon=True, forward=False, is_response=False, code=code, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message )
                return None, code, message

        if modality_x == bittensor.proto.Modality.TENSOR:
            if len(inputs_x.shape) != 3:
                code = bittensor.proto.ReturnCode.RequestShapeException
                message = "Forward message tensor input shape exception len(shape) = {} must have rank 3".format(len(inputs_x.shape))
                bittensor.logging.rpc_log( axon=True, forward=False, is_response=False, code=code, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message )
                return None, code, message

        if len(grads_dy.shape) != 3:
            code = bittensor.proto.ReturnCode.RequestShapeException
            message = "Passed gradients must have rank 3 but got {}".format(len(grads_dy.shape))
            bittensor.logging.rpc_log( axon=True, forward=False, is_response=False, code=code, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message )
            return None, code, message

        if grads_dy.shape[0] != inputs_x.shape[0] or grads_dy.shape[1] != inputs_x.shape[1]:
            code = bittensor.proto.ReturnCode.RequestShapeException
            message = "Passed gradients must same first and second dimension as passed inputs got shapes {} and {}".format(grads_dy.shape, inputs_x.shape)
            bittensor.logging.rpc_log( axon=True, forward=False, is_response=False, code=code, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message )
            return None, code, message
 
        # ---- Make nucleus backward call. ----
        bittensor.logging.rpc_log( axon=True, forward=False, is_response=False, code=bittensor.proto.ReturnCode.Success, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=None )
        outputs, code, message = self._call_backward( 
            public_key = request.hotkey, 
            inputs_x = inputs_x, 
            grads_dy = grads_dy, 
            modality = modality_x
        )
        if code != bittensor.proto.ReturnCode.Success:
            bittensor.logging.rpc_log( axon=True, forward=False, is_response=True, code=code, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message )
            return None, code, message

        # ---- Catch empty ----
        if outputs == None:
            code = bittensor.proto.ReturnCode.EmptyResponse
            message = None
            bittensor.logging.rpc_log( axon=True, forward=False, is_response=True, code=code, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message )
            return None, code, message

        # ---- Deserialize response ----
        try:
            serializer = bittensor.serializer( bittensor.proto.Serializer.MSGPACK )
            outputs_serialized = serializer.serialize( outputs, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH )
        except Exception as e:
            code = bittensor.proto.ReturnCode.ResponseSerializationException
            message = "Backward request serialization failed with error {} and inputs {}".format(e, outputs)
            bittensor.logging.rpc_log( axon=True, forward=False, is_response=True, code=code, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=None, message=message )
            return None, code, message

        # ---- Finaly return ----
        bittensor.logging.rpc_log( axon=True, forward=False, is_response=True, code=code, pubkey=request.hotkey, inputs=list(grads_dy.shape), outputs=list(outputs_serialized.shape), message=None )
        return outputs_serialized, code, message

    def attach( self, servicer:object, modality:int):
        """
            Attaches the forward and backward callbacks to the passed object.

            Returns:
                servicer (:object:`object`, `required`): 
                    object with callbacks servicer.forward and servicer.backward
        """
        self.forward_callback = self.attach_forward_callback( servicer.forward , modality)
        self.backward_callback = self.attach_backward_callback( servicer.backward , modality)

    def attach_forward_callback(self, forward_callback: Callable[ [str, torch.Tensor, int], torch.Tensor ] , modality: int):
        """ Assigns the forward_callback.

            Returns:
                forward_callback (:callabl:`Callable[ [str, torch.Tensor, int], torch.Tensor `, `required`): 
                    Forward function called on recieving a forward request.
        """
        # TODO(const): type checking.
        bittensor.axon.check_forward_callback(forward_callback,modality)
        self.forward_callback[modality] = forward_callback

    def attach_backward_callback(self, backward_callback: Callable[ [str, torch.Tensor, torch.Tensor, int], torch.Tensor ], modality: int ):
        """ Assigns the backward_callback call to this neuron.

            Returns:
                backward_callback (:callabl:`Callable[ [torch.Tensor, torch.Tensor], torch.Tensor `, `required`): 
                     Backward callback called on recieving a backward request.
        """
        # TODO(const): type checking.
        bittensor.axon.check_backward_callback(backward_callback,modality)
        self.backward_callback[modality] = backward_callback

    def update_stats_for_request(self, request, response):
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

    def __del__(self):
        r""" Called when this axon is deleted, ensures background threads shut down properly.
        """
        self.stop()

    def subscribe( 
            self, 
            use_upnpc: bool = False, 
            subtensor: 'bittensor.Subtensor' = None,
            network: str = None,
            chain_endpoint: str = None,
            timeout = 4 * bittensor.__blocktime__,
        ) -> 'Axon':
        r""" Subscribes this Axon servicing endpoint to the passed network using it's wallet.
            Args:
                use_upnpc (:type:bool, `optional`): 
                    If true, subscribes the axon attempts port forward through your router before 
                    subscribing.
                modality (:type:bool, `optional`): 
                    Which network modality are we subscribing to. Defaults to 0 for TEXT.
                subtensor (:obj:`bittensor.Subtensor`, `optional`): 
                    Chain connection through which to subscribe.
                network (default='akatsuki', type=str)
                    If subtensor is not set, uses this network flag to create the subtensor connection.
                chain_endpoint (default=None, type=str)
                    Overrides the network argument if not set.
        """   

        # Create subtensor connection.
        if subtensor == None:
            subtensor = bittensor.subtensor( network = network, chain_endpoint = chain_endpoint)

        # ---- Setup UPNPC ----
        if use_upnpc:
            try:
                self.external_port = bittensor.net.upnpc_create_port_map( port = self.port )
                bittensor.logging.success(prefix = 'UPNPC', sufix = '<red>OPEN</red>')
            except bittensor.net.UPNPCException as upnpc_exception:
                raise RuntimeError('Failed to hole-punch with upnpc with exception {}'.format( upnpc_exception ))
        else:
            self.external_port = self.port

        # ---- Get external ip ----
        try:
            self.external_ip = bittensor.net.get_external_ip()
            bittensor.logging.success(prefix = 'External IP', sufix = '<blue>{}</blue>'.format(self.external_ip))
        except Exception as E:
            raise RuntimeError('Unable to attain your external ip. Check your internet connection. error:{}', E)

        # ---- Setup Wallet. ----
        self.wallet.create()

        # ---- Subscribe to chain ----
        subscribe_success = subtensor.subscribe(
                wallet = self.wallet,
                ip = self.external_ip,
                port = self.external_port,
                modality = self.modality,
                wait_for_finalization = True,
        )
        if not subscribe_success:
            raise RuntimeError('Failed to subscribe neuron.')

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