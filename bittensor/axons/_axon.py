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
from munch import Munch
from termcolor import colored
from types import SimpleNamespace
from typing import List, Tuple, Optional, Callable

import bittensor
import bittensor.serialization as serialization
import bittensor.utils.stats as stat_utils

from loguru import logger
logger = logger.opt(colors=True)

class Axon( bittensor.grpc.BittensorServicer ):
    r""" Services Forward and Backward requests from other neurons.
    """
    def __init__( self, config: Munch, wallet: 'bittensor.wallet.Wallet', server: 'grpc._Server' ):
        r""" Initializes a new Axon tensor processing endpoint.
            
            Args:
                config (:obj:`Munch`, `optional`): 
                    axon.Axon.config()
                wallet (:obj:`bittensor.wallet.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
        """
        self.config = config
        self.wallet = wallet
        self._server = server
         
        self._forward_function = None
        self._backward_function = None

        bittensor.grpc.add_BittensorServicer_to_server( self, self._server )
        self._server.add_insecure_port('[::]:' + str( self.config.axon.local_port ))

        # Stats: Memory of network statistics, QPS and bytes in and out for instance.
        self.stats = SimpleNamespace(
            qps = stat_utils.timed_rolling_avg(0.0, 0.01),
            total_in_bytes = stat_utils.timed_rolling_avg(0.0, 0.01),
            total_out_bytes= stat_utils.timed_rolling_avg(0.0, 0.01),
            in_bytes_per_pubkey = {},
            out_bytes_per_pubkey = {},
            qps_per_pubkey = {},
        )

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
        logger.debug('-> Got Forward request: {}, size:{}', request.public_key, sys.getsizeof( request ))
        tensor, code, message = self._forward( request )
        response = bittensor.proto.TensorMessage(
            version = bittensor.__version__, 
            public_key = self.wallet.hotkey.public_key, 
            return_code = code,
            message = message,
            tensors = [tensor] if tensor is not None else [],
        )
        logger.debug('<- Got Forward response: {}, size:{}', response.public_key, sys.getsizeof( response ))
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
        logger.debug('-> Backward request: {}, size:{}', request.public_key, sys.getsizeof( request ))
        tensor, code, message = self._backward( request )
        response = bittensor.proto.TensorMessage(
            version = bittensor.__version__, 
            public_key = self.wallet.hotkey.public_key, 
            return_code = code,
            message = message,
            tensors = [tensor] if tensor is not None else [],
        )
        self.update_stats_for_request( request, response )
        logger.debug('<- Backward response: {}, size:{}', response.public_key, sys.getsizeof( response ))
        return response

    def attach( self, servicer:object ):
        """
            Attaches the forward and backward calls of the passed object.

            Returns:
                servicer (:object:`object`, `required`): 
                    object with callable functions servicer.forward and servicer.backward
        """
        self._forward_function = self.attach_forward_function( servicer.forward )
        self._backward_function = self.attach_forward_function( servicer.backward )

    def attach_forward_function(self, forward_function: Callable[ [str, torch.Tensor, int], torch.Tensor ] ):
        """ Assigns the forward_function.

            Returns:
                forward_function (:callabl:`Callable[ [str, torch.Tensor, int], torch.Tensor `, `required`): 
                    Forward function called on recieving a forward processing call on the wire.
        """
        # TODO(const): type checking.
        self._forward_function = forward_function

    def attach_backward_function(self, backward_function: Callable[ [str, torch.Tensor, torch.Tensor, int], torch.Tensor ] ):
        """ Assigns the routing_function call to this neuron.

            Returns:
                backward_function (:callabl:`Callable[ [torch.Tensor, torch.Tensor], torch.Tensor `, `required`): 
                     Backward function called on recieving a forward processing call on the wire.
        """
        # TODO(const): type checking.
        self._backward_function = backward_function

    def _call_forward(
            self, 
            public_key: str, 
            inputs_x: torch.Tensor, 
            modality: bittensor.proto.Modality
        ) -> Tuple[ torch.FloatTensor, int, str ]:
        r""" Calls the forward function subscribed by the nucleus.
            
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
        if self._forward_function == None:
            message = "Forward function is not yet subscribed on this axon."
            return None, bittensor.proto.ReturnCode.NotImplemented, message
        
        # Make forward call.
        try:
            response_tensor = self._forward_function( public_key, inputs_x, modality)
            message = "Success"
            code = bittensor.proto.ReturnCode.Success
            return response_tensor, code, message

        except Exception as e:
            response_tensor = None
            message = "Error calling forward function: {}".format(e)
            code = bittensor.proto.ReturnCode.UnknownException
            return response_tensor, code, message 


    def _call_backward(
            self, 
            public_key: str, 
            inputs_x: torch.Tensor, 
            grads_dy: torch.FloatTensor,
            modality: bittensor.proto.Modality
        ) -> Tuple[ torch.FloatTensor, int, str ]:
        r""" Calls the forward function subscribed by the nucleus.
            
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
        # Check forward has been subscribed.
        if self._backward_function == None:
            message = "Forward function is not yet subscribed on this axon."
            return None, bittensor.proto.ReturnCode.NotImplemented, message
        
        # Make forward call.
        try:
            response_tensor = self._backward_function( public_key, inputs_x, grads_dy, modality)
            message = "Success"
            code = bittensor.proto.ReturnCode.Success
            return response_tensor, code, message

        except Exception as e:
            response_tensor = None
            message = "Error calling forward function: {}".format(e)
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
        # ---- Check Empty request ----
        if len(request.tensors) == 0:
            message = "Forward request contains {} tensors, expected 1 tensor in the forward call".format(len(request.tensors))
            logger.debug('<white>Axon</white> <red>Forward Request</red> --->x <white>code</white>:<yellow>EmptyRequest</yellow>, <white>from</white>:<cyan>{}</cyan>, <white>message</white>:<red>{}</red>', request.public_key, message)
            return None, bittensor.proto.ReturnCode.EmptyRequest, message

        # ---- Check deserialization ----
        tensor_inputs = request.tensors[0]
        modality = tensor_inputs.modality
        try:
            deserializer = serialization.get_serializer( serialzer_type = tensor_inputs.serializer )
            torch_inputs = deserializer.deserialize(tensor_inputs, to_type = bittensor.proto.TensorType.TORCH)
        except Exception as e:
            message = "Request deserialization exception: {}".format(str(e))
            logger.debug('<white>Axon</white> <red>Forward Request</red> --->x <white>code</white>:<yellow>RequestDeserializationException</yellow>, <white>from</white>:<cyan>{}</cyan>, <white>message</white>:<red>{}</red>', request.public_key, message)
            return None, bittensor.proto.ReturnCode.RequestDeserializationException, message

        # ---- Check shape and modality ----
        if torch_inputs.shape[0] < 1:
            message = "Forward request batch dim exception with batch_size = {} ".format(torch_inputs.shape[0])
            logger.debug('<white>Axon</white> <red>Forward Request</red> --->x <white>code</white>:<yellow>RequestShapeException</yellow>, <white>from</white>:<cyan>{}</cyan>, <white>message</white>:<red>{}</red>', request.public_key, message)
            return None, bittensor.proto.ReturnCode.RequestShapeException, message

        if torch_inputs.shape[1] < 1:
            message = "Forward request sequence dim exception with sequence_dim = {} ".format(torch_inputs.shape[1])
            logger.debug('<white>Axon</white> <red>Forward Request</red> --->x <white>code</white>:<yellow>RequestShapeException</yellow>, <white>from</white>:<cyan>{}</cyan>, <white>message</white>:<red>{}</red>', request.public_key, message)
            return None, bittensor.proto.ReturnCode.RequestShapeException, message

        if modality == bittensor.proto.Modality.TEXT:
            if len(torch_inputs.shape) != 2:
                message = "Forward text input shape exception with len(request.shape) = {} must have rank 2.".format(len(torch_inputs.shape))
                logger.debug('<white>Axon</white> <red>Forward Request</red> --->x <white>code</white>:<yellow>RequestShapeException</yellow>, <white>from</white>:<cyan>{}</cyan>, <white>message</white>:<red>{}</red>', request.public_key, message)
                return None, bittensor.proto.ReturnCode.RequestShapeException, message
            
        if modality == bittensor.proto.Modality.IMAGE:
            if len(torch_inputs.shape) != 5:
                message =  "Forward image input shape exception for len(shape) = {}  must have rank 5".format(len(torch_inputs.shape))
                logger.debug('<white>Axon</white> <red>Forward Request</red> --->x <white>code</white>:<yellow>RequestShapeException</yellow>, <white>from</white>:<cyan>{}</cyan>, <white>message</white>:<red>{}</red>', request.public_key, message)
                return None, bittensor.proto.ReturnCode.RequestShapeException, message

        if modality == bittensor.proto.Modality.TENSOR:
            if len(torch_inputs.shape) != 3:
                message = "Forward message tensor input shape exception len(shape) = {} must have rank 3".format(len(torch_inputs.shape))
                logger.debug('<white>Axon</white> <red>Forward Request</red> --->x <white>code</white>:<yellow>RequestShapeException</yellow>, <white>from</white>:<cyan>{}</cyan>, <white>message</white>:<red>{}</red>', request.public_key, message)
                return None, bittensor.proto.ReturnCode.RequestShapeException, message

        # ---- Make nucleus forward call. ----
        logger.debug('<white>Axon</white> <green>Forward Request</green> ---> <white>from</white>:<cyan>{}</cyan>, <white>inputs</white>:<cyan>{}</cyan>', request.public_key, torch_inputs.shape)
        outputs, code, message = self._call_forward( 
            public_key = request.public_key, 
            inputs_x = torch_inputs, 
            modality = modality
        )
        if code != bittensor.proto.ReturnCode.Success:
            return None, code, message

        # ---- Serialize response ----
        try:
            serializer = serialization.get_serializer ( bittensor.proto.Serializer.MSGPACK )
            outputs_serialized = serializer.serialize ( outputs, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH )
        except Exception as e:
            logger.error(e)
            return None, bittensor.proto.ReturnCode.ResponseDeserializationException, str(e)

        # ---- Return successful response ----
        logger.debug('<white>Axon</white> <green>Forward Response</green> <--- <white>to</white>:<cyan>{}</cyan>, <white>outputs</white>:<cyan>{}</cyan>', request.public_key, outputs.shape)
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
                    message associated with forward call, potentially error, or 'success'.
                code: (:obj:`bittensor.proto.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
        """
        # ---- Check request inputs ----.
        if len(request.tensors) == 2:
            inputs_x = request.tensors[0]
            grads_dy = request.tensors[1]
            modality_x = inputs_x.modality
        else:
            message = "During backward: There are {} tensors in the request, expected 2.".format(len(request.tensors))
            logger.debug('<white>Axon</white> <red>Backward Request</red> --->x <white>code</white>:<yellow>InvalidRequest</yellow>, <white>from</white>:<cyan>{}</cyan>, <white>message</white>:<red>{}</red>', request.public_key, message)
            return None, bittensor.proto.ReturnCode.InvalidRequest, message

        # ---- Deserialize request ---
        try:
            serializer = serialization.get_serializer( inputs_x.serializer )
            inputs_x = serializer.deserialize( inputs_x, to_type = bittensor.proto.TensorType.TORCH )
            grads_dy = serializer.deserialize( grads_dy, to_type = bittensor.proto.TensorType.TORCH )
        except Exception as e:
            message = "Request serialization exception with error: {}".format(str(e))
            logger.debug('<white>Axon</white> <red>Backward Request</red> --->x <white>code</white>:<yellow>RequestDeserializationException</yellow>, <white>from</white>:<cyan>{}</cyan>, <white>message</white>:<red>{}</red>', request.public_key, message)
            return None, bittensor.proto.ReturnCode.RequestDeserializationException, message

        # ---- Check shapes ----
        if modality_x == bittensor.proto.Modality.TEXT:
            if len(inputs_x.shape) != 2:
                message = "Forward text input shape exception with len(request.shape) = {} must have rank 2.".format(len(inputs_x.shape))
                logger.debug('<white>Axon</white> <red>Backward Request</red> --->x <white>code</white>:<yellow>RequestShapeException</yellow>, <white>from</white>:<cyan>{}</cyan>, <white>message</white>:<red>{}</red>', request.public_key, message)
                return None, bittensor.proto.ReturnCode.RequestShapeException, message
            
        if modality_x == bittensor.proto.Modality.IMAGE:
            if len(inputs_x.shape) != 5:
                message =  "Forward image input shape exception for len(shape) = {}  must have rank 5".format(len(inputs_x.shape))
                logger.debug('<white>Axon</white> <red>Backward Request</red> --->x <white>code</white>:<yellow>RequestShapeException</yellow>, <white>from</white>:<cyan>{}</cyan>, <white>message</white>:<red>{}</red>', request.public_key, message)
                return None, bittensor.proto.ReturnCode.RequestShapeException, message

        if modality_x == bittensor.proto.Modality.TENSOR:
            if len(inputs_x.shape) != 3:
                message = "Forward message tensor input shape exception len(shape) = {} must have rank 3".format(len(inputs_x.shape))
                logger.debug('<white>Axon</white> <red>Backward Request</red> --->x <white>code</white>:<yellow>RequestShapeException</yellow>, <white>from</white>:<cyan>{}</cyan>, <white>message</white>:<red>{}</red>', request.public_key, message)
                return None, bittensor.proto.ReturnCode.RequestShapeException, message

        if len(grads_dy.shape) != 3:
            message = "Passed gradients must have rank 3 but got {}".format(len(grads_dy.shape))
            logger.debug('<white>Axon</white> <red>Backward Request</red> --->x <white>code</white>:<yellow>RequestShapeException</yellow>, <white>from</white>:<cyan>{}</cyan>, <white>message</white>:<red>{}</red>', request.public_key, message)
            return None, bittensor.proto.ReturnCode.RequestShapeException, message

        if grads_dy.shape[0] != inputs_x.shape[0] or grads_dy.shape[1] != inputs_x.shape[1]:
            message = "Passed gradients must same first and second dimension as passed inputs got shapes {} and {}".format(grads_dy.shape, inputs_x.shape)
            logger.debug('<white>Axon</white> <red>Backward Request</red> --->x <white>code</white>:<yellow>RequestShapeException</yellow>, <white>from</white>:<cyan>{}</cyan>, <white>message</white>:<red>{}</red>', request.public_key, message)
            return None, bittensor.proto.ReturnCode.RequestShapeException, message
 
        # ---- Make nucleus backward call. ----
        logger.debug('<white>Axon</white> <green>Backward Request</green> ---> <white>from</white>:<cyan>{}</cyan>, <white>grads_dy</white>:<cyan>{}</cyan>', request.public_key, grads_dy.shape)
        outputs, code, message = self._call_backward( 
            public_key = request.public_key, 
            inputs_x = inputs_x, 
            grads_dy = grads_dy, 
            modality = modality_x
        )
        if code != bittensor.proto.ReturnCode.Success:
            logger.debug('<white>Axon</white> <red>Backward Response</red> <--- <white>code</white>:<yellow>code</yellow>, <white>to</white>:<cyan>{}</cyan>, <white>message</white>:<red>{}</red>', request.public_key, message)
            return None, code, message

        # ---- Deserialize response ----
        try:
            serializer = serialization.get_serializer( bittensor.proto.Serializer.MSGPACK )
            outputs_serialized = serializer.serialize( outputs, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH )
        except Exception as e:
            message = "Backward request serialization failed with error {} and inputs {}".format(e, outputs)
            logger.debug('<white>Axon</white> <red>Backward Response</red> <--- <white>code</white>:<yellow>ResponseSerializationException</yellow>, <white>to</white>:<cyan>{}</cyan>, <white>message</white>:<red>{}</red>', request.public_key, message)
            return None, bittensor.proto.ReturnCode.ResponseSerializationException, message

        # ---- Finaly return ----
        logger.debug('<white>Axon</white> <green>Backward Response</green> <--- <white>code</white>:<green>Success</green>, <white>to</white>:<cyan>{}</cyan>, <white>outputs</white>:<cyan>{}</cyan>', request.public_key, outputs.shape)
        return outputs_serialized, code, message

    def update_stats_for_request(self, request, response):
        self.stats.qps.update(1)
        in_bytes = sys.getsizeof(request)
        out_bytes = sys.getsizeof(response)
        self.stats.total_in_bytes.update(in_bytes)
        self.stats.total_out_bytes.update(out_bytes)
        # ---- Check we have a stats column for this peer
        if request.public_key in self.stats.in_bytes_per_pubkey:
            self.stats.in_bytes_per_pubkey[request.public_key].update(in_bytes)
            self.stats.out_bytes_per_pubkey[request.public_key].update(out_bytes)
            self.stats.qps_per_pubkey[request.public_key].update(1)
        else:
            self.stats.in_bytes_per_pubkey[request.public_key] = stat_utils.timed_rolling_avg(in_bytes, 0.01)
            self.stats.out_bytes_per_pubkey[request.public_key] = stat_utils.timed_rolling_avg(out_bytes, 0.01)
            self.stats.qps_per_pubkey[request.public_key] = stat_utils.timed_rolling_avg(1, 0.01)

    def __str__(self):
        total_in_bytes_str = colored('\u290B {:.1f}'.format((self.stats.total_in_bytes.value * 8)/1000), 'green')
        total_out_bytes_str = colored('\u290A {:.1f}'.format((self.stats.total_out_bytes.value * 8)/1000), 'red')
        qps_str = colored("{:.3f}".format(float(self.stats.qps.value)), 'blue')
        return "(" + qps_str + "q/s|" + total_out_bytes_str + "/" + total_in_bytes_str + "kB/s" + ")"

    def __rich__(self):
        total_in_bytes_str = '[red]\u290B{:.1f}[/red]'.format((self.stats.total_in_bytes.value * 8)/1000)
        total_out_bytes_str = '[green]\u290A{:.1f}[/green]'.format((self.stats.total_out_bytes.value * 8)/1000)
        qps_str = "[blue]{:.3f}[/blue]".format(float(self.stats.qps.value))
        return "(" + qps_str + "q/s|" + total_out_bytes_str + "/" + total_in_bytes_str + "kB/s" + ")"
    
    def __to_tensorboard__(self, tensorboard, global_step):
        total_in_bytes = (self.stats.total_in_bytes.value * 8)/1000
        total_out_bytes = (self.stats.total_out_bytes.value * 8)/1000
        tensorboard.add_scalar("Axon/total_in_bytes", total_in_bytes, global_step)
        tensorboard.add_scalar("Axon/total_out_bytes", total_out_bytes, global_step)
        tensorboard.add_scalar("Axon/Queries/Sec", self.stats.qps.value, global_step)

    def __del__(self):
        r""" Called when this axon is deleted, ensures background threads shut down properly.
        """
        self.stop()

    def _serve(self):
        try:
            logger.success('Axon is serving on: {}:{}', self.config.axon.local_ip, self.config.axon.local_port)
            self._server.start()
        except (KeyboardInterrupt, SystemExit):
            self.stop()
        except Exception as e:
            logger.error(e)

    def start(self):
        r""" Starts the standalone axon GRPC server thread.
        """
        # TODO(const): should allow more than one services and these can run in different processes.
        # Destroy and create a new serving thread.
        if self._server != None:
            self._server.stop( 0 )   
        self._thread = threading.Thread( target = self._serve, daemon = True )
        self._thread.start()

    def stop(self):
        r""" Stop the axon grpc server.
        """
        if self._server != None:
            self._server.stop( 0 )
            logger.success('Axon has stopped serving on: {}:{}', self.config.axon.local_ip, self.config.axon.local_port)




