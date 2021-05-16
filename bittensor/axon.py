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
import copy
import grpc
import pandas as pd
import random
import requests
import sys
import threading
import torch
import time
import queue
import validators
import multiprocessing as mp

from concurrent import futures
from munch import Munch
from termcolor import colored
from types import SimpleNamespace
from typing import List, Tuple, Optional

import bittensor
import bittensor.utils.networking as net
import bittensor.serialization as serialization
import bittensor.utils.stats as stat_utils

from loguru import logger
logger = logger.opt(colors=True)

class Axon(bittensor.grpc.BittensorServicer):
    r"""
        Services Forward and Backward requests from other neurons.
    """
    def __init__(
            self, 
            config: Munch = None, 
            wallet: 'bittensor.wallet.Wallet' = None,
            local_port: int = None,
            local_ip: str =  None,
            max_workers: int = None, 
            forward_processing_timeout:int = None,
            backward_processing_timeout:int = None,
            **kwargs
        ):
        r""" Initializes a new Axon tensor processing endpoint.
            
            Args:
                config (:obj:`Munch`, `optional`): 
                    axon.Axon.config()
                wallet (:obj:`bittensor.wallet.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                local_port (default=8091, type=int): 
                    The port this axon endpoint is served on. i.e. 8091
                local_ip (default='127.0.0.1', type=str): 
                    The local ip this axon binds to. ie. 0.0.0.0
                max_workers (default=10, type=int): 
                    The maximum number connection handler threads working simultaneously on this endpoint. 
                        The grpc server distributes new worker threads to service requests up to this number.
                forward_processing_timeout (default=5, type=int):
                    Length of time allocated to the miner forward process for computing and returning responses
                        back to the axon.
                backward_processing_timeout (default=5, type=int):
                    Length of time allocated to the miner backward process for computing and returning responses
                        back to the axon.
        """
        # Config: Holds all config items for this items and those that are recursively defined. For instance,
        # config for the wallet and nucleus sub-objects.
        if config == None:
            config = Axon.default_config()
        config = copy.deepcopy(config); bittensor.config.Config.update_with_kwargs(config, kwargs )
        config.axon.local_port = local_port if local_port != None else config.axon.local_port
        config.axon.local_ip = local_ip if local_ip != None else config.axon.local_ip
        config.axon.max_workers = max_workers if max_workers != None else config.axon.max_workers
        config.axon.forward_processing_timeout = forward_processing_timeout if forward_processing_timeout != None else config.axon.forward_processing_timeout
        config.axon.backward_processing_timeout = backward_processing_timeout if backward_processing_timeout != None else config.axon.backward_processing_timeout
        Axon.check_config( config )
        self.config = config

        # Wallet: Holds you hotkey keypair and coldkey pub, which can be used to sign messages 
        # and subscribe to the chain.
        if wallet == None:
            wallet = bittensor.wallet.Wallet( config = self.config )
        self.wallet = wallet
        
        # Server: by default the axon serves an RPC server in its own thread using GPRC.
        # The servicer must implement Forward and Backward methods to properly communicate with
        # the other peers in the network.
        self._server = None 

        # Serving thread: A thread which runs the axon servicer passing items to the nucleus for
        # further processing.
        self._thread = None

        # Forward and Backward multiprocessing queues
        self.forward_queue = mp.Queue(self.config.axon.forward_queue_maxsize)
        self.backward_queue = mp.Queue(self.config.axon.backward_queue_maxsize)

        # Stats: Memory of network statistics, QPS and bytes in and out for instance.
        self.stats = SimpleNamespace(
            qps = stat_utils.timed_rolling_avg(0.0, 0.01),
            total_in_bytes = stat_utils.timed_rolling_avg(0.0, 0.01),
            total_out_bytes= stat_utils.timed_rolling_avg(0.0, 0.01),
            in_bytes_per_pubkey = {},
            out_bytes_per_pubkey = {},
            qps_per_pubkey = {},
        )

    @staticmethod   
    def default_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        Axon.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        r""" Adds this axon's command line arguments to the passed parser.
            Args:
                parser (:obj:`argparse.ArgumentParser`, `required`): 
                    parser argument to append args to.
        """
        bittensor.wallet.Wallet.add_args(parser)
        try:
            parser.add_argument('--axon.local_port', default=8091, type=int, 
                help='''The port this axon endpoint is served on. i.e. 8091''')
            parser.add_argument('--axon.local_ip', default='127.0.0.1', type=str, 
                help='''The local ip this axon binds to. ie. 0.0.0.0''')
            parser.add_argument('--axon.max_workers', default=10, type=int, 
                help='''The maximum number connection handler threads working simultaneously on this endpoint. 
                        The grpc server distributes new worker threads to service requests up to this number.''')
            parser.add_argument('--axon.forward_processing_timeout', default=5, type=int, 
                help='''Length of time allocated to the miner forward process for computing and returning responses
                        back to the axon.''')
            parser.add_argument('--axon.backward_processing_timeout', default=5, type=int, 
                help='''Length of time allocated to the miner backward process for computing and returning responses
                        back to the axon.''')
            parser.add_argument('--axon.forward_queue_maxsize', default=100, type=int,
                help='''Maximum number of pending forward requests queued at any time.''')
            parser.add_argument('--axon.backward_queue_maxsize', default=100, type=int, 
                help='''Maximum number of pending backward requests queued at any time.''')
        except:
            pass

    @staticmethod   
    def check_config(config: Munch):
        r""" Checks the passed config items for validity and obtains the remote ip.
            Args:
                config (:obj:`munch.Munch, `required`): 
                    config to check.
        """
        assert config.axon.local_port > 1024 and config.axon.local_port < 65535, 'config.axon.local_port must be in range [1024, 65535]'

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

    def next_forward_item( 
            self,
            timeout: int = 10 
        ) -> Tuple[Optional[mp.Pipe], Optional[str], Optional[torch.Tensor], Optional[torch.FloatTensor], Optional[int]]:
        r""" 
            Returns the next forward item from the forward queue to the caller.
            If there are no items on the queue after the timeout the response is None.
            Every call to next forward should be followed by a corresponding pong.send( outputs_y )
            
            Args:
                timeout (int, `required`): 
                    queue pull timeout,
            
            Returns:
                pong (:obj:`mp.Pipe, `optional`): 
                    multiprocessing pipe tunnel for the response or None if a timeout occurs.
                public_key (str, `optional`):
                    public key of caller or None if a timeout occurs.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed or None if a timeout occurs.
                modality ( bittensor.proto.Modality, `required`):
                    modality of inputs or None if a timeout occurs.

        """
        try:
            return self.forward_queue.get( block = True, timeout = timeout )
        except queue.Empty:
            return (None, None, None, None)

    def next_backward_item( 
            self, 
            timeout: int = 10 
        ) -> Tuple[mp.Pipe, str, torch.Tensor, torch.FloatTensor, int]:
        r""" 
            Returns the next backward item from the backward queue to the caller.
            If there are no items on the queue after the timeout the response is None.
            Every call to next backward should be followed by a corresponding pong.send( outputs_y )
            
            Args:
                timeout (int, `required`): 
                    queue pull timeout,
            
            Returns:
                pong (:obj:`mp.Pipe, `optional`): 
                    multiprocessing pipe tunnel for the response or None if a timeout occurs.
                public_key (str, `optional`):
                    public key of caller or None if a timeout occurs.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed or None if a timeout occurs.
                grads_dy ( :obj:`torch.Tensor`, `required`):
                    torch gradient inputs to be backward processed with inputs or None if a timeout occurs.
                modality ( bittensor.proto.Modality, `required`):
                    modality of inputs or None if a timeout occurs.

        """
        try:
            return self.backward_queue.get( block = True, timeout = timeout )
        except queue.Empty:
            return (None, None, None, None, None)

    def enqueue_forward_to_nucleus(
            self, 
            public_key: str, 
            inputs_x: torch.Tensor, 
            modality: bittensor.proto.Modality
        ) -> Tuple[ torch.FloatTensor, int, str ]:
        r""" Forwards the torch_inputs to the axon.forward_queue for processing by the miner threads.
        Responses are pulled from the pipe or a timeout occurs.
            
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
        logger.debug('enqueue_forward_to_nucleus: {}, inputs_x: {}', public_key, inputs_x)
        try:
            # ---- Build pipe for request ----
            ping, pong = mp.Pipe()
            forward_payload = [pong, public_key, inputs_x, modality]

            # ---- Send request to forward queue ----
            try:
                self.forward_queue.put( forward_payload, block=True, timeout = self.config.axon.forward_processing_timeout )
            except queue.Full:
                message = "Forward queue is full"
                logger.debug( message )
                return None, bittensor.proto.ReturnCode.NucleusFull, message

            # ---- Recv response from pipe ----
            if ping.poll( timeout = self.config.axon.forward_processing_timeout ):
                outputs = ping.recv()
                message = "Success"
                logger.debug( message )
                return outputs, bittensor.proto.ReturnCode.Success, message

            else:
                message = "Processing timeout"
                logger.debug( message )
                return None, bittensor.proto.ReturnCode.NucleusTimeout, message

        except Exception as e:
            message = str(e)
            logger.error( message )
            return None, bittensor.proto.ReturnCode.UnknownException, message

    def enqueue_backward_to_nucleus(
            self, 
            public_key: str, 
            inputs_x: torch.Tensor, 
            grads_dy: torch.FloatTensor,
            modality: bittensor.proto.Modality
        ) -> Tuple[ torch.FloatTensor, int, str ]:
        r""" Forwards the torch_inputs to the axon.backward_queue for processing by the miner threads.
        Responses are pulled from the pipe or a timeout occurs.
            
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
        try:
            # ---- Build pipe for request ----
            ping, pong = mp.Pipe()
            backward_payload = [pong, public_key, inputs_x, grads_dy, modality]

            # ---- Send request to queue ----
            try:
                self.backward_queue.put( backward_payload, block = True, timeout = self.config.axon.backward_processing_timeout )
            except queue.Full:
                message = "Backward queue is full"
                logger.debug( message )
                return None, bittensor.proto.ReturnCode.NucleusFull, message

            # ---- Recv response from pipe ----
            if ping.poll( timeout = self.config.axon.backward_processing_timeout ):
                outputs = ping.recv()
                message = "Success" 
                logger.debug( message )
                return outputs, bittensor.proto.ReturnCode.Success, message

            else:
                message = "Processing timeout"
                logger.debug( message )
                return None, bittensor.proto.ReturnCode.NucleusTimeout, message

        except Exception as e:
            message = str( e )
            logger.error( message )
            return None, bittensor.proto.ReturnCode.UnknownException, message
            
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
        outputs, code, message = self.enqueue_forward_to_nucleus( 
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
        outputs, code, message = self.enqueue_backward_to_nucleus( 
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
        
        self._server = grpc.server(futures.ThreadPoolExecutor( max_workers = self.config.axon.max_workers ))
        bittensor.grpc.add_BittensorServicer_to_server( self, self._server )
        self._server.add_insecure_port('[::]:' + str( self.config.axon.local_port ))  # TODO(const): should use the ip here.

        self._thread = threading.Thread( target = self._serve, daemon = True )
        self._thread.start()

    def stop(self):
        r""" Stop the axon grpc server.
        """
        if self._server != None:
            self._server.stop( 0 )
            logger.success('Axon has stopped serving on: {}:{}', self.config.axon.local_ip, self.config.axon.local_port)




