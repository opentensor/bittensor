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

from concurrent import futures
from munch import Munch
from loguru import logger
from termcolor import colored
from types import SimpleNamespace
from typing import List, Tuple, Any, Optional

import torch.multiprocessing as mp

import bittensor
import bittensor.utils.networking as net
import bittensor.serialization as serialization
import bittensor.utils.stats as stat_utils

class Axon(bittensor.grpc.BittensorServicer):
    r"""
        Services Forward and Backward requests from other neurons.
    """
    def __init__(
            self, 
            config: Munch = None, 
            wallet: 'bittensor.Wallet' = None,
            **kwargs
        ):
        r""" Initializes a new Axon tensor processing endpoint.
            
            Args:
                config (:obj:`Munch`, `optional`): 
                    axon.Axon.config()
                wallet (:obj:`bittensor.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                axon_local_port (default=8091, type=int): 
                    The port this axon endpoint is served on. i.e. 8091
                axon_local_ip (default='127.0.0.1', type=str): 
                    The local ip this axon binds to. ie. 0.0.0.0
                axon_use_upnpc (default=False, type=bool):
                    If true this axon will attempt to open a port on your router using upnpc.
                axon_external_ip (default=None, type=str):
                    The remote IP served to chain.
                        This ip is subscribed to the chain on boot and is the endpoint other peers see.
                        By default this field is None and is collected by querying a remote server during check_config. 
                        i.e. 207.12.233.1
                axon_external_port (default=None, type=str):
                    The remote port to subscribe on chain. By default this port is the same as local_port.
                        If use_upnpc is true this port is determined after the port mapping
                axon_max_workers (default=10, type=int): 
                    The maximum number connection handler threads working simultaneously on this endpoint. 
                        The grpc server distributes new worker threads to service requests up to this number.
        """
        # Config: Holds all config items for this items and those that are recursively defined. For instance,
        # config for the wallet, metagraph sub-objects.
        if config == None:
            config = Axon.default_config()
        bittensor.Config.update_split_with_kwargs(config, kwargs)
        Axon.check_config( config )
        self.config = config

        # Wallet: Holds you hotkey keypair and coldkey pub, which can be used to sign messages 
        # and subscribe to the chain.
        if wallet == None:
            wallet = bittensor.Wallet( config = self.config )
        config.wallet = wallet.config.wallet
        self.wallet = wallet
        
        # Server: by default the axon serves an RPC server in its own thread using GPRC.
        # The servicer must implement Forward and Backward methods to properly communicate with
        # the other peers in the network.
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.config.axon.max_workers))
        bittensor.grpc.add_BittensorServicer_to_server(self, self._server)
        self._server.add_insecure_port('[::]:' + str(self.config.axon.local_port))

        # Forward and Backward multiprocessing queues
        self.forward_queue = mp.Queue(1000)
        self.backward_queue = mp.Queue(1000)

        # Serving thread: A thread which runs the axon servicer passing items to the nucleus for
        # further processing.
        self._thread = None

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
        config = bittensor.Config.to_config(parser); 
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        r""" Adds this axon's command line arguments to the passed parser.
            Args:
                parser (:obj:`argparse.ArgumentParser`, `required`): 
                    parser argument to append args to.
        """
        bittensor.Wallet.add_args(parser)
        try:
            parser.add_argument('--axon.local_port', default=8091, type=int, 
                help='''The port this axon endpoint is served on. i.e. 8091''')
            parser.add_argument('--axon.local_ip', default='127.0.0.1', type=str, 
                help='''The local ip this axon binds to. ie. 0.0.0.0''')
            parser.add_argument('--axon.use_upnpc', default=False, type=bool, 
                help='''If true this axon will attempt to open a port on your router using upnpc.''')
            parser.add_argument('--axon.external_ip', default=None, type=str, 
                help='''The remote IP served to chain.
                        This ip is subscribed to the chain on boot and is the endpoint other peers see.
                        By default this field is None and is collected by querying a remote server during check_config. 
                        i.e. 207.12.233.1''')
            parser.add_argument('--axon.external_port', default=None, type=str, 
                help='''The remote port to subscribe on chain. By default this port is the same as local_port.
                        If use_upnpc is true this port is determined after the port mapping''')
            parser.add_argument('--axon.max_workers', default=10, type=int, 
                help='''The maximum number connection handler threads working simultaneously on this endpoint. 
                        The grpc server distributes new worker threads to service requests up to this number.''')
            parser.add_argument('--axon.forward_processing_timeout', default=5, type=int, 
                help='''Length of time allocated to the miner forward process for computing and returning responses
                        back to the axon.''')
            parser.add_argument('--axon.backward_processing_timeout', default=5, type=int, 
                help='''Length of time allocated to the miner backward process for computing and returning responses
                        back to the axon.''')

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
        tensor, code, message = self._forward(request)
        response = bittensor.proto.TensorMessage(
            version = bittensor.__version__, 
            public_key = self.wallet.hotkey.public_key, 
            return_code = code,
            message = message,
            tensors = [tensor] if tensor is not None else [],
        )
        # ---- Update stats for this request.
        self.update_stats_for_request(request, response)
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
        tensor, code, message = self._backward(request)
        response = bittensor.proto.TensorMessage(
            version = bittensor.__version__, 
            public_key = self.wallet.hotkey.public_key, 
            return_code = code,
            message = message,
            tensors = [tensor] if tensor is not None else [],
        )

        self.update_stats_for_request(request, response)
        return response

    def next_forward_item( 
            self,
            timeout: int = 10 
        ) -> Tuple[Optional[mp.Pipe], Optional[str], Optional[torch.Tensor], Optional[torch.FloatTensor], Optional[int]]:
        r""" Returns the next forward item from the forward queue to the caller.
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
            return self.forward_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return (None, None, None, None)

    def next_backward_item( 
            self, 
            timeout: int = 10 
        ) -> Tuple[mp.Pipe, str, torch.Tensor, torch.FloatTensor, int]:
        r""" Returns the next backward item from the backward queue to the caller.
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
            return self.backward_queue.get(block=True, timeout=timeout)
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
        try:
            # ---- Build pipe for request ----
            ping, pong = mp.Pipe()
            forward_payload = [pong, public_key, inputs_x, modality]

            # ---- Send request to forward queue ----
            try:
                self.forward_queue.put( forward_payload, block=True, timeout = self.config.axon.forward_processing_timeout )
            except queue.Full:
                return None, bittensor.proto.ReturnCode.NucleusFull, "forward queue is full"

            # ---- Recv response from pipe ----
            if ping.poll( timeout = self.config.axon.forward_processing_timeout ):
                outputs = ping.recv()
                return outputs, bittensor.proto.ReturnCode.Success, "success",
            else:
                return None, bittensor.proto.ReturnCode.NucleusTimeout, "processing timeout got here: {}".format([pong, public_key, inputs_x, modality])

        except Exception as e:
            return None, bittensor.proto.ReturnCode.UnknownException, "Unknown exception when calling nucleus forward {}".format(e)

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
                self.backward_queue.put( backward_payload, block=True, timeout = self.config.axon.backward_processing_timeout )
            except queue.Full:
                return None, bittensor.proto.ReturnCode.NucleusFull, "backward queue is full"

            # ---- Recv response from pipe ----
            if ping.poll( timeout = self.config.axon.backward_processing_timeout ):
                outputs = ping.recv()
                return outputs, bittensor.proto.ReturnCode.Success, "success" 
            else:
                return None, bittensor.proto.ReturnCode.NucleusTimeout, "processing timeout"

        except Exception as e:
            return None, bittensor.proto.ReturnCode.UnknownException, "Unknown exception when calling nucleus backward {}".format(e)
  

    def _forward(self, request):
        r""" Performs validity checks on the grpc request before calling nucleus forward.
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
            message = "forward request contains {} tensors, expected 1 tensor in the forward call".format(len(request.tensors))
            return None, bittensor.proto.ReturnCode.EmptyRequest, message

        # ---- Check deserialization ----
        tensor_inputs = request.tensors[0]
        modality = tensor_inputs.modality
        try:
            deserializer = serialization.get_serializer( serialzer_type = tensor_inputs.serializer )
            torch_inputs = deserializer.deserialize(tensor_inputs, to_type = bittensor.proto.TensorType.TORCH)
        except Exception as e:
            return None, bittensor.proto.ReturnCode.RequestDeserializationException, str(e)

        # ---- Check shape and modality ----
        if torch_inputs.shape[0] < 1:
            message = "Forward request batch dim exception with batch_size = {} ".format(torch_inputs.shape[0])
            return None, bittensor.proto.ReturnCode.RequestShapeException, message

        if torch_inputs.shape[1] < 1:
            message = "Forward request sequence dim exception with sequence_dim = {} ".format(torch_inputs.shape[1])
            return None, bittensor.proto.ReturnCode.RequestShapeException, message

        if modality == bittensor.proto.Modality.TEXT:
            if len(torch_inputs.shape) != 2:
                message = "Forward text input shape exception with len(request.shape) = {} must have rank 2.".format(len(torch_inputs.shape))
                return None, bittensor.proto.ReturnCode.RequestShapeException, message
            
        if modality == bittensor.proto.Modality.IMAGE:
            if len(torch_inputs.shape) != 5:
                message =  "Forward image input shape exception for len(shape) = {}  must have rank 5".format(len(torch_inputs.shape))
                return None, bittensor.proto.ReturnCode.RequestShapeException, message

        if modality == bittensor.proto.Modality.TENSOR:
            if len(torch_inputs.shape) != 3:
                message = "Forward message tensor input shape exception len(shape) = {} must have rank 3".format(len(torch_inputs.shape))
                return None, bittensor.proto.ReturnCode.RequestShapeException, message

        # ---- Make nucleus forward call. ----
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
            return None, bittensor.proto.ReturnCode.ResponseDeserializationException, str(e)

        # ---- Return successful response ----
        return outputs_serialized, code, message


    def _backward(self, request):
        r""" Performs validity checks on the grpc request before calling nucleus backward.
            Returns a the output, message and code from the backend backward call.
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
            return None, bittensor.proto.ReturnCode.InvalidRequest, message

        # ---- Deserialize request ---
        try:
            serializer = serialization.get_serializer( inputs_x.serializer )
            inputs_x = serializer.deserialize( inputs_x, to_type = bittensor.proto.TensorType.TORCH )
            grads_dy = serializer.deserialize( grads_dy, to_type = bittensor.proto.TensorType.TORCH )
        except Exception as e:
            return None, bittensor.proto.ReturnCode.RequestDeserializationException, str(e)

        # ---- Check shapes ----
        if modality_x == bittensor.proto.Modality.TEXT:
            if len(inputs_x.shape) != 2:
                message = "Forward text input shape exception with len(request.shape) = {} must have rank 2.".format(len(inputs_x.shape))
                return None, bittensor.proto.ReturnCode.RequestShapeException, message
            
        if modality_x == bittensor.proto.Modality.IMAGE:
            if len(inputs_x.shape) != 5:
                message =  "Forward image input shape exception for len(shape) = {}  must have rank 5".format(len(inputs_x.shape))
                return None, bittensor.proto.ReturnCode.RequestShapeException, message

        if modality_x == bittensor.proto.Modality.TENSOR:
            if len(inputs_x.shape) != 3:
                message = "Forward message tensor input shape exception len(shape) = {} must have rank 3".format(len(inputs_x.shape))
                return None, bittensor.proto.ReturnCode.RequestShapeException, message

        if len(grads_dy.shape) != 3:
            message = "Passed gradients must have rank 3 but got {}".format(len(grads_dy.shape))
            return None, bittensor.proto.ReturnCode.RequestShapeException, message

        if grads_dy.shape[0] != inputs_x.shape[0] or grads_dy.shape[1] != inputs_x.shape[1]:
            message = "Passed gradients must same first and second dimension as passed inputs got shapes {} and {}".format(grads_dy.shape, inputs_x.shape)
            return None, bittensor.proto.ReturnCode.RequestShapeException, message
 
        # ---- Make nucleus backward call. ----
        outputs, code, message = self.enqueue_backward_to_nucleus( 
            public_key = request.public_key, 
            inputs_x = inputs_x, 
            grads_dy = grads_dy, 
            modality = modality_x
        )
        logger.info('{},{},{}', outputs, code, message)
        if code != bittensor.proto.ReturnCode.Success:
            return None, code, message

        # ---- Deserialize response ----
        try:
            serializer = serialization.get_serializer( bittensor.proto.Serializer.MSGPACK )
            outputs_serialized = serializer.serialize( outputs, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH )
        except Exception as e:
            message = "Backward request serialization failed with error {} and inputs {}".format(e, outputs)
            return None, bittensor.proto.ReturnCode.ResponseSerializationException, message

        # ---- Finaly return ----
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
        total_in_bytes_str = colored('\u290B {:.1f}'.format((self.stats.total_in_bytes.value * 8)/1000), 'red')
        total_out_bytes_str = colored('\u290A {:.1f}'.format((self.stats.total_in_bytes.value * 8)/1000), 'green')
        qps_str = colored("{:.3f}".format(float(self.stats.qps.value)), 'blue')
        return "(" + qps_str + "q/s|" + total_out_bytes_str + "/" + total_in_bytes_str + "kB/s" + ")"
    
    def __to_tensorboard__(self, tensorboard, global_step):
        total_in_bytes = (self.stats.total_in_bytes.value * 8)/1000
        total_out_bytes = (self.stats.total_out_bytes.value * 8)/1000
        tensorboard.add_scalar("Axon/total_in_bytes", total_in_bytes, global_step)
        tensorboard.add_scalar("Axon/total_in_bytes", total_out_bytes, global_step)
        tensorboard.add_scalar("Axon/Queries/Sec", self.stats.qps.value, global_step)

    def __full_string__(self):
        pubkeys = list(self.stats.in_bytes_per_pubkey.keys())
        bytes_in = [avg.value * (8/1000) for avg in self.stats.in_bytes_per_pubkey.values()]
        bytes_out = [avg.value * (8/1000) for avg in self.stats.out_bytes_per_pubkey.values()]
        qps = [qps.value for qps in self.stats.qps_per_pubkey.values()]
        rows = [bytes_out, bytes_in, qps]
        df = pd.DataFrame(rows, columns=pubkeys)
        df = df.rename(index={df.index[0]: colored('\u290A kB/s', 'green')})
        df = df.rename(index={df.index[1]: colored('\u290B kB/s', 'red')})
        df = df.rename(index={df.index[2]: colored('Q/s', 'blue')})
        return '\nAxon:\n' + df.to_string(max_rows=5000, max_cols=25, line_width=1000, float_format = lambda x: '%.2f' % x, col_space=1, justify='left')

    def __del__(self):
        r""" Called when this axon is deleted, ensures background threads shut down properly.
        """
        self.stop()

    def start(self):
        r""" Starts the standalone axon GRPC server thread.
        """
        # Serving thread.
        print ('start')
        self._thread = threading.Thread(target=self._serve, daemon=False)
        self._thread.start()

    def _serve(self):
        try:
            self._server.start()
        except (KeyboardInterrupt, SystemExit):
            self.stop()
        except Exception as e:
            logger.error(e)

    def stop(self):
        r""" Stop the axon grpc server.
        """
        # Delete port maps if required.
        if self.config.axon.use_upnpc:
            try:
                net.upnpc_delete_port_map(self.config.axon.external_port)
            except net.UPNPCException:
                # Catch but continue.
                logger.error('Error while trying to destroy port map on your router.')
        if self._server != None:
            self._server.stop(0)



