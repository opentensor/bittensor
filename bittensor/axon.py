'''
The MIT License (MIT)
Copyright © 2021 Opentensor.ai

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the “Software”), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.
'''
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
from typing import List

import bittensor
import bittensor.utils.networking as net
import bittensor.serialization as serialization
import bittensor.utils.stats as stat_utils

class Axon(bittensor.grpc.BittensorServicer):
    r"""
        Services Forward and Backward requests from other neurons.
    """
    def __init__(self, config: Munch = None, wallet: 'bittenosr.wallet.Wallet' = None, nucleus: 'bittensor.nucleus.Nucleus' = None, metagraph: 'bittensor.metagraph.Metagraph' = None):
        r""" Initializes a new Axon tensor processing endpoint.
            Args:
                config (:obj:`Munch`, `optional`): 
                    axon.Axon.config()
                wallet (:obj:`bittensor.nucleus.Nucleus`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                nucleus (:obj:`bittensor.nucleus.Nucleus`, `optional`):
                    backend processing nucleus.
                metagraph (:obj:`bittensor.metagraph.Metagraph`, `optional`):
                    bittensor network metagraph.
        """
        # Config: Holds all config items for this items and those that are recursively defined. For instance,
        # config for the wallet, metagraph, and nucleus sub-objects.
        if config == None:
            config = Axon.build_config()
        self.config = config

        # Wallet: Holds you hotkey keypair and coldkey pub, which can be used to sign messages 
        # and subscribe to the chain.
        if wallet == None:
            wallet = bittensor.wallet.Wallet( config = self.config )
        self.wallet = wallet
        
        # Metagraph: Maintains a connection to the subtensor chain which updates with a sync() call.
        # The metagraph can be queried for the latest information about stake and weight matrix state.
        if metagraph == None:
            metagraph = bittensor.metagraph.Metagraph( config = self.config, wallet = self.wallet )
        self.metagraph = metagraph

        # Nucleus: Request processing object. The Axon servicer passes requests to this
        # object to perform the actual computation. Can be ripped out and replaced with various
        # computational objects.
        if nucleus == None:
            nucleus = bittensor.nucleus.Nucleus( config = self.config, wallet = self.wallet, metagraph = self.metagraph )
        self.nucleus = nucleus

        # Server: by default the axon serves an RPC server in its own thread using GPRC.
        # The servicer must implement Forward and Backward methods to properly communicate with
        # the other peers in the network.
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=self.config.axon.max_workers))
        bittensor.grpc.add_BittensorServicer_to_server(self, self._server)
        self._server.add_insecure_port('[::]:' + str(self.config.axon.local_port))

        # Local Synapse: The synapse object that will be served to the network on this endpoint.
        # Must be set by calling axon.server( synapse_object ). This object is not copied by default 
        # and therefore should be threadsafe when run with any nucleus that alos runs in a separate thread 
        # to the main training thread. 
        self.synapse = None

        # A map between public key and processing priority.
        self.priority = {}

        # Gradient queue: A queue of (input, gradient) tuples.
        self.gradients = queue.PriorityQueue(maxsize = self.config.axon.max_gradients)

        # Serving thread: A thread which runs the axon servicer passing items to the nucleus for
        # further processing.
        self._thread = None

        # Stats: Memory of network statistics, QPS and bytes in and out for instance.
        self.stats = SimpleNamespace(
            qps = stat_utils.timed_rolling_avg(0.0, 0.01),
            total_in_bytes = stat_utils.timed_rolling_avg(0.0, 0.01),
            total_out_bytes= stat_utils.timed_rolling_avg(0.0, 0.01),
            in_bytes_per_uid = {},
            out_bytes_per_uid = {},
            qps_per_uid = {},
        )

    def Forward(self, request: bittensor.proto.TensorMessage, context: grpc.ServicerContext) -> bittensor.proto.TensorMessage:
        r""" The function called by remote GRPC Forward requests from other neurons.
            Forward is equivalent to a 'forward' pass through a neural network.
            After checking request validity, this function passes the request to the nucleus for processing.
            See bittensor.proto.ReturnCode for all possible return codes.
            Args:
                request (:obj:`bittensor.proto`, `required`): 
                    Tensor request proto.
                context (:obj:`grpc.ServicerContext`, `required`): 
                    grpc server context.
            Returns:
                response: (bittensor.proto.TensorMessage): 
                    proto response carring the synapse forward output or None under failure.
        """
        # TODO(const): check signature
        # TODO(const): black and white listing.

        tensor, message, code = self._forward(request)
        response = bittensor.proto.TensorMessage(
            version = bittensor.__version__, 
            public_key = self.wallet.keypair.public_key, 
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
            See bittensor.proto.ReturnCode for all possible return codes.
            Args:
                request (:obj:`bittensor.proto`, `required`): 
                    Tensor request proto.
                context (:obj:`grpc.ServicerContext`, `required`): 
                    grpc server context.
            Returns:
                response: (bittensor.proto.TensorMessage): 
                    proto response carring the synapse backward output or None under failure.
        """
        tensor, message, code = self._backward(request)
        response = bittensor.proto.TensorMessage(
            version = bittensor.__version__, 
            public_key = self.wallet.keypair.public_key, 
            return_code = code,
            message = message,
            tensors = [tensor] if tensor is not None else [],
        )

        self.update_stats_for_request(request, response)
        return response
            

    def _forward(self, request):
        r""" Performs validity checks on the grpc request before calling nucleus forward.
            Returns the output, message and code from the backend forward call.
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

        # ---- Check synapse exists ----
        if self.synapse == None:
            message = "Remote axon not serving a synapse"
            code = bittensor.proto.ReturnCode.NotServingSynapse
            return None, message, code

        # ---- Check Empty request ----
        if len(request.tensors) == 0:
            message = "Forward request contains {} tensors, expected 1 tensor in the forward call".format(len(request.tensors))
            code = bittensor.proto.ReturnCode.EmptyRequest
            return None, message, code

        # ---- Check deserialization ----
        inputs = request.tensors[0]
        try:
            deserializer = serialization.get_serializer( serialzer_type = inputs.serializer )
            x = deserializer.deserialize(inputs, to_type = bittensor.proto.TensorType.TORCH)
        except Exception as e:
            message  = "Forward request deserialization failed with error {}".format(e)
            code = bittensor.proto.ReturnCode.RequestDeserializationException
            return None, message, code

        # ---- Check shape and modality ----
        if x.shape[0] < 1:
            message = "Froward request batch dim exception with batch_size = {} ".format(x.shape[0])
            code = bittensor.proto.ReturnCode.RequestShapeException
            return None, message, code

        if x.shape[1] < 1:
            message = "Forward request sequence dim exception with sequence_dim = {} ".format(x.shape[1])
            code =  bittensor.proto.ReturnCode.RequestShapeException
            return None, message, code

        if inputs.modality == bittensor.proto.Modality.TEXT:
            if len(x.shape) != 2:
                message = "Forward text input shape exception with len(request.shape) = {} must have rank 2.".format(len(x.shape))
                code =  bittensor.proto.ReturnCode.RequestShapeException
                return None, message, code
            
        if inputs.modality == bittensor.proto.Modality.IMAGE:
            if len(x.shape) != 5:
                message =  "Forward image input shape exception for len(shape) = {}  must have rank 5".format(len(x.shape))
                code =  bittensor.proto.ReturnCode.RequestShapeException
                return None, message, code

        if inputs.modality == bittensor.proto.Modality.TENSOR:
            if len(x.shape) != 3:
                message = "Forward message tensor input shape exception len(shape) = {} must have rank 3".format(len(x.shape))
                code = bittensor.proto.ReturnCode.RequestShapeException
                return None, message, code

        # --- Get call priority ----
        call_priority = self.get_call_priority(request)

        # ---- Make nucleus.Nucleus forward call. ----
        try:
            outputs, message, code = self.nucleus.forward(
                synapse = self.synapse.to(self.synapse.device), 
                inputs = x.to(self.synapse.device), 
                mode = inputs.modality, 
                priority = call_priority
            )

            # ---- Catch nucleus.Nucleus errors ----
            if code != bittensor.proto.ReturnCode.Success:
                return None, message, code

        except Exception as e:
            message = "Unknown exception when calling nucleus forward {}".format(e)
            code = bittensor.proto.ReturnCode.UnknownException
            return None, message, code

        # ---- Serialize response ----
        try:
            serializer = serialization.get_serializer ( bittensor.proto.Serializer.MSGPACK )
            outputs_serialized = serializer.serialize ( outputs, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH )
        
        except Exception as e:
            message = "Serializtion of forward response failed with error {} and inputs: {}".format(e, outputs)
            code = bittensor.proto.ReturnCode.ResponseDeserializationException
            return None, message, code

        # ---- Return successful response ----
        return outputs_serialized, message, code


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
        # ---- Check that we have a synapse ----.
        if self.synapse == None:
            message = "Remote axon not serving a synapse"
            code = bittensor.proto.ReturnCode.NotServingSynapse
            return None, message, code

        # ---- Check request inputs ----.
        if len(request.tensors) == 2:
            inputs_x = request.tensors[0]
            grads_dy = request.tensors[1]
            modality_x = inputs_x.modality
        else:
            message = "During backward: There are {} tensors in the request, expected 2.".format(len(request.tensors))
            code =  bittensor.proto.ReturnCode.InvalidRequest
            return None, message, code

        # ---- Deserialize request ---
        try:
            serializer = serialization.get_serializer( inputs_x.serializer )
            inputs_x = serializer.deserialize( inputs_x, to_type = bittensor.proto.TensorType.TORCH )
            grads_dy = serializer.deserialize( grads_dy, to_type = bittensor.proto.TensorType.TORCH )
                
        except Exception as e:
            message = "Backward request deserialization failed with unknown error {}".format(e)
            code =  bittensor.proto.ReturnCode.RequestDeserializationException
            return None, message, code

        # --- Get call priority ----
        try:
            call_priority = self.priority[request.public_key] + random.random()
        except:
            call_priority = 1 + random.random()

        # ---- Save gradients to buffer for later use. ---
        try:
            self.gradients.put( (call_priority, (request.public_key, inputs_x, grads_dy, modality_x)) , block=False)
        except queue.Full:
            logger.trace('gradient queue is full at size: {}', self.gradients.qsize())

        # ---- nucleus.Nucleus backward call ----
        try:
            outputs, message, code = self.nucleus.backward(
                    synapse = self.synapse, 
                    inputs_x = inputs_x, 
                    grads_dy = grads_dy, 
                    modality = modality_x,
                    priority = call_priority
            )
        except Exception as e:
            message  = "Unkown exception when calling backward with error {}".format(e)
            code =  bittensor.proto.ReturnCode.UnknownException
            return None, message, code

        # ---- Deserialize response ----
        try:
            serializer = serialization.get_serializer( bittensor.proto.Serializer.MSGPACK )
            outputs_serialized = serializer.serialize( outputs, modality = bittensor.proto.Modality.TENSOR, from_type = bittensor.proto.TensorType.TORCH )

        except Exception as e:
            message = "Backward request serialization failed with error {} and inputs {}".format(e, outputs)
            code =  bittensor.proto.ReturnCode.ResponseSerializationException
            return None, message, code

        # ---- Finaly return ----
        return outputs_serialized, message, code

    def serve(self, synapse: 'bittensor.synapse.Synapse'):
        r""" Set the synapse being served on this axon endpoint. 
            This object's call_forward and call_backward will be 
            called on incoming Forward and Backward requests respectively.

            Args:
                synapse (:obj:`bittensor.synapse.synapse.Synapse`, `required`): 
                    synpase object to serve.
        """
        self.synapse = synapse

    def set_priority(self, neurons: List[bittensor.proto.Neuron], priority: torch.FloatTensor):
        r""" Set the serving priority for requests on the served synapse. 
            Float values must are normalized to 1.
            
            Args:
                neurons (:obj:`List[bittensor.proto.Neuron]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                priority (:obj:`torch.FloatTnsor` of shape :obj:`(num_neurons)`, `required`): 
                    call priority for neurons on endpoint.
        """
        assert priority.shape[0] == len(neurons), 'priority for neurons must of the same length'
        if torch.sum(priority) != 0:
            priority = torch.true_divide(priority, torch.sum(priority))
        priority_map = {}
        for neuron, priority in list(zip(neurons, priority.tolist())):
            priority_map[neuron.public_key] = priority
        self.priority = priority_map

    def get_call_priority(self, request: bittensor.proto.TensorMessage):
        if request.public_key in self.priority:
            call_priority = self.priority[request.public_key]
        else:
            try:
                uid = self.metagraph.state.uid_for_pubkey[request.public_key]
                idx = int(self.metagraph.uids_to_indices(torch.tensor([uid])).item())
                call_priority = self.metagraph.incentive[idx]
            except Exception as e:
                call_priority = 0.0
        call_priority += random.random() * 0.0001
        return call_priority


    def update_stats_for_request(self, request, response):
        self.stats.qps.update(1)
        in_bytes = sys.getsizeof(request)
        out_bytes = sys.getsizeof(response)
        self.stats.total_in_bytes.update(in_bytes)
        self.stats.total_out_bytes.update(out_bytes)
        if request.public_key in self.metagraph.state.uid_for_pubkey:
            # ---- Check we have a stats column for this peer
            request_uid = self.metagraph.state.uid_for_pubkey[request.public_key]
            if request_uid in self.stats.in_bytes_per_uid:
                self.stats.in_bytes_per_uid[request_uid].update(in_bytes)
                self.stats.out_bytes_per_uid[request_uid].update(out_bytes)
                self.stats.qps_per_uid[request_uid].update(1)
            else:
                self.stats.in_bytes_per_uid[request_uid] = stat_utils.timed_rolling_avg(in_bytes, 0.01)
                self.stats.out_bytes_per_uid[request_uid] = stat_utils.timed_rolling_avg(out_bytes, 0.01)
                self.stats.qps_per_uid[request_uid] = stat_utils.timed_rolling_avg(1, 0.01)


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

    def __full_str__(self):
        uids = list(self.stats.in_bytes_per_uid.keys())
        bytes_in = [avg.value * (8/1000) for avg in self.stats.in_bytes_per_uid.values()]
        bytes_out = [avg.value * (8/1000) for avg in self.stats.in_bytes_per_uid.values()]
        qps = [qps.value for qps in self.stats.qps_per_uid.values()]
        rows = [bytes_out, bytes_in, qps]
        df = pd.DataFrame(rows, columns=uids)
        df = df.rename(index={df.index[0]: colored('\u290A kB/s', 'green')})
        df = df.rename(index={df.index[1]: colored('\u290B kB/s', 'red')})
        df = df.rename(index={df.index[2]: colored('Q/s', 'blue')})
        return '\nAxon:\n' + df.to_string(max_rows=5000, max_cols=25, line_width=1000, float_format = lambda x: '%.2f' % x, col_space=1, justify='left')

    @staticmethod   
    def build_config() -> Munch:
        # Parses and returns a config Munch for this object.
        parser = argparse.ArgumentParser(); 
        Axon.add_args(parser) 
        config = bittensor.config.Config.to_config(parser); 
        Axon.check_config(config)
        return config

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        r""" Adds this axon's command line arguments to the passed parser.
            Args:
                parser (:obj:`argparse.ArgumentParser`, `required`): 
                    parser argument to append args to.
        """
        bittensor.nucleus.Nucleus.add_args(parser)
        bittensor.metagraph.Metagraph.add_args(parser) # Also adds for wallet.
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
            parser.add_argument('--axon.max_gradients', default=100, type=int, 
                help='''The max number of lingering gradients stored in the gradient queue.
                        Gradients passed from other peers accumulate on this endpoint and queue in axon.gradients.''')
        except:
            pass

    @staticmethod   
    def check_config(config: Munch):
        r""" Checks the passed config items for validity and obtains the remote ip.
            Args:
                config (:obj:`munch.Munch, `required`): 
                    config to check.
        """
        bittensor.nucleus.Nucleus.check_config(config)
        bittensor.metagraph.Metagraph.check_config(config) # Also checks for wallet.
        assert config.axon.local_port > 1024 and config.axon.local_port < 65535, 'config.axon.local_port must be in range [1024, 65535]'

        # Attain external ip.
        if config.axon.external_ip == None:
            try:
                config.axon.external_ip = net.get_external_ip()
            except net.ExternalIPNotFound as external_port_exception:
                logger.error('Axon failed in its attempt to attain your external ip. Check your internet connection.')
                raise external_port_exception

        if config.axon.external_port == None:
            # Optionally: use upnpc to map your router to the local host.
            if config.axon.use_upnpc:
                # Open a port on your router
                logger.info('UPNPC: ON')
                try:
                    config.axon.external_port = net.upnpc_create_port_map(local_port = config.axon.local_port)
                except net.UPNPCException as upnpc_exception:
                    logger.error('Axon failed in its attempt to attain your external ip. Check your internet connection.')
                    raise upnpc_exception
            # Falls back to using your provided local_port.
            else:
                logger.info('UPNPC: OFF')
                config.axon.external_port = config.axon.local_port

        logger.info('Using external endpoint: {}:{}', config.axon.external_ip, config.axon.external_port)
        logger.info('Using local endpoint: {}:{}', config.axon.local_ip, config.axon.local_port)

    def __del__(self):
        r""" Called when this axon is deleted, ensures background threads shut down properly.
        """
        self.stop()

    def start(self):
        r""" Starts the standalone axon GRPC server thread.
        """
        # Serving thread.
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
        logger.info('Shutting down the nucleus.Nucleus...')
        if self.nucleus != None:
            self.nucleus.stop()
        if self._server != None:
            self._server.stop(0)



