import argparse
import grpc
import random
import requests
import sys
import threading
import torch
import queue
import validators

from concurrent import futures
from munch import Munch
from loguru import logger
from types import SimpleNamespace
from typing import List

import bittensor
import bittensor.utils.networking as net
import bittensor.serialization import serialization
from bittensor.nucleus import Nucleus
from bittensor.synapse import Synapse
from bittensor import bittensor_pb2
from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor.serializer import PyTorchSerializer


class Axon(bittensor_grpc.BittensorServicer):
    r"""
    A bittensor Axon serves a grpc endpoint which provides access to a single bittensor.synapse.Synapse 
    It recieves Forward and Backward requests and process the corresponding Synapse.call_forward and Synapse.call_backward.
    
    """
    def __init__(self, config, nucleus):
        r""" Initializes a new Axon endpoint with passed config and keypair.
            Args:
                config (:obj:`Munch`, `required`): 
                    bittensor Munch config.
                nucleus (:obj:`bittensor.nucleus.Nucleus`, `required`):
                    backend processing nucleus.
        """
        self._config = config
        self.__keypair = config.session.keypair
        self._nucleus = nucleus

        # Init server objects.
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=self._config.axon.max_workers))
        bittensor_grpc.add_BittensorServicer_to_server(self, self._server)
        self._server.add_insecure_port('[::]:' + str(self._config.axon.local_port))

        # Local synapse to serve.
        self.synapse = None
        self.priority = {}
        self._next_unknown_priority_increment = 0 

        # Gradient queue
        self.gradients = queue.PriorityQueue(maxsize = self._config.axon.max_gradients)

        # Serving thread.
        self._thread = None

        # Stats.
        #TODO(\u290B,\u290A)
        self.stats = SimpleNamespace(
            forward_in_bytes_per_second = 0.0,
            backward_in_bytes_per_second = 0.0,
            forward_out_bytes_per_second = 0.0,
            backward_out_bytes_per_second = 0.0,
        )

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser):
        r""" Adds this axon's command line arguments to the passed parser.
            Args:
                parser (:obj:`argparse.ArgumentParser`, `required`): 
                    parser argument to append args to.
        """
        parser.add_argument('--axon.local_port', default=8091, type=int, help='Port to serve axon')
        parser.add_argument('--axon.local_ip', default='127.0.0.1', type=str, help='IP this axon binds to.')
        parser.add_argument('--axon.use_upnpc', default=False, type=bool, help='Will we attempt to use upnpc to open a port on your router.')
        parser.add_argument('--axon.external_ip', default=None, type=str, help='Remote IP to serve to chain.')
        parser.add_argument('--axon.external_port', default=None, type=str, help='Remote Port to serve to chain.')
        parser.add_argument('--axon.max_workers', default=10, type=int, help='Max number connection handler threads working simultaneously.')
        parser.add_argument('--axon.max_gradients', default=100, type=int, help='Max number of lingering gradient stored in the gradient queue')

    @staticmethod   
    def check_config(config: Munch):
        r""" Checks the passed config items for validity and obtains the remote ip.
            Args:
                config (:obj:`munch.Munch, `required`): 
                    config to check.
        """
        assert config.axon.local_port > 1024 and config.axon.local_port < 65535, 'config.axon.local_port must be in range [1024, 65535]'

        # Attain external ip.
        try:
            config.axon.external_ip = net.get_external_ip()
        except net.ExternalIPNotFound as external_port_exception:
            logger.error('Axon failed in its attempt to attain your external ip. Check your internet connection.')
            raise external_port_exception

        # Optionally: use upnpc to map your router to the local host.
        if config.axon.use_upnpc:
            # Open a port on your router
            try:
                config.axon.external_port = net.upnpc_create_port_map(local_port = config.axon.local_port)
            except net.UPNPCException as upnpc_exception:
                logger.error('Axon failed in its attempt to attain your external ip. Check your internet connection.')
                raise upnpc_exception
        # Falls back to using your provided local_port.
        else:
            config.axon.external_port = config.axon.local_port

        logger.info('Public Endpoint: {}:{}', config.axon.external_ip, config.axon.external_port)
        logger.info('Local Endpoint: {}:{}', config.axon.local_ip, config.axon.local_port)

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
        if self._config.axon.use_upnpc:
            try:
                net.upnpc_create_port_map(self._config.axon.external_port)
            except net.UPNPCException:
                # Catch but continue.
                logger.error('Error while trying to destroy port map on your router.')
        logger.info('Shutting down the Nucleus...')
        self._nucleus.stop()
        if self._server != None:
            self._server.stop(0)


    def serve(self, synapse: Synapse):
        r""" Set the synapse being served on this axon endpoint. 
            This object's call_forward and call_backward will be 
            called on incoming Forward and Backward requests respectively.

            Args:
                synapse (:obj:`bittensor.synapse.Synapse`, `required`): 
                    synpase object to serve.
        """
        self.synapse = synapse

    def set_priority(self, neurons: List[bittensor_pb2.Neuron], priority: torch.FloatTensor):
        r""" Set the serving priority for requests on the served synapse. 
            Float values must are normalized to 1.
            
            Args:
                neurons (:obj:`List[bittensor_pb2.Neuron]` of shape :obj:`(num_neurons)`, `required`):
                    List of remote neurons which match length of x. Tensors from x are sent forward to these neurons.

                priority (:obj:`torch.FloatTnsor` of shape :obj:`(num_neurons)`, `required`): 
                    call priority for neurons on endpoint.
        """
        assert priority.shape[0] == len(neurons), 'priority for neurons must of the same length'
        if torch.sum(priority) != 0 and torch.sum(priority) != 0:
            priority = priority / torch.sum(priority)
        priority_map = {}
        for neuron, priority in list(zip(neurons, priority.tolist())):
            priority_map[neuron.public_key] = priority
        self.priority = priority_map

    def Forward(self, request: bittensor_pb2.TensorMessage, context: grpc.ServicerContext) -> bittensor_pb2.TensorMessage:
        r""" The function called by remote GRPC Forward requests from other neurons.
            Forward is equivalent to a 'forward' pass through a neural network.
            After checking request validity, passes the request to the nucleus for processing.
            See bittensor_pb2.ReturnCode for all possible return codes.
            Args:
                request (:obj:`bittensor_pb2`, `required`): 
                    Tensor request proto.
                context (:obj:`grpc.ServicerContext`, `required`): 
                    grpc server context.
            Returns:
                response: (bittensor_pb2.TensorMessage): 
                    proto response carring the synapse forward output or None under failure.
        """
        if request.version in bittensor.__compatability__:
            tensor, message, code = self._forward(request)
            response = bittensor_pb2.TensorMessage(
                version = bittensor.__version__, 
                public_key = self.__keypair.public_key, 
                return_code = code,
                message = message,
                tensors = [tensor] if tensor is not None else [],
            )

        # Catch incompatible request versions.
        else:
            code = bittensor_pb2.ReturnCode.RequestIncompatibleVersion
            message = "request version must be in {}".format(bittensor.__compatability__)
            response = bittensor_pb2.TensorMessage(
                version = bittensor.__version__, 
                public_key = self.__keypair.public_key, 
                return_code = code,
                message = message,
            )

        return response


    def Backward(self, request: bittensor_pb2.TensorMessage, context: grpc.ServicerContext) -> bittensor_pb2.TensorMessage:
        r""" The function called by remote GRPC Backward requests from other neurons.
            Backward is equivalent to a 'backward' gradient descent pass through a neural network.
            After checking request validity, passes the request to the nucleus for processing.
            See bittensor_pb2.ReturnCode for all possible return codes.
            Args:
                request (:obj:`bittensor_pb2`, `required`): 
                    Tensor request proto.
                context (:obj:`grpc.ServicerContext`, `required`): 
                    grpc server context.
            Returns:
                response: (bittensor_pb2.TensorMessage): 
                    proto response carring the synapse backward output or None under failure.
        """
        if request.version in bittensor.__compatability__:
            tensor, message, code = self._backward(request)
            response = bittensor_pb2.TensorMessage(
                version = bittensor.__version__, 
                public_key = self.__keypair.public_key, 
                return_code = code,
                message = message,
                tensors = [tensor] if tensor is not None else [],
            )

        # Catch incompatible request versions.
        else:
            code = bittensor_pb2.ReturnCode.RequestIncompatibleVersion
            message = "request version must be in {}".format(self._compatible_request_versions)
            response = bittensor_pb2.TensorMessage(
                version = bittensor.__version__, 
                public_key = self.__keypair.public_key, 
                return_code = code,
                message = message,
            )
        return response
            

    def _forward(self, request):
        r""" Performs validity checks on the grpc request before calling nucleus forward.
            Returns a the output, message and code from the backend forward call.
            Args:
                request (:obj:`bittensor_pb2`, `required`): 
                    Tensor request proto.
            Returns:
                response: (:obj:`bittensor_pb2.Tensor, `required`): 
                    serialized tensor response from the nucleus call or None.
                message: (str, `required`): 
                    message associated with forward call, potentially error, or 'success'.
                code: (:obj:`bittensor_pb2.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
        """
        # ---- Check synapse exists ----
        if self.synapse == None:
            message = "Remote axon not serving a synapse"
            code = bittensor_pb2.ReturnCode.NotServingSynapse
            return None, message, code

        # ---- Check Empty request ----
        if len(request.tensors) == 0:
            message = "Forward request contains {} tensors, expected 1 tensor in the forward call".format(len(request.tensors))
            code = bittensor_pb2.ReturnCode.EmptyRequest
            return None, message, code

        # ---- Check deserialization ----
        inputs = request.tensors[0]
        try:
            deserializer = serialization.get_serializer( serialzer_type = inputs.serialzer )
            x = deserializer.deserialize(inputs, to_type = bittensor_pb2.TensorType.TORCH)
        except Exception as e:
            message  = "Forward request deserialization failed with error {}".format(e)
            code = bittensor_pb2.ReturnCode.RequestDeserializationException
            return None, message, code

        # ---- Check shape and modality ----
        if x.shape[0] < 1:
            message = "Froward request batch dim exception with batch_size = {} ".format(x.shape[0])
            code = bittensor_pb2.ReturnCode.RequestShapeException
            return None, message, code

        if x.shape[1] < 1:
            message = "Forward request sequence dim exception with sequence_dim = {} ".format(x.shape[1])
            code =  bittensor_pb2.ReturnCode.RequestShapeException
            return None, message, code

        if inputs.modality == bittensor_pb2.Modality.TEXT:
            if len(x.shape) != 2:
                message = "Forward text input shape exception with len(request.shape) = {} must have rank 2.".format(len(x.shape))
                code =  bittensor_pb2.ReturnCode.RequestShapeException
                return None, message, code
            
        if inputs.modality == bittensor_pb2.Modality.IMAGE:
            if len(x.shape) != 5:
                message =  "Forward image input shape exception for len(shape) = {}  must have rank 5".format(len(x.shape))
                code =  bittensor_pb2.ReturnCode.RequestShapeException
                return None, message, code

        if inputs.modality == bittensor_pb2.Modality.TENSOR:
            if len(x.shape) != 3:
                message = "Forward message tensor input shape exception len(shape) = {} must have rank 3".format(len(x.shape))
                code = bittensor_pb2.ReturnCode.RequestShapeException
                return None, message, code

        # --- Get call priority ----
        try:
            call_priority = self.priority[request.public_key] + random.random()
        except:
            call_priority = 1 + random.random()

        # ---- Make Nucleus forward call. ----
        try:
            outputs, message, code = self._nucleus.forward(
                synapse = self.synapse, 
                inputs = x, 
                mode = inputs.modality, 
                priority = call_priority
            )

            # ---- Catch Nucleus errors ----
            if code != bittensor_pb2.ReturnCode.Success:
                return None, message, code

        except Exception as e:
            message = "Unknown exception when calling nucleus forward {}".format(e)
            code = bittensor_pb2.ReturnCode.UnknownException
            return None, message, code

        # ---- Serialize response ----
        try:
            serializer = serialization.get_serializer ( bittensor_pb2.Serializer.PICKLE )
            outputs_serialized = serializer.serialize ( outputs, from_type = bittensor_pb2.TensorType.TORCH )
        
        except Exception as e:
            message = "Serializtion of forward response failed with error {} and inputs: {}".format(e, outputs)
            code = bittensor_pb2.ReturnCode.ResponseDeserializationException
            return None, message, code

        # ---- Return successful response ----
        return outputs_serialized, message, code


    def _backward(self, request):
        r""" Performs validity checks on the grpc request before calling nucleus backward.
            Returns a the output, message and code from the backend backward call.
            Args:
                request (:obj:`bittensor_pb2`, `required`): 
                    Tensor request proto.
            Returns:
                response: (:obj:`bittensor_pb2.Tensor, `required`): 
                    serialized tensor response from the nucleus call or None.
                message: (str, `required`): 
                    message associated with forward call, potentially error, or 'success'.
                code: (:obj:`bittensor_pb2.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
        """
        # ---- Check that we have a synapse ----.
        if self.synapse == None:
            message = "Remote axon not serving a synapse"
            code = bittensor_pb2.ReturnCode.NotServingSynapse
            return None, message, code

        # ---- Check request inputs ----.
        if len(request.tensors) == 2:
            inputs_x = request.tensors[0]
            grads_dy = request.tensors[1]
            modality_x = inputs_x.modality
        else:
            message = "During backward: There are {} tensors in the request, expected 2.".format(len(request.tensors))
            code =  bittensor_pb2.ReturnCode.InvalidRequest
            return None, message, code

        # ---- Deserialize request ---
        try:
            deserializer = serialization.get_serializer( inputs_x.serializer )
            inputs_x = deserializer.deserialize( inputs_x, to_type = serialization.SerializationTypes.TORCH )
            grads_dy = deserializer.deserialize( grads_dy, to_type = serialization.SerializationTypes.TORCH )
                
        except Exception as e:
            message = "Backward request deserialization failed with unknown error {}".format(e)
            code =  bittensor_pb2.ReturnCode.RequestDeserializationException
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
            logger.trace('gradient queue is full at size: {}', self.gradient_queue.qsize())

        # ---- Nucleus backward call ----
        try:
            outputs, message, code = self._nucleus.backward(
                    synapse = self.synapse, 
                    inputs_x = inputs_x, 
                    grads_dy = grads_dy, 
                    modality = modality_x,
                    priority = call_priority
            )
        except Exception as e:
            message  = "Unkown exception when calling backward with error {}".format(e)
            code =  bittensor_pb2.ReturnCode.UnknownException
            return None, message, code

        # ---- Deserialize response ----
        try:
            serializer = serialization.get_serializer( bittensor_pb2.Serializer.PICKLE )
            outputs_serialized = serializer.serialize( outputs, from_type = serialization.SerializationTypes.TORCH ))

        except Exception as e:
            message = "Backward request serialization failed with error {} and inputs {}".format(e, outputs_serialized)
            code =  bittensor_pb2.ReturnCode.ResponseSerializationException
            return None, message, code

        # ---- Finaly return ----
        return outputs_serialized, message, code



