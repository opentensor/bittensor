import argparse
import grpc
import random
import requests
import threading
import torch
import validators

from concurrent import futures
from munch import Munch
from loguru import logger

import bittensor
from bittensor.synapse import Synapse
from bittensor import bittensor_pb2
from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor.serializer import PyTorchSerializer
from bittensor.exceptions.Exceptions import DeserializationException, InvalidRequestException, RequestShapeException, SerializationException, NonExistentSynapseException

def obtain_ip(config: Munch) -> Munch:
    if config.axon.remote_ip != None:
        return config
    try:
        value = requests.get('https://api.ipify.org').text
    except:
        logger.error("CONIG: Could not retrieve public facing IP from IP API.")
        raise SystemError('CONFIG: Could not retrieve public facing IP from IP API.')
    if not validators.ipv4(value):
        logger.error("CONFIG: Response from IP API is not a valid IP with ip {}", value)
        raise SystemError('CONFIG: Response from IP API is not a valid IP with ip {}'.format(value))
    config.axon.remote_ip = value
    return config

class Axon(bittensor_grpc.BittensorServicer):
    r"""
    The Axon serves a bittensor.synapse.Synapse to recieve remote Forward & Backward calls on the network.
    
    """
    def __init__(self, config, keypair):
        r""" Serves a Synapse to the axon server replacing the previous Synapse if exists.

            Args:
                config (:obj:`Munch`, `required`): 
                    bittensor Munch config.
                keypair (:obj:`substrateinterface.Keypair`, `required`): 
                    bittensor keypair.
        """
        self._config = config
        self.__keypair = keypair

        # Init server objects.
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        bittensor_grpc.add_BittensorServicer_to_server(self, self._server)
        self._server.add_insecure_port('[::]:' + str(self._config.axon.port))

        # Local synapse.
        self._synapse = None

        # Serving thread.
        self._thread = None

    @staticmethod   
    def add_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--axon.port', default=8091, type=int, 
                            help='Port to serve axon')

        parser.add_argument('--axon.remote_ip', default=None, type=str, 
                            help='Remote IP to serve to chain.')
        return parser

    @staticmethod   
    def check_config(config: Munch) -> Munch:
        config = obtain_ip(config)
        assert config.axon.port > 1024 and config.axon.port < 65535, 'config.axon.port must be in range [1024, 65535]'
        return config

    def __del__(self):
        r""" Called when this axon is deleted, ensures background threads shut down properly.
        """
        self.stop()

    def start(self):
        r""" Starts the standalone axon GRPC server thread.
        """
        self._thread = threading.Thread(target=self._serve, daemon=True)
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
        if self._server != None:
            self._server.stop(0)

    def serve(self, synapse: Synapse):
        r""" Serves a Synapse to the axon server replacing the previous Synapse if exists.

            Args:
                synapse (:obj:`bittensor.Synapse`, `required`): 
                    synpase object to serve on the axon server.
        """
        self._synapse = synapse

    def Forward(self, request: bittensor_pb2.TensorMessage,
                context: grpc.ServicerContext):

        r""" Function called by remote GRPC Forward requests by other neurons.

            Args:
                request (:obj:`bittensor_pb2`, `required`): 
                    Tensor request Proto.
                context (:obj:`grpc.ServicerContext`, `required`): 
                    grpc server context.
        """
        try:
            # Single tensor requests only.
            if len(request.tensors) > 0:
                inputs = request.tensors[0]
            else:
                raise InvalidRequestException("Forward: This request contains {} tensors, expected 1 tensor in the forward call".format(len(request.tensors)))

            # Deserialize the modality inputs to tensor.
            try:
                x = PyTorchSerializer.deserialize(inputs)
            except SerializationException as e:
                logger.warning("Exception occured: {}".format(e))
                raise SerializationException("Deserialization of inputs failed. Inputs: {}".format(inputs))

            if x.shape[0] < 1:
                raise RequestShapeException(
                    "request batch dim exception with batch_size = {} ".format(
                        x.shape[0]))

            if x.shape[1] < 1:
                raise RequestShapeException(
                    "request sequence dim exception with sequence_dim = {} ".
                    format(x.shape[1]))

            if inputs.modality == bittensor_pb2.Modality.TEXT:
                if len(x.shape) != 2:
                    raise RequestShapeException(
                        "text input shape exception with len(request.shape) = {} "
                        .format(len(x.shape)))

            elif inputs.modality == bittensor_pb2.Modality.IMAGE:
                if len(x.shape) != 5:
                    raise RequestShapeException(
                        "image input shape exception for len(shape) = {} ".
                        format(len(x.shape)))

            elif inputs.modality == bittensor_pb2.Modality.TENSOR:
                if len(x.shape) != 3:
                    raise RequestShapeException(
                        "tensor input shape exception len(shape) = {} ".format(
                            len(x.shape)))

            # Call forward network. May call NotImplementedError:
            y = self._synapse.call_forward(x, inputs.modality)

            # Serialize.
            y_serialized = [PyTorchSerializer.serialize_tensor(y)]

            # Build response.
            response = bittensor_pb2.TensorMessage(
                version=bittensor.__version__,
                public_key=self.__keypair.public_key,
                tensors=y_serialized)

        except (RequestShapeException, NonExistentSynapseException,
                SerializationException, NotImplementedError, InvalidRequestException) as _:
            # Build null response.
            response = bittensor_pb2.TensorMessage(
                version=bittensor.__version__,
                public_key=self.__keypair.public_key)

        bittensor.session.tbwriter.write_axon_network_data('Forward Call Response Message Size (MB)', response.ByteSize() / 1024)
        return response

    def Backward(self, request: bittensor_pb2.TensorMessage,
                 context: grpc.ServicerContext):
        # TODO (const): optionally check signature.
        # TODO (const): Exceptions.
        # Return null response if the target does not exist.
        try:
            # Make local call.
            if len(request.tensors) != 2:
                raise InvalidRequestException("Backward: There are {} tensors in the request, expected 2.".format(len(request.tensors)))
            else:
                try:
                    x = PyTorchSerializer.deserialize(request.tensors[0])
                    dy = PyTorchSerializer.deserialize(request.tensors[1])
                except DeserializationException as _: 
                    raise DeserializationException("Failed to deserialize {} and {}".format(request.tensors[0], request.tensors[1]))
                
                try:
                    dx = self._synapse.call_backward(x, dy)
                    dx_serialized = PyTorchSerializer.serialize_tensor(dx)
                    response = bittensor_pb2.TensorMessage(
                        version=bittensor.__version__,
                        public_key=self.__keypair.public_key,
                        tensors=[dx_serialized])

                except SerializationException as _:
                    raise SerializationException("Failed to serialize.")
                except NotImplementedError as _:
                    raise NotImplementedError("call_backward is not implemented for this synapse")


        except (InvalidRequestException, NonExistentSynapseException, SerializationException, DeserializationException) as e:
                logger.warning("Exception occured: {}. Sending null response back.".format(e))
                # Build null response.
                response = bittensor_pb2.TensorMessage(
                    version=bittensor.__version__,
                    public_key=self.__keypair.public_key)
        
        bittensor.session.tbwriter.write_axon_network_data('Backward Call Response Message Size (MB)', response.ByteSize() / 1024)
        return response