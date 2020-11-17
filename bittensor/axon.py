from concurrent import futures
import grpc
from loguru import logger
import random
import threading
import torch

import bittensor
from bittensor import synapse
from bittensor import bittensor_pb2
from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor.serializer import PyTorchSerializer
from bittensor.exceptions.Exceptions import DeserializationException, InvalidRequestException, RequestShapeException, SerializationException, NonExistentSynapseException


class Axon(bittensor_grpc.BittensorServicer):
    """ Processes Fwd and Bwd requests for a set of local Synapses """

    def __init__(self, config, keypair):
        self._config = config
        self.__keypair = keypair

        # Init server objects.
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        bittensor_grpc.add_BittensorServicer_to_server(self, self._server)
        self._server.add_insecure_port('[::]:' + str(self._config.session_settings.axon_port))

        # Local synapses
        self._local_synapses = {}

        # Serving thread.
        self._thread = None

    def __del__(self):
        """ Delete the synapse terminal. """
        self.stop()

    def start(self):
        """ Start the synapse terminal server. """
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
        """ Stop the synapse terminal server """
        self._server.stop(0)

    def serve(self, synapse: bittensor.synapse.Synapse):
        """ Adds an Synapse to the serving set """
        self._local_synapses[synapse.synapse_key()] = synapse

    def Forward(self, request: bittensor_pb2.TensorMessage,
                context: grpc.ServicerContext):
        try:
            if request.synapse_key not in self._local_synapses:
                raise NonExistentSynapseException("There is no record of this caller's synapse key ({}) in the local synapses".format(request.synapse_key))

            synapse = self._local_synapses[request.synapse_key]

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
            y = synapse.call_forward(x, inputs.modality)

            # Serialize.
            y_serialized = [PyTorchSerializer.serialize_tensor(y)]

            # Build response.
            response = bittensor_pb2.TensorMessage(
                version=bittensor.__version__,
                neuron_key=self.__keypair.public_key,
                synapse_key=request.synapse_key,
                tensors=y_serialized)

        except (RequestShapeException, NonExistentSynapseException,
                SerializationException, NotImplementedError, InvalidRequestException) as _:
            # Build null response.
            response = bittensor_pb2.TensorMessage(
                version=bittensor.__version__,
                neuron_key=self.__keypair.public_key,
                synapse_key=request.synapse_key)

        return response

    def Backward(self, request: bittensor_pb2.TensorMessage,
                 context: grpc.ServicerContext):
        # TODO (const): optionally check signature.
        # TODO (const): Exceptions.
        # Return null response if the target does not exist.
        try:
            if request.synapse_key not in self._local_synapses:
                raise NonExistentSynapseException("Backward: There is no record of this caller's synapse key ({}) in the local synapses".format(request.synapse_key))

            synapse = self._local_synapses[request.synapse_key]

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
                    dx = synapse.call_backward(x, dy)
                    dx_serialized = PyTorchSerializer.serialize_tensor(dx)

                    response = bittensor_pb2.TensorMessage(
                        version=bittensor.__version__,
                        neuron_key=self.__keypair.public_key,
                        synapse_key=request.synapse_key,
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
                    neuron_key=self.__keypair.public_key,
                    synapse_key=request.synapse_key)
        
        return response