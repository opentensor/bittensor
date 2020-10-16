from concurrent import futures
from loguru import logger

import grpc
import random
import threading
import torch

from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor import bittensor_pb2
from bittensor.serializer import PyTorchSerializer
from bittensor.exceptions.ResponseExceptions import RequestShapeException
import bittensor

class Axon(bittensor_grpc.BittensorServicer):
    """ Processes Fwd and Bwd requests for a set of local Synapses """
    def __init__(self, config: bittensor.Config):
        self._config = config

        # Init server objects.
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        bittensor_grpc.add_BittensorServicer_to_server(self, self._server)
        self._server.add_insecure_port('[::]:' + str(self._config.axon_port))

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

    def serve(self, synapse: bittensor.Synapse):
        """ Adds an Synapse to the serving set """
         # Create a new bittensor_pb2.Synapse proto.
        synapse_proto = bittensor_pb2.Synapse(
            version = bittensor.__version__, 
            neuron_key = self._config.neuron_key, 
            synapse_key = synapse.synapse_key(), 
            address = self._config.remote_ip, 
            port = self._config.axon_port, 
        )
        self._local_synapses[synapse.synapse_key()] = synapse
    
    def Forward(self, request: bittensor_pb2.TensorMessage, context: grpc.ServicerContext):
        # TODO (const): optionally check signature.
        # Return null response if the target does not exist.
        if request.synapse_key not in self._local_synapses:
            return bittensor_pb2.TensorMessage()
        
        # Find synapse.
        synapse = self._local_synapses[request.synapse_key]
        
        # Deserialize and decode.
        inputs = request.tensors[0]
        
        # Deserialize the modality inputs to tensor.
        x = PyTorchSerializer.deserialize( inputs )

        # Check shaping contraints.
        try:
            if x.shape[0] < 1:
                raise RequestShapeException("request batch dim exception with batch_size = {} ".format(x.shape[0]))

            if x.shape[1] < 1:
                raise RequestShapeException("request sequence dim exception with sequence_dim = {} ".format(x.shape[1]))

            if inputs.modality == bittensor_pb2.Modality.TEXT:
                if len(x.shape) != 2:
                    raise RequestShapeException("text input shape exception with len(request.shape) = {} ".format(len(x.shape)))

            elif inputs.modality == bittensor_pb2.Modality.IMAGE:
                if len(x.shape) != 5:
                    raise RequestShapeException("image input shape exception for len(shape) = {} ".format(len(x.shape)))

            elif inputs.modality == bittensor_pb2.Modality.TENSOR:
                if len(x.shape) != 3:
                    raise RequestShapeException("tensor input shape exception len(shape) = {} ".format(len(x.shape)))
        
            # Call forward network. May call NotImplementedError:
            y = synapse.call_forward(x, inputs.modality)

            # Serialize.
            y_serialized = [PyTorchSerializer.serialize_tensor(y)]

            # Build response.
            response = bittensor_pb2.TensorMessage(
                version = bittensor.__version__,
                neuron_key = self._config.neuron_key,
                synapse_key = request.synapse_key,
                tensors = y_serialized
            )
                
        except (RequestShapeException, NotImplementedError) as _:
            # Build null response.
            response = bittensor_pb2.TensorMessage (
                version = bittensor.__version__,
                neuron_key = self._config.neuron_key,
                synapse_key = request.synapse_key
            )
        
        return response

    def Backward(self, request: bittensor_pb2.TensorMessage, context: grpc.ServicerContext):
        # TODO (const): optionally check signature.
        # Return null response if the target does not exist.
        if request.synapse_key not in self._local_synapses:
            return bittensor_pb2.TensorMessage()
        synapse = self._local_synapses[request.synapse_key]
                
        # Make local call.
        x = PyTorchSerializer.deserialize(request.tensors[0])
        dy = PyTorchSerializer.deserialize(request.tensors[1])        
        dx = synapse.call_backward(x, dy)    
        dx_serialized = PyTorchSerializer.serialize_tensor(dx)

        response = bittensor_pb2.TensorMessage(
            version = bittensor.__version__,
            neuron_key = self._config.neuron_key,
            synapse_key = request.synapse_key,
            tensors = [dx_serialized])
        return response
