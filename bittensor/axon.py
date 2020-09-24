from concurrent import futures
from loguru import logger

import grpc
import random
import threading
import torch

from bittensor import bittensor_pb2_grpc as bittensor_grpc
from bittensor import bittensor_pb2
from bittensor.serializer import PyTorchSerializer
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
        synapse_proto = synapse.to_proto()
        self._local_synapses[synapse_proto.synapse_key] = synapse
    
    def Forward(self, request: bittensor_pb2.TensorMessage, context: grpc.ServicerContext):
        # TODO (const): optionally check signature.
        # Return null response if the target does not exist.
        if request.synapse_key not in self._local_synapses:
            return bittensor_pb2.TensorMessage()
        synapse = self._local_synapses[request.synapse_key]
        
        # Deserializer and decode.
        dtype = request.tensors[0].tensor_def.dtype        
        if dtype == bittensor_pb2.DataType.STRING:
            x = PyTorchSerializer.deserialize_string(request.tensors[0])
            y = synapse.encode_string(x)
        else:
            x = PyTorchSerializer.deserialize_tensor(request.tensors[0])
            y = synapse.encode_tensor(x)
        
        # Make local call to synapse.
        z = synapse.call_forward(y)
        
        # Serializer and pass to network.
        z_serialized = PyTorchSerializer.serialize_tensor(z)

        response = bittensor_pb2.TensorMessage(
            version = bittensor.__version__,
            neuron_key = self._config.neuron_key,
            synapse_key = request.synapse_key,
            tensors = [z_serialized])
        
        return response

    def Backward(self, request: bittensor_pb2.TensorMessage, context: grpc.ServicerContext):
        # TODO (const): optionally check signature.
        # Return null response if the target does not exist.
        if request.synapse_key not in self._local_synapses:
            return bittensor_pb2.TensorMessage()
        synapse = self._local_synapses[request.synapse_key]
                
        # Make local call.
        x = PyTorchSerializer.deserialize_tensor(request.tensors[0])
        dy = PyTorchSerializer.deserialize_tensor(request.tensors[1])        
        dx = synapse.call_backward(x, dy)    
        dx_serialized = PyTorchSerializer.serialize_tensor(dx)

        response = bittensor_pb2.TensorMessage(
            version = bittensor.__version__,
            neuron_key = self._config.neuron_key,
            synapse_key = request.synapse_key,
            tensors = [dx_serialized])
        return response
