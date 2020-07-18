from concurrent import futures
from loguru import logger

import grpc
import random
import threading
import torch

from opentensor import opentensor_pb2_grpc as opentensor_grpc
from opentensor import opentensor_pb2
import opentensor

class Axon(opentensor_grpc.OpentensorServicer):
    """ Processes Fwd and Bwd requests for a set of local Synapses """
    def __init__(self, identity, port):
        self._identity = identity

        # Init server objects.
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        opentensor_grpc.add_OpentensorServicer_to_server(self, self._server)
        self._server.add_insecure_port('[::]:' + str(port))

        # Local synapses
        self._synapses = {}

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

    def subscribe(self, synapse_proto: opentensor_pb2.Synapse, synapse: opentensor.Synapse):
        """ Adds an Synapse to the serving set """
        self._synapses[synapse_proto.identity] = synapse

    def Fwd(self, request, context):
        version = request.version
        neuron_key = request.neuron_key
        source_id = request.source_id
        target_id = request.target_id
        #nounce = request.nounce
        x_tensor = request.tensors[0]

        # Return null response if the target does not exist.
        if target_id not in self._synapses:
            return opentensor_pb2.TensorMessage()

        synapse = self._synapses[target_id]
        x = opentensor.Serializer.deserialize(x_tensor)
        y = synapse (x)
        y_tensor = opentensor.Serializer.serialize(y)

        response = opentensor_pb2.TensorMessage(
            version=version,
            neuron_key=self._identity.public_key(),
            source_id=target_id,
            target_id=source_id,
            tensors=[y_tensor])
        return response

    def Bwd(self, request, context):
        pass