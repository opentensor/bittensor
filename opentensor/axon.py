from concurrent import futures
from loguru import logger

import grpc
import random
import threading
import torch
from torch.utils.tensorboard import SummaryWriter

from opentensor import opentensor_pb2_grpc as opentensor_grpc
from opentensor import opentensor_pb2
import opentensor


class Axon():
    """ Implementation of an axon. A single ip/port tensor processing unit """
    def __init__(self):
        pass

    def indef(self) -> opentensor_pb2.TensorDef:
        """ Returns the opentensor_pb2.TensorDef for the input """
        raise NotImplementedError

    def outdef(self) -> opentensor_pb2.TensorDef:
        """ Returns the opentensor_pb2.TensorDef for the output """
        raise NotImplementedError

    def forward(self, key, tensor) -> torch.Tensor:
        """ Processes the tensor from the sent key """
        raise NotImplementedError

    def backward(self, key, tensor) -> torch.Tensor:
        """ Processes the gradient from the sent key """
        raise NotImplementedError


class AxonTerminal(opentensor_grpc.OpentensorServicer):
    """ Processes Fwd and Bwd requests for a set of local Axons """
    def __init__(self, identity, port, writer: SummaryWriter):
        self._identity = identity
        self._writer = writer

        # Init server objects.
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        opentensor_grpc.add_OpentensorServicer_to_server(self, self._server)
        self._server.add_insecure_port('[::]:' + str(port))

        # Local axons
        self._axons = {}

        # Serving thread.
        self._thread = None

    def __del__(self):
        """ Delete the axon terminal. """
        self.stop()

    def start(self):
        """ Start the axon terminal server. """
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
        """ Stop the axon terminal server """
        self._server.stop(0)

    def subscribe(self, axon_proto: opentensor_pb2.Axon, axon: Axon):
        """ Adds an Axon to the serving set """
        self._axons[axon_proto.identity] = axon

    def Fwd(self, request, context):
        version = request.version
        neuron_key = request.neuron_key
        source_id = request.source_id
        target_id = request.target_id
        #nounce = request.nounce
        tensor = request.tensors[0]

        tensor = opentensor.Serializer.deserialize(tensor)
        assert target_id in self._axons
        axon = self._axons[target_id]
        tensor = axon.forward(source_id, tensor)
        tensor = opentensor.Serializer.serialize(tensor)

        response = opentensor_pb2.TensorMessage(
            version=version,
            neuron_key=self._identity.public_key(),
            source_id=target_id,
            target_id=source_id,
            tensors=[tensor])
        return response

    def Bwd(self, request, context):
        self._metagraph.Bwd(request, context)
