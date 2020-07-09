from concurrent import futures
from typing import List

import math
import grpc
import random
import threading
import time

from opentensor import opentensor_pb2
from opentensor import opentensor_pb2_grpc as opentensor_grpc
import opentensor


class Metagraph(opentensor_grpc.MetagraphServicer):
    """
        A continuously updating metagraph state object. Uses the locally
        trained weights to keep this set prunned.
    """
    def __init__(self,
                 max_size: int = 1000000,
                 port: int = random.randint(1000, 10000)):
        # Max size of the graph (number of axons)
        self._max_size = max_size
        # List of graph axons.
        # TODO(const) access mutex
        self._axons = []
        # Local axon list
        self._local_axons = []
        # A map from axon identity to a learned score.
        self._weights = {}

        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        opentensor_grpc.add_OpentensorServicer_to_server(self, self._server)
        self._server.add_insecure_port('[::]:' + str(port))

        # Update thread.
        self._update_thread = None
        self._server_thread = None
        self._running = False

    def get(self, n: int) -> List[opentensor_pb2.Axon]:
        """ Returns min(n, len(axons)) axon from the graph sorted by score."""
        min_n = min(len(self._axons, n))
        return self._axons[:n]

    def setweights(self, axons: List[opentensor_pb2.Axon],
                   weights: List[float]):
        """ Set local scores for each passed node """
        for idx, axon in enumerate(axons):
            self._weights[axon.identity] = weights[idx]

    def getweights(self, axons: List[opentensor_pb2.Axon]) -> List[float]:
        """ Get local weights for a list of axons """
        result = []
        for ax in axons:
            if ax.identity not in self._weights:
                result.append(0.0)
            else:
                result.append(self._weights[ax.identity])
        return result

    def subscribe(self, axon: opentensor.Axon):
        """ Adds a local node to the graph """
        # TODO (const) remove items.
        axon_proto = opentensor_pb2.Axon(version=1.0,
                                         public_key=axon.public_key(),
                                         identity=axon.identity(),
                                         address=axon.address(),
                                         port=axon.port(),
                                         indef=axon.indef(),
                                         outdef=axon.outdef(),
                                         definition=axon.definition())
        self._axons.append(axon_proto)
        self._local_axons.append(axon_proto)
        self._weights[axon_proto.identity] = math.inf

    def Gossip(self, request, context):
        self._sink(request)
        return self._make_axon_batch(10)

    def _update(self):
        """ Internal update thread. Keeps the metagraph up to date. """
        while self._running:
            self._gossip()
            time.sleep(10)

    def _gossip(self):
        """ Sends gossip query to random node in cache """
        batch = self._make_axon_batch(10)
        axon_choice = random.choice(self._axons)

        # Make query.
        version = 1.0
        address = axon_choice.address + ":" + (axon_choice.metagraph_port + 1)
        channel = grpc.insecure_channel(address)
        stub = opentensor_grpc.MetagraphStub(channel)
        response = stub.Gossip(batch)

        # Sink the results to the cache.
        self._sink(response)

    def _make_axon_batch(self, k: int):
        """ Builds a random batch of cache elements of size k """
        # TODO (const) sign message.
        # TODO (const) create new_neuron entries for local endpoints.
        # Create batch of random neuron definitions.
        assert k > 0
        k = min(len(self._axons), 50)
        batch = random.sample(self._axons, k)
        batch = opentensor_pb2.AxonBatch(axons=batch)
        return batch

    def _sink(self, batch: opentensor_pb2.AxonBatch):
        """ Updates storage with gossiped neuron info. """
        # TODO(const) score based on POW, timestamp, and trust.
        # TODO(const) check signatures.
        # TODO(const) sink weights as well.
        # TODO(const) write to disk if need be and replace heap cache.
        # TODO(const) check size contraints.
        for axon in batch:
            if axon.identity not in self._axons:
                self._weights[axon.identity] = 0.0
                self._axons[axon.identity] = axon
            else:
                # TODO (const) check if newer.
                pass

    def __del__(self):
        self.stop()

    def _serve(self):
        self._server.start()

    def stop(self):
        """ Stops the gossip thread """
        self._running = False
        if self._update_thread:
            self._update_thread.join()
        if self._server_thread:
            self._server_thread.join()

    def start(self):
        """ Starts the gossip thread """
        self._running = True
        self._update_thread = threading.Thread(target=self._update,
                                               daemon=True)
        self._server_thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
