from concurrent import futures
from loguru import logger
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
                 identity: opentensor.Identity,
                 max_size: int = 1000000,
                 port: int = random.randint(1000, 10000),
                 remote_ip: str = 'localhost',
                 bootstrap: str = None):
        # Opentensor identity
        self._identity = identity
        # remote ip.
        self._remote_ip = remote_ip
        # Max size of the graph (number of axons)
        self._max_size = max_size
        # Address-port string endpoints.
        self._peers = set()
        if bootstrap:
            self._peers.add(bootstrap)
        # List of graph axons.
        # TODO(const) access mutex
        self._axons = {}
        # A map from axon identity to a learned score.
        self._weights = {}

        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        opentensor_grpc.add_MetagraphServicer_to_server(self, self._server)
        self._server.add_insecure_port('[::]:' + str(port))

        # Update thread.
        self._update_thread = None
        self._server_thread = None
        self._running = False

    def get(self, n: int) -> List[opentensor_pb2.Axon]:
        """ Returns min(n, len(axons)) axon from the graph sorted by score."""
        # TODO (const) sort axon array
        axon_list = list(self._axons.values())
        min_n = min(len(axon_list), n)
        return axon_list[:n]

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

    def subscribe(self, axon_proto: opentensor_pb2.Axon):
        """ Adds a local node to the graph """
        # TODO (const) remove items.
        logger.info('subscribe', axon_proto)
        self._axons[axon_proto.identity] = axon_proto
        self._weights[axon_proto.identity] = math.inf

    def Gossip(self, request, context):
        self._sink(request)
        return self._make_axon_batch(10)

    def _gossip(self):
        """ Sends ip query to random node in cache """
        if len(self._peers) == 0:
            return
        batch = self._make_axon_batch(10)
        metagraph_address = random.choice(list(self._peers))

        # Switch to loop for local nodes.
        if metagraph_address.split(':')[0] == self._remote_ip:
            metagraph_address = 'localhost:' + metagraph_address.split(':')[1]

        # Make query.
        logger.info('gossip -> {}', metagraph_address)
        try:
            version = 1.0
            channel = grpc.insecure_channel(metagraph_address)
            stub = opentensor_grpc.MetagraphStub(channel)
            response = stub.Gossip(batch)

            # Sink the results to the cache.
            self._sink(response)
        except:
            self._peers.remove(metagraph_address)

    def _make_axon_batch(self, k: int):
        """ Builds a random batch of cache elements of size k """
        # TODO (const) sign message.
        # TODO (const) create new_neuron entries for local endpoints.
        # Create batch of random neuron definitions.
        assert k > 0
        axon_list = list(self._axons.values())
        k = min(len(axon_list), 50)
        batch = random.sample(axon_list, k)
        batch = opentensor_pb2.AxonBatch(axons=batch)
        return batch

    def _sink(self, batch: opentensor_pb2.AxonBatch):
        """ Updates storage with gossiped neuron info. """
        # TODO(const) score based on POW, timestamp, and trust.
        # TODO(const) check signatures.
        # TODO(const) sink weights as well.
        # TODO(const) write to disk if need be and replace heap cache.
        # TODO(const) check size contraints.
        for axon in batch.axons:
            if axon.identity not in self._axons:
                self._weights[axon.identity] = 0.0
                self._axons[axon.identity] = axon
                self._peers.add(axon.address + ':' + axon.m_port)
            else:
                # TODO (const) check if newer.
                pass

    def __del__(self):
        self.stop()

    def _update(self):
        """ Internal update thread. Keeps the metagraph up to date. """
        try:
            while self._running:
                self._gossip()
                print(self._axons)
                time.sleep(10)
        except (KeyboardInterrupt, SystemExit):
            logger.info('stop metagraph')
            self._running = False
            self.stop()

    def _serve(self):
        try:
            self._server.start()
        except (KeyboardInterrupt, SystemExit):
            self.stop()
        except Exception as e:
            logger.error(e)

    def stop(self):
        """ Stops the gossip thread """
        self._running = False
        if self._update_thread:
            self._update_thread.join()
        self._server.stop(0)

    def start(self):
        """ Starts the gossip thread """
        self._running = True
        self._update_thread = threading.Thread(target=self._update,
                                               daemon=True)
        self._server_thread = threading.Thread(target=self._serve, daemon=True)
        self._update_thread.start()
        self._server_thread.start()
