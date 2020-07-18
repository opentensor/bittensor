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
        # Max size of the graph (number of synapses)
        self._max_size = max_size
        # Address-port string endpoints.
        self._peers = set()
        if bootstrap:
            self._peers.add(bootstrap)
        # List of graph synapses.
        # TODO(const) access mutex
        self._synapses = {}
        
        # A map from synapse identity to a learned score.
        self._weights = {}
        
        # A map from synapse identity to time of addition.
        self._ToA = {}
        self._ttl = 60 * 5 # 5 minutes.

        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        opentensor_grpc.add_MetagraphServicer_to_server(self, self._server)
        self._server.add_insecure_port('[::]:' + str(port))

        # Update thread.
        self._update_thread = None
        self._server_thread = None
        self._running = False

    def get(self, n: int) -> List[opentensor_pb2.Synapse]:
        """ Returns min(n, len(synapses)) synapse from the graph sorted by score."""
        # TODO (const) sort synapse array
        synapse_list = list(self._synapses.values())
        min_n = min(len(synapse_list), n)
        return synapse_list[:n]

    def setweights(self, synapses: List[opentensor_pb2.Synapse],
                   weights: List[float]):
        """ Set local scores for each passed node """
        for idx, synapse in enumerate(synapses):
            self._weights[synapse.identity] = weights[idx]

    def getweights(self, synapses: List[opentensor_pb2.Synapse]) -> List[float]:
        """ Get local weights for a list of synapses """
        result = []
        for ax in synapses:
            if ax.identity not in self._weights:
                result.append(0.0)
            else:
                result.append(self._weights[ax.identity])
        return result

    def subscribe(self, synapse_proto: opentensor_pb2.Synapse):
        """ Adds a local node to the graph """
        # TODO (const) remove items.
        logger.info('subscribe', synapse_proto)
        self._synapses[synapse_proto.identity] = synapse_proto
        self._weights[synapse_proto.identity] = math.inf

    def Gossip(self, request, context):
        self._sink(request)
        return self._make_synapse_batch(10)

    def remove(self, uid):
        if uid in self._synapses:
            del self._synapses[uid]
        if uid in self._weights:
            del self._weights[uid]
        if uid in self._ToA:
            del self._ToA[uid]
            
    def add(self, synapse: opentensor_pb2.Synapse):
        self._synapses[synapse.identity] = synapse
        if synapse.identity not in self._weights:
            self._weights[synapse] = 0.0
        self._ToA[synapse.identity] = time.time()
        
    def _clean(self):
        now = time.time()
        for uid in list(self._ToA):
            if now - self._ToA[uid] > self._ttl:
                self.remove(uid)
                
    def _gossip(self):
        """ Sends ip query to random node in cache """
        if len(self._peers) == 0:
            return
        batch = self._make_synapse_batch(10)
        metagraph_address = random.choice(list(self._peers))

        # Switch to loop for local nodes.
        realized_address = metagraph_address 
        if metagraph_address.split(':')[0] == self._remote_ip:
            realized_address = 'localhost:' + metagraph_address.split(':')[1]

        # Make query.
        logger.info('gossip -> {}', metagraph_address)
        try:
            version = 1.0
            channel = grpc.insecure_channel(realized_address)
            stub = opentensor_grpc.MetagraphStub(channel)
            response = stub.Gossip(batch)

            # Sink the results to the cache.
            self._sink(response)
        except:
            self._peers.remove(metagraph_address)

    def _make_synapse_batch(self, k: int):
        """ Builds a random batch of cache elements of size k """
        # TODO (const) sign message.
        # TODO (const) create new_neuron entries for local endpoints.
        # Create batch of random neuron definitions.
        try:
            synapse_list = list(self._synapses.values())
            k = min(len(synapse_list), 50)
            batch = random.sample(synapse_list, k)
            batch = opentensor_pb2.SynapseBatch(synapses=batch)
            return batch
        except:
            batch = opentensor_pb2.SynapseBatch()

    def _sink(self, batch: opentensor_pb2.SynapseBatch):
        """ Updates storage with gossiped neuron info. """
        # TODO(const) score based on POW, timestamp, and trust.
        # TODO(const) check signatures.
        # TODO(const) sink weights as well.
        # TODO(const) write to disk if need be and replace heap cache.
        # TODO(const) check size contraints.
        for synapse in batch.synapses:
            self.add(synapse)
        
    def __del__(self):
        self.stop()

    def _update(self):
        """ Internal update thread. Keeps the metagraph up to date. """
        try:
            while self._running:
                self._gossip()
                self._clean()
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

    @property
    def peers(self):
        return self._peers

    @property
    def synapses(self):
        return self._synapses

    @property
    def weights(self):
        return self._weights
