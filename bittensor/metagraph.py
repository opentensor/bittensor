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
    def __init__(self, config: bittensor.Config):
        """Initializes a new Metagraph POW-cache object.
        Args:
            config (opentensor.Config): An opentensor cache config object.
        """
        # Internal state
        self._peers = set()
        self._synapses = {}
        self._weights = {}
        self._heartbeat = {}
        
        # Opentensor config
        self._config = config
        if len(self._config.bootstrap) > 0:
            self._peers.add(self._config.bootstrap)
  
        # Init server objects.
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        opentensor_grpc.add_MetagraphServicer_to_server(self, self._server)
        self._server.add_insecure_port('[::]:' + str(self._config.metagraph_port))
        
        # Update thread.
        self._update_thread = None
        self._server_thread = None
        self._running = False
        
    def subscribe(self, synapse_proto: opentensor_pb2.Synapse):
        self._synapses[synapse_proto.synapse_key] = synapse_proto
        self._weights[synapse_proto.synapse_key] = math.inf
        self._heartbeat[synapse_proto.synapse_key] = time.time()
        
    def get_synapses(self, n: int) -> List[opentensor_pb2.Synapse]:
        """ Returns min(n, len(synapses)) synapse from the graph sorted by score.
        Args:
            n (int): min(n, len(synapses)) synapses to return.

        Returns:
            List[opentensor_pb2.Synapse]: List of synapse endpoints.
        """
        # TODO (const) sort synapse array
        synapse_list = list(self._synapses.values())
        min_n = min(len(synapse_list), n)
        return synapse_list[:min_n]
    
    def get_peers(self, n: int) -> List[str]:
        """ Return min(n, len(peers)) peer endpoints from the active set.

        Args:
            n (int): min(n, len(synapses)) peers to return.

        Returns:
            List[str]: List of peers.
        """
        peer_list = list(self._peers.values())
        min_n = min(len(peer_list), n)
        return peer_list[:min_n]
    
    def _sink(self, request: opentensor_pb2.GossipBatch):
        """Sinks a gossip request to the metagraph.

        Args:
            request (opentensor_pb2.SynapseBatch): [description]
        """
        for peer in request.peers:
            self._peers.add(peer)
        for synapse in request.synapses:
            self._synapses[synapse.synapse_key] = synapse
            self._heatbeat[synapse.synapse_key] = time.time()      

    def Gossip(self, request: opentensor_pb2.GossipBatch, context):
        synapses = self._get_synapses(1000)
        peers = self._get_peers(10)
        self._sink(request)
        response = opentensor_pb2.GossipBatch(peer=peers, synapses=synapses)
        return response
    
    def do_gossip(self):
        """ Sends gossip query to random peer"""
        if len(self._peers) == 0:
            return
        synapses = self._get_synapses(1000)
        peers = self._get_peers(10)
        metagraph_address = random.choice(list(self._peers))
        realized_address = metagraph_address
        if metagraph_address.split(':')[0] == self._config.remote_ip:
            realized_address = 'localhost:' + str(metagraph_address.split(":")[1])
        try:
            version = opentensor.PROTOCOL_VERSION
            channel = grpc.insecure_channel(realized_address)
            stub = opentensor_grpc.MetagraphStub(channel)
            request = opentensor_pb2.GossipBatch(peers=peers, synapses=batch)
            response = stub.Gossip(request)
            self._sink(response)
        except:
            # Faulty peer.
            self._peers.remove(metagraph_address)
        
    def do_clean(self, ttl: int):
        """Cleans lingering metagraph elements
        Args:
            ttl (int): time to live.
        """
        now = time.time()
        for uid in list(self._synapses):
            if now - self._heartbeat[uid] > ttl:
                del self._synapses[uid]
                del self._heartbeat[uid]
        
    def _update(self):
        """ Internal update thread. Keeps the metagraph up to date. """
        try:
            while self._running:
                self.do_gossip()
                self.do_clean(60*60)
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
        
    def __del__(self):
        self.stop()

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
    def synapses(self):
        return self._synapses

    @property
    def weights(self):
        return self._weights
    
    @property
    def peers(self):
        return self._peers
        
    def setweights(self, synapses: List[opentensor_pb2.Synapse],
                   weights: List[float]):
        """ Set local scores for each passed node """
        for idx, synapse in enumerate(synapses):
            self._weights[synapse.synapse_key] = weights[idx]

    def getweights(self, synapses: List[opentensor_pb2.Synapse]) -> List[float]:
        """ Get local weights for a list of synapses """
        result = []
        for syn in synapses:
            if syn.synapse_key not in self._weights:
                result.append(0.0)
            else:
                result.append(self._weights[syn.synapse_key])
        return result
    
