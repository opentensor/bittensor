
import math
import grpc
import random
import threading
import torch
import time
import os
import bittensor
import bittensor.synapse 

from typing import List
from concurrent import futures
from loguru import logger
from bittensor import bittensor_pb2
from bittensor import bittensor_pb2_grpc as bittensor_grpc

class Metagraph(bittensor_grpc.MetagraphServicer):

    def __init__(self, config: bittensor.Config):
        """Initializes a new Metagraph POW-cache object.
        Args:
            config (bittensor.Config): An bittensor cache config object.
        """
        # Internal state
        self._peers = []
        self._synapses = {}
        self._weights = {}
        self._heartbeat = {}
        self._ttl = 1800 # 30 minute time-to-live for testing
        # bittensor config
        self._config = config
        self.bootpeer = self._config.get_bootpeer()
        if self.bootpeer:
            self.bootstrap_peer = {
                                    "address": self.bootpeer,
                                    "neuron_key": None,
                                    "heartbeat": time.time()
            }
            self._peers.append(self.bootstrap_peer)


        # Init server objects.
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=100))
        bittensor_grpc.add_MetagraphServicer_to_server(self, self._server)
        self._server.add_insecure_port('[::]:' +
                                       str(self._config.metagraph_port))

        # Update thread.
        self._update_thread = None
        self._server_thread = None
        self._running = False

        if os.environ.get('https_proxy'):
            del os.environ['https_proxy']
        if os.environ.get('http_proxy'):
            del os.environ['http_proxy']

    def synapses(self, n: int = 1000) -> List[bittensor_pb2.Synapse]:
        """ Returns min(n, len(synapses)) synapse from the graph sorted by score.
        Args:
            n (int): min(n, len(synapses)) synapses to return.

        Returns:
            List[bittensor_pb2.Synapse]: List of synapse endpoints.
        """
        # TODO (const) sort synapse array
        synapse_list = list(self._synapses.values())
        min_n = min(len(synapse_list), n)
        return synapse_list[:min_n]

    def peers(self, n: int = 10) -> List[str]:
        """ Return min(n, len(peers)) peer endpoints from the active set.

        Args:
            n (int): min(n, len(synapses)) peers to return.

        Returns:
            List[str]: List of peers.
        """
        peer_list = list(self._peers)
        min_n = min(len(peer_list), n)
        return peer_list[:min_n]

    def _sink(self, payload: bittensor_pb2.GossipBatch):
        """Sinks a gossip request to the metagraph.

        Args:
            request (bittensor_pb2.SynapseBatch): [description]
        """
        known_peer_keys = [p['neuron_key'] for p in self._peers]
        
        # Special case: If we only have one peer with no neuron key, then this payload 
        # is coming from that peer, and its neuron key should be set for our records.
        if len(self._peers) == 1:
            if not self._peers[0]["neuron_key"]:
                self._peers[0]["neuron_key"] = payload.source_neuron_key

        try:
            for i in range(len(payload.peers)):
                peer_stats = {
                                "address": payload.peers[i].address,
                                "neuron_key": payload.peers[i].neuron_key,
                                "heartbeat": time.time()
                            }

                if payload.peers[i].neuron_key not in known_peer_keys:
                    self._peers.append(peer_stats)
                else:
                   # First, let's find the source peer -- that is, the peer that sent this request.
                   source_peer_neuron_key = payload.source_neuron_key
                   # Let's make sure we are not the source peer 
                   if source_peer_neuron_key != self._config.neuron_key:
                        # We are not the source, so we should update the source's heartbeat
                        # Find the source node
                        src_idx_local = next((index for (index, d) in enumerate(self._peers) if d["neuron_key"] == payload.source_neuron_key), None)
                        src_idx_remote = next((index for (index,d) in enumerate(payload.peers) if d.neuron_key == payload.source_neuron_key), None)
                        # If the src idx doesn't exist here, it means we're talking to a metagraph. 
                        if src_idx_local:
                            now = time.time()
                            # Update the source peer's last heartbeat in our records if they haven't exceeded the ttl
                            # otherwise delete
                            if now - payload.peers[src_idx_remote].heartbeat <= self._ttl:
                                self._peers[src_idx_local].update({"heartbeat": time.time()})
        except Exception as e:
            logger.error("Exception occured: {}".format(e))

        for synapse in payload.synapses:
            self._synapses[synapse.synapse_key] = synapse
            self._heartbeat[synapse.synapse_key] = time.time()

    def Gossip(self, request: bittensor_pb2.GossipBatch, context):
        synapses = self.synapses(1000)
        peers = self.peers(10)
        self._sink(request)
        response = bittensor_pb2.GossipBatch(peers=peers, synapses=synapses, source_neuron_key=self._config.neuron_key)
        return response

    def do_gossip(self):
        """ Sends gossip query to random peer"""
        if len(self._peers) == 0:
            return

        synapses = self.synapses(1000)
        peers = self.peers(10)
        random_peer = random.choice(list(self._peers))
        random_peer_address = random_peer['address']
        realized_address = random_peer_address
        if random_peer_address.split(':')[0] == self._config.remote_ip:
            realized_address = 'localhost:' + str(
                random_peer_address.split(":")[1])
        
        retries = 0
        peer_reached = False
        backoff = 1
        while (retries < 3):
            try:
                channel = grpc.insecure_channel(realized_address, options=(('grpc.enable_http_proxy', 0),))
                stub = bittensor_grpc.MetagraphStub(channel)
                request = bittensor_pb2.GossipBatch(peers=peers, synapses=synapses, source_neuron_key=self._config.neuron_key)
                response = stub.Gossip(request)
                channel.close()
                self._sink(response)
                peer_reached = True
            except Exception as e:
                # Faulty peer.
                logger.warning("Faulty peer!: {}".format(random_peer))
                logger.error("ERROR: {}".format(e))
                #self._peers.remove(random_peer)
                time.sleep(backoff * 2)
                retries += 1
                backoff += 1
                logger.info("Retry number: {}".format(retries))
                continue
            break
        
        if not peer_reached:
            logger.error("Peer {} is unreachable".format(random_peer))
            self._peers.remove(random_peer)


    def do_clean(self, ttl: int):
        """Cleans lingering metagraph elements
        Args:
            ttl (int): time to live. (in minutes)
        """
        now = time.time()
        for uid in list(self._synapses):
            if now - self._heartbeat[uid] > ttl:
                del self._synapses[uid]
                del self._heartbeat[uid]

        for peer in list(self._peers):
            time_elapsed_since_last_heartbeat = now - peer['heartbeat']
            if time_elapsed_since_last_heartbeat > self._ttl:
                logger.info("It appears peer {} has dropped off, last heartbeat was {:.2f} minutes ago".format(peer, time_elapsed_since_last_heartbeat/60))
                self._peers.remove(peer)

    def _update(self):
        """ Internal update thread. Keeps the metagraph up to date. """
        try:
            while self._running:
                self.do_gossip()
                if len(self._peers) > 0:
                    self.do_clean(self._ttl)
        
        except (KeyboardInterrupt, SystemExit) as e:
            logger.info('stop metagraph')
            self._running = False
            self.stop()
            raise e

    def _serve(self):
        try:
            self._server.start()
        except (KeyboardInterrupt, SystemExit) as ex:
            self.stop()
            raise ex
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
        self._update_thread = threading.Thread(target=self._update, daemon=True)
        self._server_thread = threading.Thread(target=self._serve, daemon=True)
        self._update_thread.start()
        self._server_thread.start()

    def getweights(self, synapses: List[bittensor_pb2.Synapse]) -> torch.Tensor:
        """Get the weights for list of Synapse endpoints.

        Args:
            synapses (List[bittensor_pb2.Synapse]): Synapses to get weights for.

        Returns:
            [type]: Weights set for each synapse.
        """
        return torch.Tensor(self._getweights(synapses))

    def _getweights(self, synapses: List[bittensor_pb2.Synapse]) -> List[float]:
        """ Get local weights for a list of synapses """
        result = []
        for syn in synapses:
            if syn.synapse_key not in self._weights:
                result.append(0.0)
            else:
                result.append(self._weights[syn.synapse_key])
        return result

    def setweights(self, synapses: List[bittensor_pb2.Synapse],
                   weights: torch.Tensor):
        """Sets the weights for these synapses given equal length list of weights.

        Args:
            synapses (List[bittensor_pb2.Synapse]): Synapses to set weights.
            weights (torch.Tensor): Weights to set.
        """
        weights = weights.cpu().detach().numpy().tolist()
        self._setweights(synapses, weights)

    def _setweights(self, synapses: List[bittensor_pb2.Synapse],
                    weights: List[float]):
        """ Set local scores for each passed node """
        for idx, synapse in enumerate(synapses):
            self._weights[synapse.synapse_key] = weights[idx]

    def subscribe(self, synapse):
        """Subscribes a synapse class object to the metagraph.

        Args:
            module (bittensor.Synapse): bittensor.Synapse class object to subscribe.
        """
        # Create a new bittensor_pb2.Synapse proto.
        synapse_proto = bittensor_pb2.Synapse(
            version=bittensor.__version__,
            neuron_key=self._config.neuron_key,
            synapse_key=synapse.synapse_key(),
            address=self._config.remote_ip,
            port=self._config.axon_port,
        )
        self._subscribe(synapse_proto)

    def _subscribe(self, synapse_proto: bittensor_pb2.Synapse):
        self._synapses[synapse_proto.synapse_key] = synapse_proto
        self._weights[synapse_proto.synapse_key] = math.inf
        self._heartbeat[synapse_proto.synapse_key] = time.time()

    @property
    def weights(self):
        return self._weights
