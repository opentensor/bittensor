import random
import time
import threading

from opentensor import opentensor_pb2


class Gossip():
    def __init__(self, metagraph: opentensor.Metagraph, heartbeat: int):
        self._metagraph = metagraph
        self._heartbeat = heartbeat
        self._thread = None
        self._running = False

    def __del__(self):
        self.stop()

    def start(self):
        """ Starts the gossip thread """
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """ Stops the gossip thread """
        self._running = False
        if self._thread:
            self._thread.join()

    def _run(self):
        """ Main loop """
        while self._running:
            self._do_gossip()
            time.sleep(self._heartbeat)

    def _do_gossip(self):
        """ Build gossip query for random node and initiates gossip """
        nodes = self._metagraph.nodes()
        neuron = opentensor_pb2.Neuron(public_key=self._metagraph.public_key(),
                                       nodes=nodes)
        subgraph = opentensor_pb2.Metagraph(neurons=[neuron])
        node_choice = random.choice(nodes)
        self._metagraph.send_gossip(node_choice, subgraph)
