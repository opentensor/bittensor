from opentensor import opentensor_pb2
import opentensor

from typing import List, Tuple

import math

# TODO (const) should be inheritable.
class Storage():
    def __init__(self, max_size):
        self._max_size = max_size
        # TODO(const) should be implemented using sortedmap
        self._nodes = {} # map from node.identity to score and node.
        self._weights = {} # map from [node.identity] to weight. 
        # TODO(const) needs check for overflow and then sink to disk
        self._N = 0

    def getweights(self, nodes: List[opentensor_pb2.Node]) -> List[float]:
        """ Get local weights for a list of nodes. """
        # TODO(const) cache overflow.
        weights = []
        for n in nodes:
            if n.identity not in self._weights:
                weights.append(0.0)
            else:
                weights.append(self._weights[n.identity])
        return weights

    def setweights(self, nodes: List[opentensor_pb2.Node], weights: List[float]):
        """ Set weights for nodes """
        for n, w in list(zip(nodes, weights)):
            self._weights[n.identity] = w

    def addlocal(self, node: opentensor_pb2.Node):
        """ Adds a single node to the graph """
        self._nodes[node.identity] = (math.inf, node)

    def add(self, graph: opentensor_pb2.Metagraph):
        """ Updates storage with gossiped metagraph. """
        # TODO(const) score based on POW, timestamp, and trust.
        # TODO(const) check signatures.
        # TODO(const) sink weights as well.
        # TODO(const) write to disk if need be and replace heap cache.
        # TODO(const) check size contraints.
        for neuron in graph:
            for node in neuron.nodes:
                if node.identity not in self._nodes:
                    score = 0.0
                    self._N += 1
                else:
                    score = self._nodes[node.identity]
                self._nodes[node.identity] = (score, node)

    def get(self, n: long) -> List[Tuple(float, opentensor_pb2.Node)]:
        """ Returns min(n, len(storage)) nodes from storage sorted by score."""
        # TODO(const) needs get without score function
        # TODO(const) needs get from disk if exists on dist
        # TODO(const) shoudl be sortedmap i.e. RB tree.
        return sorted(list(self._nodes.values()[:min(n, self._N)))

    def getweights(self, nodes: List[opentensor_pb2.Node]) -> List[float]:
        weights = []
        for n in nodes:
            if n.identity not in self._weights[self.identity.public_key()]:
                weights.append(0.0)
            else:
                weights.append(self._weights[self.identity.public_key()][
                    n.identity].value)
 

                                                return pass

    def score(self, nodes: List[opentensor_pb2.Node], score: List[float]):
        """ Set local scores for each passed node """
        for idx, node in enumerate(nodes):
            if node.identity in self._nodes:
                self._nodes[node.identity] = (score[idx], node)



        pass
