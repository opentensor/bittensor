from opentensor import opentensor_pb2_grpc as opentensor_grpc
import opentensor


class Storage():
    def __init__(self, max_size):
        self._max_size = max_size
        pass

    def recv_gossip(self, graph: opentensor_pb2.Metagraph):
        pass

    def add(self, nodes: List[opentensor_pb2.Node]):
        pass

    def get(self) -> List[opentensor_pb2.Node]:
        pass

    def weights(self, nodes: List[opentensor_pb2.Node], weights: List[float]):
        pass
