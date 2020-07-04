from opentensor import opentensor_pb2_grpc as opentensor_grpc
from opentensor import opentensor_pb2
import opentensor

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization

from concurrent import futures
from torch import nn
from typing import List
from loguru import logger

import os
import ast
import grpc
import hashlib
import sys
import threading
import torch
import numpy as np

def new_node_id():
    return os.urandom(12)

class Metagraph(nn.Module):

    def __init__(self, public_key, address, port, metagraph_address, start = False):
        super().__init__()    
    
        self.public_key = public_key
        self.address = address
        self.port = port
        self.metagraph_address = metagraph_address
        
        # Get channel to metagraph proxy.
        channel = grpc.insecure_channel(self.metagraph_address)
        self.stub = opentensor_grpc.MetagraphProxyStub(channel)
   
        # State maps.
        self._nodes = {}
        self._local_nodes = {}

        # Dendrite
        self._dendrite = opentensor.Dendrite(self)

        # Build grpc server.
        self._servicer = opentensor.Synapse(self)
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        opentensor_grpc.add_OpentensorServicer_to_server(self._servicer, self._server)
        serve_address = self.address + ":" + self.port
        self._server.add_insecure_port(serve_address)
        
        self._thread = None
        if start:
            self.start()

    def nodes(self) -> List[opentensor_pb2.Node]:
        return list(self._nodes.values())

    def Fwd(self, source_id, target_id, tensor): 
        assert (target_id in self._local_nodes)
        node = self._local_nodes[target_id]
        tensor = node.fwd(source_id, tensor)
        return tensor
    
    def Bwd(self, request, context):
        pass

    def forward(self, x: List[torch.Tensor], nodes: List[opentensor_pb2.Node]):
        return self._dendrite.forward(x, nodes) 

    def subscribe(self, Node):
        node_id = new_node_id()
        assert (node_id not in self._nodes)
        node_proto = opentensor_pb2.Node(
            version = 1.0, 
            public_key = self.public_key,
            identity = node_id,
            address = self.address,
            port = self.port,
        )
        self._nodes[node_id] = node_proto
        self._local_nodes[node_id] = Node
        self._subscribe(node_proto)

    def _subscribe(self, node_proto):
        try:
            response = self.stub.Subscribe(node_proto)
        except:
            pass

    def updates(self, metagraph):
        for node_proto in metagraph.nodes:
            if node_proto.node_id in self._nodes_for_node_i:
                continue
            self._nodes_for_node_id[node_proto.node_id] = node_proto

    def refresh(self):
        try:
            request = proto_pb2.MetagraphRequest(version=1.0)
            metagraph = self.stub.GetMetagraph(request)
            self.update(metagraph) 
        except:
            pass
    
    def __del__(self):
        self.stop()

    def start(self):
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def stop(self):
        self._server.stop(0)

    def _serve(self):
        print ('serving metagraph ...')
        self._server.start()

