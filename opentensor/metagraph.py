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
import binascii
import grpc
import hashlib
import sys
import threading
import torch
import numpy as np

class Metagraph(nn.Module):

    def __init__(self, identity, address, port, proxy='localhost:8899', start = False):
        super().__init__()    
    
        self.identity = identity
        self.address = address
        self.port = port
        self.metagraph_address = proxy
        
        # Get channel to metagraph proxy.
        channel = grpc.insecure_channel(self.metagraph_address)
        self.stub = opentensor_grpc.MetagraphProxyStub(channel)
   
        # State maps.
        self._nodes = {}
        self._local_nodes = {}
        self._local_node_protos = {}

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

    def subscribe(self, Node: opentensor.Node):
        node_identity = opentensor.Identity().public_key()
        assert (node_identity not in self._nodes)
        node_proto = opentensor_pb2.Node(
            version = 1.0, 
            public_key = self.identity.public_key(),
            identity = node_identity,
            address = self.address,
            port = self.port,
        )
        self._nodes[node_identity] = node_proto
        self._local_node_protos[node_identity] = node_proto
        self._local_nodes[node_identity] = Node
        self._proxy_subscribe(node_proto)

    def _proxy_subscribe(self, node_proto: opentensor_pb2.Node):
        response = self.stub.Subscribe(node_proto)

    def _update(self, state):
        for node in state.nodes:
            self._nodes[node.identity] = node

    def refresh(self):
        for node_proto in self._local_node_protos.values():
            self._proxy_subscribe(node_proto)
   
        request = opentensor_pb2.ACK()
        state = self.stub.GetMetagraph(request)
        self._update(state) 
    
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

