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
import random
import sys
import threading
import torch
import numpy as np

class Metagraph(nn.Module):

    def __init__(self, identity: opentensor.Identity, start=True):
        super().__init__()    
   
        # Network identity key object.
        self.identity = identity
       
        # Inward connection handler.
        self._axon_address = 'localhost'
        self._axon_port = str(random.randint(8000, 30000))
        self._axon = opentensor.Axon(self)
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        opentensor_grpc.add_OpentensorServicer_to_server(self._axon, self._server)
        self._server.add_insecure_port('[::]:' + self._axon_port)

        # Network proxy stub.
        self._proxy_address = 'localhost:8899'
        channel = grpc.insecure_channel(self._proxy_address)
        self.stub = opentensor_grpc.MetagraphProxyStub(channel)

        # Dendrite: outward connection handler.
        self._dendrite = opentensor.Dendrite(self)

        # Internal state.
        self._nodes = {}
        self._local_nodes = {}
        self._local_weights = {}
        self._local_node_protos = {}

        # Build grpc server.
        self._thread = None

        if start:
            self.start()
        
    
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

    def setWeight(self, target: opentensor_pb2.Node, value):
        weight = opentensor_pb2.Weight (
            source = self.identity.public_key(),
            target = target.public_key,
            value = value
        )
        self._local_weights[target.public_key] = weight

    def getWeight(self, target: opentensor_pb2.Node):
        if target.public_key in self._local_weights:
            return self._local_weights[target.public_key].value
        return 0.0
        
    def subscribe(self, Node: opentensor.Node):
        node_identity = opentensor.Identity().public_key()
        assert (node_identity not in self._nodes)
        node_proto = opentensor_pb2.Node(
            version = 1.0, 
            public_key = self.identity.public_key(),
            identity = node_identity,
            address = self._axon_address,
            port = self._axon_port    
        )
        self._nodes[node_identity] = node_proto
        self._local_node_protos[node_identity] = node_proto
        self._local_nodes[node_identity] = Node
        self.refresh()
    
    def refresh(self):
        # Send graph.
        neuron = opentensor_pb2.Neuron (
                public_key = self.identity.public_key(),
                nodes = self._local_node_protos.values(),
                weights = self._local_weights.values(),
        ) 
        try:
            self.stub.Subscribe(neuron)
        except:
            return
        # Pull graph.
        request = opentensor_pb2.ACK()
        state = self.stub.GetMetagraph(request)
        for neuron in state.neurons:
            for node in neuron.nodes:
                self._nodes[node.identity] = node
    
    
