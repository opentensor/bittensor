from opentensor import opentensor_pb2 as proto_pb2

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opentensor.node import Node
from opentensor.keys import Keys
from opentensor.identity import Identity
from opentensor.axon import Axon
from opentensor.dendrite import Dendrite
from opentensor.metagraph import Metagraph
from opentensor.gate import Gate
from opentensor.dispatcher import Dispatcher

from io import BytesIO
import joblib
import torch
import pickle
import umsgpack

class SerializerBase:
    @staticmethod
    def dumps(obj: object) -> bytes:
        raise NotImplementedError()

    @staticmethod
    def loads(buf: bytes) -> object:
        raise NotImplementedError()

class Serializer(SerializerBase):
    
    @staticmethod
    def serialize(obj: torch.Tensor) -> proto_pb2.Tensor:
        data_buffer = Serializer.dumps(obj)
        tensor = proto_pb2.Tensor(buffer=data_buffer)
        return tensor
    
    @staticmethod
    def deserialize(proto: proto_pb2.Tensor) -> torch.Tensor:    
        torch_tensor = Serializer.loads(proto.buffer) 
        return torch_tensor
    
    @staticmethod
    def dumps(obj: object) -> bytes:
        s = BytesIO()
        torch.save(obj, s, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        return s.getvalue()
    
    @staticmethod
    def loads(buf: bytes) -> object:
        return torch.load(BytesIO(buf))
