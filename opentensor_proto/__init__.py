from opentensor_proto import opentensor_pb2 as proto_pb2

import pickle
from io import BytesIO

import joblib
import torch
import umsgpack

import torch

class SerializerBase:
    @staticmethod
    def dumps(obj: object) -> bytes:
        raise NotImplementedError()

    @staticmethod
    def loads(buf: bytes) -> object:
        raise NotImplementedError()

class PytorchSerializer(SerializerBase):

    @staticmethod
    def serialize(obj: torch.Tensor) -> proto_pb2.Tensor:
        data_buffer = PytorchSerializer.dumps(obj)
        tensor = proto_pb2.Tensor(buffer=data_buffer)
        return tensor
    
    @staticmethod
    def deserialize(proto: proto_pb2.Tensor) -> torch.Tensor:    
        torch_tensor = PytorchSerializer.loads(proto.buffer) 
        return torch_tensor

    @staticmethod
    def dumps(obj: object) -> bytes:
        s = BytesIO()
        torch.save(obj, s, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        return s.getvalue()

    @staticmethod
    def loads(buf: bytes) -> object:
        return torch.load(BytesIO(buf))

