from opentensor import opentensor_pb2 as proto_pb2
from io import BytesIO
import torch
import pickle


class SerializerBase:
    @staticmethod
    def todef(obj: object) -> proto_pb2.TensorDef:
        raise NotImplementedError()

    @staticmethod
    def serialize(obj: object) -> proto_pb2.Tensor:
        raise NotImplementedError()

    @staticmethod
    def deserialize(proto: proto_pb2.Tensor) -> object:
        raise NotImplementedError()

    @staticmethod
    def dumps(obj: object) -> bytes:
        raise NotImplementedError()

    @staticmethod
    def loads(buf: bytes) -> object:
        raise NotImplementedError()


class Serializer(SerializerBase):
    @staticmethod
    def zeros_for_def(in_tensor: torch.Tensor,
                      tensor_def: proto_pb2.TensorDef) -> torch.Tensor:
        batch_dim = in_tensor.shape[0]
        out_shape = tensor_def.shape[1:]
        shape = tuple([batch_dim] + out_shape)
        if tensor_def.dtype == proto_pb2.DataType.FLOAT32:
            return torch.zeros(shape, dtype=torch.float32)

        elif tensor_def.dtype == proto_pb2.DataType.FLOAT64:
            return torch.zeros(shape, dtype=torch.float64)

        elif tensor_def.dtype == proto_pb2.DataType.INT32:
            return torch.zeros(shape, dtype=torch.int32)

        elif tensor_def.dtype == proto_pb2.DataType.INT64:
            return torch.zeros(shape, dtype=torch.int64)
        else:
            raise ValueError

    @staticmethod
    def todef(obj: torch.Tensor) -> proto_pb2.TensorDef:
        if obj.dtype == torch.float32:
            dtype = proto_pb2.DataType.FLOAT32

        elif obj.dtype == torch.float64:
            dtype = proto_pb2.DataType.FLOAT64

        elif obj.dtype == torch.int32:
            dtype = proto_pb2.DataType.INT32

        elif obj.dtype == torch.int64:
            dtype = proto_pb2.DataType.INT64

        else:
            dtype = proto_pb2.DataType.UNKNOWN

        return proto_pb2.TensorDef(shape=list(obj.shape), dtype=dtype)

    @staticmethod
    def serialize(obj: torch.Tensor) -> proto_pb2.Tensor:
        data_buffer = Serializer.dumps(obj)
        shape = list(obj.shape)
        tensor = proto_pb2.Tensor(buffer=data_buffer,
                                  tensor_def=Serializer.todef(obj))
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
