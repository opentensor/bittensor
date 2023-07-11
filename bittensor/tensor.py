
import torch
import base64
import msgpack
import pydantic
import msgpack_numpy
from typing import Dict, Optional, Tuple, Union, List, Callable

TORCH_DTYPES = {
    'torch.float32': torch.float32,
    'torch.float64': torch.float64,
    'torch.int64': torch.int64,
}

class Tensor( pydantic.BaseModel ):

    class Config:
        validate_assignment = True

    def deserialize( self ) -> torch.Tensor:
        shape = tuple(self.shape)
        buffer_bytes = base64.b64decode( self.buffer.encode('utf-8') )
        numpy_object = msgpack.unpackb( buffer_bytes, object_hook = msgpack_numpy.decode ).copy()
        torch_object = torch.as_tensor( numpy_object ).view( shape )
        return torch_object.type( TORCH_DTYPES[self.dtype] )

    def serialize( tensor: torch.Tensor ) -> 'Tensor':
        dtype = str( tensor.dtype )
        shape = list( tensor.shape )
        torch_numpy = tensor.cpu().detach().numpy().copy()
        data_buffer = base64.b64encode(msgpack.packb( torch_numpy, default = msgpack_numpy.encode )).decode('utf-8')
        return Tensor(
            buffer = data_buffer,
            shape = shape,
            dtype = dtype,
        )

    # Placeholder for tensor data.
    buffer: str = pydantic.Field(
        title = 'buffer',
        description = 'Tensor buffer data',
        examples = '0x321e13edqwds231231231232131',
        allow_mutation = False,
        repr = False
    )

    # Placeholder for tensor dtype.
    dtype: str = pydantic.Field(
        title = 'dtype',
        description = 'Tensor data type',
        examples = 'torch.float32',
        allow_mutation = False,
        repr = True
    )

    # Defines the shape of the tensor.
    shape: List[int] = pydantic.Field(
        title = 'shape',
        description = 'Tensor shape.',
        examples = '[10,10]',
        allow_mutation = False,
        repr = True
    )