
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

def cast_dtype( raw: Union[None, torch.dtype, str ]) -> str:
    if not raw:
        return None
    if isinstance(raw, torch.dtype ):
        return TORCH_DTYPES[raw]
    elif isinstance( raw, str ):
        assert raw in TORCH_DTYPES, f"{str} not a valid torch type in dict {TORCH_DTYPES}"
        return raw
    else:
        raise Exception( f"{raw} of type {type(raw)} does not have valid type in Union[None, torch.dtype, str ]")
    

def cast_shape( raw: Union[None, List[int], str ]) -> str:
    if not raw:
        return None
    elif isinstance( raw, list ):
        if len( raw ) == 0:
            return raw
        elif isinstance( raw[0], int ):
            return raw
        else:
            raise Exception( f"{raw} list elements are not of type int")
    elif isinstance( raw, str ):
        shape = list(map(int,raw.split('[')[1].split(']')[0].split(',')))
        return shape
    else:
        raise Exception( f"{raw} of type {type(raw)} does not have valid type in Union[None, List[int], str ]")


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
    buffer: Optional[str] = pydantic.Field(
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
    _extract_dtype = pydantic.validator('dtype', pre=True, allow_reuse=True)(cast_dtype)

    # Defines the shape of the tensor.
    shape: List[int] = pydantic.Field(
        title = 'shape',
        description = 'Tensor shape.',
        examples = '[10,10]',
        allow_mutation = False,
        repr = True
    )
    _extract_shape = pydantic.validator('shape', pre=True, allow_reuse=True)(cast_shape)
