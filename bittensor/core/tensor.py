import base64
from typing import Optional, Union

import msgpack
import msgpack_numpy
import numpy as np
from pydantic import ConfigDict, BaseModel, Field, field_validator

from bittensor.utils.registration import torch, use_torch


class DTypes(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.torch: bool = False
        self.update(
            {
                "float16": np.float16,
                "float32": np.float32,
                "float64": np.float64,
                "uint8": np.uint8,
                "int16": np.int16,
                "int8": np.int8,
                "int32": np.int32,
                "int64": np.int64,
                "bool": bool,
            }
        )

    def __getitem__(self, key):
        self._add_torch()
        return super().__getitem__(key)

    def __contains__(self, key):
        self._add_torch()
        return super().__contains__(key)

    def _add_torch(self):
        if self.torch is False:
            torch_dtypes = {
                "torch.float16": torch.float16,
                "torch.float32": torch.float32,
                "torch.float64": torch.float64,
                "torch.uint8": torch.uint8,
                "torch.int16": torch.int16,
                "torch.int8": torch.int8,
                "torch.int32": torch.int32,
                "torch.int64": torch.int64,
                "torch.bool": torch.bool,
            }
            self.update(torch_dtypes)
            self.torch = True


dtypes = DTypes()


def cast_dtype(raw: Union[None, np.dtype, "torch.dtype", str]) -> Optional[str]:
    """
    Casts the raw value to a string representing the `numpy data type <https://numpy.org/doc/stable/user/basics.types.html>`_, or the `torch data type <https://pytorch.org/docs/stable/tensor_attributes.html>`_ if using torch.

    Parameters:
        raw: The raw value to cast.

    Returns:
        The string representing the numpy/torch data type.

    Raises:
        Exception: If the raw value is of an invalid type.
    """
    if not raw:
        return None
    if use_torch() and isinstance(raw, torch.dtype):
        return dtypes[raw]
    elif isinstance(raw, np.dtype):
        return dtypes[raw]
    elif isinstance(raw, str):
        if use_torch():
            assert raw in dtypes, f"{raw} not a valid torch type in dict {dtypes}"
            return raw
        else:
            assert raw in dtypes, f"{raw} not a valid numpy type in dict {dtypes}"
            return raw
    else:
        raise Exception(
            f"{raw} of type {type(raw)} does not have a valid type in Union[None, numpy.dtype, torch.dtype, str]"
        )


def cast_shape(raw: Union[None, list[int], str]) -> Optional[Union[str, list]]:
    """
    Casts the raw value to a string representing the tensor shape.

    Parameters:
        raw: The raw value to cast.

    Returns:
        The string representing the tensor shape.

    Raises:
        Exception: If the raw value is of an invalid type or if the list elements are not of type int.
    """
    if not raw:
        return None
    elif isinstance(raw, list):
        if len(raw) == 0 or isinstance(raw[0], int):
            return raw
        else:
            raise Exception(f"{raw} list elements are not of type int")
    elif isinstance(raw, str):
        shape = list(map(int, raw.split("[")[1].split("]")[0].split(",")))
        return shape
    else:
        raise Exception(
            f"{raw} of type {type(raw)} does not have a valid type in Union[None, list[int], str]"
        )


class tensor:
    def __new__(cls, tensor: Union[list, "np.ndarray", "torch.Tensor"]):
        if isinstance(tensor, list) or isinstance(tensor, np.ndarray):
            tensor = torch.tensor(tensor) if use_torch() else np.array(tensor)
        return Tensor.serialize(tensor_=tensor)


class Tensor(BaseModel):
    """
    Represents a Tensor object.

    Parameters:
        buffer: Tensor buffer data.
        dtype: Tensor data type.
        shape: Tensor shape.
    """

    model_config = ConfigDict(validate_assignment=True)

    def tensor(self) -> Union[np.ndarray, "torch.Tensor"]:
        return self.deserialize()

    def tolist(self) -> list[object]:
        return self.deserialize().tolist()

    def numpy(self) -> "np.ndarray":
        return (
            self.deserialize().detach().numpy() if use_torch() else self.deserialize()
        )

    def deserialize(self) -> Union["np.ndarray", "torch.Tensor"]:
        """
        Deserializes the Tensor object.

        Returns:
            np.array or torch.Tensor: The deserialized tensor object.

        Raises:
            Exception: If the deserialization process encounters an error.
        """
        shape = tuple(self.shape)
        buffer_bytes = base64.b64decode(self.buffer.encode("utf-8"))
        numpy_object = msgpack.unpackb(
            buffer_bytes, object_hook=msgpack_numpy.decode
        ).copy()
        if use_torch():
            torch_object = torch.as_tensor(numpy_object)
            # Reshape does not work for (0) or [0]
            if not (len(shape) == 1 and shape[0] == 0):
                torch_object = torch_object.reshape(shape)
            return torch_object.type(dtypes[self.dtype])
        else:
            # Reshape does not work for (0) or [0]
            if not (len(shape) == 1 and shape[0] == 0):
                numpy_object = numpy_object.reshape(shape)
            return numpy_object.astype(dtypes[self.dtype])

    @staticmethod
    def serialize(tensor_: Union["np.ndarray", "torch.Tensor"]) -> "Tensor":
        """
        Serializes the given tensor.

        Parameters:
            tensor_: The tensor to serialize.

        Returns:
            The serialized tensor.

        Raises:
            Exception: If the serialization process encounters an error.
        """
        dtype = str(tensor_.dtype)
        shape = list(tensor_.shape)
        if len(shape) == 0:
            shape = [0]
        tensor__ = tensor_.cpu().detach().numpy().copy() if use_torch() else tensor_
        data_buffer = base64.b64encode(
            msgpack.packb(tensor__, default=msgpack_numpy.encode)
        ).decode("utf-8")
        return Tensor(buffer=data_buffer, shape=shape, dtype=dtype)

    # Represents the tensor buffer data.
    buffer: Optional[str] = Field(
        default=None,
        title="buffer",
        description="Tensor buffer data. This field stores the serialized representation of the tensor data.",
        examples=["0x321e13edqwds231231231232131"],
        frozen=True,
        repr=False,
    )

    # Represents the data type of the tensor.
    dtype: str = Field(
        title="dtype",
        description="Tensor data type. This field specifies the data type of the tensor, such as numpy.float32 or torch.int64.",
        examples=["np.float32"],
        frozen=True,
        repr=True,
    )

    # Represents the shape of the tensor.
    shape: list[int] = Field(
        title="shape",
        description="Tensor shape. This field defines the dimensions of the tensor as a list of integers, such as [10, 10] for a 2D tensor with shape (10, 10).",
        examples=[10, 10],
        frozen=True,
        repr=True,
    )

    # Extract the represented shape of the tensor.
    _extract_shape = field_validator("shape", mode="before")(cast_shape)

    # Extract the represented data type of the tensor.
    _extract_dtype = field_validator("dtype", mode="before")(cast_dtype)
