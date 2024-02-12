# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import base64
import json
from typing import Dict, Optional, Tuple, Union, List, Callable

import msgpack
import msgpack_numpy
import numpy as np
from pydantic import BaseModel, Field
from pydantic.functional_validators import model_validator,field_validator
import torch


TORCH_DTYPES = {
    torch.float16: "torch.float16",
    torch.float32: "torch.float32",
    torch.float64: "torch.float64",
    torch.uint8: "torch.uint8",
    torch.int16: "torch.int16",
    torch.int8: "torch.int8",
    torch.int32: "torch.int32",
    torch.int64: "torch.int64",
    torch.bool: "torch.bool",
    torch.complex32: "torch.complex32",
    torch.complex64: "torch.complex64",
    torch.complex128: "torch.complex128",
}


def cast_dtype(raw: Union[None, torch.dtype, str]) -> Union[str, None]:
    """
    Casts the raw value to a string representing the torch data type.
    """
    if raw is None:
        return None
    if isinstance(raw, torch.dtype):
        return TORCH_DTYPES.get(raw, str(raw))
    if isinstance(raw, str):
        if raw in TORCH_DTYPES.values():
            return raw
        raise ValueError(f"{raw} is not a valid torch type in {TORCH_DTYPES.values()}")
    raise TypeError(
        f"{raw} of type {type(raw)} is not a valid type in Union[None, torch.dtype, str]"
    )


def cast_shape(raw: Union[None, List[int], str]) -> str:
    """
    Casts the raw value to a string representing the tensor shape.
    """
    if raw is None:
        return "None"
    elif isinstance(raw, list):
        if all(isinstance(item, int) for item in raw):
            return str(raw)
        raise ValueError(f"{raw} list elements are not all of type int")
    elif isinstance(raw, str):
        return raw
    raise TypeError(
        f"{raw} of type {type(raw)} is not a valid type in Union[None, List[int], str]"
    )


class Tensor(BaseModel):
    """
    A model representing a tensor with support for scalars, vectors, matrices, and higher-dimensional data.

    Attributes:
        scalar (Optional[int]): Scalar value of the tensor, if applicable.
        vector (Optional[List[int]]): Vector representation of the tensor, if applicable.
        matrix (Optional[List[List[Union[int, float]]]]): Matrix representation of the tensor, if applicable.
        tensor (Optional[np.ndarray]): Higher-dimensional tensor data, if applicable.
        buffer (Optional[str]): Serialized tensor data for storage or transmission.
        dtype (str): Data type of the tensor elements.
        shape (List[int]): Dimensions of the tensor.
    """
    scalar: Optional[int] = Field(
        default=None,
        title="Scalar",
        description="Scalar value of the tensor.",
        examples=[1],
        frozen=True,
        repr=True,
    )
    vector: Optional[List[int]] = Field(
        default=None,
        title="Vector",
        description="Vector representation of the tensor.",
        examples=[[1, 2, 3, 4, 5]],
        frozen=True,
        repr=True,
    )
    matrix: Optional[List[List[Union[int, float]]]] = Field(
        default=None,
        title="Matrix",
        description="Matrix representation of the tensor.",
        examples=[[[1, 2], [3, 4], [5, 6]]],
        frozen=True,
        repr=True,
    )
    tensor: Optional[np.ndarray] = Field(
        default=None,
        title="Tensor",
        description="Higher-dimensional tensor data.",
        examples=[np.array([[[1, 2, 3], [4, 5, 6]]])],
        frozen=True,
        repr=True,
    )
    buffer: Optional[str] = Field(
        default=None,
        title="Buffer",
        description="Serialized tensor data for storage or transmission.",
        examples=["0x321e13edqwds231231231232131"],
        frozen=True,
        repr=False,
    )
    dtype: str = Field(
        default="float32",
        title="Data Type",
        description="Data type of the tensor elements.",
        examples=["float32"],
        frozen=True,
        repr=True,
    )
    shape: List[int] = Field(
        default_factory=list,
        title="Shape",
        description="Dimensions of the tensor.",
        examples=[10, 10],
        frozen=True,
        repr=True,
    )

    @field_validator("tensor")
    def validate_tensor_shape(cls, v):
        if not isinstance(v, np.ndarray):
            raise TypeError("Tensor must be a numpy array")
        return v

    @field_validator("matrix")
    def must_be_rectangular(cls, v):
        if not all(len(row) == len(v[0]) for row in v):
            raise ValueError("Matrix must be rectangular")
        return v

    @field_validator("dtype")
    def validate_dtype(self, value):
        return cast_dtype(value)

    @field_validator("shape")
    def validate_shape(self, value):
        return self.cast_shape(value)

    def totensor(self) -> torch.Tensor:
        return self.deserialize()

    def tolist(self) -> List[object]:
        return self.deserialize().tolist()

    def numpy(self) -> "np.ndarray":
        return self.deserialize().detach().numpy()

    def cast_shape(
        self, raw: Union[None, int, List[int], List[List[int]], str]
    ) -> None:
        """
        Detects the type of tensor shape (scalar, vector, matrix, etc.), validates it,
        and saves it to the corresponding attribute.

        Args:
            raw (Union[None, int, List[int], List[List[int]], str]): The raw value to cast.

        Raises:
            ValueError: If the raw value is of an invalid type or if list elements are not of the correct type.
        """
        if raw is None:
            return
        elif isinstance(raw, int):
            self.scalar = raw
        elif isinstance(raw, list):
            if all(isinstance(item, int) for item in raw):
                self.vector = raw
            elif all(
                isinstance(row, list) and all(isinstance(item, int) for item in row)
                for row in raw
            ):
                self.matrix = raw
            else:
                raise ValueError(
                    "Invalid tensor shape: elements are not of the correct type"
                )
        elif isinstance(raw, str):
            try:
                # Transforming the string into a JSON-like format
                json_like_str = raw.replace("(", "[").replace(")", "]").replace(" ", "")
                parsed = json.loads(json_like_str)
                self.cast_shape(parsed)
            except (json.JSONDecodeError, ValueError):
                raise ValueError("Invalid string representation of tensor shape")
        else:
            raise ValueError(f"Invalid type for tensor shape: {type(raw)}")

    @staticmethod
    def serialize(tensor: "torch.Tensor") -> "Tensor":
        """
        Serializes the given tensor.

        When you serialize tensor data using msgpack (and possibly compress or encode it with base64), you're
        serializing the tensor's data content. To preserve the shape (and potentially the data type)
        of the tensor, you typically need to explicitly include the shape (and dtype) as part of the serialized data.
        This means you serialize a structure containing both the tensor data and its shape (and dtype), and during
        deserialization, you reconstruct the tensor using this information. If the shape is not explicitly handled,
        deserializing the raw data back into a tensor would not automatically restore the original shape.

        Args:
            tensor (torch.Tensor): The tensor to serialize.

        Returns:
            Tensor: The serialized tensor.

        Raises:
            Exception: If the serialization process encounters an error.
        """
        dtype = str(tensor.dtype)
        shape = list(tensor.shape)
        if len(shape) == 0:
            shape = [0]
        torch_numpy = tensor.cpu().detach().numpy().copy()
        data_buffer = base64.b64encode(
            msgpack.packb(torch_numpy, default=msgpack_numpy.encode)
        ).decode("utf-8")
        return Tensor(buffer=data_buffer, shape=shape, dtype=dtype)

    def deserialize(self) -> torch.Tensor:
        """
        Deserializes the Tensor object.

        Returns:
            torch.Tensor: The deserialized tensor object.

        Raises:
            ValueError: If the deserialization process encounters an error.
        """
        # Decode the buffer
        buffer_bytes = base64.b64decode(self.buffer.encode("utf-8"))
        numpy_object = msgpack.unpackb(
            buffer_bytes, object_hook=msgpack_numpy.decode
        ).copy()
        torch_object = torch.as_tensor(numpy_object)

        # Handle the shape
        if self.scalar is not None:
            # If it's a scalar, no need to reshape
            pass
        elif self.vector is not None:
            torch_object = torch_object.reshape(self.vector)
        elif self.matrix is not None:
            torch_object = torch_object.reshape(self.matrix)
        # TODO: higher dimensions
        return torch_object.type(TORCH_DTYPES[self.dtype])

    class Config:
        """
        Configuration class for the ModelConfig Pydantic model.

        Attributes:
            validate_assignment (bool): Enables validation of attribute assignments.

        Note:
            In Pydantic v2, the 'ConfigDict' used in earlier versions is deprecated.
            This 'Config' inner class is used instead to configure the behavior of the model.
        """

        validate_assignment = True


class TensorFactory:
    """
    Factory class for creating Tensor objects from various input types.
    """

    @staticmethod
    def create(tensor: Union[list, np.ndarray, torch.Tensor]) -> Tensor:
        """
        Creates a Tensor object from a given input which can be a list, numpy array, or a torch tensor.

        Args:
            tensor (Union[list, numpy.ndarray, torch.Tensor]): The input tensor data.

        Returns:
            Tensor: The serialized Tensor object.
        """
        if isinstance(tensor, (list, np.ndarray)):
            tensor = torch.tensor(tensor)
        return Tensor.serialize(tensor=tensor)
