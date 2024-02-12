import pytest
from typing import Union, List
import torch
from bittensor.tensor import cast_dtype, cast_shape, TORCH_DTYPES, Tensor


# Unit tests for cast_dtype
def test_cast_dtype_none():
    assert cast_dtype(None) is None


def test_cast_dtype_torch_dtype():
    assert cast_dtype(torch.float32) == "float32"


def test_cast_dtype_str_valid():
    assert cast_dtype("float32") == "float32"


def test_cast_dtype_str_invalid():
    with pytest.raises(ValueError):
        cast_dtype("invalid_dtype")


def test_cast_dtype_type_error():
    with pytest.raises(TypeError):
        cast_dtype(123)


# Unit tests for cast_shape
def test_cast_shape_none():
    assert cast_shape(None) == "None"


def test_cast_shape_list_int():
    assert cast_shape([1, 2, 3]) == "[1, 2, 3]"


def test_cast_shape_str():
    assert cast_shape("1, 2, 3") == "1, 2, 3"


def test_cast_shape_list_non_int():
    with pytest.raises(ValueError):
        cast_shape([1, "two", 3])


def test_cast_shape_type_error():
    with pytest.raises(TypeError):
        cast_shape(123)


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (torch.float16, "torch.float16"),
        (torch.float32, "torch.float32"),
        (torch.float64, "torch.float64"),
        (torch.uint8, "torch.uint8"),
        (torch.int16, "torch.int16"),
        (torch.int8, "torch.int8"),
        (torch.int32, "torch.int32"),
        (torch.int64, "torch.int64"),
        (torch.bool, "torch.bool"),
        (torch.complex32, "torch.complex32"),
        (torch.complex64, "torch.complex64"),
        (torch.complex128, "torch.complex128"),
    ],
)
def test_cast_dtype_with_torch_dtype(dtype, expected):
    assert cast_dtype(dtype) == expected


@pytest.mark.parametrize(
    "dtype_str, expected",
    [
        ("torch.float16", "torch.float16"),
        ("torch.float32", "torch.float32"),
        ("torch.float64", "torch.float64"),
        ("torch.uint8", "torch.uint8"),
        ("torch.int16", "torch.int16"),
        ("torch.int8", "torch.int8"),
        ("torch.int32", "torch.int32"),
        ("torch.int64", "torch.int64"),
        ("torch.bool", "torch.bool"),
        ("torch.complex32", "torch.complex32"),
        ("torch.complex64", "torch.complex64"),
        ("torch.complex128", "torch.complex128"),
    ],
)
def test_cast_dtype_with_string(dtype_str, expected):
    assert cast_dtype(dtype_str) == expected


@pytest.mark.parametrize(
    "invalid_dtype",
    [
        "nonexistent_dtype",
        123,  # Non-string, non-dtype value
        [],  # Another non-string, non-dtype value
    ],
)
def test_cast_dtype_invalid(invalid_dtype):
    if isinstance(invalid_dtype, str):
        with pytest.raises(ValueError):
            cast_dtype(invalid_dtype)
    else:
        with pytest.raises(TypeError):
            cast_dtype(invalid_dtype)


@pytest.mark.parametrize(
    "input_shape, expected_output",
    [
        (None, "None"),
        ([1, 2, 3], "[1, 2, 3]"),
        ([10, 20], "[10, 20]"),
        ("1, 2, 3", "1, 2, 3"),  # Direct string input
        ("[10, 20]", "[10, 20]"),  # String representation of a list
    ],
)
def test_cast_shape_valid(input_shape, expected_output):
    assert cast_shape(input_shape) == expected_output


@pytest.mark.parametrize(
    "invalid_shape",
    [
        [1, "two", 3],  # Mixed types, should raise ValueError
        ["1", "2", "3"],  # All strings, should raise ValueError
        {},  # Wrong type (dict), should raise TypeError
        (1, 2, 3),  # Wrong type (tuple), should raise TypeError
    ],
)
def test_cast_shape_invalid(invalid_shape):
    if isinstance(invalid_shape, list) and any(
        isinstance(item, str) for item in invalid_shape
    ):
        with pytest.raises(ValueError):
            cast_shape(invalid_shape)
    else:
        with pytest.raises(TypeError):
            cast_shape(invalid_shape)


@pytest.mark.parametrize(
    "valid_str_shape",
    [
        "[]",  # Empty list as a string
        "[100]",  # Single element list as a string
        "[2, 2, 2]",  # Multiple elements list as a string
    ],
)
def test_cast_shape_str_valid(valid_str_shape):
    assert cast_shape(valid_str_shape) == valid_str_shape


@pytest.mark.parametrize(
    "complex_input",
    [
        [100, -100],  # Negative numbers
        [2147483647],  # INT_MAX
        [0],  # Zero
    ],
)
def test_cast_shape_complex_valid(complex_input):
    assert cast_shape(complex_input) == str(complex_input)


def test_tensor_scalar_initialization():
    tensor = Tensor(scalar=5, dtype='int32', shape=[1])
    assert tensor.scalar == 5
    assert tensor.dtype == 'int32'
    assert tensor.shape == [1]


def test_tensor_vector_initialization():
    tensor = Tensor(vector=[1, 2, 3], dtype='int32', shape=[3])
    assert tensor.vector == [1, 2, 3]
    assert tensor.dtype == 'int32'
    assert tensor.shape == [3]


def test_tensor_matrix_initialization():
    tensor = Tensor(matrix=[[1, 2], [3, 4]], dtype='float32', shape=[2, 2])
    assert tensor.matrix == [[1, 2], [3, 4]]
    assert tensor.dtype == 'float32'
    assert tensor.shape == [2, 2]

def test_tensor_tensor_initialization():
    np_array = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
    tensor = Tensor(tensor=np_array, dtype='float32', shape=[1, 2, 2])
    assert np.array_equal(tensor.tensor, np_array)
    assert tensor.dtype == 'float32'
    assert tensor.shape == [1, 2, 2]
