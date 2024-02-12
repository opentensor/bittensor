



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
