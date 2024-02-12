



# Unit tests for cast_dtype
def test_cast_dtype_none():
    assert cast_dtype(None) is None

def test_cast_dtype_torch_dtype():
    assert cast_dtype(torch.float32) == "float32"

def test_cast_dtype_str_valid():
    assert cast_dtype("float32") == "float32"
