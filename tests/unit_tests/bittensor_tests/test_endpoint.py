import bittensor 
import torch
import random
import sys

test_wallet = bittensor.wallet (
    path = '/tmp/pytest',
    name = 'pytest',
    hotkey = 'pytest',
)
test_wallet.new_coldkey( use_password=False, overwrite = True )
test_wallet.new_hotkey( use_password=False, overwrite = True )


endpoint = None

def test_create_endpoint():
    global endpoint
    endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '0.0.0.0',
        ip_type = 4,
        port = 12345,
        hotkey = test_wallet.hotkey.ss58_address,
        coldkey = test_wallet.coldkey.ss58_address,
        modality = 0
    )
    assert endpoint.check_format() == True
    endpoint.assert_format()
    assert endpoint.version == bittensor.__version_as_int__
    assert endpoint.uid == 0
    assert endpoint.ip == '0.0.0.0'
    assert endpoint.ip_type == 4
    assert endpoint.port == 12345
    assert endpoint.hotkey == test_wallet.hotkey.ss58_address
    assert endpoint.coldkey == test_wallet.coldkey.ss58_address
    assert endpoint.modality == 0

def test_endpoint_fails_checks():
    test_endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = -1,
        ip = '0.0.0.0',
        ip_type = 4,
        port = 12345,
        hotkey = test_wallet.hotkey.ss58_address,
        coldkey = test_wallet.coldkey.ss58_address,
        modality = 0
    )
    assert test_endpoint.check_format() == False
    test_endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 4294967296,
        ip = '0.0.0.0',
        ip_type = 4,
        port = 12345,
        hotkey = test_wallet.hotkey.ss58_address,
        coldkey = test_wallet.coldkey.ss58_address,
        modality = 0
    )
    assert test_endpoint.check_format() == False
    test_endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '0.0.0.0',
        ip_type = 5,
        port = 12345,
        hotkey = test_wallet.hotkey.ss58_address,
        coldkey = test_wallet.coldkey.ss58_address,
        modality = 0
    )
    assert test_endpoint.check_format() == False
    test_endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '0.0.0.0',
        ip_type = 4,
        port = 12345222,
        hotkey = test_wallet.hotkey.ss58_address,
        coldkey = test_wallet.coldkey.ss58_address,
        modality = 0
    )
    assert test_endpoint.check_format() == False
    test_endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '0.0.0.0',
        ip_type = 4,
        port = 2142,
        hotkey = test_wallet.hotkey.ss58_address + "sssd",
        coldkey = test_wallet.coldkey.ss58_address,
        modality = 0
    )
    assert test_endpoint.check_format() == False
    test_endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '0.0.0.0',
        ip_type = 4,
        port = 2142,
        hotkey = test_wallet.hotkey.ss58_address,
        coldkey = test_wallet.coldkey.ss58_address + "sssd",
        modality = 0
    )
    assert test_endpoint.check_format() == False
    test_endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '0.0.0.0',
        ip_type = 4,
        port = 2142,
        hotkey = test_wallet.hotkey.ss58_address,
        coldkey = test_wallet.coldkey.ss58_address,
        modality = 2
    )
    assert test_endpoint.check_format() == False

def test_endpoint_to_tensor():
    tensor_endpoint = endpoint.to_tensor()
    assert list(tensor_endpoint.shape) == [250]
    converted_endpoint = bittensor.endpoint.from_tensor( tensor_endpoint )
    assert converted_endpoint == endpoint
    assert converted_endpoint.check_format() == True

def test_endpoint_to_tensor():
    tensor_endpoint = endpoint.to_tensor()
    assert list(tensor_endpoint.shape) == [250]
    converted_endpoint = bittensor.endpoint.from_tensor( tensor_endpoint )
    assert converted_endpoint == endpoint
    assert torch.equal(tensor_endpoint, converted_endpoint.to_tensor())
    assert converted_endpoint.check_format() == True

def test_thrash_equality_of_endpoint():
    n_tests = 10000
    for _ in range(n_tests):
        new_endpoint = bittensor.endpoint(
            version = random.randint(0,999),
            uid = random.randint(0,4294967295-1),
            ip = str(random.randint(0,250)) + '.' +  str(random.randint(0,250)) + '.' + str(random.randint(0,250)) + '.' + str(random.randint(0,250)),
            ip_type = random.choice( [4,6] ),
            port = random.randint(0,64000),
            hotkey = test_wallet.hotkey.ss58_address,
            coldkey = test_wallet.coldkey.ss58_address,
            modality = 0
        )
        assert new_endpoint.check_format() == True
        tensor_endpoint = new_endpoint.to_tensor()
        assert list(tensor_endpoint.shape) == [250]
        converted_endpoint = bittensor.endpoint.from_tensor( tensor_endpoint )
        assert converted_endpoint == new_endpoint
        assert torch.equal(tensor_endpoint, converted_endpoint.to_tensor())





