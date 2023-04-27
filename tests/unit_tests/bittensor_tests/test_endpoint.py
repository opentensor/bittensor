# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

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

import random

import pytest
import torch

import bittensor
from bittensor.utils.test_utils import get_random_unused_port

test_wallet = bittensor.wallet.mock()

def test_create_endpoint():
    endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '0.0.0.0',
        ip_type = 4,
        port = 12345,
        hotkey = test_wallet.hotkey.ss58_address,
        coldkey = test_wallet.coldkey.ss58_address,
        modality = 0,
        protocol = 0,
    )
    assert endpoint.check_format() == True
    assert endpoint.assert_format()
    assert endpoint.version == bittensor.__version_as_int__
    assert endpoint.uid == 0
    assert endpoint.ip == '0.0.0.0'
    assert endpoint.ip_type == 4
    assert endpoint.port == 12345
    assert endpoint.hotkey == test_wallet.hotkey.ss58_address
    assert endpoint.coldkey == test_wallet.coldkey.ss58_address
    assert endpoint.modality == 0

def test_endpoint_fails_checks():
    with pytest.raises(AssertionError):
        test_endpoint = bittensor.endpoint(
            version = bittensor.__version_as_int__,
            uid = -1,
            ip = '0.0.0.0',
            ip_type = 4,
            port = 12345,
            hotkey = test_wallet.hotkey.ss58_address,
            coldkey = test_wallet.coldkey.ss58_address,
            modality = 0,
            protocol = 0,
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
    endpoint = bittensor.endpoint(
        version = bittensor.__version_as_int__,
        uid = 0,
        ip = '0.0.0.0',
        ip_type = 4,
        port = get_random_unused_port(),
        hotkey = test_wallet.hotkey.ss58_address,
        coldkey = test_wallet.coldkey.ss58_address,
        modality = 0,
        protocol = 0,
    )
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
            modality = 0,
            protocol = 0,
        )
        assert new_endpoint.check_format() == True
        tensor_endpoint = new_endpoint.to_tensor()
        assert list(tensor_endpoint.shape) == [250]
        converted_endpoint = bittensor.endpoint.from_tensor( tensor_endpoint )
        assert converted_endpoint == new_endpoint
        assert torch.equal(tensor_endpoint, converted_endpoint.to_tensor())





