# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation
# Copyright © 2023 Opentensor Technologies Inc

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

import pytest
import unittest
import bittensor
from typing import Any
from unittest import IsolatedAsyncioTestCase
from starlette.requests import Request
from unittest.mock import MagicMock
from bittensor.axon import AxonMiddleware


def test_attach():
    # Create a mock AxonServer instance
    server = bittensor.axon()

    # Define the Synapse type
    class Synapse(bittensor.Synapse):
        pass

    # Define the functions with the correct signatures
    def forward_fn(synapse: Synapse) -> Any:
        pass

    def blacklist_fn(synapse: Synapse) -> bool:
        return True

    def priority_fn(synapse: Synapse) -> float:
        return 1.0

    def verify_fn(synapse: Synapse) -> None:
        pass

    # Test attaching with correct signatures
    server.attach(forward_fn, blacklist_fn, priority_fn, verify_fn)

    # Define functions with incorrect signatures
    def wrong_blacklist_fn(synapse: Synapse) -> int:
        return 1

    def wrong_priority_fn(synapse: Synapse) -> int:
        return 1

    def wrong_verify_fn(synapse: Synapse) -> bool:
        return True

    # Test attaching with incorrect signatures
    with pytest.raises(AssertionError):
        server.attach(forward_fn, wrong_blacklist_fn, priority_fn, verify_fn)

    with pytest.raises(AssertionError):
        server.attach(forward_fn, blacklist_fn, wrong_priority_fn, verify_fn)

    with pytest.raises(AssertionError):
        server.attach(forward_fn, blacklist_fn, priority_fn, wrong_verify_fn)


def test_attach():
    # Create a mock AxonServer instance
    server = bittensor.axon()

    # Define the Synapse type
    class Synapse:
        pass

    # Define a class that inherits from Synapse
    class InheritedSynapse(bittensor.Synapse):
        pass

    # Define a function with the correct signature
    def forward_fn(synapse: InheritedSynapse) -> Any:
        pass

    # Test attaching with correct signature and inherited class
    server.attach(forward_fn)

    # Define a class that does not inherit from Synapse
    class NonInheritedSynapse:
        pass

    # Define a function with an argument of a class not inheriting from Synapse
    def wrong_forward_fn(synapse: NonInheritedSynapse) -> Any:
        pass

    # Test attaching with incorrect class inheritance
    with pytest.raises(AssertionError):
        server.attach(wrong_forward_fn)


def test_log_and_handle_error():
    from bittensor.axon import log_and_handle_error

    synapse = SynapseMock()

    synapse = log_and_handle_error(synapse, Exception("Error"), 500, 100)
    assert synapse.axon.status_code == 500
    assert synapse.axon.status_message == "Error"
    assert synapse.axon.process_time is not None


def test_create_error_response():
    from bittensor.axon import create_error_response

    synapse = SynapseMock()
    synapse.axon.status_code = 500
    synapse.axon.status_message = "Error"

    response = create_error_response(synapse)
    assert response.status_code == 500
    assert response.body == b'{"message":"Error"}'


# Mock synapse class for testing


class AxonMock:
    def __init__(self):
        self.status_code = None
        self.forward_class_types = {}
        self.blacklist_fns = {}
        self.priority_fns = {}
        self.forward_fns = {}
        self.verify_fns = {}
        self.thread_pool = bittensor.PriorityThreadPoolExecutor(max_workers=1)


class SynapseMock(bittensor.Synapse):
    pass


def verify_fn_pass(synapse):
    pass


def verify_fn_fail(synapse):
    raise Exception("Verification failed")


def blacklist_fn_pass(synapse):
    return False, ""


def blacklist_fn_fail(synapse):
    return True, ""


def priority_fn_pass(synapse) -> float:
    return 0.0


def priority_fn_timeout(synapse) -> float:
    return 2.0


@pytest.fixture
def middleware():
    # Mock AxonMiddleware instance with empty axon object
    axon = AxonMock()
    return AxonMiddleware(None, axon)


@pytest.mark.asyncio
async def test_verify_pass(middleware):
    synapse = SynapseMock()
    middleware.axon.verify_fns = {"SynapseMock": verify_fn_pass}
    await middleware.verify(synapse)
    assert synapse.axon.status_code != 401


@pytest.mark.asyncio
async def test_verify_fail(middleware):
    synapse = SynapseMock()
    middleware.axon.verify_fns = {"SynapseMock": verify_fn_fail}
    with pytest.raises(Exception):
        await middleware.verify(synapse)
    assert synapse.axon.status_code == 401


@pytest.mark.asyncio
async def test_blacklist_pass(middleware):
    synapse = SynapseMock()
    middleware.axon.blacklist_fns = {"SynapseMock": blacklist_fn_pass}
    await middleware.blacklist(synapse)
    assert synapse.axon.status_code != 403


@pytest.mark.asyncio
async def test_blacklist_fail(middleware):
    synapse = SynapseMock()
    middleware.axon.blacklist_fns = {"SynapseMock": blacklist_fn_fail}
    with pytest.raises(Exception):
        await middleware.blacklist(synapse)
    assert synapse.axon.status_code == 403


@pytest.mark.asyncio
async def test_priority_pass(middleware):
    synapse = SynapseMock()
    middleware.axon.priority_fns = {"SynapseMock": priority_fn_pass}
    await middleware.priority(synapse)
    assert synapse.axon.status_code != 408


class TestAxonMiddleware(IsolatedAsyncioTestCase):
    def setUp(self):
        # Create a mock app
        self.mock_app = MagicMock()
        # Create a mock axon
        self.mock_axon = MagicMock()
        self.mock_axon.uuid = "1234"
        self.mock_axon.forward_class_types = {
            "request_name": bittensor.Synapse,
        }
        self.mock_axon.wallet.hotkey.sign.return_value = bytes.fromhex("aabbccdd")
        # Create an instance of AxonMiddleware
        self.axon_middleware = AxonMiddleware(self.mock_app, self.mock_axon)
        return self.axon_middleware

    @pytest.mark.asyncio
    async def test_preprocess(self):
        # Mock the request
        request = MagicMock(spec=Request)
        request.url.path = "/request_name"
        request.client.port = "5000"
        request.client.host = "192.168.0.1"
        request.headers = {}

        synapse = await self.axon_middleware.preprocess(request)

        # Check if the preprocess function fills the axon information into the synapse
        assert synapse.axon.version == str(bittensor.__version_as_int__)
        assert synapse.axon.uuid == "1234"
        assert synapse.axon.nonce is not None
        assert synapse.axon.status_message == "Success"
        assert synapse.axon.status_code == "100"
        assert synapse.axon.signature == "0xaabbccdd"

        # Check if the preprocess function fills the dendrite information into the synapse
        assert synapse.dendrite.port == "5000"
        assert synapse.dendrite.ip == "192.168.0.1"

        # Check if the preprocess function sets the request name correctly
        assert synapse.name == "request_name"


if __name__ == "__main__":
    unittest.main()
