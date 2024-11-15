import json
import os
from collections import deque
import random
from unittest.mock import patch
import websocket

import pytest
from bittensor.core import settings
from bittensor.core.subtensor import Subtensor

with open(os.path.join(os.path.dirname(__file__), '..', 'helpers', 'refined-output.json')) as f:
    WEBSOCKET_DICT = eval(f.read())


class FakeWebsocket(websocket.WebSocket):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.received = deque()

    def send(self, payload: str, *args, **kwargs):
        print("I SENT IT")
        received = json.loads(payload)
        id_ = received.pop("id")
        self.received.append((received, id_))

    def recv(self):
        item, _id = self.received.pop()
        print(item, type(item))
        response = WEBSOCKET_DICT[(item["method"], json.dumps(item["params"]))]
        response["id"] = _id
        return json.dumps(response)

    def close(self, *args, **kwargs):
        pass


@pytest.fixture
def subtensor():
    subtensor_ = Subtensor(websocket=FakeWebsocket())
    yield subtensor_
    print("Subtensor closed")


@pytest.mark.parametrize(
    "input_, runtime_api, method, output",
    [
        (
            [1],
            "NeuronInfoRuntimeApi",
            "get_neurons_lite",
            "0x0100"
        ),
        (
            [],
            "SubnetRegistrationRuntimeApi",
            "get_network_registration_cost",
            "0x"
        )
    ]
)
# def test_encode_params(subtensor, input_, runtime_api, method, output):
#     call_definition = settings.TYPE_REGISTRY["runtime_api"][runtime_api]["methods"][
#         method
#     ]
#     result = subtensor._encode_params(call_definition=call_definition, params=input_)
#     assert result == output


# def test_blocks_since_last_update(subtensor):
#     netuid = 1
#     updates = subtensor._get_hyperparameter(param_name="LastUpdate", netuid=netuid)
#     assert isinstance(updates, (list, None))
#     if updates is not None:
#         uid = random.randint(0, len(updates)-1)
#         query = subtensor.blocks_since_last_update(netuid=netuid, uid=uid)
#         assert isinstance(query, int)


def test_get_all_subnets_info(subtensor):
    result = subtensor.get_all_subnets_info()
    assert isinstance(result, list)
    assert result[0].owner_ss58 == "5C4hrfjw9DjXZTzV3MwzrrAr9P1MJhSrvWGWqi1eSuyUpnhM"
    assert result[1].kappa == 32767
    assert result[1].max_weight_limit == 65535
    assert result[1].blocks_since_epoch == 10


# def test_metagraph(subtensor):
#     result = subtensor.metagraph(23)
#     assert result.n == 19
#     assert result.netuid == 23
#     assert result.block == 3242195


# def test_get_netuids_for_hotkey(subtensor):
#     result = subtensor.get_netuids_for_hotkey("5DkzsviNQr4ZePXMmEfNPDcE7cQ9cVyepmQbgUw6YT3odcwh")
#     assert result == [23]

def test_get_current_block(subtensor):
    result = subtensor.get_current_block()
    assert result == 3242195