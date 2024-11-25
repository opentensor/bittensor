# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from collections import deque
import json
from typing import Union

from websockets.sync.client import ClientConnection, ClientProtocol
from websockets.uri import parse_uri

from bittensor_wallet.mock.wallet_mock import MockWallet as _MockWallet
from bittensor_wallet.mock.wallet_mock import get_mock_coldkey
from bittensor_wallet.mock.wallet_mock import get_mock_hotkey
from bittensor_wallet.mock.wallet_mock import get_mock_wallet

from bittensor.utils.balance import Balance
from bittensor.core.chain_data import AxonInfo, NeuronInfo, PrometheusInfo
from tests.helpers.integration_websocket_data import WEBSOCKET_RESPONSES


def __mock_wallet_factory__(*_, **__) -> _MockWallet:
    """Returns a mock wallet object."""

    mock_wallet = get_mock_wallet()

    return mock_wallet


class CLOSE_IN_VALUE:
    value: Union[float, int, Balance]
    tolerance: Union[float, int, Balance]

    def __init__(
        self,
        value: Union[float, int, Balance],
        tolerance: Union[float, int, Balance] = 0.0,
    ) -> None:
        self.value = value
        self.tolerance = tolerance

    def __eq__(self, __o: Union[float, int, Balance]) -> bool:
        # True if __o \in [value - tolerance, value + tolerance]
        # or if value \in [__o - tolerance, __o + tolerance]
        return (
            (self.value - self.tolerance) <= __o <= (self.value + self.tolerance)
        ) or ((__o - self.tolerance) <= self.value <= (__o + self.tolerance))


def get_mock_neuron(**kwargs) -> NeuronInfo:
    """
    Returns a mock neuron with the given kwargs overriding the default values.
    """

    mock_neuron_d = dict(
        # TODO fix the AxonInfo here — it doesn't work
        {
            "netuid": -1,  # mock netuid
            "axon_info": AxonInfo(
                block=0,
                version=1,
                ip=0,
                port=0,
                ip_type=0,
                protocol=0,
                placeholder1=0,
                placeholder2=0,
            ),
            "prometheus_info": PrometheusInfo(
                block=0, version=1, ip=0, port=0, ip_type=0
            ),
            "validator_permit": True,
            "uid": 1,
            "hotkey": "some_hotkey",
            "coldkey": "some_coldkey",
            "active": 0,
            "last_update": 0,
            "stake": {"some_coldkey": 1e12},
            "total_stake": 1e12,
            "rank": 0.0,
            "trust": 0.0,
            "consensus": 0.0,
            "validator_trust": 0.0,
            "incentive": 0.0,
            "dividends": 0.0,
            "emission": 0.0,
            "bonds": [],
            "weights": [],
            "stake_dict": {},
            "pruning_score": 0.0,
            "is_null": False,
        }
    )

    mock_neuron_d.update(kwargs)  # update with kwargs

    if kwargs.get("stake") is None and kwargs.get("coldkey") is not None:
        mock_neuron_d["stake"] = {kwargs.get("coldkey"): 1e12}

    if kwargs.get("total_stake") is None:
        mock_neuron_d["total_stake"] = sum(mock_neuron_d["stake"].values())

    mock_neuron = NeuronInfo._neuron_dict_to_namespace(mock_neuron_d)

    return mock_neuron


def get_mock_neuron_by_uid(uid: int, **kwargs) -> NeuronInfo:
    return get_mock_neuron(
        uid=uid, hotkey=get_mock_hotkey(uid), coldkey=get_mock_coldkey(uid), **kwargs
    )


class FakeWebsocket(ClientConnection):
    def __init__(self, *args, seed, **kwargs):
        protocol = ClientProtocol(parse_uri("ws://127.0.0.1:9945"))
        super().__init__(socket=None, protocol=protocol, **kwargs)
        self.seed = seed
        self.received = deque()

    def send(self, payload: str, *args, **kwargs):
        received = json.loads(payload)
        id_ = received.pop("id")
        self.received.append((received, id_))

    def recv(self, *args, **kwargs):
        item, _id = self.received.pop()
        try:
            response = WEBSOCKET_RESPONSES[self.seed][item["method"]][
                json.dumps(item["params"])
            ]
            response["id"] = _id
            return json.dumps(response)
        except (KeyError, TypeError):
            print("ERROR", self.seed, item["method"], item["params"])
            raise

    def close(self, *args, **kwargs):
        pass
