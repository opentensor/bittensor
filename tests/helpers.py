# The MIT License (MIT)
# Copyright © 2023 Opentensor Foundation

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

from typing import Union
from bittensor import Balance, NeuronInfo, AxonInfo, PrometheusInfo, Keypair, __ss58_format__
from scalecodec import ss58_encode

class CLOSE_IN_VALUE():
    value: Union[float, int, Balance]
    tolerance: Union[float, int, Balance]

    def __init__(self, value: Union[float, int, Balance], tolerance: Union[float, int, Balance] = 0.0) -> None:
        self.value = value
        self.tolerance = tolerance

    def __eq__(self, __o: Union[float, int, Balance]) -> bool:
        # True if __o \in [value - tolerance, value + tolerance]
        # or if value \in [__o - tolerance, __o + tolerance]
        return ((self.value - self.tolerance) <= __o and __o <= (self.value + self.tolerance)) or \
                ((__o - self.tolerance) <= self.value and self.value <= (__o + self.tolerance))


def get_mock_keypair( uid: int ) -> Keypair:
    return Keypair(ss58_encode(int.to_bytes(uid, 32, 'big', signed=False), __ss58_format__))

def get_mock_hotkey( uid: int ) -> str:
    return get_mock_keypair(uid).ss58_address

def get_mock_coldkey( uid: int ) -> str:
    return get_mock_keypair(uid).ss58_address

def get_mock_neuron(**kwargs) -> NeuronInfo:
    """
    Returns a mock neuron with the given kwargs overriding the default values.
    """

    mock_neuron_d = dict({
                "netuid": -1, # mock netuid
                "axon_info": AxonInfo(
                    block = 0,
                    version = 1,
                    ip = 0,
                    port = 0,
                    ip_type = 0,
                    protocol = 0,
                    placeholder1 = 0,
                    placeholder2 = 0
                ),
                "prometheus_info": PrometheusInfo(
                    block = 0,
                    version = 1,
                    ip = 0,
                    port = 0,
                    ip_type = 0
                ),
                "validator_permit": True,
                "uid":1,
                "hotkey":'some_hotkey',
                "coldkey":'some_coldkey',
                "active":0,
                "last_update":0,
                "stake": {
                    "some_coldkey": 1e12
                },
                "total_stake":1e12,
                "rank":0.0,
                "trust":0.0,
                "consensus":0.0,
                "incentive":0.0,
                "dividends":0.0,
                "emission":0.0,
                "bonds":[],
                "weights":[],
                "is_null":False
            })

    mock_neuron_d.update(kwargs) # update with kwargs

    if kwargs.get('stake') is None and kwargs.get('coldkey') is not None:
        mock_neuron_d['stake'] = { kwargs.get('coldkey'): 1e12 }

    if kwargs.get('total_stake') is None:
        mock_neuron_d['total_stake'] = sum(mock_neuron_d['stake'].values())

    mock_neuron = NeuronInfo._neuron_dict_to_namespace(
        mock_neuron_d
    )

    return mock_neuron
