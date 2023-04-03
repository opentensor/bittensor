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

from typing import Union, Optional
from bittensor import Balance, NeuronInfo, AxonInfo, PrometheusInfo, Keypair, __ss58_format__
from scalecodec import ss58_encode
from rich.console import Console
from rich.text import Text

from Crypto.Hash import keccak

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


def get_mock_keypair( uid: int, test_name: Optional[str] = None ) -> Keypair:
    """
    Returns a mock keypair from a uid and optional test_name.
    If test_name is not provided, the uid is the only seed.
    If test_name is provided, the uid is hashed with the test_name to create a unique seed for the test.
    """
    if test_name is not None:
        hashed_test_name: bytes = keccak.new(digest_bits=256, data=test_name.encode('utf-8')).digest()
        hashed_test_name_as_int: int = int.from_bytes(hashed_test_name, byteorder='big', signed=False)
        uid = uid + hashed_test_name_as_int

    return Keypair.create_from_seed( seed_hex = int.to_bytes(uid, 32, 'big', signed=False), ss58_format = __ss58_format__)

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
                "validator_trust": 0.0,
                "incentive":0.0,
                "dividends":0.0,
                "emission":0.0,
                "bonds":[],
                "weights":[],
                "stake_dict": {},
                "pruning_score": 0.0,
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

def get_mock_neuron_by_uid( uid: int, **kwargs ) -> NeuronInfo:
    return get_mock_neuron(
        uid = uid,
        hotkey = get_mock_hotkey(uid),
        coldkey = get_mock_coldkey(uid),
        **kwargs
    )

class MockStatus:
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def start(self):
        pass

    def stop(self):
        pass

class MockConsole:
    """
    Mocks the console object for status and print.
    Captures the last print output as a string.
    """
    captured_print = None

    def status(self, *args, **kwargs):
        return MockStatus()
    
    def print(self, *args, **kwargs):
        console = Console(width = 1000, no_color=True, markup=False) # set width to 1000 to avoid truncation
        console.begin_capture()
        console.print(*args, **kwargs)
        self.captured_print = console.end_capture()

    def clear(self, *args, **kwargs):
        pass

    @staticmethod
    def remove_rich_syntax(text: str) -> str:
        """
        Removes rich syntax from the given text.
        Removes markup and ansi syntax.
        """
        output_no_syntax = Text.from_ansi(
            Text.from_markup(
                text
            ).plain
        ).plain

        return output_no_syntax