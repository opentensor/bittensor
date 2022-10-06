import json
import random
import unittest
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock

import bittensor
from bittensor.utils.fast_sync import FastSync

U64MAX = 18446744073709551615
U32MAX = 4294967295

class TestFastSync(unittest.TestCase):
    def test_load_neurons_from_metagraph_file(self):
        """
        We expect a JSON array of:
        {
            "hotkey": str,
            "coldkey": str,
            "uid": int,
            "ip": str,
            "ip_type": int,
            "port": int,
            "stake": str(int),
            "rank": str(int),
            "emission": str(int),
            "incentive": str(int),
            "consensus": str(int),
            "trust": str(int),
            "dividends": str(int),
            "modality": int,
            "last_update": str(int),
            "version": int,
            "priority": str(int),
            "weights": [
                [int, int],
            ],
            "bonds": [
                [int, str(int)],
            ],
        }
        """

        fake_neurons: List[SimpleNamespace] = [
            SimpleNamespace(
                hotkey="",
                coldkey="",
                uid=0,
                ip="",
                ip_type=0,
                port=0,
                stake=0,
                rank=0,
                emission=str(random.randint(0, 100) * bittensor.__rao_per_tao__),
                incentive=str(random.randint(0, 100) * bittensor.__rao_per_tao__),
                consensus=str(random.randint(0, U64MAX)),
                trust=str(random.randint(0, U64MAX)),
                dividends=str(random.randint(0, U64MAX)),
                modality=0,
                last_update=str(random.randint(0, 10000)),
                version=0,
                priority=str(random.randint(0, U64MAX)),
                weights=[
                    [0, random.randint(0, U32MAX)],
                ],
                bonds=[
                    [0, str(random.randint(0, 100) * bittensor.__rao_per_tao__)],
                ],
            )
        ]

        fake_neuron_json_data = json.dumps(fake_neurons)
        neurons_loaded = FastSync._load_neurons_from_metragraph_file_data(fake_neuron_json_data)

        # Check that the loaded neurons are the same as the fake neurons
        loaded_neuron = neurons_loaded[0].__dict__
        for key, _ in loaded_neuron:
            assert loaded_neuron[key] == fake_neurons[0].__dict__[key]


    def test_verify_metagraph_data(self):
        pass