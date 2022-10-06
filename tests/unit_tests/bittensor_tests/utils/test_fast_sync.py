import json
import random
import unittest
from types import SimpleNamespace
from typing import List
from unittest.mock import patch
import pytest

import bittensor
from bittensor.utils.fast_sync import FastSync, FastSyncFormatException, FastSyncFileException, FastSyncOSNotSupportedException, FastSyncNotFoundException

U64MAX = 18446744073709551615
U32MAX = 4294967295

class TestLoadNeurons(unittest.TestCase):
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

    def test_load_neurons_from_metagraph_file_bad_data_missing_fields(self):
        fake_neurons: List[SimpleNamespace] = [
            SimpleNamespace(
                #hotkey="",
                #coldkey="", # Missing hotkey and coldkey
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
        with pytest.raises(FastSyncFormatException):
            _ = FastSync._load_neurons_from_metragraph_file_data(fake_neuron_json_data)

    def test_load_neurons_from_metagraph_file_bad_data_bad_numbers(self):
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
                consensus=123, # should be str
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
        with pytest.raises(FastSyncFormatException):
            _ = FastSync._load_neurons_from_metragraph_file_data(fake_neuron_json_data)
    
    def test_load_neurons_from_metagraph_file_json_error(self):
        bad_json_string = "bad json string"
        with pytest.raises(FastSyncFormatException):
            _ = FastSync._load_neurons_from_metragraph_file_data(bad_json_string)

    def test_load_neurons_file_os_error(self):
        with patch("builtins.open", side_effect=OSError):
            with pytest.raises(FastSyncFileException):
                _ = FastSync.load_neurons("") # Should raise an OSError


    def test_load_neurons_from_metagraph_file_no_file(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            with pytest.raises(FastSyncFileException):
                _ = FastSync.load_neurons("") # Should raise a FileNotFoundError
       
class TestSupportCheck(unittest.TestCase):
    def test_os_not_supported_windows(self):
        with patch("FastSync.platform", return_value="win32"): # Windows is not supported
            with pytest.raises(FastSyncOSNotSupportedException):
                _ = FastSync.verify_os_support()

    def test_os_not_supported_other(self):
        with patch("FastSync.platform", return_value="someotheros"): # Some other os that is not supported
            with pytest.raises(FastSyncOSNotSupportedException):
                _ = FastSync.verify_os_support()

    def test_os_not_supported_freebsd(self):
        with patch("FastSync.platform", return_value="freebsd"): # freebsd is not supported
            with pytest.raises(FastSyncOSNotSupportedException):
                _ = FastSync.verify_os_support()
    
    def test_os_supported_linux(self):
        with patch("FastSync.platform", return_value="linux"): # linux is supported
            _ = FastSync.verify_os_support()

    def test_os_supported_macos(self):
        with patch("FastSync.platform", return_value="darwin"): # darwin is macos and is supported
            _ = FastSync.verify_os_support()
    
    def test_binary_not_found(self):
        with patch("os.path.exists", return_value=False): # Binary does not exist
            with pytest.raises(FastSyncNotFoundException):
                _ = FastSync.verify_binary_exists()

        with patch("os.path.exists", return_value=True):
            with patch("os.path.isfile", return_value=False): # Binary exists but is not a file
                with pytest.raises(FastSyncNotFoundException):
                    _ = FastSync.verify_binary_exists()
        
    def test_binary_found(self):
        with patch("os.path.exists", return_value=True):
            with patch("os.path.isfile", return_value=True):
                _ = FastSync.verify_binary_exists() # no exception should be raised
