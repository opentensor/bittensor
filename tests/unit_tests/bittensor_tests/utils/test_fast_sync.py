import json
import random
import unittest
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, patch

import bittensor
import pytest
from bittensor.utils.fast_sync import (FastSync, FastSyncFileException,
                                       FastSyncFormatException,
                                       FastSyncNotFoundException,
                                       FastSyncOSNotSupportedException,
                                       FastSyncRuntimeException)

U64MAX = 18446744073709551615
U32MAX = 4294967295
RAOPERTAO = bittensor.__rao_per_tao__

class TestLoadNeurons(unittest.TestCase):
    def test_load_neurons_from_metagraph_file(self):
        """
        We expect a JSON array of:
        {
            "hotkey": str,
            "coldkey": str,
            "uid": int,
            "active": int,
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
                hotkey="5FTWCbNmsywinyF38vFRDJRKHono5ssbtzEq8naPbSPpkpnd",
                coldkey="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm",
                uid=0,
                active=0,
                ip="",
                ip_type=0,
                port=0,
                stake="0",
                rank="0",
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
            ).__dict__
        ]

        # create a JSON string of the list
        fake_neuron_json_data = json.dumps(fake_neurons)
        # load the neurons from the JSON string
        neurons_loaded = FastSync._load_neurons_from_metragraph_file_data(fake_neuron_json_data)

        # get the fake_neuron and adjust the values as needed
        fake_neuron = fake_neurons[0]
        fake_neuron['stake'] = int(fake_neuron['stake']) / RAOPERTAO
        fake_neuron['rank'] = int(fake_neuron['rank']) / U64MAX
        fake_neuron['emission'] = int(fake_neuron['emission']) / RAOPERTAO
        fake_neuron['incentive'] = int(fake_neuron['incentive']) / U64MAX
        fake_neuron['consensus'] = int(fake_neuron['consensus']) / U64MAX
        fake_neuron['trust'] = int(fake_neuron['trust']) / U64MAX
        fake_neuron['dividends'] = int(fake_neuron['dividends']) / U64MAX

        fake_neuron['last_update'] = int(fake_neuron['last_update'])
        fake_neuron['priority'] = int(fake_neuron['priority'])
        fake_neuron['bonds'] = [ [bond[0], int(bond[1])] for bond in fake_neuron['bonds'] ]

        # Check that the loaded neurons are the same as the fake neurons
        loaded_neuron = neurons_loaded[0].__dict__
        for key in loaded_neuron:
            assert loaded_neuron[key] == fake_neuron[key]

    def test_load_neurons_from_metagraph_file_bad_data_missing_fields(self):
        fake_neurons: List[SimpleNamespace] = [
            SimpleNamespace(
                #hotkey="5FTWCbNmsywinyF38vFRDJRKHono5ssbtzEq8naPbSPpkpnd",
                #coldkey="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm", # Missing hotkey and coldkey
                uid=0,
                active=0,
                ip="",
                ip_type=0,
                port=0,
                stake="0",
                rank="0",
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
            ).__dict__
        ]

        fake_neuron_json_data = json.dumps(fake_neurons)
        with pytest.raises(FastSyncFormatException):
            _ = FastSync._load_neurons_from_metragraph_file_data(fake_neuron_json_data)

    def test_load_neurons_from_metagraph_file_bad_data_bad_numbers(self):
        fake_neurons: List[SimpleNamespace] = [
            SimpleNamespace(
                hotkey="5FTWCbNmsywinyF38vFRDJRKHono5ssbtzEq8naPbSPpkpnd",
                coldkey="5DD26kC2kxajmwfbbZmVmxhrY9VeeyR1Gpzy9i8wxLUg6zxm", 
                uid=0,
                active=0,
                ip="",
                ip_type=0,
                port=0,
                stake="0",
                rank="0",
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
            ).__dict__
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
        with patch("bittensor.utils.fast_sync.FastSync.get_platform", return_value="win32"): # Windows is not supported
            with pytest.raises(FastSyncOSNotSupportedException):
                _ = FastSync.verify_os_support()

    def test_os_not_supported_other(self):
        with patch("bittensor.utils.fast_sync.FastSync.get_platform", return_value="someotheros"): # Some other os that is not supported
            with pytest.raises(FastSyncOSNotSupportedException):
                _ = FastSync.verify_os_support()

    def test_os_not_supported_freebsd(self):
        with patch("bittensor.utils.fast_sync.FastSync.get_platform", return_value="freebsd"): # freebsd is not supported
            with pytest.raises(FastSyncOSNotSupportedException):
                _ = FastSync.verify_os_support()
    
    def test_os_supported_linux(self):
        with patch("bittensor.utils.fast_sync.FastSync.get_platform", return_value="linux"): # linux is supported
            _ = FastSync.verify_os_support()

    def test_os_supported_macos(self):
        with patch("bittensor.utils.fast_sync.FastSync.get_platform", return_value="darwin"): # darwin is macos and is supported
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

class TestFailureAndFallback(unittest.TestCase):
    def test_fast_sync_fails_fallback_to_regular_sync(self):
        mock_self_subtensor = bittensor.subtensor(_mock=True)
        mock_self_subtensor.use_fast_sync = True

        class ExitEarly(Exception):
            pass
        
        with patch("bittensor.Subtensor.get_n", return_value=4096): # make sure it has neurons
            with patch('bittensor.Subtensor.neuron_for_uid', side_effect=ExitEarly): # raise an ExitEarly exception when neuron_for_uid is called
                with patch("bittensor.utils.fast_sync.FastSync.verify_fast_sync_support", side_effect=FastSyncOSNotSupportedException): # mock OS not supported
                    with pytest.raises(ExitEarly): # neuron_for_uid should be called because fast sync failed due to OS not being supported
                        mock_self_subtensor.neurons()
                mock_self_subtensor.use_fast_sync = True
                
                with patch("bittensor.utils.fast_sync.FastSync.verify_fast_sync_support", side_effect=FastSyncNotFoundException): # mock binary not found
                    with pytest.raises(ExitEarly): # neuron_for_uid should be called because fast sync failed due to binary not being found
                        mock_self_subtensor.neurons()
                mock_self_subtensor.use_fast_sync = True

                with patch("bittensor.utils.fast_sync.FastSync.verify_fast_sync_support", return_value=None): # mock support check passes

                    with patch("bittensor.utils.fast_sync.FastSync.sync_neurons", side_effect=FastSyncRuntimeException): # mock fast sync runtime error
                        with pytest.raises(ExitEarly): # neuron_for_uid should be called because fast sync failed due to runtime error
                            mock_self_subtensor.neurons()
                    mock_self_subtensor.use_fast_sync = True

                    with patch("bittensor.utils.fast_sync.FastSync.sync_neurons", return_value=None): # mock sync succeeds
                        with patch("bittensor.utils.fast_sync.FastSync.load_neurons", side_effect=FastSyncFormatException): # mock fast sync format error
                            with pytest.raises(ExitEarly): # neuron_for_uid should be called because fast sync failed due to format error
                                mock_self_subtensor.neurons()

if __name__ == '__main__':
    unittest.main()