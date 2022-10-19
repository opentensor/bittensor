import unittest
from unittest.mock import patch
import pytest
import bittensor
from subtensorapi import FastSync
from subtensorapi.exceptions import *

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