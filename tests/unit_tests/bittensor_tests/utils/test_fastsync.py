import unittest
from unittest.mock import patch
import pytest
import bittensor
try:
    from subtensorapi.exceptions import *
    fastsync_imported = True
except ImportError:
    fastsync_imported = False

class TestFailureAndFallback(unittest.TestCase):
    def setUp(self) -> None:
        if not fastsync_imported:
            self.skipTest("FastSync not installed")

    def test_fast_sync_fails_fallback_to_regular_sync(self):
        config = bittensor.Config()
        config.subtensor = bittensor.Config()
        config.subtensor.use_fast_sync = True
        mock_self_subtensor = bittensor.subtensor(_mock=True, config=config)

        class ExitEarly(BaseException):
            pass
        
        with patch("bittensor.Subtensor.get_n", return_value=4096): # make sure it has neurons
            with patch('bittensor.Subtensor.neuron_for_uid', side_effect=ExitEarly): # raise an ExitEarly exception when neuron_for_uid is called
                with patch("subtensorapi.FastSync.verify_fast_sync_support", side_effect=FastSyncOSNotSupportedException): # mock OS not supported
                    with pytest.raises(ExitEarly): # neuron_for_uid should be called because fast sync failed due to OS not being supported
                        mock_self_subtensor.neurons()
                mock_self_subtensor.use_fast_sync = True
                
                with patch("subtensorapi.FastSync.verify_fast_sync_support", side_effect=FastSyncNotFoundException): # mock binary not found
                    with pytest.raises(ExitEarly): # neuron_for_uid should be called because fast sync failed due to binary not being found
                        mock_self_subtensor.neurons()
                mock_self_subtensor.use_fast_sync = True

                with patch("subtensorapi.FastSync.verify_fast_sync_support", return_value=None): # mock support check passes

                    with patch("subtensorapi.FastSync.sync_fd", side_effect=FastSyncRuntimeException): # mock fast sync runtime error
                        with pytest.raises(ExitEarly): # neuron_for_uid should be called because fast sync failed due to runtime error
                            mock_self_subtensor.neurons()
                    mock_self_subtensor.use_fast_sync = True

class TestFeatureFlag(unittest.TestCase):
    def setUp(self) -> None:
        if not fastsync_imported:
            self.skipTest("FastSync not installed")

    def test_feature_off(self):
        mock_self_subtensor = bittensor.subtensor(_mock=True)
        mock_self_subtensor.use_fast_sync = False # feature flag is off

        class ExitEarly(BaseException):
            pass

        class MockException(BaseException):
            pass
        
        with patch("subtensorapi.FastSync.verify_fast_sync_support", side_effect=MockException): # fail if fast_sync support is checked

            with patch("bittensor.Subtensor.get_n", return_value=4096): # make sure it has neurons
                with patch('bittensor.Subtensor.neuron_for_uid', side_effect=ExitEarly):
                    with pytest.raises(ExitEarly): # neuron_for_uid should be called because feature flag is off
                        mock_self_subtensor.neurons()
    
            mock_self_subtensor.use_fast_sync = False

            with patch('bittensor.Subtensor.blockAtRegistration_all_pysub', side_effect=ExitEarly):
                with pytest.raises(ExitEarly): # blockAtRegistration_all_pysub should be called because feature flag is off
                    mock_self_subtensor.blockAtRegistration_all()

    def test_feature_on(self):
        mock_self_subtensor = bittensor.subtensor(_mock=True)
        mock_self_subtensor.use_fast_sync = True # feature flag is on

        class ExitEarly(BaseException):
            pass

        class MockException(BaseException):
            pass
        
        with patch("subtensorapi.FastSync.verify_fast_sync_support", side_effect=ExitEarly): # fail if fast_sync support is checked

            with patch("bittensor.Subtensor.get_n", return_value=4096): # make sure it has neurons
                with patch('bittensor.Subtensor.neuron_for_uid', side_effect=MockException):
                    # neuron_for_uid should not called because feature flag is off
                    with pytest.raises(ExitEarly): # should check fast_sync support
                        mock_self_subtensor.neurons()
    
            mock_self_subtensor.use_fast_sync = True

            with patch('bittensor.Subtensor.blockAtRegistration_all_pysub', side_effect=MockException):
                # blockAtRegistration_all_pysub should not called because feature flag is off
                with pytest.raises(ExitEarly): # should check fast_sync support
                    mock_self_subtensor.blockAtRegistration_all()


if __name__ == '__main__':
    unittest.main()