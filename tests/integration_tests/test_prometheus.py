import bittensor

import pytest
import unittest
from unittest.mock import MagicMock



class TestPrometheus(unittest.TestCase):

    def setUp(self):
        class success():
            def __init__(self):
                self.is_success = True
            def process_events(self):
                return True
        class fail():
            def __init__(self):
                self.is_success = False
                self.error_message = 'Mock failure'
            def process_events(self):
                return True
        self.subtensor = bittensor.subtensor(network = 'mock')
        self.wallet = bittensor.wallet.mock()
        self.success = success()
        self.fail = fail()

    def test_init_prometheus_success(self):
        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = self.success)
        assert bittensor.prometheus(wallet = self.wallet, subtensor = self.subtensor, netuid=3)

    def test_init_prometheus_failed(self):
        self.subtensor.substrate.submit_extrinsic = MagicMock(return_value = self.fail)
        with pytest.raises(Exception):
            bittensor.prometheus(wallet = self.wallet, subtensor = self.subtensor, netuid=3)
