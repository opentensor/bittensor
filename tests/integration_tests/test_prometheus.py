import bittensor
import torch
import unittest
import pytest 
from bittensor._subtensor.subtensor_mock import mock_subtensor


class TestMetagraph(unittest.TestCase):

    def setUp(self):
        self.wallet = bittensor.wallet.mock()
    
    def test_init_prometheus_success(self):
        bittensor.prometheus(wallet = self.wallet)
    
    def test_init_prometheus_missing_param(self):
        # type error missing wallet param
        with pytest.raises(Exception):
            bittensor.prometheus()
