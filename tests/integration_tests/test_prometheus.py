import bittensor

import pytest
import unittest
from unittest.mock import MagicMock, patch
from bittensor._subtensor.subtensor_mock import MockSubtensor
from tests.helpers import _get_mock_wallet

_subtensor_mock: MockSubtensor = bittensor.subtensor(network="mock", _mock=True)


def setUpModule():
    _subtensor_mock.reset()

    _subtensor_mock.create_subnet(netuid=3)

    _subtensor_mock.set_difficulty(netuid=3, difficulty=0)


class TestPrometheus(unittest.TestCase):
    def setUp(self):
        self.subtensor = bittensor.subtensor(network="mock")
        self.wallet = _get_mock_wallet()

    def test_init_prometheus_success(self):
        with patch.object(
            self.subtensor, "_do_serve_prometheus", return_value=(True, None)
        ):
            with patch("prometheus_client.start_http_server"):
                self.assertTrue(
                    bittensor.prometheus(
                        wallet=self.wallet, subtensor=self.subtensor, netuid=3
                    )
                )

    def test_init_prometheus_failed(self):
        with patch.object(
            self.subtensor, "_do_serve_prometheus", return_value=(False, "Mock failure")
        ):
            with patch("prometheus_client.start_http_server"):
                with pytest.raises(Exception):
                    bittensor.prometheus(
                        wallet=self.wallet, subtensor=self.subtensor, netuid=3
                    )
