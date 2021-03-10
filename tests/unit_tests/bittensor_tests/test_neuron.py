
from loguru import logger
from munch import Munch
import bittensor
import pytest
from unittest.mock import MagicMock

def test_create_neuron():
    bittensor.init()
    assert bittensor.neuron != None

def test_assert_components():
    neuron = bittensor.neuron
    assert neuron.subtensor != None
    assert neuron.wallet != None
    assert neuron.config != None
    assert neuron.metagraph != None
    assert neuron.dendrite != None
    assert neuron.axon != None