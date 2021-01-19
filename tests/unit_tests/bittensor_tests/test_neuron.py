
from loguru import logger
import bittensor
from bittensor.neuron import Neuron
from bittensor.config import Config
from bittensor.subtensor.interface import Keypair
from munch import Munch

neuron = None

def test_create_neuron():
    neuron = Neuron()

if __name__ == "__main__":
    test_create_neuron()