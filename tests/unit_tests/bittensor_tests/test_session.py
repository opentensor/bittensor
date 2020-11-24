
from loguru import logger
import bittensor
from bittensor.config import Config
from substrateinterface import SubstrateInterface, Keypair

def new_session():
    # 1. Init Config item.
    config = Config.load_from_args(neuron_path='bittensor/neurons/mnist')
    logger.info(Config.toString(config))
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
    session = bittensor.init(config, keypair)
    return session

def test_new_session():
    new_session()

if __name__ == "__main__":
    test_new_session()