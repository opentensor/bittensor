
from loguru import logger
import bittensor
from bittensor.config import Config
from bittensor.subtensor import Keypair

def new_session():# 1. Init Config item.
    config = Config.load()
    Config.toString(config)
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
    session = bittensor.init(config, keypair)
    return session

def test_new_session():
    new_session()

if __name__ == "__main__":
    test_new_session()