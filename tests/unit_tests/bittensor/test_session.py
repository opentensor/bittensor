
from loguru import logger
import bittensor
from bittensor.config import Config
from substrateinterface import SubstrateInterface, Keypair

def new_session():# 1. Init Config item.
    config = Config.load()
    Config.toString(config)
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
    session = bittensor.init(config, keypair)
    return session

def test_new_session():
    new_session()

def test_new_session_start_stop():
    session = new_session()
    session.start()
    session.stop()

def test_new_session_subscribe_unsubscribe():
    with new_session() as session:
        logger.info('with sess {}', session)


if __name__ == "__main__":
    #test_new_session()
    #test_new_session_start_stop()
    test_new_session_subscribe_unsubscribe()