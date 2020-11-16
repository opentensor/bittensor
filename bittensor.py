import argparse
import bittensor
import os
import time

from bittensor.config import Config

from loguru import logger
from importlib.machinery import SourceFileLoader
from substrateinterface import SubstrateInterface, Keypair

def main():

    # 1. Init Config item.
    logger.info('Load config.')
    try:
        config = Config.load()
    except Exception as e:
        logger.error("Invalid configuration. Aborting with error {}", e)
        quit(-1)
    logger.info('Config: {')
    Config.toString(config)
    logger.info('} \n')

    
    # 2. Load keypair.
    logger.info('Load keyfile')
    # TODO(const): check path to keypem exists.
    try:
        mnemonic = Keypair.generate_mnemonic()
        keypair = Keypair.create_from_mnemonic(mnemonic)
    except:
        logger.error('Unable to load keypair with error {}', e)
        quit(-1)

    # 3. Load Neuron module.
    logger.info('Load neuron module.')
    full_neuron_path = os.getcwd() + config.neuron.neuron_path + '/neuron.py'
    if not os.path.isfile(full_neuron_path):
        raise FileNotFoundError('Cannot find neuron.py on path:', full_neuron_path)
        quit(-1)
    try:
        neuron_module = SourceFileLoader("Neuron", full_neuron_path).load_module()
    except Exception as e:
        logger.error('Unable to load Neuron module at {} with error {}', full_neuron_path, e)
        quit(-1)

    # 4. Main try catch.
    neuron = None
    session = None
    try:
        # 4.1 Init the bittensor session
        session = bittensor.init(config, keypair)

        # 4.2 Start the bittensor session.
        session.start()
        session.subscribe()

        # 4.3 Start the bittensor neuron.
        neuron = neuron_module.Neuron( config, session )
        neuron.start()
    except Exception as e:
        logger.error("Exception while running neuron occured. Message: {}".format(e))
        raise
    
    # 5. Wrap up and tear down.  
    if neuron != None:
        neuron.stop()
    if session != None:
        session.unsubscribe()
        session.stop()

if __name__ == "__main__":
    main()
