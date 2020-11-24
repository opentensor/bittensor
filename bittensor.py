import argparse
import bittensor
import os
import time
import asyncio
from  bittensor.utils.asyncio import Asyncio

from bittensor.config import Config

from loguru import logger
from importlib.machinery import SourceFileLoader
from bittensor.subtensor import Keypair

def main():

    # 1. Load Config.
    logger.info('Load Config ...')
    config = Config.load()
    logger.info(config.toJSON())

    # 2. Load Keypair.
    logger.info('Load Keyfile ...')
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
   
    # 3. Load Neuron.
    logger.info('Load Neuron ... ')
    neuron_module = SourceFileLoader("Neuron", os.getcwd() + '/' + config.neuron.neuron_path + '/neuron.py').load_module()
    neuron = neuron_module.Neuron( config )

    # 4. Load Session.
    logger.info('Build Session ... ')
    session = bittensor.init(config, keypair)

    # 5. Start Neuron.
    logger.info('Start ... ')
    with session:
        Asyncio.init()
        Asyncio.start_in_thread(neuron.start, session)
        Asyncio.run_forever()








if __name__ == "__main__":
    main()
