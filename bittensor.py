import argparse
import bittensor
import os
import time
import asyncio
from  bittensor.utils.asyncio import Asyncio

from loguru import logger
from importlib.machinery import SourceFileLoader
from bittensor.subtensor import Keypair

from bittensor.config import Config

def main():

    # 1. Load passed --neuron_path for custom config.
    parser = argparse.ArgumentParser()
    parser.add_argument('--neuron_path', required=True, help='directoty path to a neuron.py class.')
    params = parser.parse_known_args()[0]

    # 2. Load Config.
    logger.info('Load Config ...')
    config = Config.load(params.neuron_path)
    logger.info(Config.toString(config))

    # 3. Load Keypair.
    logger.info('Load Keyfile ...')
    mnemonic = Keypair.generate_mnemonic()
    keypair = Keypair.create_from_mnemonic(mnemonic)
   
    # 4. Load Neuron.
    logger.info('Load Neuron ... ')
    neuron_module = SourceFileLoader("Neuron", os.getcwd() + '/' + params.neuron_path + '/neuron.py').load_module()
    neuron = neuron_module.Neuron( config )

    # 5. Load Session.
    logger.info('Build Session ... ')
    session = bittensor.init(config, keypair)

    # 6. Start Neuron.
    logger.info('Start ... ')
    with session:
        Asyncio.init()
        Asyncio.start_in_thread(neuron.start, session)
        Asyncio.run_forever()








if __name__ == "__main__":
    main()
